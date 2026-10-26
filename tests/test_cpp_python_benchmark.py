import math
import os
import statistics
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = Path(__file__).with_name("benchmark_cpp.cpp")
WORKLOADS = {
    "small": (1, 2, 8, 1000),
    "medium": (4, 8, 32, 100),
    "large": (8, 24, 96, 10),
}


def make_weights(hidden_size, input_size, gate):
    return [
        [
            ((((gate + 1) * (row + 2) * (column + 3)) % 17) - 8) * 0.0125
            for column in range(hidden_size + input_size)
        ]
        for row in range(hidden_size)
    ]


def make_bias(hidden_size, gate):
    return [
        (gate - row) * 0.01 + (1.0 if gate == 0 else 0.0)
        for row in range(hidden_size)
    ]


def make_input(step, input_size):
    return [
        ((((step + 1) * (column + 3)) % 19) - 9) * 0.025
        for column in range(input_size)
    ]


def activate(weights, bias, current_input, previous_hidden, use_tanh):
    concatenated = previous_hidden + current_input
    values = [
        bias[row]
        + sum(weight * value for weight, value in zip(weights[row], concatenated))
        for row in range(len(weights))
    ]
    if use_tanh:
        return [math.tanh(value) for value in values]
    return [1.0 / (1.0 + math.exp(-value)) for value in values]


def evaluate(
    input_size, hidden_size, sequence_length, matrices, biases, sample_index=0
):
    hidden = [0.0] * hidden_size
    cell = [0.0] * hidden_size
    for step in range(sequence_length):
        current_input = make_input(step + sample_index, input_size)
        forget = activate(matrices[0], biases[0], current_input, hidden, False)
        input_gate = activate(matrices[1], biases[1], current_input, hidden, False)
        candidate = activate(matrices[2], biases[2], current_input, hidden, True)
        cell = [
            forget[index] * cell[index] + input_gate[index] * candidate[index]
            for index in range(hidden_size)
        ]
        output_gate = activate(matrices[3], biases[3], current_input, hidden, False)
        hidden = [
            output_gate[index] * math.tanh(cell[index])
            for index in range(hidden_size)
        ]
    return hidden


class CppPythonBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="lstm-benchmark-")
        cls.binary = Path(cls.temp_dir.name) / "benchmark_cpp"
        compiler = os.environ.get("CXX", "g++")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                str(CPP_SOURCE),
                "-o",
                str(cls.binary),
            ],
            cwd=ROOT,
            check=True,
        )
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        if cls.results:
            print("\nAccuracy-adjusted runtime benchmark (lower is better)")
            print(
                f"{'case':<8} {'CPP us':>12} {'Python us':>12} "
                f"{'CPP acc.':>10} {'Python acc.':>12} {'CPP score':>12} "
                f"{'Python score':>14} {'speedup':>9}"
            )
            for result in cls.results:
                print(
                    f"{result['name']:<8} {result['cpp_us']:>12.2f} "
                    f"{result['python_us']:>12.2f} "
                    f"{result['cpp_accuracy']:>9.2f}% "
                    f"{result['python_accuracy']:>11.2f}% "
                    f"{result['cpp_score_us']:>12.2f} "
                    f"{result['python_score_us']:>14.2f} "
                    f"{result['speedup']:>8.2f}x"
                )
        cls.temp_dir.cleanup()

    def assert_workload(self, name):
        input_size, hidden_size, sequence_length, iterations = WORKLOADS[name]
        matrices = [
            make_weights(hidden_size, input_size, gate) for gate in range(4)
        ]
        biases = [make_bias(hidden_size, gate) for gate in range(4)]
        prediction = evaluate(
            input_size, hidden_size, sequence_length, matrices, biases
        )

        squared_error = 0.0
        for sample in range(8):
            last_input = make_input(sample + sequence_length - 1, input_size)
            sample_prediction = evaluate(
                input_size,
                hidden_size,
                sequence_length,
                matrices,
                biases,
                sample,
            )
            for index, actual in enumerate(sample_prediction):
                target = math.tanh(0.7 * last_input[index % input_size] + 0.03 * index)
                squared_error += (actual - target) ** 2
        python_mse = squared_error / (8 * hidden_size)

        cpp_measurements = []
        for _ in range(3):
            cpp_output = subprocess.run(
                [
                    str(self.binary),
                    str(input_size),
                    str(hidden_size),
                    str(sequence_length),
                    str(iterations),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()
            cpp_measurements.append(tuple(map(float, cpp_output)))
        cpp_seconds = statistics.median(item[0] for item in cpp_measurements)
        cpp_mse, cpp_prediction = cpp_measurements[0][1:3]

        self.assertAlmostEqual(cpp_prediction, prediction[0], delta=1e-12)
        self.assertAlmostEqual(cpp_mse, python_mse, delta=1e-12)
        for _, repeated_mse, repeated_prediction, _ in cpp_measurements[1:]:
            self.assertAlmostEqual(repeated_mse, cpp_mse, delta=1e-12)
            self.assertAlmostEqual(repeated_prediction, cpp_prediction, delta=1e-12)

        evaluate(input_size, hidden_size, sequence_length, matrices, biases)
        python_timings = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(iterations):
                evaluate(input_size, hidden_size, sequence_length, matrices, biases)
            python_timings.append((time.perf_counter() - start) / iterations)

        cpp_seconds_per_eval = cpp_seconds / iterations
        python_seconds_per_eval = statistics.median(python_timings)
        cpp_accuracy = 1.0 / (1.0 + cpp_mse)
        python_accuracy = 1.0 / (1.0 + python_mse)
        cpp_score = cpp_seconds_per_eval / cpp_accuracy
        python_score = python_seconds_per_eval / python_accuracy
        self.__class__.results.append(
            {
                "name": name,
                "cpp_us": cpp_seconds_per_eval * 1e6,
                "python_us": python_seconds_per_eval * 1e6,
                "cpp_accuracy": cpp_accuracy * 100,
                "python_accuracy": python_accuracy * 100,
                "cpp_score_us": cpp_score * 1e6,
                "python_score_us": python_score * 1e6,
                "speedup": python_score / cpp_score,
            }
        )

    def test_small_sequence(self):
        self.assert_workload("small")

    def test_medium_sequence(self):
        self.assert_workload("medium")

    def test_large_sequence(self):
        self.assert_workload("large")


if __name__ == "__main__":
    unittest.main(verbosity=2)
