#include "../../src/gates.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Workload {
    std::size_t inputSize;
    std::size_t hiddenSize;
    std::size_t sequenceLength;
    std::size_t iterations;
};

using Vector = std::vector<double>;
using WeightMatrix = std::vector<Vector>;

static WeightMatrix makeWeights(
    std::size_t hiddenSize, std::size_t inputSize, std::size_t gate) {
    WeightMatrix weights(hiddenSize, Vector(hiddenSize + inputSize));
    for (std::size_t row = 0; row < hiddenSize; ++row) {
        for (std::size_t column = 0; column < hiddenSize + inputSize; ++column) {
            const int value = static_cast<int>(
                ((gate + 1) * (row + 2) * (column + 3)) % 17) - 8;
            weights[row][column] = value * 0.0125;
        }
    }
    return weights;
}

static Vector makeBias(std::size_t hiddenSize, std::size_t gate) {
    Vector bias(hiddenSize);
    for (std::size_t row = 0; row < hiddenSize; ++row) {
        bias[row] = (static_cast<double>(gate) - static_cast<double>(row)) * 0.01;
    }
    if (gate == 0) {
        for (double& value : bias) {
            value += 1.0;
        }
    }
    return bias;
}

static Vector makeInput(std::size_t step, std::size_t inputSize) {
    Vector input(inputSize);
    for (std::size_t column = 0; column < inputSize; ++column) {
        const int value = static_cast<int>(((step + 1) * (column + 3)) % 19) - 9;
        input[column] = value * 0.025;
    }
    return input;
}

static Vector evaluate(
    const Workload& workload, const Vector biases[4],
    const std::vector<WeightMatrix>& matrices, std::size_t sampleIndex = 0) {
    Vector hidden(workload.hiddenSize, 0.0);
    Vector cell(workload.hiddenSize, 0.0);
    for (std::size_t step = 0; step < workload.sequenceLength; ++step) {
        const Vector input = makeInput(step + sampleIndex, workload.inputSize);
        const Vector forget = forwardPropGates::forgetGate(
            matrices[0], input, biases[0], hidden);
        const Vector inputGate = forwardPropGates::inputGate(
            matrices[1], input, biases[1], hidden);
        const Vector candidate = forwardPropGates::candidateCellState(
            matrices[2], input, biases[2], hidden);
        cell = forwardPropGates::updateCellState(cell, forget, inputGate, candidate);
        hidden = forwardPropGates::outputGate(
            matrices[3], input, biases[3], hidden, cell);
    }
    return hidden;
}

static Workload parseArguments(int argc, char** argv) {
    if (argc != 5) {
        throw std::invalid_argument(
            "Expected input size, hidden size, sequence length, and iteration count");
    }
    Workload workload{
        static_cast<std::size_t>(std::stoul(argv[1])),
        static_cast<std::size_t>(std::stoul(argv[2])),
        static_cast<std::size_t>(std::stoul(argv[3])),
        static_cast<std::size_t>(std::stoul(argv[4]))};
    if (workload.inputSize == 0 || workload.hiddenSize == 0 ||
        workload.sequenceLength == 0 || workload.iterations == 0) {
        throw std::invalid_argument("All workload sizes and iterations must be positive");
    }
    return workload;
}

int main(int argc, char** argv) {
    try {
        const Workload workload = parseArguments(argc, argv);
        std::vector<WeightMatrix> matrices;
        Vector biases[4];
        for (std::size_t gate = 0; gate < 4; ++gate) {
            matrices.push_back(makeWeights(workload.hiddenSize, workload.inputSize, gate));
            biases[gate] = makeBias(workload.hiddenSize, gate);
        }

        const Vector prediction = evaluate(workload, biases, matrices);
        double totalSquaredError = 0.0;
        constexpr std::size_t accuracySamples = 8;
        for (std::size_t sample = 0; sample < accuracySamples; ++sample) {
            const std::size_t lastStep =
                sample + workload.sequenceLength - 1;
            const Vector input = makeInput(lastStep, workload.inputSize);
            const Vector samplePrediction =
                evaluate(workload, biases, matrices, sample);
            for (std::size_t index = 0; index < workload.hiddenSize; ++index) {
                const double target = std::tanh(
                    0.7 * input[index % workload.inputSize] +
                    0.03 * static_cast<double>(index));
                const double difference = samplePrediction[index] - target;
                totalSquaredError += difference * difference;
            }
        }
        const double mse = totalSquaredError /
            static_cast<double>(accuracySamples * workload.hiddenSize);

        volatile double result = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t iteration = 0; iteration < workload.iterations; ++iteration) {
            const Vector current = evaluate(workload, biases, matrices);
            result = current[iteration % workload.hiddenSize];
        }
        const auto finish = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(finish - start).count();

        std::cout << std::setprecision(17) << seconds << ' ' << mse << ' '
                  << prediction.front() << ' ' << result << '\n';
    } catch (const std::exception& error) {
        std::cerr << "benchmark_cpp: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
