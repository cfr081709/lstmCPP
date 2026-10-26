# lstmCPP

`lstmCPP` is a small, from-scratch C++20 implementation of a single-layer
Long Short-Term Memory (LSTM) sequence regressor. It is intended for learning
and experimentation, with no external C++ dependencies.

The repository also includes C++ gate checks, a Python/C++ inference benchmark,
and an optional helper for downloading a public Google stock prediction
dataset. The model demo and benchmark use generated synthetic data; downloading
the optional dataset is not required to build or test the model.

## Contents

- [Requirements](#requirements)
- [Build and run the demo](#build-and-run-the-demo)
- [Run tests](#run-tests)
- [Model overview](#model-overview)
- [Using the model API](#using-the-model-api)
- [Benchmark methodology](#benchmark-methodology)
- [Download and organize the optional dataset](#download-and-organize-the-optional-dataset)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)

## Requirements

- C++20 compiler: GCC or Clang
- Python 3.10 or newer to run the Python benchmark and optional dataset helper
- `kagglehub` only for downloading the optional dataset

The C++ implementation, demo, and gate tests have no third-party dependencies.
The Python benchmark uses only the standard library.

## Build and run the demo

From the repository root:

```sh
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic src/main.cpp -o lstm
./lstm
```

The demo trains a one-input, one-hidden-unit model on generated scalar
regression samples and prints its last training MSE, test MSE, and prediction
for `0.5`. The output can vary if the model, data, compiler, or training
configuration changes. The current constructor seed is fixed for repeatable
initialization.

To build with Clang, replace `g++` with `clang++`. On Windows, use a
C++20-enabled GCC/Clang toolchain and run the resulting executable (for example,
`lstm.exe`).

## Run tests

### C++ gate tests

```sh
g++ -std=c++20 -Wall -Wextra -Wpedantic src/test.cpp -o gate-tests
./gate-tests
```

These checks use assertions for activation functions, gate outputs and
gradients, and invalid vector dimensions.

### Python/C++ benchmark tests

```sh
python3 -m unittest discover -v -s tests/test_1 -p 'test_*.py'
```

The test compiles its C++ benchmark executable into a temporary directory and
removes it afterward. It runs three workload sizes and verifies that Python
and C++ produce matching predictions and mean squared errors before reporting
timings. A working C++20 compiler must be available on `PATH`. Set `CXX` to
select a different compiler:

```sh
CXX=clang++ python3 -m unittest discover -v -s tests/test_1 -p 'test_*.py'
```

The benchmark uses synthetic data and requires no Kaggle account, dataset,
NumPy, or pandas. The dataset-helper tests use a mocked download and do not
access the network.

## Model overview

The `model` class is currently defined in `src/main.cpp`. For each input vector
in a sequence, it computes the standard LSTM operations:

1. Forget gate: choose how much prior cell state to retain.
2. Input gate and candidate state: choose and calculate the new cell content.
3. Cell state update: combine retained and candidate cell information.
4. Output gate: calculate the current hidden state.

The hidden and cell state start at zero for every sequence. After processing
the full sequence, the final hidden state is the prediction. Training minimizes
mean squared error (MSE) with backpropagation through time (BPTT). The model
uses a single learning rate and processes samples in their existing order.

The reusable gate calculations and their backward-gradient helpers are in
[`src/gates.hpp`](src/gates.hpp); sigmoid and `tanh` functions are in
[`src/mathFunctions.hpp`](src/mathFunctions.hpp).

## Using the model API

The model and nested dataset types are defined in `src/main.cpp`. A sample
contains a sequence of input vectors and one target vector:

```cpp
model::Sample sample{
    {{0.1}, {0.2}, {0.3}}, // three time steps, one feature per step
    {0.25}                 // one target value
};
model::Dataset dataset{sample};
```

For this sample, construct the model with one input feature and one hidden
unit. The target vector length must equal the hidden size.

The public methods are:

| Method | Behavior |
| --- | --- |
| `model(inputSize, hiddenSize, learningRate = 0.01, seed = 42)` | Construct and initialize the model. |
| `train(dataset, epochs)` | Train on the dataset and return one average training MSE per epoch. |
| `eval(sequence)` | Return the final hidden-state prediction for one sequence. |
| `test(dataset)` | Return the average MSE for the dataset without updating model parameters. |

Input and hidden sizes and the learning rate must be positive; the learning
rate must also be finite. Sequences and datasets must not be empty. Every input
vector must match the configured input size, every target must match the
hidden size, and the epoch count passed to `train` must be positive. Invalid
arguments throw `std::invalid_argument`.

`model` currently shares `src/main.cpp` with the demonstration `main()` and is
not yet exposed as a standalone library/header. To use the class from another
program, first separate the model implementation from the demo entry point.

## Benchmark methodology

The benchmark compares the same hand-implemented LSTM gate equations in C++
and Python, using identical deterministic weights, biases, and sequence
inputs. The cases are:

| Case | Input features | Hidden units | Time steps |
| --- | ---: | ---: | ---: |
| Small | 1 | 2 | 8 |
| Medium | 4 | 8 | 32 |
| Large | 8 | 24 | 96 |

Each implementation's prediction is evaluated against generated synthetic
targets. The reported accuracy is a normalized score derived from MSE:

```text
accuracy = 1 / (1 + MSE)
accuracy-adjusted runtime = median inference time / accuracy
```

Runtime is reported per sequence in microseconds; lower accuracy-adjusted
runtime is better. The speedup shown is Python's adjusted runtime divided by
C++'s adjusted runtime. The benchmark takes the median of three timed runs per
language and case.

This score combines runtime with this benchmark's synthetic-target MSE. It is
not classification accuracy, a measure of generalization, or an overall
judgment of language performance. Results depend on machine, compiler,
compiler optimization, and system load. Compare results only under similar
conditions; correctness parity is asserted separately from timing.

## Download and organize the optional dataset

The optional helper downloads
`shreenidhihipparagi/google-stock-prediction` through KaggleHub and stages an
unchanged copy under `data/raw/google-stock-prediction/`. Raw downloaded data
is ignored by Git and should not be committed.

Create and activate a Python environment, then install the helper dependency:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Download to the default location:

```sh
python tests/test_1/data.py
```

Choose a different destination with `--output-dir`:

```sh
python tests/test_1/data.py --output-dir /path/to/google-stock-data
```

The helper preserves the downloaded directory structure and file names. It
does not transform, clean, or load the data into the LSTM. KaggleHub may
require Kaggle credentials depending on access and local configuration; see
the [KaggleHub documentation](https://github.com/Kaggle/kagglehub) if download
authentication is needed. Dataset contents and availability are controlled by
the dataset publisher.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/main.cpp` | LSTM model implementation and synthetic training demo |
| `src/gates.hpp` | Forward and backward gate calculations |
| `src/mathFunctions.hpp` | Sigmoid and `tanh` activation functions |
| `src/test.cpp` | C++ gate and gradient checks |
| `tests/test_1/test_cpp_python_benchmark.py` | Three Python/C++ parity and timing tests |
| `tests/test_1/benchmark_cpp.cpp` | C++ executable source compiled by the benchmark |
| `tests/test_1/data.py` | Optional KaggleHub dataset downloader and organizer |
| `tests/test_1/test_data.py` | Offline tests for the dataset helper |
| `data/raw/` | Local downloaded raw data; ignored by Git |

## Limitations

- One LSTM layer; no batching or variable-length sequence masking.
- Regression with MSE; no classification-specific output layer or loss.
- Output size is tied to hidden size.
- Training processes samples sequentially without shuffling.
- No model persistence, optimizer configuration, or command-line training
  options.
- Model implementation and demo entry point share one source file.
- The Kaggle dataset helper only downloads and stages raw files; it does not
  connect that dataset to training.
