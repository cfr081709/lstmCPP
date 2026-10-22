# lstmCPP

A small, from-scratch C++ implementation of a Long Short-Term Memory (LSTM)
network. It demonstrates the LSTM gates, sequence processing, and training with
backpropagation through time (BPTT).

## What it does

The `model` class in `src/main.cpp` implements a single-layer sequence
regressor:

- Takes a non-empty sequence of input vectors.
- Produces one prediction vector after processing the full sequence.
- Trains with mean squared error (MSE) and BPTT.
- Evaluates a dataset by returning its average MSE.

Each gate has its own weights and bias: forget, input, candidate cell state,
and output. The hidden state and cell state are initialized to zero for each
sequence. The target vector must have the same size as the model's hidden
state.

This project is intended for learning and experimentation, not production ML.
It currently has no separate model library/header, optimizer options, model
save/load support, or output layer independent of the hidden size.

## Requirements

- A C++20-compatible compiler, such as `g++` or `clang++`
- No external dependencies

The optional Python data helper in `tests/test_1/data.py` requires Python 3 and
the packages listed in `requirements.txt`. Set up its environment from the
repository root with:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate it with `.venv\Scripts\activate` instead.

## Build and run

From the repository root:

```sh
g++ -std=c++20 -Wall -Wextra -Wpedantic src/main.cpp -o lstm
./lstm
```

The demo trains on a small regression dataset and prints the final training
MSE, test MSE, and a prediction.

Build and run the gate verification tests separately:

```sh
g++ -std=c++20 -Wall -Wextra -Wpedantic src/test.cpp -o gate-tests
./gate-tests
```

The gate tests check known outputs and gradients, including zero and nonzero
weight cases.

## Model API

The class currently lives in `src/main.cpp`.

- `model(inputSize, hiddenSize, learningRate = 0.01, seed = 42)` constructs the
  model. The seed makes weight initialization reproducible.
- `train(dataset, epochs)` updates the model and returns the average training
  MSE recorded for each epoch.
- `eval(sequence)` returns the prediction vector for one sequence.
- `test(dataset)` returns the average MSE for a dataset without updating the
  model.

The dataset types are nested in `model`:

```cpp
model::Sample sample{
    {{0.1}, {0.2}, {0.3}}, // sequence of input vectors
    {0.25}                 // target vector
};
model::Dataset dataset{sample};
```

For this sample shape, construct a model with `inputSize = 1`; the target
dimension must match `hiddenSize`. Every sequence in a dataset must be
non-empty, and every input vector must match the configured input size.
Invalid dimensions, empty sequences, empty datasets, and zero training epochs
are rejected with `std::invalid_argument`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/main.cpp` | Model implementation and runnable training demo |
| `src/gates.hpp` | Forward and backward LSTM gate calculations |
| `src/mathFunctions.hpp` | Sigmoid and `tanh` activation functions |
| `src/test.cpp` | Gate calculation verification tests |

## Current limitations

- Single LSTM layer; no batching or variable-length masking
- Regression only, using MSE
- The output dimension is tied to the hidden size
- Training processes samples sequentially and does not shuffle them
- The model class and demo `main()` share `src/main.cpp`, so it is not yet
  packaged as a reusable library

## Possible next steps

- Separate the model class from the demo entry point
- Add tests for model training and inference
- Add model persistence and configurable output layers
