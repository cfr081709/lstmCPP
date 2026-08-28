# lstmCPP

A from-scratch C++ implementation of a single LSTM (Long Short-Term Memory) cell's **forward pass**, built to understand the internal gate mechanics of an LSTM at the math level.

## Scope

This implements the standard LSTM forward-pass equations for one cell at one timestep:

- **Forget gate** — decides what to discard from the previous cell state
- **Input gate** — decides how much new information to add
- **Candidate cell state** — proposes new content via `tanh`
- **Cell state update** — combines forget/input/candidate into the new cell state `C_t`
- **Output gate** — decides what part of the new cell state becomes the hidden state `h_t`

**This is not a trainable neural network.** There is no backpropagation, no gradient computation, and no weight learning. Weights and biases must be supplied externally (e.g., pre-set, random, or loaded from another trained model). This project is for learning/demonstrating LSTM internals, not for production ML use.

## Requirements

- A C++20-compatible compiler (uses `std::numbers::e`)
- No external dependencies beyond the standard library

## Build

```bash
g++ -std=c++20 -Wall -Wextra test.cpp -o test
```

## Run

```bash
./test
```

This runs a verification suite with two cases:
1. **Zero case** — all-zero weights/biases/inputs, where outputs are hand-predictable (sigmoid gates → 0.5, tanh-based outputs → 0.0)
2. **Nonzero case** — distinct weights and biases per dimension, to confirm indexing, concatenation order, and activation functions are correct

## Files

| File | Purpose |
|---|---|
| `gates.hpp` | `forwardPropGates` class — static methods implementing each LSTM gate |
| `mathFunctions.hpp` | `sigmoidFunction` and `tanhFunction` implementations |
| `test.cpp` | Verification test harness with zero and nonzero test cases |
| `scaling.hpp` | *(describe purpose here)* |

## Known limitations

- Single cell, single timestep only — no sequence/multi-timestep chaining yet
- No input validation — mismatched vector/matrix dimensions will not throw, and may read invalid memory
- No training, gradient descent, or backpropagation
- Minimal test coverage (two hand-picked cases, not a formal test suite)

## Roadmap

- [ ] Chain cells across multiple timesteps in a sequence
- [ ] Add input validation / dimension checks
- [ ] Expand test coverage (edge cases, negative/mixed-sign inputs)
- [ ] Consider backpropagation / training support