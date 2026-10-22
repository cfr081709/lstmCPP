#pragma once

#include "mathFunctions.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

using Matrix = const std::vector<std::vector<double>>&;
using Vec = const std::vector<double>&;

namespace gateDetail {
inline void requireSameSize(Vec first, Vec second, const char* message) {
    if (first.size() != second.size()) {
        throw std::invalid_argument(message);
    }
}

inline void requireSameSize(Vec first, Vec second, Vec third, const char* message) {
    requireSameSize(first, second, message);
    requireSameSize(first, third, message);
}

inline std::vector<double> concatenate(Vec previousOutput, Vec currentInput) {
    std::vector<double> values;
    values.reserve(previousOutput.size() + currentInput.size());
    values.insert(values.end(), previousOutput.begin(), previousOutput.end());
    values.insert(values.end(), currentInput.begin(), currentInput.end());
    return values;
}

inline std::vector<double> affine(Matrix weights, Vec input, Vec bias) {
    if (weights.size() != bias.size()) {
        throw std::invalid_argument("Weight row count must match bias size");
    }

    std::vector<double> result(weights.size());
    for (std::size_t row = 0; row < weights.size(); ++row) {
        if (weights[row].size() != input.size()) {
            throw std::invalid_argument("Weight row width must match concatenated input size");
        }
        double sum = bias[row];
        for (std::size_t column = 0; column < input.size(); ++column) {
            sum += weights[row][column] * input[column];
        }
        result[row] = sum;
    }
    return result;
}

inline std::vector<double> gateValues(
    Matrix weights, Vec currentInput, Vec bias, Vec previousOutput, bool useTanh) {
    const auto concatenated = concatenate(previousOutput, currentInput);
    auto values = affine(weights, concatenated, bias);
    for (double& value : values) {
        value = useTanh ? mathFunctions::tanhFunction(value)
                        : mathFunctions::sigmoidFunction(value);
    }
    return values;
}
}

class forwardPropGates {
public:
    static std::vector<double> forgetGate(
        Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput) {
        return gateDetail::gateValues(weightMatrix, currentInput, biasTerm, previousOutput, false);
    }

    static std::vector<double> inputGate(
        Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput) {
        return gateDetail::gateValues(weightMatrix, currentInput, biasTerm, previousOutput, false);
    }

    static std::vector<double> candidateCellState(
        Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput) {
        return gateDetail::gateValues(weightMatrix, currentInput, biasTerm, previousOutput, true);
    }

    static std::vector<double> canidateCellState(
        Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput) {
        return candidateCellState(weightMatrix, currentInput, biasTerm, previousOutput);
    }

    static std::vector<double> updateCellState(
        Vec previousCellState,
        Vec forgetGateOutput,
        Vec inputGateOutput,
        Vec canidateCellStateOutput) {
        gateDetail::requireSameSize(
            previousCellState, forgetGateOutput, inputGateOutput,
            "Cell state and gate vectors must have matching sizes");
        gateDetail::requireSameSize(
            previousCellState, canidateCellStateOutput,
            "Cell state and candidate vectors must have matching sizes");

        std::vector<double> newCellState(previousCellState.size());
        for (std::size_t i = 0; i < newCellState.size(); ++i) {
            newCellState[i] = forgetGateOutput[i] * previousCellState[i]
                + inputGateOutput[i] * canidateCellStateOutput[i];
        }
        return newCellState;
    }

    static std::vector<double> outputGateValues(
        Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput) {
        return gateDetail::gateValues(weightMatrix, currentInput, biasTerm, previousOutput, false);
    }

    static std::vector<double> outputGate(
        Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput, Vec cellState) {
        auto output = outputGateValues(weightMatrix, currentInput, biasTerm, previousOutput);
        gateDetail::requireSameSize(output, cellState,
            "Output gate and cell state vectors must have matching sizes");

        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] *= mathFunctions::tanhFunction(cellState[i]);
        }
        return output;
    }
};

class backwardPropGates {
public:
    static std::vector<double> forgetGate(
        Vec forgetGateOutput, Vec previousCellState, Vec cellStateGradient) {
        gateDetail::requireSameSize(forgetGateOutput, previousCellState, cellStateGradient,
            "Forget gate, previous cell state, and cell gradient sizes must match");
        std::vector<double> gradient(forgetGateOutput.size());
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] = cellStateGradient[i] * previousCellState[i]
                * forgetGateOutput[i] * (1.0 - forgetGateOutput[i]);
        }
        return gradient;
    }

    static std::vector<double> inputGate(
        Vec inputGateOutput, Vec canidateCellStateOutput, Vec cellStateGradient) {
        gateDetail::requireSameSize(inputGateOutput, canidateCellStateOutput, cellStateGradient,
            "Input gate, candidate, and cell gradient sizes must match");
        std::vector<double> gradient(inputGateOutput.size());
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] = cellStateGradient[i] * canidateCellStateOutput[i]
                * inputGateOutput[i] * (1.0 - inputGateOutput[i]);
        }
        return gradient;
    }

    static std::vector<double> candidateCellState(
        Vec canidateCellStateOutput, Vec inputGateOutput, Vec cellStateGradient) {
        gateDetail::requireSameSize(canidateCellStateOutput, inputGateOutput, cellStateGradient,
            "Candidate, input gate, and cell gradient sizes must match");
        std::vector<double> gradient(canidateCellStateOutput.size());
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] = cellStateGradient[i] * inputGateOutput[i]
                * (1.0 - canidateCellStateOutput[i] * canidateCellStateOutput[i]);
        }
        return gradient;
    }

    static std::vector<double> canidateCellState(
        Vec canidateCellStateOutput, Vec inputGateOutput, Vec cellStateGradient) {
        return candidateCellState(canidateCellStateOutput, inputGateOutput, cellStateGradient);
    }

    static std::vector<double> updateCellState(Vec forgetGateOutput, Vec cellStateGradient) {
        gateDetail::requireSameSize(forgetGateOutput, cellStateGradient,
            "Forget gate and cell gradient sizes must match");
        std::vector<double> previousCellStateGradient(forgetGateOutput.size());
        for (std::size_t i = 0; i < previousCellStateGradient.size(); ++i) {
            previousCellStateGradient[i] = cellStateGradient[i] * forgetGateOutput[i];
        }
        return previousCellStateGradient;
    }

    static std::vector<double> outputGate(
        Vec outputGateOutput, Vec cellState, Vec hiddenStateGradient) {
        gateDetail::requireSameSize(outputGateOutput, cellState, hiddenStateGradient,
            "Output gate, cell state, and hidden gradient sizes must match");
        std::vector<double> gradient(outputGateOutput.size());
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            const double activatedCellState = mathFunctions::tanhFunction(cellState[i]);
            gradient[i] = hiddenStateGradient[i] * activatedCellState
                * outputGateOutput[i] * (1.0 - outputGateOutput[i]);
        }
        return gradient;
    }

    static std::vector<double> cellStateFromOutputGate(
        Vec outputGateOutput, Vec cellState, Vec hiddenStateGradient) {
        gateDetail::requireSameSize(outputGateOutput, cellState, hiddenStateGradient,
            "Output gate, cell state, and hidden gradient sizes must match");
        std::vector<double> gradient(cellState.size());
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            const double activatedCellState = mathFunctions::tanhFunction(cellState[i]);
            gradient[i] = hiddenStateGradient[i] * outputGateOutput[i]
                * (1.0 - activatedCellState * activatedCellState);
        }
        return gradient;
    }

    static std::vector<double> cellStateFromOutputGate(
        Vec outputGateOutput, Vec cellState, Vec hiddenStateGradient, Vec futureCellStateGradient) {
        gateDetail::requireSameSize(outputGateOutput, cellState, hiddenStateGradient,
            "Output gate, cell state, and hidden gradient sizes must match");
        gateDetail::requireSameSize(cellState, futureCellStateGradient,
            "Cell state and future cell gradient sizes must match");
        auto gradient = cellStateFromOutputGate(outputGateOutput, cellState, hiddenStateGradient);
        for (std::size_t i = 0; i < gradient.size(); ++i) {
            gradient[i] += futureCellStateGradient[i];
        }
        return gradient;
    }
};
