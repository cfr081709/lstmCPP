#include "gates.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

void printVector(const string& label, const vector<double>& v) {
    cout << label << ": ";
    for (double val : v) {
        cout << val << " ";
    }
    cout << endl;
}

int test() {
    assert(mathFunctions::sigmoidFunction(-1000.0) == 0.0);
    assert(mathFunctions::sigmoidFunction(1000.0) == 1.0);

    vector<double> previousCellState{2.0, -1.0};
    vector<double> forgetGateOutput{0.5, 0.25};
    vector<double> inputGateOutput{0.4, 0.6};
    vector<double> canidateCellStateOutput{0.2, -0.5};
    vector<double> cellStateGradient{3.0, 2.0};

    auto forgetGradient = backwardPropGates::forgetGate(
        forgetGateOutput, previousCellState, cellStateGradient);
    auto inputGradient = backwardPropGates::inputGate(
        inputGateOutput, canidateCellStateOutput, cellStateGradient);
    auto candidateGradient = backwardPropGates::canidateCellState(
        canidateCellStateOutput, inputGateOutput, cellStateGradient);
    auto previousCellStateGradient = backwardPropGates::updateCellState(
        forgetGateOutput, cellStateGradient);

    assert(std::abs(forgetGradient[0] - 1.5) < 1e-12);
    assert(std::abs(forgetGradient[1] - -0.375) < 1e-12);
    assert(std::abs(inputGradient[0] - 0.144) < 1e-12);
    assert(std::abs(inputGradient[1] - -0.24) < 1e-12);
    assert(std::abs(candidateGradient[0] - 1.152) < 1e-12);
    assert(std::abs(candidateGradient[1] - 0.9) < 1e-12);
    assert(std::abs(previousCellStateGradient[0] - 1.5) < 1e-12);
    assert(std::abs(previousCellStateGradient[1] - 0.5) < 1e-12);

    vector<double> outputGateOutput{0.7, 0.3};
    vector<double> cellState{0.5, -0.25};
    vector<double> hiddenStateGradient{1.5, -2.0};
    auto outputGradient = backwardPropGates::outputGate(
        outputGateOutput, cellState, hiddenStateGradient);
    auto outputCellStateGradient = backwardPropGates::cellStateFromOutputGate(
        outputGateOutput, cellState, hiddenStateGradient);
    vector<double> futureCellStateGradient{0.25, -0.5};
    auto totalCellStateGradient = backwardPropGates::cellStateFromOutputGate(
        outputGateOutput, cellState, hiddenStateGradient, futureCellStateGradient);
    for (size_t i = 0; i < cellState.size(); ++i) {
        double activatedCellState = std::tanh(cellState[i]);
        double expectedOutputGradient = hiddenStateGradient[i] * activatedCellState
            * outputGateOutput[i] * (1.0 - outputGateOutput[i]);
        double expectedCellStateGradient = hiddenStateGradient[i] * outputGateOutput[i]
            * (1.0 - activatedCellState * activatedCellState);
        assert(std::abs(outputGradient[i] - expectedOutputGradient) < 1e-12);
        assert(std::abs(outputCellStateGradient[i] - expectedCellStateGradient) < 1e-12);
        assert(std::abs(totalCellStateGradient[i]
            - (expectedCellStateGradient + futureCellStateGradient[i])) < 1e-12);
    }

    vector<vector<double>> outputWeights{{0.0, 0.0}, {0.0, 0.0}};
    vector<double> outputBias{0.0, 1.0};
    vector<double> outputInput{0.0};
    vector<double> previousOutput{0.0};
    vector<double> outputCell{0.5, -0.25};
    auto rawOutputGate = forwardPropGates::outputGateValues(
        outputWeights, outputInput, outputBias, previousOutput);
    auto hiddenOutput = forwardPropGates::outputGate(
        outputWeights, outputInput, outputBias, previousOutput, outputCell);
    assert(std::abs(rawOutputGate[0] - 0.5) < 1e-12);
    assert(std::abs(rawOutputGate[1] - mathFunctions::sigmoidFunction(1.0)) < 1e-12);
    assert(std::abs(hiddenOutput[0] - rawOutputGate[0] * std::tanh(outputCell[0])) < 1e-12);
    assert(std::abs(hiddenOutput[1] - rawOutputGate[1] * std::tanh(outputCell[1])) < 1e-12);

    bool rejectedInvalidShape = false;
    try {
        forwardPropGates::forgetGate(outputWeights, outputInput, vector<double>{0.0}, previousOutput);
    } catch (const invalid_argument&) {
        rejectedInvalidShape = true;
    }
    assert(rejectedInvalidShape);

    int hiddenSize = 10;
    int inputSize = 10;
    int concatSize = hiddenSize + inputSize;

    // All-zero weight matrix, all-zero bias, all-zero previousOutput/currentInput
    vector<vector<double>> zeroMatrix(hiddenSize, vector<double>(concatSize, 0.0));
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < concatSize; ++j) {
            zeroMatrix[i][j] = 0.0;
        }
    }

    vector<double> zeroBias(hiddenSize, 0.0);
    vector<double> zeroInput(inputSize, 0.0);
    vector<double> zeroPrevOutput(hiddenSize, 0.0);
    vector<double> zeroPrevCellState(hiddenSize, 0.0);

    // Expect all outputs to be 0.5 (sigmoid(0)) for forget/input/output gates
    vector<double> forgetOut = forwardPropGates::forgetGate(zeroMatrix, zeroInput, zeroBias, zeroPrevOutput);
    printVector("forgetGate (expect all 0.5)", forgetOut);

    vector<double> inputOut = forwardPropGates::inputGate(zeroMatrix, zeroInput, zeroBias, zeroPrevOutput);
    printVector("inputGate (expect all 0.5)", inputOut);

    // Expect all outputs to be 0 (tanh(0)) for candidate cell state
    vector<double> candidateOut = forwardPropGates::canidateCellState(zeroMatrix, zeroInput, zeroBias, zeroPrevOutput);
    printVector("canidateCellState (expect all 0.0)", candidateOut);

    // Expect all outputs to be 0 since forget*0 + input*0 = 0
    vector<double> newCellState = forwardPropGates::updateCellState(zeroPrevCellState, forgetOut, inputOut, candidateOut);
    printVector("updateCellState (expect all 0.0)", newCellState);

    // Expect all outputs to be 0 since tanh(0) = 0
    vector<double> hiddenState = forwardPropGates::outputGate(zeroMatrix, zeroInput, zeroBias, zeroPrevOutput, newCellState);
    printVector("outputGate (expect all 0.0)", hiddenState);

    cout << endl;

    // Nonzero test case: distinct weights and bias per row, to check ordering and per-row bias
    vector<vector<double>> nonzeroMatrix(hiddenSize, vector<double>(concatSize));
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < concatSize; ++j) {
            nonzeroMatrix[i][j] = 0.01 * (i + 1) * (j + 1);
        }
    }

    vector<double> nonzeroBias(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        nonzeroBias[i] = 0.1 * (i + 1);
    }

    vector<double> nonzeroInput(inputSize);
    for (int i = 0; i < inputSize; ++i) {
        nonzeroInput[i] = i + 1;
    }

    vector<double> nonzeroPrevOutput(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        nonzeroPrevOutput[i] = -(i + 1);
    }

    vector<double> nonzeroPrevCellState(hiddenSize, 1.0);

    vector<double> forgetOut2 = forwardPropGates::forgetGate(nonzeroMatrix, nonzeroInput, nonzeroBias, nonzeroPrevOutput);
    printVector("forgetGate (nonzero case)", forgetOut2);

    vector<double> inputOut2 = forwardPropGates::inputGate(nonzeroMatrix, nonzeroInput, nonzeroBias, nonzeroPrevOutput);
    printVector("inputGate (nonzero case)", inputOut2);

    vector<double> candidateOut2 = forwardPropGates::canidateCellState(nonzeroMatrix, nonzeroInput, nonzeroBias, nonzeroPrevOutput);
    printVector("canidateCellState (nonzero case)", candidateOut2);

    vector<double> newCellState2 = forwardPropGates::updateCellState(nonzeroPrevCellState, forgetOut2, inputOut2, candidateOut2);
    printVector("updateCellState (nonzero case)", newCellState2);

    vector<double> hiddenState2 = forwardPropGates::outputGate(nonzeroMatrix, nonzeroInput, nonzeroBias, nonzeroPrevOutput, newCellState2);
    printVector("outputGate (nonzero case)", hiddenState2);

    return 0;
}

int main() {
    return test();
}