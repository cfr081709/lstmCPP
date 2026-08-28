#include "gates.hpp"

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