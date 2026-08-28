#include "mathFunctions.hpp"

#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

using Matrix = const vector<vector<double>>&;
using Vec = const vector<double>&;

class forwardPropGates {
    public:
        static vector<double> forgetGate(Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput) {
                // Concatenate currentInput and previousOutput
                vector<double> concat;
                concat.reserve(currentInput.size() + previousOutput.size());
                concat.insert(concat.end(), previousOutput.begin(), previousOutput.end());
                concat.insert(concat.end(), currentInput.begin(), currentInput.end());

                // Multiply weight matrix by the concatenated vector
                int weightMatrixSize = weightMatrix.size();
                int inputDim = concat.size();
                vector<double> forgetOutput(weightMatrixSize, 0.0);
                for (int i = 0; i < weightMatrixSize; ++i) {
                    double sum = biasTerm[i];
                    for (int j = 0; j < inputDim; ++j) {
                        sum += weightMatrix[i][j] * concat[j];
                    }
                    forgetOutput[i] = mathFunctions::sigmoidFunction(sum);
                }

                return forgetOutput;
            }

        static vector<double> inputGate(Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput){
                // Concatenate currentInput and  previousOutput
                vector<double> concat;
                concat.reserve(currentInput.size() + previousOutput.size());
                concat.insert(concat.end(), previousOutput.begin(), previousOutput.end());
                concat.insert(concat.end(), currentInput.begin(), currentInput.end());

                // Multiply weight matrix by the concatenated vector
                int weightMatrixSize = weightMatrix.size();
                int inputDim = concat.size();
                vector<double> inputGateOutput(weightMatrixSize, 0.0);
                for (int i = 0; i < weightMatrixSize; ++i) {
                    double sum = biasTerm[i];
                    for (int j = 0; j < inputDim; ++j) {
                        sum += weightMatrix[i][j] * concat[j];
                    }
                    inputGateOutput[i] = mathFunctions::sigmoidFunction(sum);
                }

                return inputGateOutput;
            }

        static vector<double> canidateCellState(Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput){
                // Concatenate currentInput and previousOutput
                vector<double> concat;
                concat.reserve(currentInput.size() + previousOutput.size());
                concat.insert(concat.end(), previousOutput.begin(), previousOutput.end());
                concat.insert(concat.end(), currentInput.begin(), currentInput.end());

                // Multiply weight matrix by the concatenated vector
                int weightMatrixSize = weightMatrix.size();
                int inputDim = concat.size();
                vector<double> canCellStateOutput(weightMatrixSize, 0.0);
                for (int i = 0; i < weightMatrixSize; ++i) {
                    double sum = biasTerm[i];
                    for (int j = 0; j < inputDim; ++j) {
                        sum += weightMatrix[i][j] * concat[j];
                    }
                    canCellStateOutput[i] = mathFunctions::tanhFunction(sum);
                }

                return canCellStateOutput;
            }

        static vector<double> updateCellState(Vec previousCellState, Vec forgetGateOutput, Vec inputGateOutput, Vec canidateCellStateOutput){
                // Create newCellState vector with a size equal to that of the preivous cell state
                int size = previousCellState.size();
                vector<double> newCellState(size);

                // Compute the new cell state at each position i
                for(int i = 0; i < size; ++i){
                    newCellState[i] = ((forgetGateOutput[i] * previousCellState[i]) + (inputGateOutput[i] * canidateCellStateOutput[i]));
                }

                return newCellState;
            }

        static vector<double> outputGate(Matrix weightMatrix, Vec currentInput, Vec biasTerm, Vec previousOutput, Vec cellState){
                // Concatenate currentInput and previousOutput
                vector<double> concat;
                concat.reserve(currentInput.size() + previousOutput.size());
                concat.insert(concat.end(), previousOutput.begin(), previousOutput.end());
                concat.insert(concat.end(), currentInput.begin(), currentInput.end());

                // Multiply weight matrix by the concatenated vector
                int weightMatrixSize = weightMatrix.size();
                int inputDim = concat.size();
                vector<double> output(weightMatrixSize, 0.0);
                for (int i = 0; i < weightMatrixSize; ++i) {
                    double sum = biasTerm[i];
                    for (int j = 0; j < inputDim; ++j) {
                        sum += weightMatrix[i][j] * concat[j];
                    }
                    output[i] = mathFunctions::sigmoidFunction(sum);
                }

                vector<double> hiddenState(cellState.size());

                for(int j = 0; j < hiddenState.size(); ++j){
                    hiddenState[j] = (output[j] * (mathFunctions::tanhFunction(cellState[j])));
                }

                return hiddenState;
            }
};