#include "gates.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

class model {
public:
    using Vector = std::vector<double>;
    using Sequence = std::vector<Vector>;

    struct Sample {
        Sequence inputs;
        Vector target;
    };

    using Dataset = std::vector<Sample>;

    model(std::size_t inputSize, std::size_t hiddenSize,
          double learningRate = 0.01, unsigned int seed = 42)
        : inputSize_(inputSize),
          hiddenSize_(hiddenSize),
          learningRate_(learningRate),
          forget_(hiddenSize, inputSize + hiddenSize),
          input_(hiddenSize, inputSize + hiddenSize),
          candidate_(hiddenSize, inputSize + hiddenSize),
          output_(hiddenSize, inputSize + hiddenSize) {
        if (inputSize_ == 0 || hiddenSize_ == 0) {
            throw std::invalid_argument("Input and hidden sizes must be positive");
        }
        if (!std::isfinite(learningRate_) || learningRate_ <= 0.0) {
            throw std::invalid_argument("Learning rate must be finite and positive");
        }

        std::mt19937 generator(seed);
        const double bound = std::sqrt(1.0 / static_cast<double>(inputSize_ + hiddenSize_));
        std::uniform_real_distribution<double> distribution(-bound, bound);
        initializeWeights(forget_, distribution, generator);
        initializeWeights(input_, distribution, generator);
        initializeWeights(candidate_, distribution, generator);
        initializeWeights(output_, distribution, generator);
        std::fill(forget_.bias.begin(), forget_.bias.end(), 1.0);
    }

    std::vector<double> train(const Dataset& dataset, std::size_t epochs) {
        validateDataset(dataset);
        if (epochs == 0) {
            throw std::invalid_argument("Epoch count must be positive");
        }

        std::vector<double> losses;
        losses.reserve(epochs);
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            double totalLoss = 0.0;
            for (const Sample& sample : dataset) {
                std::vector<StepCache> cache = forwardSequence(sample.inputs);
                totalLoss += meanSquaredError(cache.back().hidden, sample.target);

                Gradients gradients(*this);
                backpropagate(cache, sample.target, gradients);
                applyGradients(gradients, 1.0 / static_cast<double>(sample.inputs.size()));
            }
            losses.push_back(totalLoss / static_cast<double>(dataset.size()));
        }
        return losses;
    }

    Vector eval(const Sequence& sequence) const {
        validateSequence(sequence);
        Vector hidden(hiddenSize_, 0.0);
        Vector cell(hiddenSize_, 0.0);
        for (const Vector& currentInput : sequence) {
            StepCache step = forwardStep(currentInput, hidden, cell);
            hidden = std::move(step.hidden);
            cell = std::move(step.cell);
        }
        return hidden;
    }

    double test(const Dataset& dataset) const {
        validateDataset(dataset);
        double totalLoss = 0.0;
        for (const Sample& sample : dataset) {
            totalLoss += meanSquaredError(eval(sample.inputs), sample.target);
        }
        return totalLoss / static_cast<double>(dataset.size());
    }

private:
    struct GateParameters {
        std::vector<Vector> weights;
        Vector bias;

        GateParameters(std::size_t rows, std::size_t columns)
            : weights(rows, Vector(columns, 0.0)), bias(rows, 0.0) {}
    };

    struct StepCache {
        Vector input;
        Vector previousHidden;
        Vector previousCell;
        Vector forget;
        Vector inputGate;
        Vector candidate;
        Vector outputGate;
        Vector cell;
        Vector hidden;
    };

    struct GateGradient {
        std::vector<Vector> weights;
        Vector bias;

        GateGradient(std::size_t rows, std::size_t columns)
            : weights(rows, Vector(columns, 0.0)), bias(rows, 0.0) {}
    };

    struct Gradients {
        GateGradient forget;
        GateGradient input;
        GateGradient candidate;
        GateGradient output;

        explicit Gradients(const model& owner)
            : forget(owner.hiddenSize_, owner.inputSize_ + owner.hiddenSize_),
              input(owner.hiddenSize_, owner.inputSize_ + owner.hiddenSize_),
              candidate(owner.hiddenSize_, owner.inputSize_ + owner.hiddenSize_),
              output(owner.hiddenSize_, owner.inputSize_ + owner.hiddenSize_) {}
    };

    std::size_t inputSize_;
    std::size_t hiddenSize_;
    double learningRate_;
    GateParameters forget_;
    GateParameters input_;
    GateParameters candidate_;
    GateParameters output_;

    template <typename Distribution, typename Generator>
    static void initializeWeights(
        GateParameters& parameters, Distribution& distribution, Generator& generator) {
        for (Vector& row : parameters.weights) {
            for (double& weight : row) {
                weight = distribution(generator);
            }
        }
    }

    void validateSequence(const Sequence& sequence) const {
        if (sequence.empty()) {
            throw std::invalid_argument("Input sequence must not be empty");
        }
        for (const Vector& input : sequence) {
            if (input.size() != inputSize_) {
                throw std::invalid_argument("Input vector size does not match model input size");
            }
        }
    }

    void validateDataset(const Dataset& dataset) const {
        if (dataset.empty()) {
            throw std::invalid_argument("Dataset must not be empty");
        }
        for (const Sample& sample : dataset) {
            validateSequence(sample.inputs);
            if (sample.target.size() != hiddenSize_) {
                throw std::invalid_argument(
                    "Target vector size must match model hidden size");
            }
        }
    }

    StepCache forwardStep(
        const Vector& currentInput, const Vector& previousHidden,
        const Vector& previousCell) const {
        StepCache step;
        step.input = currentInput;
        step.previousHidden = previousHidden;
        step.previousCell = previousCell;
        step.forget = forwardPropGates::forgetGate(
            forget_.weights, currentInput, forget_.bias, previousHidden);
        step.inputGate = forwardPropGates::inputGate(
            input_.weights, currentInput, input_.bias, previousHidden);
        step.candidate = forwardPropGates::candidateCellState(
            candidate_.weights, currentInput, candidate_.bias, previousHidden);
        step.cell = forwardPropGates::updateCellState(
            previousCell, step.forget, step.inputGate, step.candidate);
        step.outputGate = forwardPropGates::outputGateValues(
            output_.weights, currentInput, output_.bias, previousHidden);
        step.hidden.resize(hiddenSize_);
        for (std::size_t i = 0; i < hiddenSize_; ++i) {
            step.hidden[i] = step.outputGate[i] * std::tanh(step.cell[i]);
        }
        return step;
    }

    std::vector<StepCache> forwardSequence(const Sequence& sequence) const {
        std::vector<StepCache> cache;
        cache.reserve(sequence.size());
        Vector hidden(hiddenSize_, 0.0);
        Vector cell(hiddenSize_, 0.0);
        for (const Vector& currentInput : sequence) {
            cache.push_back(forwardStep(currentInput, hidden, cell));
            hidden = cache.back().hidden;
            cell = cache.back().cell;
        }
        return cache;
    }

    static double meanSquaredError(const Vector& prediction, const Vector& target) {
        double loss = 0.0;
        for (std::size_t i = 0; i < prediction.size(); ++i) {
            const double difference = prediction[i] - target[i];
            loss += difference * difference;
        }
        return loss / static_cast<double>(prediction.size());
    }

    static void accumulateGateGradient(
        const GateParameters& parameters, GateGradient& gradient,
        const Vector& preActivationGradient, const Vector& concatenatedInput,
        Vector& previousHiddenGradient) {
        for (std::size_t row = 0; row < parameters.weights.size(); ++row) {
            gradient.bias[row] += preActivationGradient[row];
            for (std::size_t column = 0; column < concatenatedInput.size(); ++column) {
                gradient.weights[row][column] +=
                    preActivationGradient[row] * concatenatedInput[column];
                previousHiddenGradient[column] +=
                    parameters.weights[row][column] * preActivationGradient[row];
            }
        }
    }

    void backpropagate(
        const std::vector<StepCache>& cache, const Vector& target,
        Gradients& gradients) const {
        Vector hiddenGradient(hiddenSize_, 0.0);
        Vector cellGradient(hiddenSize_, 0.0);
        const double outputScale = 2.0 / static_cast<double>(hiddenSize_);

        for (std::size_t i = 0; i < hiddenSize_; ++i) {
            hiddenGradient[i] = outputScale * (cache.back().hidden[i] - target[i]);
        }

        for (std::size_t time = cache.size(); time-- > 0;) {
            const StepCache& step = cache[time];
            Vector forgetGradient(hiddenSize_);
            Vector inputGradient(hiddenSize_);
            Vector candidateGradient(hiddenSize_);
            Vector outputGradient(hiddenSize_);
            Vector previousCellGradient(hiddenSize_);

            Vector concatenatedInput = step.previousHidden;
            concatenatedInput.insert(
                concatenatedInput.end(), step.input.begin(), step.input.end());
            Vector concatenatedGradient(inputSize_ + hiddenSize_, 0.0);

            for (std::size_t i = 0; i < hiddenSize_; ++i) {
                const double activatedCell = std::tanh(step.cell[i]);
                const double cellFromHidden =
                    hiddenGradient[i] * step.outputGate[i] *
                    (1.0 - activatedCell * activatedCell);
                cellGradient[i] += cellFromHidden;

                outputGradient[i] = hiddenGradient[i] * activatedCell *
                    step.outputGate[i] * (1.0 - step.outputGate[i]);
                forgetGradient[i] = cellGradient[i] * step.previousCell[i] *
                    step.forget[i] * (1.0 - step.forget[i]);
                inputGradient[i] = cellGradient[i] * step.candidate[i] *
                    step.inputGate[i] * (1.0 - step.inputGate[i]);
                candidateGradient[i] = cellGradient[i] * step.inputGate[i] *
                    (1.0 - step.candidate[i] * step.candidate[i]);
                previousCellGradient[i] = cellGradient[i] * step.forget[i];
            }

            accumulateGateGradient(
                forget_, gradients.forget, forgetGradient,
                concatenatedInput, concatenatedGradient);
            accumulateGateGradient(
                input_, gradients.input, inputGradient,
                concatenatedInput, concatenatedGradient);
            accumulateGateGradient(
                candidate_, gradients.candidate, candidateGradient,
                concatenatedInput, concatenatedGradient);
            accumulateGateGradient(
                output_, gradients.output, outputGradient,
                concatenatedInput, concatenatedGradient);

            hiddenGradient.assign(
                concatenatedGradient.begin(),
                concatenatedGradient.begin() + static_cast<std::ptrdiff_t>(hiddenSize_));
            cellGradient = std::move(previousCellGradient);
        }
    }

    void applyGateGradient(
        GateParameters& parameters, const GateGradient& gradient, double scale) {
        for (std::size_t row = 0; row < parameters.weights.size(); ++row) {
            parameters.bias[row] -= learningRate_ * scale * gradient.bias[row];
            for (std::size_t column = 0; column < parameters.weights[row].size(); ++column) {
                parameters.weights[row][column] -=
                    learningRate_ * scale * gradient.weights[row][column];
            }
        }
    }

    void applyGradients(const Gradients& gradients, double scale) {
        applyGateGradient(forget_, gradients.forget, scale);
        applyGateGradient(input_, gradients.input, scale);
        applyGateGradient(candidate_, gradients.candidate, scale);
        applyGateGradient(output_, gradients.output, scale);
    }
};

int main() {
    model network(1, 1, 0.03);
    model::Dataset trainingData;
    model::Dataset testData;

    for (int value = -8; value <= 8; ++value) {
        const double input = static_cast<double>(value) / 8.0;
        model::Sample sample{{{input}}, {input}};
        if (value % 4 == 0) {
            testData.push_back(sample);
        } else {
            trainingData.push_back(sample);
        }
    }

    const std::vector<double> losses = network.train(trainingData, 1000);
    std::cout << "Training MSE: " << losses.back() << '\n'
              << "Test MSE: " << network.test(testData) << '\n'
              << "Prediction for 0.5: "
              << network.eval({{0.5}}).front() << '\n';
    return 0;
}
