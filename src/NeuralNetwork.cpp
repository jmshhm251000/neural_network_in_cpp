#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(std::vector<int>& layerSizes, float learningRate)
	: learningRate(learningRate)
{
	for (int i = 0; i < layerSizes.size() - 1; ++i) {
		layers.emplace_back(std::make_unique<Layer>(layerSizes.at(i), layerSizes.at(i + 1)));
	}
}

std::vector<double> NeuralNetwork::propagateForward(const std::vector<double>& inputs) {
	std::vector<double> currentOutputs = inputs;

	for (const auto& layer : this->layers) {
		currentOutputs = layer->propagateForward(currentOutputs);
	}

	return activation_functions::softmax(currentOutputs);
}

void NeuralNetwork::propagateBackward(const std::vector<double>& expectedOutput, const std::vector<double>& inputs) {
	layers.back()->computeDeltaOutputLayer(expectedOutput);

	for (int i = layers.size() - 2; i >= 0; --i) {
		layers.at(i)->computeDeltaHiddenLayer(*layers[i + 1]);
	}

	layers.at(0)->update(learningRate, inputs);
	
	for (size_t i = 1; i < layers.size(); ++i) {
		layers.at(i)->update(learningRate, layers[i - 1]->outputs);
	}
}

int NeuralNetwork::classify(const std::vector<double>& inputs) {
	std::vector<double> outputs = this->propagateForward(inputs);

	return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

double NeuralNetwork::mse(double outputActivation, double expectedOutput) {
	double error = outputActivation - expectedOutput;

	return error * error;
}

double NeuralNetwork::crossEntropyLoss(const std::vector<double>& predicted, const std::vector<double>& expected) {
	double loss = 0.0;
	const double epsilon = 1e-12; // avoid log(0)

	for (size_t i = 0; i < predicted.size(); ++i) {
		loss += -expected[i] * std::log(predicted[i] + epsilon);
	}

	return loss;
}

double NeuralNetwork::cost(const DataPoint& dataPoint) {
	std::vector<double> outputs = this->propagateForward(dataPoint.inputs);
	return crossEntropyLoss(outputs, dataPoint.expectedOutputs);
}

double NeuralNetwork::cost(const std::vector<DataPoint>& data) {
	double totalCost = 0.0;

	for (const auto& dataPoint : data) {
		totalCost += cost(dataPoint);
	}

	return totalCost / data.size();
}

void NeuralNetwork::learnByCentralDifference(const std::vector<DataPoint>& trainingData) {
	const double h = 0.0001;

	for (auto& layer : layers) {
		for (int nodeIn = 0; nodeIn < layer->weights.size(); ++nodeIn) {
			for (int nodeOut = 0; nodeOut < layer->weights[nodeIn].size(); nodeOut++) {
				layer->weights.at(nodeIn).at(nodeOut) += h;
				double costPlus = cost(trainingData);

				layer->weights.at(nodeIn).at(nodeOut) -= 2 * h;
				double costMinus = cost(trainingData);

				layer->weights.at(nodeIn).at(nodeOut) += h;

				layer->costGradientW.at(nodeIn).at(nodeOut) = (costPlus - costMinus) / (2 * h);
			}
		}

		for (int nodeOut = 0; nodeOut < layer->biases.size(); ++nodeOut) {
			layer->biases.at(nodeOut) += h;
			double costPlus = cost(trainingData);

			layer->biases.at(nodeOut) -= 2 * h;
			double costMinus = cost(trainingData);

			layer->biases.at(nodeOut) += h;

			layer->costGradientB.at(nodeOut) = (costPlus - costMinus) / (2 * h);
		}
	}

	for (auto& layer : layers) {
		layer->updateWeightsandBiases(learningRate);
	}
}

void NeuralNetwork::trainByCentralDifference(const std::vector<DataPoint>& trainingData) {
	for (int epoch = 0; epoch < 10000; ++epoch) {
		learnByCentralDifference(trainingData);

		if (epoch % 100 == 0) {
			double cost = this->cost(trainingData);
			std::cout << "Epoch " << epoch << ", Cost: " << cost;

			int correct = 0;
			for (const auto& d : trainingData) {
				int pred = propagateForward(d.inputs)[0] >= 0.5 ? 1 : 0;
				int actual = static_cast<int>(d.expectedOutputs[0]);
				if (pred == actual) correct++;
			}
			double acc = static_cast<double>(correct) / trainingData.size();
			std::cout << ", Accuracy: " << acc * 100 << "%" << std::endl;
		}
	}
}

void NeuralNetwork::train(const std::vector<DataPoint>& trainingData, int length, bool print) {
	for (int epoch = 0; epoch < length; ++epoch) {
		int correct = 0;
		double totalCost = 0.0;

		for (const auto& data : trainingData) {
			std::vector<double> pred = propagateForward(data.inputs);
			propagateBackward(data.expectedOutputs, data.inputs);

			if (epoch % 5 == 0) {
				totalCost += crossEntropyLoss(pred, data.expectedOutputs);

				int predictedClass = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
				int actualClass = std::distance(data.expectedOutputs.begin(), std::max_element(data.expectedOutputs.begin(), data.expectedOutputs.end()));

				if (predictedClass == actualClass)
					++correct;
			}
		}

		if (epoch % 5 == 0) {
			double avgCost = totalCost / trainingData.size();
			double acc = static_cast<double>(correct) / trainingData.size();

			std::cout << "Epoch " << epoch << ", Cost: " << avgCost << ", Accuracy: " << acc * 100 << "%" << std::endl;
		}
	}

	int correct = 0;
	for (const auto& data : trainingData) {
		std::vector<double> pred = propagateForward(data.inputs);
		propagateBackward(data.expectedOutputs, data.inputs);
		int predictedClass = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
		int actualClass = std::distance(data.expectedOutputs.begin(), std::max_element(data.expectedOutputs.begin(), data.expectedOutputs.end()));

		if (print) {
			std::cout << data.inputs << " | " << predictedClass << " | " << actualClass << std::endl;
		}

		if (predictedClass == actualClass)
			++correct;
	}
	double acc = static_cast<double>(correct) / trainingData.size();
	std::cout << "Accuracy: " << acc * 100 << "%" << std::endl;
}
