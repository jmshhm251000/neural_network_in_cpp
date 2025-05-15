#include <Layer.hpp>
#include <random>
#include <ctime>

Layer::Layer(int numNodesIn, int numNodesOut)
	: numNodesIn(numNodesIn), numNodesOut(numNodesOut) {

	weights.resize(numNodesIn, std::vector<double>(numNodesOut, 0.0));
	biases.resize(numNodesOut, 0.0);

	costGradientW.resize(numNodesIn, std::vector<double>(numNodesOut, 0.0));
	costGradientB.resize(numNodesOut, 0.0);

	zValues.resize(numNodesOut, 0.0);
	outputs.resize(numNodesOut, 0.0);
	delta.resize(numNodesOut, 0.0);

	randomWeights();
}

void Layer::randomWeights() {
	std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	for (int i = 0; i < numNodesIn; ++i) {
		for (int j = 0; j < numNodesOut; ++j) {
			weights[i][j] = distribution(generator);
		}
	}

	for (int j = 0; j < numNodesOut; ++j) {
		biases[j] = distribution(generator);
	}
}

void Layer::update(double learningRate, const std::vector<double>& previousOutputs) {
	for (int i = 0; i < numNodesIn; ++i) {
		for (int j = 0; j < numNodesOut; ++j) {
			weights.at(i).at(j) -= learningRate * delta.at(j) * previousOutputs.at(i);
		}
	}

	for (int j = 0; j < numNodesOut; ++j) {
		biases.at(j) -= learningRate * delta.at(j);
	}
}

std::vector<double> Layer::propagateForward(std::vector<double>& inputs) {
	for (int nodeOut = 0; nodeOut < numNodesOut; ++nodeOut) {
		double weightedSum = biases.at(nodeOut);

		for (int nodeIn = 0; nodeIn < numNodesIn; ++nodeIn) {
			weightedSum += inputs.at(nodeIn) * weights.at(nodeIn).at(nodeOut);
		}
		zValues.at(nodeOut) = weightedSum;
		outputs.at(nodeOut) = activation_functions::sigmoid(weightedSum);
	}

	return outputs;
}

void Layer::updateWeightsandBiases(double learningRate) {
	for (int nodeOut = 0; nodeOut < numNodesOut; ++nodeOut) {
		biases.at(nodeOut) -= costGradientB.at(nodeOut) * learningRate;
		for (int nodeIn = 0; nodeIn < numNodesIn; ++nodeIn) {
			weights.at(nodeIn).at(nodeOut) -= costGradientW.at(nodeIn).at(nodeOut) * learningRate;
		}
	}
}

void Layer::computeDeltaHiddenLayer(const Layer& nextLayer) {
	for (int j = 0; j < numNodesOut; ++j) {
		double sum = 0.0;

		for (int k = 0; k < nextLayer.numNodesOut; ++k) {
			sum += nextLayer.weights.at(j).at(k) * nextLayer.delta.at(k);
		}

		delta[j] = sum * activation_functions::sigmoid_derivative(zValues.at(j));
	}
}

void Layer::computeDeltaOutputLayer(const std::vector<double>& expectedOutput) {
	for (int i = 0; i < numNodesOut; ++i) {
		delta.at(i) = outputs.at(i) - expectedOutput.at(i);
	}
}
