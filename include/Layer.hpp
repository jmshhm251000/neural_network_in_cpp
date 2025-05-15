#pragma once

#include <iostream>
#include <vector>
#include <ActivationFunctions.hpp>

struct DataPoint {
	std::vector<double> inputs;
	std::vector<double> expectedOutputs;
};

class Layer {
private:
	int numNodesIn;
	int numNodesOut;

	std::vector<std::vector<double>> weights;
	std::vector<double> biases;
	std::vector<std::vector<double>> costGradientW;
	std::vector<double> costGradientB;
	std::vector<double> zValues;
	std::vector<double> outputs;
	std::vector<double> delta;

	friend class NeuralNetwork;

public:
	Layer(int numNodesIn, int numNodesOut);

	std::vector<double> propagateForward(std::vector<double>& inputs);
	void randomWeights();
	void update(double learningRate, const std::vector<double>& previousOutputs);
	void updateWeightsandBiases(double learningRate);
	void computeDeltaHiddenLayer(const Layer& nextLayer);
	void computeDeltaOutputLayer(const std::vector<double>& expectedOutput);
};