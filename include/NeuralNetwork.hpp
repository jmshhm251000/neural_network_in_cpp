#pragma once

#include <Layer.hpp>
#include <algorithm>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
	os << "{";
	for (size_t i = 0; i < vec.size(); ++i) {
		if (i > 0) os << ", ";
		os << vec[i];
	}
	os << "}";
	return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<std::vector<T>>& vec) {
	os << "{ ";
	for (const auto& inner_vec : vec) {
		os << inner_vec << ", ";
	}
	os << "}";
	return os;
}

class NeuralNetwork {
private:
	std::vector<std::unique_ptr<Layer>> layers;
	float learningRate;

public:
	NeuralNetwork(std::vector<int>& layerSizes, float learningRate = float(0.005));

	std::vector<double> propagateForward(const std::vector<double>& inputs);
	void propagateBackward(const std::vector<double>& expectedOutput, const std::vector<double>& inputs);

	int classify(const std::vector<double>& inputs);
	double mse(double outputActivation, double expectedOutput);
	double crossEntropyLoss(const std::vector<double>& predicted, const std::vector<double>& expected);
	double cost(const DataPoint& dataPoint);
	double cost(const std::vector<DataPoint>& data);
	void learnByCentralDifference(const std::vector<DataPoint>& trainingData);
	void trainByCentralDifference(const std::vector<DataPoint>& trainingData);
	void train(const std::vector<DataPoint>& trainingData, int length = 20, bool print = false);
};