#pragma once
#include <cmath>

namespace activation_functions {

	inline double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	inline double sigmoid_derivative(double x) {
		double s = sigmoid(x);
		return s * (1 - s);
	}

	inline double relu(double x) {
		return x > 0.0 ? x : 0.0;
	}

	inline double relu_derivative(double x) {
		return x > 0.0 ? 1.0 : 0.0;
	}

	inline double tanh(double x) {
		return std::tanh(x);
	}

	inline double tanh_derivative(double x) {
		double t = std::tanh(x);
		return 1.0 - t * t;
	}

	inline double leaky_relu(double x, double alpha = 0.01) {
		return x > 0.0 ? x : alpha * x;
	}

	inline double leaky_relu_derivative(double x, double alpha = 0.01) {
		return x > 0.0 ? 1.0 : alpha;
	}

	inline double elu(double x, double alpha = 1.0) {
		return x >= 0.0 ? x : alpha * (std::exp(x) - 1.0);
	}

	inline double elu_derivative(double x, double alpha = 1.0) {
		return x >= 0.0 ? 1.0 : alpha * std::exp(x);
	}

	inline std::vector<double> softmax(const std::vector<double>& z) {
		std::vector<double> result(z.size());

		double maxVal = *std::max_element(z.begin(), z.end());

		double sum = 0.0;
		for (double val : z)
			sum += std::exp(val - maxVal);

		for (size_t i = 0; i < z.size(); ++i)
			result[i] = std::exp(z[i] - maxVal) / sum;

		return result;
	}

	inline double softplus(double x) {
		return std::log(1.0 + std::exp(x));
	}

	inline double softplus_derivative(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	inline double swish(double x) {
		return x * sigmoid(x);
	}

	inline double swish_derivative(double x) {
		double s = sigmoid(x);
		return s + x * s * (1 - s);
	}
}