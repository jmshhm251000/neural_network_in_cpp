#include <iostream>
#include <NeuralNetwork.hpp>
#include <random>
#include <algorithm>

// Modified Data Generation: one-hot encoding for XOR (2 expected outputs)
std::vector<DataPoint> generateXORData(int numSamples = 3000) {
    std::vector<DataPoint> data;
    std::default_random_engine rng;
    std::uniform_real_distribution<double> noise(-0.1, 0.1);
    std::uniform_int_distribution<int> bit(0, 1);

    for (int i = 0; i < numSamples; ++i) {
        int a = bit(rng);
        int b = bit(rng);

        double x1 = a + noise(rng); // noisy version of 0 or 1
        double x2 = b + noise(rng);

        int xorResult = a ^ b;

        // One-hot encode: if xorResult == 0 -> [1, 0], if xorResult == 1 -> [0, 1]
        std::vector<double> expected(2, 0.0);
        if (xorResult == 0)
            expected[0] = 1.0;
        else
            expected[1] = 1.0;

        data.push_back({ {x1, x2}, expected });
    }

    return data;
}

int main() {
    // Generate training data with two expected outputs per data point
    std::vector<DataPoint> trainingData = generateXORData();

    // Update neural network architecture: 2 inputs, one hidden layer with 4 neurons, and 2 outputs
    std::vector<int> layerSizes = { 2, 4, 2 };
    NeuralNetwork nn(layerSizes, 0.1f);

    // Train the network (using central difference method)
    nn.train(trainingData);

    std::cout << nn.propagateForward({0, 0}).at(0) << "," << nn.propagateForward({0, 0}).at(1);
    std::cout << nn.propagateForward({ 1, 0 }).at(0) << "," << nn.propagateForward({ 1, 0 }).at(1);
    std::cout << nn.propagateForward({ 0, 1 }).at(0) << "," << nn.propagateForward({ 0, 1 }).at(1);
    std::cout << nn.propagateForward({ 1, 1 }).at(0) << "," << nn.propagateForward({ 1, 1 }).at(1);
}
