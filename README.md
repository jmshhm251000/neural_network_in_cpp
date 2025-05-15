# Neural Network (C++)

This is a simple implementation of a feedforward neural network in C++. It demonstrates how to structure a neural network using object-oriented programming and includes support for basic activation functions and layers.

## Project Structure

neuralnetwork/
├── include/
│ ├── ActivationFunctions.hpp # Defines activation functions like sigmoid, ReLU, etc.
│ ├── Layer.hpp # Layer class: weights, biases, forward/backward pass
│ └── NeuralNetwork.hpp # NeuralNetwork class: overall architecture and training loop
└── src/
├── Layer.cpp # Implementation of the Layer class
├── NeuralNetwork.cpp # Implementation of the NeuralNetwork class
└── main.cpp # Entry point for testing or training the model


## Build Instructions

1. **Install CMake** (v3.10 or later)
2. **Build the project**
