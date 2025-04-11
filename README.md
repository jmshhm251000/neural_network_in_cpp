# neural_network_in_cpp

This project is a feedforward neural network implemented in C++ with support for training using backpropagation and central difference.

## Folder Structure

├── include/
  └──ActivationFunctions.hpp
  └──Layer.hpp
  └──NeuralNetwork.hpp
├── src/
  └──Layer.cpp
  └──NeuralNetwork.cpp
├── main.cpp
├── makefile
└── README.md

## Build Instructions

### Requirements
- g++ with C++17 support
- `make` or `mingw32-make` (on Windows)

### To Build

```bash
make

### To Clean

make clean

### Run

./myprogram
