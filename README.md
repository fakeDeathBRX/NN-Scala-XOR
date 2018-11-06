# Multi Layer Perceptron - Scala
This project is a very simple implementation of a MLP (1 hidden layer) on Scala.

## Usage
Simply run `scala main.scala`.

## How does it work
It trains a neural network based on the `sin` and `cos` values of an angle, and the target output is that angle (2 inputs -> 1 output).
Note that the network can be configured with X amounts of nodes in each layer.

## Configuration
* `NeuralNetwork(intputNodes, hiddenNodes, outputNodes)` declares a neural network.
* `epoch` stands for the number of times that the neural net will be trained.
* `samples` stands for how much times the epoch loop will run before logging the error.
* `decrease` stands the interval that the learning rate will decrease by half.
