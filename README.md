# Neural Network - Scala
This project is a very simple implementation of MLP (1 hidden layer) on Scala. It uses the XOR gate as a training and testing set.

## XOR Gate
| In | In | Out |
|---|---|---|
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 0 | 0 | 0 |
| 1 | 1 | 0 |

## Activation function
For the back-end, it uses the hyperbolic tangent, `tanh`, and for printing out the value, the outputs are passed thought the step function:

| X | Y |
| :---: | :---: |
| <= 0.5 | 0.0 |
| > 0.5 | 1.0 |
