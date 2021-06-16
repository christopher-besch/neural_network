# Neural Networks and Deep Learning

## [Markus' Sister Project](https://github.com/MarcasRealAccount/NeuralNetwork)

Some Experiments with Neural Networks and Deep Learning

Based on [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)

# Dependencies

- armadillo
- LAPACK and BLAS have to be installed on your system.
   On Debian based systems you can use `sudo apt-get install libblas-dev liblapack-dev` to install them.

# A Brief History of how to Traverse Hyper-Parameter Space

## General
- use simpler version of problem and network first -> speed up

## Eta
- find threshold of immediate (in first few epochs) decrease, no rise or oscillation of training cost
- order of magnitude: start with 0.01, multiply or divide by 10
- compare multiple values directly above found magnitude
- iterate a few times
- use early stopping or half threshold

## Lambda
- validation accuracy

## Mini-Batch Size
- validation accuracy
