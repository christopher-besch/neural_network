#pragma once
#include "hyper/data.h"
#include "net/net.h"

namespace NeuralNet {
// return number of correct results of neural network
// neuron in final layer with highest activation determines result
float total_accuracy(const Network& net, const Data* data, std::function<float(const arma::fvec& y, const arma::fvec& a)> evaluater);

// return summed and regularized cost of all data sets in <data>
float total_cost(const Network& net, const Data* data, float lambda_l1, float lambda_l2);

// return output of network with input a
// input is vector as matrix
// a gets changed
arma::fmat feedforward(const Network& net, arma::fmat a);

// when full_run -> e.g. print current epoch
void update_learn_status(const Network& net, HyperParameter& hy);
} // namespace NeuralNet
