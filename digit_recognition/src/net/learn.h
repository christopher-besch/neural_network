#pragma once
#include "data.h"
#include "net.h"
#include "pch.h"

// return number of correct results of neural network
// neuron in final layer with highest activation determines result
size_t total_accuracy(const Network& net, const Data* data);

// return summed and regularized cost of all data sets in <data>
float total_cost(const Network& net, const Data* data, float lambda_l1, float lambda_l2);

// return output of network with input a
// input is vector as matrix
// a gets changed
arma::fmat feedforward(const Network& net, arma::fmat a);

// stochastic gradient descent
void sgd(Network& net, HyperParameter& hy);

// update weights and biases
// eta = learning rate
// mu = momentum co-efficient
// lambda = regularization parameter
void update_mini_batch(Network&                   net,
                       const arma::subview<float> x,
                       const arma::subview<float> y,
                       float                      eta,
                       float                      mu,
                       float                      lambda_l1,
                       float                      lambda_l2,
                       size_t                     n);

// set nabla_b and nabla_w to sum of delta_nabla_b and delta_nabla_w representing gradient of cost function for all data sets in batch
// layer-by-layer, congruent to net->biases and net->weights
void backprop(const Network&             net,
              const arma::subview<float> x,
              const arma::subview<float> y,
              std::vector<arma::fvec>&   nabla_b,
              std::vector<arma::fmat>&   nabla_w);
