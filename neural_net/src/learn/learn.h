#pragma once
#include "hyper/data.h"
#include "net/net.h"

namespace NeuralNet {
// stochastic gradient descent
void sgd(Network& net, HyperParameter& hy);

// update weights and biases
// eta = learning rate
// mu = momentum co-efficient
// lambda = regularization parameter
void update_mini_batch(Network&                   net,
                       const arma::subview<float> x,
                       const arma::subview<float> y,
                       std::vector<arma::fmat>&   vel_biases,
                       std::vector<arma::fmat>&   vel_weights,
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
} // namespace NeuralNet
