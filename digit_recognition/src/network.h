#pragma once
#include "pch.h"

class Network
{
private:
    size_t              m_num_layers;
    std::vector<size_t> m_sizes; // one element per layer; amount of neurons
    // todo: does have to be matrix?
    std::vector<arma::fmat> m_biases;  // column vector for each layer, except input layer; represented as single-column matrix
    std::vector<arma::fmat> m_weights; // matrix for each space between layers; w_(j,k) = weigth from k-th in first layer to j-th in second layer

public:
    // sizes of layers, first is input, last is output
    Network(const std::vector<size_t>& sizes);

    // vectorized
    arma::fvec sigmoid(const arma::fvec& z)
    {
        return 1.0 / (1.0 + arma::exp(-z));
    }

    // return output of network with input a
    // input is vector as matrix
    // a gets changed
    arma::fmat& feedforward(arma::fmat& a);

    // stochastic gradient descent
    // eta = learning rate
    void sgd(std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
             size_t epochs, size_t mini_batch_size, float eta,
             const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data = {});

    // update weights and biases
    void update_mini_batch(const std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
                           size_t offset, size_t length, size_t eta);

    float evaluate(const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data);

    void backprop(const arma::fvec& x, const arma::fvec& y, std::vector<arma::fmat>& delta_nabla_b, std::vector<arma::fmat>& delta_nabla_w);

    std::string to_string();
};
