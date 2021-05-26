#pragma once
#include "pch.h"

class Network
{
private:
    size_t m_num_layers;
    // one element per layer; amount of neurons
    std::vector<size_t> m_sizes;
    // column vector for each layer, except input layer; represented as single-column matrix/
    // m_biases[i] are for i+1-th layer
    std::vector<arma::fvec> m_biases;
    // matrix for each space between layers
    // m_weights[i] are between i-th and i+1-th layer
    // w_(j,k) = weigth from k-th in first layer to j-th in second layer
    std::vector<arma::fmat> m_weights;

public:
    // sizes of layers, first is input, last is output
    Network(const std::vector<size_t>& sizes);

    // return output of network with input a
    // input is vector as matrix
    // a gets changed
    arma::fmat feedforward(arma::fmat a);

    // stochastic gradient descent
    // eta = learning rate
    void sgd(std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
             size_t epochs, size_t mini_batch_size, float eta,
             const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data = {});

    // update weights and biases
    void update_mini_batch(const std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
                           size_t offset, size_t length, float eta);

    // adjust nabla_b and nabla_w according to delta_nabla_b and delta_nabla_w representing gradient of cost function
    // layer-by-layer, congruent to m_biases and m_weights
    void backprop(const arma::fvec& x, const arma::fvec& y, std::vector<arma::fvec>& nabla_b, std::vector<arma::fmat>& nabla_w);

    // return number of correct results of neural network
    // neuron in final layer with highest activation determines result
    size_t evaluate(const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data);

    // return vector of partial derivatives \partial C_x / \partial a
    arma::fvec cost_derivative(const arma::fvec& output_activations, const arma::fvec& y)
    {
        return output_activations - y;
    }

    // vectorized
    arma::fvec sigmoid(const arma::fvec& z)
    {
        return 1.0 / (1.0 + arma::exp(-z));
    }
    // derivative of sigmoid function
    arma::fvec sigmoid_prime(const arma::fvec& z)
    {
        return sigmoid(z) % (1 - sigmoid(z));
    }

    std::string to_string();
};
