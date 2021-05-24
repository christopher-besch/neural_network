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
    Network(const std::vector<size_t>& sizes)
    {
        m_num_layers = sizes.size();
        m_sizes      = sizes;

        // one for each layer except input layer
        m_biases.reserve(m_num_layers - 1);
        for (size_t layer_idx = 1; layer_idx < m_num_layers; ++layer_idx)
        {
            m_biases.emplace_back(m_sizes[layer_idx], 1, arma::fill::randn);
        }

        // one for each space between layers
        m_weights.reserve(m_num_layers - 1);
        for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
        {
            // from next layer to current layer
            m_weights.emplace_back(m_sizes[left_layer_idx + 1], m_sizes[left_layer_idx], arma::fill::randn);
        }
    }

    // vectorized
    arma::fvec sigmoid(const arma::fvec& z)
    {
        return 1.0 / (1.0 + arma::exp(-z));
    }

    // return output of network with input a
    // input is vector as matrix
    // a gets changed
    arma::fmat& feedforward(arma::fmat& a)
    {
        // loop over each layer
        for (size_t right_layer_idx = 0; right_layer_idx < m_num_layers - 1; ++right_layer_idx)
        {
            // only care about first column
            a = sigmoid(m_weights[right_layer_idx] * a + m_biases[right_layer_idx]);
        }
        return a;
    }

    // stochastic gradient descent
    // eta = learning rate
    void sgd(const std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
             size_t epochs, size_t mini_batch_size, float eta,
             const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data = {})
    {
        size_t n      = training_data.size();
        size_t n_test = test_data.size();

        for (size_t e; e < epochs; ++e)
        {
            std::shuffle(training_data.begin(), training_data.end(), std::mt19937);
        }
    }

    std::string to_string()
    {
        std::stringstream buffer;
        buffer << "<Network: sizes: ";
        for (size_t size : m_sizes)
            buffer << size << " ";
        buffer << std::endl
               << "biases:" << std::endl;
        for (const arma::fmat& bias : m_biases)
            buffer << bias << std::endl;
        buffer << "weights:" << std::endl;
        for (const arma::fmat& weight : m_weights)
            buffer << weight << std::endl;
        buffer << ">";
        return buffer.str();
    }
};
