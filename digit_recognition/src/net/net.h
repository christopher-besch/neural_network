#pragma once
#include "pch.h"

#include "costs.h"

struct Network
{
    size_t num_layers;
    // one element per layer; amount of neurons
    std::vector<size_t> sizes;
    // column vector for each layer, except input layer; represented as single-column matrix/
    // biases[i] are for i+1-th layer
    std::vector<arma::fvec> biases;
    // matrix for each space between layers
    // weights[i] are between i-th and i+1-th layer
    // w_(j,k) = weight from k-th in first layer to j-th in second layer
    std::vector<arma::fmat> weights;

    // used for momentum-based gradient descent
    std::vector<arma::fmat> vel_biases;
    std::vector<arma::fmat> vel_weights;

    std::shared_ptr<Cost> cost;
};

// todo: fix
// std::string Network::to_str() const
// {
//     std::stringstream buffer;
//     buffer << "<Network: sizes: ";
//     for (size_t size : m_sizes)
//         buffer << size << " ";
//     buffer << std::endl
//            << "biases:" << std::endl;
//     for (const arma::fvec& bias : m_biases)
//         buffer << bias.n_rows << " " << bias.n_cols << std::endl;
//     buffer << "weights:" << std::endl;
//     for (const arma::fmat& weight : m_weights)
//         buffer << weight.n_rows << " " << weight.n_cols << std::endl;
//     buffer << ">";
//     return buffer.str();
// }
