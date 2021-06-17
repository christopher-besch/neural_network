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

inline std::ostream& operator<<(std::ostream& out, const Network& net)
{
    out << "<Network: sizes: ";
    for (size_t size : net.sizes)
        out << size << " ";
    out << std::endl
        << "biases:" << std::endl;
    for (const arma::fvec& bias : net.biases)
        out << bias.n_rows << " " << bias.n_cols << std::endl;
    out << "weights:" << std::endl;
    for (const arma::fmat& weight : net.weights)
        out << weight.n_rows << " " << weight.n_cols << std::endl;
    out << ">";
    return out;
}
