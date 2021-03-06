#pragma once
#include <armadillo>

namespace NeuralNet {
// vectorized
// one column per data set
inline arma::fmat sigmoid(const arma::fmat& z) {
    return 1.0f / (1.0f + arma::exp(-z));
}

// derivative of sigmoid function
// one column per data set
inline arma::fmat sigmoid_prime(const arma::fmat& z) {
    return sigmoid(z) % (1.0f - sigmoid(z));
}
} // namespace NeuralNet
