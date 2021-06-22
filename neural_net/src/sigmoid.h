#pragma once
#include <armadillo>

namespace NeuralNet {
// vectorized
// one column per data set
inline arma::fmat sigmoid(const arma::fmat& z) {
    return 1.0 / (1.0 + arma::exp(-z));
}

// derivative of sigmoid function
// one column per data set
inline arma::fmat sigmoid_prime(const arma::fmat& z) {
    return sigmoid(z) % (1 - sigmoid(z));
}
} // namespace NeuralNet
