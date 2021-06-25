#pragma once

#include <armadillo>

namespace NeuralNet {
namespace DefaultEvaluater {
// return 1 if correct else 0
float classifier(const arma::fvec& y, const arma::fvec& a);
} // namespace DefaultEvaluater
} // namespace NeuralNet
