#pragma once

#include <armadillo>

namespace NeuralNet {
// used for classifiers; set indices to highest value in respective vector
void get_highest_index(const arma::fvec& y, const arma::fvec& a, size_t& correct_number, size_t& selected_number);

namespace DefaultEvaluater {
// return 1 if correct else 0
float classifier(const arma::fvec& y, const arma::fvec& a);

float all_round_correct(const arma::fvec& y, const arma::fvec& a);
} // namespace DefaultEvaluater
} // namespace NeuralNet
