#include "evaluator.h"

#include "pch.h"

namespace NeuralNet {
namespace DefaultEvaluater {
// return 1 if correct else 0
float classifier(const arma::fvec& y, const arma::fvec& a) {
    // determine highest confidences
    float  highest_correct_confidence;
    size_t correct_number    = -1;
    bool   got_first_correct = false;
    float  highest_selected_confidence;
    size_t selected_number    = -1;
    bool   got_first_selected = false;

    for(size_t idx = 0; idx < a.n_rows; ++idx) {
        if(!got_first_correct || y[idx] > highest_correct_confidence) {
            highest_correct_confidence = y[idx];
            correct_number             = idx;
            got_first_correct          = true;
        }
        if(!got_first_selected || a[idx] > highest_selected_confidence) {
            highest_selected_confidence = a[idx];
            selected_number             = idx;
            got_first_selected          = true;
        }
    }
    return selected_number == correct_number;
}

float all_round_correct(const arma::fvec& y, const arma::fvec& a) {
    for(int i = 0; i < 8; ++i) {
        if(std::round(a[i]) != y[i])
            return false;
    }
    return true;
}
} // namespace DefaultEvaluater
} // namespace NeuralNet
