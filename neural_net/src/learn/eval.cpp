#include "eval.h"

#include "pch.h"

namespace NeuralNet {
float total_accuracy(const Network& net, const Data* data, std::function<float(const arma::fvec& y, const arma::fvec& a)> evaluater) {
    float sum = 0;
    // go over all data sets
    for(size_t i = 0; i < data->get_x().n_cols; ++i) {
        arma::subview<float> x = data->get_x().col(i);
        arma::subview<float> y = data->get_y().col(i);
        arma::fvec           a = feedforward(net, x);

        sum += evaluater(y, a);
    }
    return sum;
}

float total_cost(const Network& net, const Data* data, float lambda_l1, float lambda_l2) {
    float cost = 0.0f;
    // go over all data sets
    for(size_t i = 0; i < data->get_x().n_cols; ++i) {
        arma::subview<float> x = data->get_x().col(i);
        arma::subview<float> y = data->get_y().col(i);
        arma::fvec           a = feedforward(net, x);

        cost += net.cost->fn(a, y);
    }
    // take average
    cost /= data->get_x().n_cols;

    // L1 regularization
    if(lambda_l1) {
        // sum of absolute of all weights
        float sum = 0.0f;
        for(const arma::fmat& w: net.weights) {
            sum += arma::accu(arma::abs(w));
        }
        cost += (lambda_l1 / data->get_x().n_cols) * sum;
    }
    // L2 regularization
    if(lambda_l2) {
        // sum of squares of all weights
        float sum = 0.0f;
        for(const arma::fmat& w: net.weights) {
            // euclidean norm:
            // || a b ||
            // || c d ||2 = sqrt(a**2 + b**2 + c**2 + d**2)
            float norm = arma::norm(w);
            // the square root has to be removed
            sum += norm * norm;
        }
        cost += 0.5f * (lambda_l2 / data->get_x().n_cols) * sum;
    }
    return cost;
}

arma::fmat feedforward(const Network& net, arma::fmat a) {
    // loop over each layer
    for(size_t left_layer_idx = 0; left_layer_idx < net.num_layers - 1; ++left_layer_idx) {
        //                                 <- actually of right layer
        arma::fmat biases_mat = net.biases[left_layer_idx] * arma::fmat(1, a.n_cols, arma::fill::ones);
        a                     = sigmoid(net.weights[left_layer_idx] * a + biases_mat);
    }
    return a;
}

void update_learn_status(const Network& net, HyperParameter& hy) {
    // evaluate status
    if(hy.monitor_test_cost) {
        float cost = total_cost(net, hy.test_data, hy.lambda_l1, hy.lambda_l2);
        hy.test_costs.push_back(cost);
        log_learn_extra("\tCost on test data: {}", cost);
    }
    if(hy.monitor_test_accuracy) {
        float accuracy = total_accuracy(net, hy.test_data, net.evaluator);
        hy.test_accuracies.push_back(accuracy);
        size_t n_test = hy.test_data->get_y().n_cols;
        log_learn_extra("\tAccuracy on test data: {} / {}", accuracy, n_test);
    }

    if(hy.monitor_eval_cost) {
        float cost = total_cost(net, hy.eval_data, hy.lambda_l1, hy.lambda_l2);
        hy.eval_costs.push_back(cost);
        log_learn_extra("\tCost on eval data: {}", cost);
    }
    if(hy.monitor_eval_accuracy) {
        float accuracy = total_accuracy(net, hy.eval_data, net.evaluator);
        hy.eval_accuracies.push_back(accuracy);
        size_t n_eval = hy.eval_data->get_y().n_cols;
        log_learn_extra("\tAccuracy on eval data: {} / {}", accuracy, n_eval);
    }

    if(hy.monitor_train_cost) {
        float cost = total_cost(net, hy.training_data, hy.lambda_l1, hy.lambda_l2);
        hy.train_costs.push_back(cost);
        log_learn_extra("\tCost on training data: {}", cost);
    }
    if(hy.monitor_train_accuracy) {
        float accuracy = total_accuracy(net, hy.training_data, net.evaluator);
        hy.train_accuracies.push_back(accuracy);
        size_t n_train = hy.training_data->get_x().n_cols;
        log_learn_extra("\tAccuracy on training data: {} / {}", accuracy, n_train);
    }
}
} // namespace NeuralNet
