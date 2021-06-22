#include "learn.h"

#include "pch.h"

#include <random>

namespace NeuralNet {
size_t total_accuracy(const Network& net, const Data* data) {
    size_t sum = 0;
    // go over all data sets
    for(size_t i = 0; i < data->get_x().n_cols; ++i) {
        arma::subview<float> x = data->get_x().col(i);
        arma::subview<float> y = data->get_y().col(i);
        arma::fvec           a = feedforward(net, x);

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
        sum += selected_number == correct_number;
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
        //                               <- actually of right layer
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
        float accuracy = total_accuracy(net, hy.test_data);
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
        float accuracy = total_accuracy(net, hy.eval_data);
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
        float accuracy = total_accuracy(net, hy.training_data);
        hy.train_accuracies.push_back(accuracy);
        size_t n_train = hy.training_data->get_x().n_cols;
        log_learn_extra("\tAccuracy on training data: {} / {}", accuracy, n_train);
    }
}

void sgd(Network& net, HyperParameter& hy) {
    // info block
    log_learn_general("Using stochastic gradient descent:\n{}", hy.to_str());

    auto begin = std::chrono::high_resolution_clock::now();
    hy.is_valid();
    // current eta may get changed over time
    float eta = hy.init_eta;
    // don't divide by 0
    float  stop_eta = hy.stop_eta_fraction ? hy.init_eta / hy.stop_eta_fraction : 0;
    size_t n        = hy.training_data->get_x().n_cols;

    size_t epoch = 0;
    // gets reset after reducing eta
    size_t epochs_since_last_reduction = 0;
    // go over epochs
    bool quit = false;
    while(!quit) {
        // learn
        Data this_training_data = hy.training_data->get_shuffled();
        // go over mini batches
        for(size_t offset = 0; offset < n; offset += hy.mini_batch_size) {
            // make last batch smaller if necessary
            size_t length = offset + hy.mini_batch_size >= n ? n - offset : hy.mini_batch_size;
            update_mini_batch(net, this_training_data.get_mini_x(offset, length),
                              this_training_data.get_mini_y(offset, length), eta, hy.mu, hy.lambda_l1, hy.lambda_l2, n);
        }
        log_learn_extra("Epoch {} training complete", epoch);
        update_learn_status(net, hy);

        // learning rate schedule
        // when there aren't enough epochs yet, don't do anything
        if(hy.learning_schedule_type != LearningScheduleType::None &&
           epochs_since_last_reduction >= hy.no_improvement_in) {
            float sum_delta;
            switch(hy.learning_schedule_type) {
            case LearningScheduleType::TestAccuracy:
                sum_delta = get_sum_delta(hy.test_accuracies.end() - hy.no_improvement_in, hy.test_accuracies.end());
                break;
            case LearningScheduleType::EvalAccuracy:
                sum_delta = get_sum_delta(hy.eval_accuracies.end() - hy.no_improvement_in, hy.eval_accuracies.end());
                break;
            case LearningScheduleType::None:
                raise_critical("learning rate schedule broken");
                break;
            }
            // no improvement?
            if(sum_delta <= 0.0f) {
                eta /= 2;
                // reset
                epochs_since_last_reduction = 0;

                log_learn_general("No improvement ({}) in last {} epochs; reducing learning rate to: {}", sum_delta,
                                  hy.no_improvement_in, eta);

                if(hy.stop_eta_fraction != -1.0f && eta < stop_eta) {
                    log_learn_general("Learning rate dropped below 1/{}; learning terminated.", hy.stop_eta_fraction);
                    quit = true;
                }
            }
            // else
            //      log_learn_extra("accuracy improvement in last {} epochs: {}", hy.no_improvement_in, sum_delta);
        }
        if(epoch >= hy.max_epochs) {
            log_learn_general("Maximum epochs exceeded; learning terminated");
            quit = true;
        }
        ++epoch;
        ++epochs_since_last_reduction;
    }
    // report
    auto      end        = std::chrono::high_resolution_clock::now();
    long long delta_time = (end - begin).count();
    hy.learn_time        = delta_time;
    if(delta_time > 1e9)
        log_learn_extra("learning time: {} seconds", std::to_string(delta_time / 1e9f));
    else if(delta_time > 1e6)
        log_learn_extra("learning time: {} milliseconds", std::to_string(delta_time / 1e6f));
    else if(delta_time > 1e3)
        log_learn_extra("learning time: {} microseconds", std::to_string(delta_time / 1e3f));
    else
        log_learn_extra("learning time: {} nanoseconds", std::to_string(delta_time));
}

void update_mini_batch(Network& net, const arma::subview<float> x, const arma::subview<float> y, float eta, float mu,
                       float lambda_l1, float lambda_l2, size_t n) {
    // sums of gradients <- how do certain weights and biases change the cost
    // layer-wise
    std::vector<arma::fvec> nabla_b(net.biases.size());
    std::vector<arma::fmat> nabla_w(net.weights.size());
    // set to size of weights and biases
    // start at all 0
    for(size_t i = 0; i < nabla_b.size(); ++i) {
        nabla_b[i] = arma::fvec(net.biases[i].n_rows, arma::fill::zeros);
        nabla_w[i] = arma::fmat(net.weights[i].n_rows, net.weights[i].n_cols, arma::fill::zeros);
    }

    // use backprop to calculate gradient -> de-/increase delta
    // use matrix or multiple vectors
#if 1
    backprop(net, x, y, nabla_b, nabla_w);
#else
    for(size_t idx = 0; idx < x.n_cols; ++idx) {
        backprop(x.col(idx), y.col(idx), nabla_b, nabla_w);
    }
#endif

    // optimization
    float eta_over_length  = eta / x.n_cols;
    float lambda_l1_over_n = lambda_l1 / n;
    float lambda_l2_over_n = lambda_l2 / n;
    // update weights and biases
    // move in opposite direction -> reduce cost
    for(size_t i = 0; i < net.weights.size(); ++i) {
        // include weight decay
        // L1 regularization
        net.vel_weights[i] -= eta * lambda_l1_over_n * arma::sign(net.weights[i]);
        // L2 regularization
        net.vel_weights[i] *= 1.0f - eta * lambda_l2_over_n;

        // update velocity
        // apply momentum co-efficient
        net.vel_weights[i] *= mu;
        net.vel_biases[i] *= mu;
        // approx gradient with mini batch
        net.vel_weights[i] -= eta_over_length * nabla_w[i];
        net.vel_biases[i] -= eta_over_length * nabla_b[i];

        // apply velocity
        net.weights[i] += net.vel_weights[i];
        net.biases[i] += net.vel_biases[i];
    }
}

void backprop(const Network& net, const arma::subview<float> x, const arma::subview<float> y,
              std::vector<arma::fvec>& nabla_b, std::vector<arma::fmat>& nabla_w) {
    // activations layer by layer <- needed by backprop algorithm
    // one per layer
    std::vector<arma::fmat> activations {x};
    activations.reserve(net.num_layers);
    // list of inputs for sigmoid function
    // one for each layer, except input
    std::vector<arma::fmat> zs;
    zs.reserve(net.num_layers - 1);

    // feedforward
    for(size_t left_layer_idx = 0; left_layer_idx < net.num_layers - 1; ++left_layer_idx) {
        // extend biases with as many copied columns as there are data sets in the batch
        //                               <- actually right layer
        arma::fmat biases_mat = net.biases[left_layer_idx] * arma::fmat(1, x.n_cols, arma::fill::ones);
        // weighted input
        zs.emplace_back(net.weights[left_layer_idx] * activations[activations.size() - 1] + biases_mat);
        activations.emplace_back(sigmoid(zs[zs.size() - 1]));
    }

    // calculate error for last layer (BP1)
    arma::fmat error = net.cost->error(zs[zs.size() - 1], activations[activations.size() - 1], y);
    // get gradient with respect to biases (BP3)
    // sum errors from each data set together -> sum into single column
    nabla_b[nabla_b.size() - 1] += arma::sum(error, 1);
    // get gradient with respect to weights (BP4)
    nabla_w[nabla_w.size() - 1] += error * activations[activations.size() - 2].t();

    // for all other layers
    // start at penultimate element of zs and go back to first
    for(int64_t layer_idx = net.num_layers - 3; layer_idx >= 0; --layer_idx) {
        // get input for sigmoid function of current layer
        arma::fmat sp = sigmoid_prime(zs[layer_idx]);
        // calculate error for current layer with error from layer to the right (BP2)
        error = (net.weights[layer_idx + 1].t() * error) % sp;

        // update gradient like with last layer
        nabla_b[layer_idx] += arma::sum(error, 1);
        // activations is one longer than zs
        nabla_w[layer_idx] += error * activations[layer_idx].t();
    }
}
} // namespace NeuralNet
