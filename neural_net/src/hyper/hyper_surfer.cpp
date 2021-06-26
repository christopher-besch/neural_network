#include "hyper_surfer.h"

#include "learn/learn.h"
#include "net/setup.h"
#include "pch.h"

namespace NeuralNet {
inline void test(const Network& net, HyperParameter& hy) {
    Network current_net = net;
    hy.reset_results();
    sgd(current_net, hy);
}

// use multiple different sets of weights and take average
inline float test_eval_accuracies(Network net, HyperParameter& hy, size_t amount) {
    float delta_sum = 0;
    for(size_t i = 0; i < amount; ++i) {
        default_weight_reset(net);
        test(net, hy);
        delta_sum += get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());
    }
    return delta_sum;
}

// use multiple different sets of weights and take average
inline float test_eval_accuracies_over_time(Network net, HyperParameter& hy, size_t amount) {
    float delta_over_time_sum = 0;
    for(size_t i = 0; i < amount; ++i) {
        default_weight_reset(net);
        test(net, hy);
        delta_over_time_sum += get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end()) / hy.learn_time;
    }
    return delta_over_time_sum;
}

// use multiple different sets of weights
// true when majority is decreasing
inline bool test_train_costs_decrease(Network net, HyperParameter& hy, size_t amount) {
    float decreases = 0;
    for(size_t i = 0; i < amount; ++i) {
        default_weight_reset(net);
        test(net, hy);
        decreases += strictly_monotone_decrease(hy.train_costs);
    }
    return decreases / amount;
}

void coarse_hyper_surf(const Network& net, HyperParameter& hy) {
    log_hyper_general("Coarse Hyper Surfing...");
    // using no learning rate schedule
    size_t               max_epochs_buffer             = hy.max_epochs;
    LearningScheduleType learning_schedule_type_buffer = hy.learning_schedule_type;
    hy.learning_schedule_type                          = LearningScheduleType::None;
    log_hyper_general("Find order of magnitude of eta threshold...");
    coarse_eta_surf(net, hy);

    log_hyper_general("Coarse lambda surf...");
    hy.lambda_l2 = 1.0f;
    default_coarse_surf(net, hy, hy.lambda_l2);

    log_hyper_general("Surf mini batch size...");
    mini_batch_size_surf(net, hy);

    hy.max_epochs             = max_epochs_buffer;
    hy.learning_schedule_type = learning_schedule_type_buffer;

    // report
    log_hyper_general("Coarse Hyper surf complete:");
    log_hyper_general("\tinit_eta: {}", hy.init_eta);
    log_hyper_general("\tlambda_l2: {}", hy.lambda_l2);
    log_hyper_general("\tmini_batch_size: {}", hy.mini_batch_size);
}

void default_coarse_surf(const Network& net, HyperParameter& hy, float& h_parameter, size_t first_epochs, size_t max_tries, size_t amount) {
    hy.reset_monitor();
    hy.max_epochs            = first_epochs;
    hy.monitor_eval_accuracy = true;

    // determine direction of improvement
    h_parameter /= 10.0f;
    float left_delta = test_eval_accuracies(net, hy, amount);
    h_parameter *= 100.0f;
    float right_delta = test_eval_accuracies(net, hy, amount);

    // set first delta
    float last_delta = left_delta > right_delta ? left_delta : right_delta;
    bool  go_right   = left_delta > right_delta ? false : true;
    // go two steps in right direction
    if(left_delta > right_delta) {
        h_parameter /= 100.0f;
        log_hyper_extra("parameter should be smaller than default; has been set to: {}", h_parameter);
    } else
        log_hyper_extra("parameter should be bigger than default; has been set to: {}", h_parameter);

    for(size_t i = 0; i < max_tries; ++i) {
        // does step in right direction still improve?
        h_parameter *= go_right ? 10.0f : 1.0f / 10.0f;
        log_hyper_extra("Changing parameter to {}", h_parameter);
        float new_delta = test_eval_accuracies(net, hy, amount);
        if(new_delta < last_delta) {
            // go one step back
            h_parameter *= go_right ? 1.0f / 10.0f : 10.0f;
            log_hyper_extra("Best coarse value for parameter found: {}", h_parameter);
            return;
        }
    }
    raise_critical("Failed to find order of magnitude for parameter.");
}

void mini_batch_size_surf(const Network& net, HyperParameter& hy, size_t first_epochs, size_t depth, size_t amount) {
    hy.reset_monitor();
    hy.monitor_eval_accuracy = true;
    hy.max_epochs            = first_epochs;

    // can't be any smaller than online learning or bigger than training data
    size_t min = 1;
    size_t max = hy.training_data->get_x().n_cols;
    for(size_t i = 0; i < depth; ++i) {
        size_t middle = hy.mini_batch_size;
        // between min and middle
        size_t left_value = middle - (middle - min) / 2;
        // between middle and max
        size_t right_value = middle + (max - middle) / 2;

        // evaluate left value
        hy.init_eta *= static_cast<float>(hy.mini_batch_size) / static_cast<float>(left_value);
        hy.mini_batch_size         = left_value;
        float left_delta_over_time = test_eval_accuracies_over_time(net, hy, amount);

        // evaluate right value
        hy.init_eta *= static_cast<float>(hy.mini_batch_size) / static_cast<float>(right_value);
        hy.mini_batch_size          = right_value;
        float right_delta_over_time = test_eval_accuracies_over_time(net, hy, amount);

        // find best improvement per time
        if(left_delta_over_time > right_delta_over_time) {
            // leave min
            max = middle;
            log_hyper_extra("smaller side [{}; {}] is better for mini batch size", min, max);
        } else {
            min = middle;
            // leave max
            log_hyper_extra("bigger side [{}; {}] is better for mini batch size", min, max);
        }
        size_t new_mini_batch_size = min + (max - min) / 2;
        // scale eta anti-proportional to mini batch size
        hy.init_eta *= static_cast<float>(hy.mini_batch_size) / static_cast<float>(new_mini_batch_size);
        hy.mini_batch_size = new_mini_batch_size;
        // only size_t <- fraction not possible
        if(max - min < 2)
            break;
    }
    log_hyper_extra("found best mini batch size: {}; set eta threshold to {}", hy.mini_batch_size, hy.init_eta);
}

void coarse_eta_surf(const Network& net, HyperParameter& hy, float start_eta, size_t first_epochs, size_t max_tries, size_t amount) {
    hy.reset_monitor();
    hy.init_eta           = start_eta;
    hy.max_epochs         = first_epochs;
    hy.monitor_train_cost = true;

    bool last_decreased = false;
    bool first          = true;
    for(size_t i = 0; i < max_tries; ++i) {
        // strictly monotone decreasing in most cases
        bool decrease = test_train_costs_decrease(net, hy, amount);
        if(first) {
            last_decreased = decrease;
            first          = false;
        }

        if(decrease) {
            if(last_decreased) {
                hy.init_eta *= 10;
                log_hyper_extra("Still decreasing, raising eta to {}", hy.init_eta);
            } else {
                // first time decreasing -> threshold found
                log_hyper_extra("First decrease detected; coarse eta found: {}", hy.init_eta);
                return;
            }
        } else {
            if(last_decreased) {
                // last value did work but this one doesn't
                hy.init_eta /= 10;
                log_hyper_extra("Decrease stopped; coarse eta found: {}", hy.init_eta);
                return;
            } else {
                hy.init_eta /= 10;
                log_hyper_extra("eta still too big, reducing eta to {}", hy.init_eta);
            }
        }
    }
    raise_critical("Failed to find order of magnitude of eta.");
}

void bounce_hyper_surf(const Network& net, HyperParameter& hy, size_t fine_surfs, size_t surf_depth) {
    log_hyper_general("Bounce Hyper Surf...");
    hy.mu = 0.5f;
    for(size_t i = 0; i < fine_surfs; ++i) {
        // use new weights each time
        Network this_net = net;
        default_weight_reset(this_net);

        log_hyper_general("{}. fine mu adjustment...", i);
        default_fine_surf(this_net, hy, hy.mu, 0.0f, 1.0f, surf_depth);
        log_hyper_general("mu set to: {}", hy.mu);

        log_hyper_general("{}. fine init_eta adjustment...", i);
        default_fine_surf(this_net, hy, hy.init_eta, hy.init_eta / 2.0f, hy.init_eta * 2.0f, surf_depth);
        log_hyper_general("init_eta set to: {}", hy.init_eta);

        log_hyper_general("{}. fine lambda adjustment...", i);
        default_fine_surf(this_net, hy, hy.lambda_l2, hy.lambda_l2 / 2.0f, hy.lambda_l2 * 2.0f, surf_depth);
        log_hyper_general("lambda_l2 set to: {}", hy.lambda_l2);
    }
    log_hyper_general("Bounce Hyper surf complete:");
    log_hyper_general("\tmu: {}", hy.mu);
    log_hyper_general("\tinit_eta: {}", hy.init_eta);
    log_hyper_general("\tlambda_l2: {}", hy.lambda_l2);
}

void default_fine_surf(const Network& net, HyperParameter& hy, float& h_parameter, float min, float max, size_t depth) {
    hy.monitor_eval_accuracy = true;
    for(size_t i = 0; i < depth; ++i) {
        float middle = h_parameter;
        // between min and middle
        float left_value = middle - (middle - min) / 2;
        // between middle and max
        float right_value = middle + (max - middle) / 2;

        // evaluate left value
        h_parameter = left_value;
        test(net, hy);
        float left_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        // evaluate right value
        h_parameter = right_value;
        test(net, hy);
        float right_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        if(left_delta > right_delta) {
            // leave min
            max = middle;
            log_hyper_extra("smaller side [{}; {}] is better", min, max);
        } else {
            min = middle;
            // leave max
            log_hyper_extra("bigger side [{}; {}] is better", min, max);
        }
        h_parameter = min + (max - min) / 2;
    }
}
} // namespace NeuralNet
