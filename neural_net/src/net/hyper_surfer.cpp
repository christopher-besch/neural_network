#include "pch.h"

#include "hyper_surfer.h"
#include "learn.h"

namespace NeuralNet
{
inline void test(const Network& net, HyperParameter& hy)
{
    Network current_net = net;
    hy.reset_results();
    sgd(current_net, hy);
}

void hyper_surf(const Network& net, HyperParameter& hy, size_t fine_surfs, size_t surf_depth)
{
    log_general("Hyper surfing...");
    // using no learning rate schedule
    size_t no_improvement_in_buffer = hy.no_improvement_in;
    size_t max_epochs_buffer        = hy.max_epochs;
    hy.no_improvement_in            = 0;
    // initial parameters, good default values
    hy.mini_batch_size = 50;
    log_general("Find order of magnitude of eta threshold...");
    coarse_eta_surf(net, hy);
    // use constant eta for further hyper surfing
    hy.init_eta /= 2.0f;

    log_general("Coarse surf Lambda...");
    hy.lambda_l2 = 1.0f;
    default_coarse_surf(net, hy, hy.lambda_l2);

    log_general("Surf mini batch size...");
    mini_batch_size_surf(net, hy);

    log_general("bounce hyper surfing...");
    bounce_hyper_surf(net, hy, 30, fine_surfs, surf_depth);

    // use default options again
    hy.no_improvement_in = no_improvement_in_buffer;
    hy.max_epochs        = max_epochs_buffer;
    hy.init_eta *= 2.0f;

    log_general(R"(Hyper Surfer terminated:\n
\teta threshold: {}\n
\tL2 regularization parameter: {}\n
\tmomentum co-efficient: {}\n
\tmini batch size: {})",
                hy.init_eta, hy.lambda_l2, hy.mu, hy.mini_batch_size);
}

void bounce_hyper_surf(const Network& net, HyperParameter& hy, size_t first_epochs, size_t fine_surfs, size_t surf_depth)
{
    hy.mu = 0.5f;
    hy.reset_monitor();
    for (size_t i = 0; i < fine_surfs; ++i)
    {
        log_general("{}. fine mu adjustment", i);
        default_fine_surf(net, hy, hy.mu, 0.0f, 1.0f, first_epochs, surf_depth);
        log_general("{}. fine eta adjustment", i);
        default_fine_surf(net, hy, hy.init_eta, hy.init_eta / 1.5f, hy.init_eta * 1.5f, first_epochs, surf_depth);
        log_general("{}. fine lambda adjustment", i);
        default_fine_surf(net, hy, hy.lambda_l2, hy.lambda_l2 / 1.5f, hy.lambda_l2 * 1.5f, first_epochs, surf_depth);
    }
}

void default_coarse_surf(const Network& net, HyperParameter& hy, float& h_parameter, size_t first_epochs, size_t max_tries)
{
    hy.reset_monitor();
    hy.max_epochs            = first_epochs;
    hy.monitor_eval_accuracy = true;

    // determine direction of improvement
    h_parameter /= 10.0f;
    test(net, hy);
    float left_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());
    h_parameter *= 100.0f;
    test(net, hy);
    float right_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

    // set first delta
    float last_delta = left_delta > right_delta ? left_delta : right_delta;
    bool  go_right   = left_delta > right_delta ? false : true;
    // go two steps in right direction
    if (left_delta > right_delta)
    {
        h_parameter /= 100.0f;
        log_extra("parameter should be smaller than default; has been set to: {}", h_parameter);
    }
    else
        log_extra("parameter should be bigger than default; has been set to: {}", h_parameter);

    for (size_t i = 0; i < max_tries; ++i)
    {
        // does step in right direction still improve?
        h_parameter *= go_right ? 10.0f : 1.0f / 10.0f;
        log_extra("Changing parameter to {}", h_parameter);
        test(net, hy);
        float new_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());
        if (new_delta < last_delta)
        {
            // go one step back
            h_parameter *= go_right ? 1.0f / 10.0f : 10.0f;
            log_extra("Best coarse value for parameter found: {}", h_parameter);
            return;
        }
    }
    raise_critical("Failed to find order of magnitude for parameter.");
}

void mini_batch_size_surf(const Network& net, HyperParameter& hy, size_t first_epochs, size_t depth)
{
    hy.reset_monitor();
    hy.monitor_eval_accuracy = true;
    hy.max_epochs            = first_epochs;

    // can't be any smaller than online learning or bigger than training data
    size_t min = 1;
    size_t max = hy.training_data->get_x().n_cols;
    for (size_t i = 0; i < depth; ++i)
    {
        size_t middle = hy.mini_batch_size;
        // between min and middle
        size_t left_value = middle - (middle - min) / 2;
        // between middle and max
        size_t right_value = middle + (max - middle) / 2;

        log_extra("evaluate left value for mini batch size");
        hy.init_eta *= static_cast<float>(hy.mini_batch_size) / static_cast<float>(left_value);
        hy.mini_batch_size = left_value;
        test(net, hy);
        long long left_time  = hy.learn_time;
        float     left_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        log_extra("evaluate right value for mini batch size");
        hy.init_eta *= static_cast<float>(hy.mini_batch_size) / static_cast<float>(right_value);
        hy.mini_batch_size = right_value;
        test(net, hy);
        long long right_time  = hy.learn_time;
        float     right_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        // find best improvement per time
        if ((left_delta / left_time) > (right_delta / right_time))
        {
            // leave min
            max = middle;
            log_extra("smaller side [{}; {}] is better for mini batch size", min, max);
        }
        else
        {
            min = middle;
            // leave max
            log_extra("bigger side [{}; {}] is better for mini batch size", min, max);
        }
        size_t new_mini_batch_size = min + (max - min) / 2;
        // scale eta anti-proportional to mini batch size
        hy.init_eta *= static_cast<float>(hy.mini_batch_size) / static_cast<float>(new_mini_batch_size);
        hy.mini_batch_size = new_mini_batch_size;
        // only size_t <- fraction not possible
        if (max - min < 2)
            break;
    }
    log_extra("found best mini batch size: {}; set eta threshold to {}", hy.mini_batch_size, hy.init_eta);
}

void coarse_eta_surf(const Network& net, HyperParameter& hy, float start_eta, size_t first_epochs, size_t max_tries)
{
    hy.reset_monitor();
    hy.init_eta           = start_eta;
    hy.max_epochs         = first_epochs;
    hy.monitor_train_cost = true;

    bool last_decreased = false;
    bool first          = true;
    for (size_t i = 0; i < max_tries; ++i)
    {
        test(net, hy);
        if (first)
        {
            last_decreased = strictly_monotone_decrease(hy.train_costs);
            first          = false;
        }

        if (strictly_monotone_decrease(hy.train_costs))
        {
            if (last_decreased)
            {
                hy.init_eta *= 10;
                log_extra("Still decreasing, raising eta to {}", hy.init_eta);
            }
            else
            {
                // first time decreasing -> threshold found
                log_extra("First decrease detected; coarse eta found: {}", hy.init_eta);
                return;
            }
        }
        else
        {
            if (last_decreased)
            {
                // last value did work but this one doesn't
                hy.init_eta /= 10;
                log_extra("Decrease stopped; coarse eta found: {}", hy.init_eta);
                return;
            }
            else
            {
                hy.init_eta /= 10;
                log_extra("eta still too big, reducing eta to {}", hy.init_eta);
            }
        }
    }
    raise_critical("Failed to find order of magnitude of eta.");
}

void default_fine_surf(const Network& net, HyperParameter& hy, float& h_parameter, float min, float max, size_t first_epochs, size_t depth)
{
    hy.reset_monitor();
    hy.max_epochs            = first_epochs;
    hy.monitor_eval_accuracy = true;

    for (size_t i = 0; i < depth; ++i)
    {
        float middle = h_parameter;
        // between min and middle
        float left_value = middle - (middle - min) / 2;
        // between middle and max
        float right_value = middle + (max - middle) / 2;

        log_extra("evaluate left value");
        h_parameter = left_value;
        test(net, hy);
        float left_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        log_extra("evaluate right value");
        h_parameter = right_value;
        test(net, hy);
        float right_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        if (left_delta > right_delta)
        {
            // leave min
            max = middle;
            log_extra("smaller side [{}; {}] is better", min, max);
        }
        else
        {
            min = middle;
            // leave max
            log_extra("bigger side [{}; {}] is better", min, max);
        }
        h_parameter = min + (max - min) / 2;
    }
}
} // namespace NeuralNet
