#include "pch.h"

#include "hyper_surfer.h"
#include "learn.h"

void hyper_surf(const Network& net, HyperParameter& hy, size_t fine_surfs, size_t surf_depth)
{
    std::cout << "// hyper surfing" << std::endl;
    // using no learning rate schedule
    size_t no_improvement_in_buffer = hy.no_improvement_in;
    size_t max_epochs_buffer        = hy.max_epochs;
    hy.no_improvement_in            = 0;
    // initial parameters, good default values
    hy.mini_batch_size = 50;
    coarse_eta_surf(net, hy);
    // use constant eta for further hyper surfing
    hy.init_eta /= 2.0f;
    std::cout << std::endl;

    std::cout << "// Find order of magnitude of best L2 lambda" << std::endl;
    hy.lambda_l2 = 1.0f;
    default_coarse_surf(net, hy, hy.lambda_l2);
    std::cout << "// best coarse Lambda for L2 regularization found: " << hy.lambda_l2 << std::endl;
    std::cout << std::endl;

    // todo: mini_batch_size missing

    bounce_hyper_surf(net, hy, 10, fine_surfs, surf_depth);

    // use default options again
    hy.no_improvement_in = no_improvement_in_buffer;
    hy.max_epochs        = max_epochs_buffer;
    hy.init_eta *= 2.0f;

    std::cout << "// Hyper Surfer terminated:" << std::endl;
    std::cout << "\teta threshold: " << hy.init_eta << std::endl;
    std::cout << "\tL2 regularization lambda: " << hy.lambda_l2 << std::endl;
    std::cout << "\tmomentum co-efficient: " << hy.mu << std::endl;
    std::cout << "\tmini-batch size: " << hy.mini_batch_size << std::endl;
    std::cout << std::endl;
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
        std::cout << "// parameter should be smaller than default; has been set to: " << h_parameter << std::endl;
    }
    else
        std::cout << "// parameter should be bigger than default; has been set to: " << h_parameter << std::endl;

    for (size_t i = 0; i < max_tries; ++i)
    {
        // does step in right direction still improve?
        h_parameter *= go_right ? 10.0f : 1.0f / 10.0f;
        std::cout << "// Changing parameter to " << h_parameter << std::endl;
        test(net, hy);
        float new_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());
        if (new_delta < last_delta)
        {
            // go one step back
            h_parameter *= go_right ? 1.0f / 10.0f : 10.0f;
            return;
        }
    }
    raise_error("Failed to find order of magnitude for parameter.");
}

void coarse_eta_surf(const Network& net, HyperParameter& hy, float start_eta, size_t first_epochs, size_t max_tries)
{
    std::cout << "// Find order of magnitude of eta threshold" << std::endl;
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
                std::cout << "// Still decreasing, raising eta to " << hy.init_eta << std::endl;
            }
            else
            {
                // first time decreasing -> threshold found
                std::cout << "// First decrease detected; coarse eta found: " << hy.init_eta << std::endl;
                return;
            }
        }
        else
        {
            if (last_decreased)
            {
                // last value did work but this one doesn't
                hy.init_eta /= 10;
                std::cout << "// Decrease stopped; coarse eta found: " << hy.init_eta << std::endl;
                return;
            }
            else
            {
                hy.init_eta /= 10;
                std::cout << "// eta still too big, reducing eta to " << hy.init_eta << std::endl;
            }
        }
    }
    raise_error("Failed to find order of magnitude of eta.");
}

void default_fine_surf(const Network& net, HyperParameter& hy, float& h_parameter, float min, float max, size_t first_epochs, size_t depth)
{
    hy.reset_monitor();
    hy.max_epochs            = first_epochs;
    hy.monitor_eval_accuracy = true;

    for (int i = 0; i < depth; ++i)
    {
        float middle = h_parameter;
        // between min and middle
        float left_value = middle - (middle - min) / 2;
        // between middle and max
        float right_value = middle + (max - middle) / 2;

        std::cout << "// evaluate left value" << std::endl;
        h_parameter = left_value;
        test(net, hy);
        float left_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        std::cout << "// evaluate right value" << std::endl;
        h_parameter = right_value;
        test(net, hy);
        float right_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

        if (left_delta > right_delta)
        {
            min = min;
            max = middle;
            std::cout << "// smaller side [" << min << "; " << max << "] is better" << std::endl;
        }
        else
        {
            min = middle;
            max = max;
            std::cout << "// bigger side [" << min << "; " << max << "] is better" << std::endl;
        }
        h_parameter = min + (max - min) / 2;
    }
}

void bounce_hyper_surf(const Network& net, HyperParameter& hy, size_t first_epochs, size_t fine_surfs, size_t surf_depth)
{
    hy.mu = 0.5f;
    hy.reset_monitor();
    for (size_t i = 0; i < fine_surfs; ++i)
    {
        std::cout << "// " << i << ". fine mu adjustment" << std::endl;
        default_fine_surf(net, hy, hy.mu, 0.0f, 1.0f, first_epochs, surf_depth);
        std::cout << std::endl;
        std::cout << "// " << i << ". fine eta adjustment" << std::endl;
        default_fine_surf(net, hy, hy.init_eta, hy.init_eta / 2.0f, hy.init_eta * 2.0f, first_epochs, surf_depth);
        std::cout << std::endl;
        std::cout << "// " << i << ". fine lambda adjustment" << std::endl;
        default_fine_surf(net, hy, hy.lambda_l2, hy.lambda_l2 / 2.0f, hy.lambda_l2 * 2.0f, first_epochs, surf_depth);
        std::cout << std::endl;
    }
}
