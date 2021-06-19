#include "pch.h"

#include "hyper_surfer.h"
#include "learn.h"

void hyper_surf(const Network& net, HyperParameter& hy)
{
    std::cout << "// hyper surfing" << std::endl;
    // missing:
    // - mini_batch_size
    // - mu

    // initial parameters
    hy.mini_batch_size = 50;
    // hy.lambda_l1       = 0.0f;
    // hy.lambda_l2       = 0.0f;
    // hy.mu              = 0.0f;
    gross_eta_surf(net, hy);
    std::cout << "\n"
              << std::endl;
    gross_lambda_surf(net, hy);
    std::cout << "\n"
              << std::endl;
    // good default values
    hy.max_epochs        = 100;
    hy.no_improvement_in = 10;
    hy.stop_eta_fraction = 1024.0f;
    hy.lambda_l1         = 0.0f;
}

void gross_eta_surf(const Network& net, HyperParameter& hy, float start_eta, size_t first_epochs, size_t max_tries)
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
            last_decreased = strictly_monotone_decreasing(hy.train_costs);
            first          = false;
        }

        if (strictly_monotone_decreasing(hy.train_costs))
        {
            if (last_decreased)
            {
                hy.init_eta *= 10;
                std::cout << "// Still decreasing, raising eta to " << hy.init_eta << std::endl;
            }
            else
            {
                // first time decreasing -> threshold found
                std::cout << "// First decrease detected; gross eta found: " << hy.init_eta << std::endl;
                return;
            }
        }
        else
        {
            if (last_decreased)
            {
                // last value did work but this one doesn't
                hy.init_eta /= 10;
                std::cout << "// Decrease stopped; gross eta found: " << hy.init_eta << std::endl;
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

void gross_lambda_surf(const Network& net, HyperParameter& hy, size_t max_tries)
{
    std::cout << "// Find order of magnitude of best L2 lambda" << std::endl;
    hy.reset_monitor();
    hy.lambda_l2             = 1.0f;
    hy.monitor_eval_accuracy = true;

    // determine direction of improvement
    hy.lambda_l2 /= 10;
    test(net, hy);
    float left_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());
    hy.lambda_l2 *= 100;
    test(net, hy);
    float right_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());

    // set first delta
    float last_delta = left_delta > right_delta ? left_delta : right_delta;
    bool  go_right   = left_delta > right_delta ? false : true;
    // go two steps in right direction
    if (left_delta > right_delta)
        hy.lambda_l2 /= 100;

    for (size_t i = 0; i < max_tries; ++i)
    {
        // does step in right direction still improve?
        hy.lambda_l2 *= go_right ? 10 : 1 / 10;
        std::cout << "// Changing L2 lambda to " << hy.lambda_l2 << std::endl;
        test(net, hy);
        float new_delta = get_sum_delta(hy.eval_accuracies.begin(), hy.eval_accuracies.end());
        if (new_delta < last_delta)
        {
            // go one step back
            hy.lambda_l2 *= go_right ? 1 / 10 : 10;
            std::cout << "// Gross lambda for L2 regularization found: " << hy.lambda_l2 << std::endl;
            return;
        }
    }
    raise_error("Failed to find order of magnitude of L2 lambda.");
}

// void gross_lambda_close_in(const Network& net, HyperParameter& hy, float old_delta, size_t max_tries)
// {
//     if (!--max_tries)
//         return;
//     test(net, hy);
//     if (new_delta > old_delta)
//         hy.lambda_l2
//
//         gross_lambda_close_in(net, hy, new_delta, learning_rate, max_tries);
// }
