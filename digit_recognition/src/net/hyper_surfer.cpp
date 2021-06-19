#include "pch.h"

#include "hyper_surfer.h"
#include "learn.h"

bool strictly_monotone_decreasing(std::vector<float> values)
{
    float last_value = values[0];
    for (int i = 1; i < values.size(); ++i)
    {
        float current_value = values[i];
        if (current_value >= last_value)
            return false;
        last_value = current_value;
    }
    return true;
}

void hyper_surf(const Network& net, HyperParameter& hy)
{
    // missing:
    // - mini_batch_size
    // - mu
    // - lambda
    gross_eta_surf(net, hy);
    gross_lambda_surf(net, hy);
    // good default values
    hy.max_epochs        = 100;
    hy.no_improvement_in = 10;
    hy.stop_eta_fraction = 1024.0f;
    hy.lambda_l1         = 0.0f;
}

void gross_lambda_surf(const Network& net, HyperParameter& hy, size_t max_tries)
{
    std::cout << "// find order of magnitude of best L2 lambda" << std::endl;
    hy.reset_monitor();
    hy.lambda_l2             = 1.0f;
    hy.monitor_eval_accuracy = true;
}

void gross_eta_surf(const Network& net, HyperParameter& hy, float start_eta, size_t first_epochs, size_t max_tries)
{
    std::cout << "// find order of magnitude of eta threshold" << std::endl;
    hy.reset_monitor();
    hy.eta                = start_eta;
    hy.max_epochs         = first_epochs;
    hy.monitor_train_cost = true;

    bool last_decreased = false;
    bool first          = true;
    for (size_t i = 0; i < max_tries; ++i)
    {
        Network current_net = net;
        hy.reset_results();
        sgd(current_net, hy);
        if (first)
        {
            last_decreased = strictly_monotone_decreasing(hy.train_costs);
            first          = false;
        }

        if (strictly_monotone_decreasing(hy.train_costs))
        {
            if (last_decreased)
            {
                // still decreasing, can go even higher
                hy.init_eta *= 10;
            }
            else
            {
                // first time decreasing -> threshold found
                std::cout << "first decrease detected; gross eta found: " << hy.init_eta << std::endl;
                return;
            }
        }
        else
        {
            if (last_decreased)
            {
                // last value did work but this one doesn't
                hy.init_eta /= 10;
                std::cout << "decrease stopped; gross eta found: " << hy.init_eta << std::endl;
                return;
            }
            else
            {
                // still too big
                hy.init_eta /= 10;
            }
        }
    }
    raise_error("Failed to find order of magnitude of eta.");
}
