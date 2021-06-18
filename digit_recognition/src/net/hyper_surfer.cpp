#include "pch.h"

#include "hyper_surfer.h"
#include "learn.h"

// has to receive at least one element
bool monotone_decreasing(std::vector<float> values)
{
    float last_value = values[0];
    for (int i = 1; i < values.size(); ++i)
    {
        float current_value = values[i];
        if (current_value < last_value)
            return false;
        last_value = current_value;
    }
    return true;
}

void hyper_surf(const Network& net, HyperParameter& hy)
{
    hy.eta = 0.01f;
    gross_eta_surf(net, hy);
    hy.reset_results();
}

void gross_eta_surf(Network& net, HyperParameter& hy, size_t iterations)
{
    for (size_t i = 0; i < iterations; ++i)
    {
        Network current_net = net;
        hy.reset_results();
        sgd(current_net, hy);
    }
}
