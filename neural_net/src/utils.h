#pragma once
#include <vector>

namespace NeuralNet
{
// has to receive at least one element
inline bool strictly_monotone_decrease(std::vector<float> values)
{
    float last_value = values[0];
    for (size_t i = 1; i < values.size(); ++i)
    {
        float current_value = values[i];
        if (current_value >= last_value)
            return false;
        last_value = current_value;
    }
    return true;
}

// sum up deltas between values
// has to receive at least two elements
inline float get_sum_delta(std::vector<float>::iterator begin, std::vector<float>::iterator end)
{
    float sum_delta  = 0.0f;
    float last_value = *begin;
    for (++begin; begin < end; ++begin)
    {
        float current_value = *begin;
        sum_delta += current_value - last_value;
        last_value = current_value;
    }
    return sum_delta;
}
} // namespace NeuralNet
