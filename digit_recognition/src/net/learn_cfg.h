#include "pch.h"

struct LearnCFG
{
    bool monitor_test_cost      = false;
    bool monitor_test_accuracy  = false;
    bool monitor_train_cost     = false;
    bool monitor_train_accuracy = false;

    // results
    std::vector<float> test_costs, test_accuracies,
        train_costs, train_accuracies;
};
