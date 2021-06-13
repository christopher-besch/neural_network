#include "pch.h"

struct LearnCFG
{
    bool monitor_eval_cost      = false;
    bool monitor_eval_accuracy  = false;
    bool monitor_train_cost     = false;
    bool monitor_train_accuracy = false;

    // results
    std::vector<float> eval_costs, eval_accuracies,
        train_costs, train_accuracies;
};
