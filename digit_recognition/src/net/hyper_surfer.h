#pragma once
#include "learn.h"
#include "net.h"
#include "pch.h"

inline void test(const Network& net, HyperParameter& hy)
{
    Network current_net = net;
    hy.reset_results();
    sgd(current_net, hy);
}

// do everything in the surfers power to create the best vlaues for the hyper parameters possible
// manual optimization is gernerally better but these values can be used as a starting point
void hyper_surf(const Network& net, HyperParameter& hy);

// find order of magnitude of initial eta
// find threshold of decrease in first epochs
void gross_eta_surf(const Network& net, HyperParameter& hy, float start_eta = 0.01f, size_t first_epochs = 5, size_t max_tries = 100);

// find order of magnitude of bets L2 lambda
void gross_lambda_surf(const Network& net, HyperParameter& hy, size_t max_tries = 100);
