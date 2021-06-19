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
void hyper_surf(const Network& net, HyperParameter& hy, size_t fine_surfs = 2, size_t surf_depth = 4);

// using eval accuracy to find best order of magnitude of supplied parameter
void default_coarse_surf(const Network& net, HyperParameter& hy, float& h_parameter, size_t first_epochs = 5, size_t max_tries = 100);

// find order of magnitude of initial eta
// find threshold of decrease in first epochs
void coarse_eta_surf(const Network& net, HyperParameter& hy, float start_eta = 0.01f, size_t first_epochs = 5, size_t max_tries = 100);

// using eval accuracy to fine tune supplied parameter between min and max
// using h_parameter as initial pivot
// taking probe in between [min; middle] and [middle; max]
// use better result as next [min; max] with middle in exact middle
void default_fine_surf(const Network& net, HyperParameter& hy, float& h_parameter, float min, float max, size_t first_epochs, size_t depth);

// optimize one hyper parameter after another, closing in to good values
void bounce_hyper_surf(const Network& net, HyperParameter& hy, size_t first_epochs, size_t fine_surfs, size_t surf_depth);
