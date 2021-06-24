#pragma once
#include "learn.h"
#include "net.h"

namespace NeuralNet {
////////////
// coarse //
////////////
// optimize coarse
// changing: monitors, init_eta, lambda_l2, mini_batch_size
void coarse_hyper_surf(const Network& net, HyperParameter& hy);

// using eval accuracy to find best order of magnitude of supplied parameter
// changing: monitors, max_epochs, h_parameter
void default_coarse_surf(const Network& net, HyperParameter& hy, float& h_parameter, size_t first_epochs = 10, size_t max_tries = 35);

// find mini batch size with least amount of time required
// scale init_eta anti-proportionally to mini_batch_size
// changing: monitors, max_epochs, min_batch_size, init_eta
void mini_batch_size_surf(const Network& net, HyperParameter& hy, size_t first_epochs = 10, size_t depth = 15);

// find order of magnitude of initial eta
// find threshold of decrease in first epochs
// changing: monitors, init_eta, max_epochs
void coarse_eta_surf(const Network& net, HyperParameter& hy, float start_eta = 0.01f, size_t first_epochs = 10, size_t max_tries = 35);

//////////
// fine //
//////////
// optimize one hyper parameter after another, closing in to good values
// changing: monitors, mu, init_eta, lambda_l2
void bounce_hyper_surf(const Network& net, HyperParameter& hy, size_t fine_surfs, size_t surf_depth);

// using eval accuracy to fine tune supplied parameter between min and max
// using h_parameter as initial pivot
// taking probe in between [min; middle] and [middle; max]
// use better result as next [min; max] with middle in exact middle
// changing: monitors, h_parameter
void default_fine_surf(const Network& net, HyperParameter& hy, float& h_parameter, float min, float max, size_t depth);
} // namespace NeuralNet
