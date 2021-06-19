#pragma once
#include "net.h"
#include "pch.h"

// has to receive at least one element
bool strictly_monotone_decreasing(std::vector<float> values);

void hyper_surf(const Network& net, HyperParameter& hy);

// find order of magnitude of initial eta
// find threshold of decrease in first epochs
void gross_eta_surf(const Network& net, HyperParameter& hy, float start_eta = 0.01f, size_t first_epochs = 5, size_t max_tries = 100);

// find order of magnitude of bets L2 lambda
void gross_lambda_surf(const Network& net, HyperParameter& hy, size_t max_tries = 100);
