#pragma once
#include "net.h"
#include "pch.h"

bool monotone_decreasing(std::vector<float> values);

void hyper_surf(const Network* net, HyperParameter& hy);

void gross_eta_surf(const Network& net, HyperParameter& hy);
