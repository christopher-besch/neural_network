#pragma once
#include "costs.h"
#include "net.h"

namespace NeuralNet
{
// sizes of layers, first is input, last is output
// beware of memory leaks
void create_network(Network& net, const std::vector<size_t>& sizes, std::shared_ptr<Cost> cost = std::make_shared<CrossEntropyCost>());

// laod from json
// beware of memory leaks
void load_json_network(Network& net, const std::string& json_path);

void save_json(const Network& net, const std::string& path);

// init weights with gaussian distribution, mean 0, standard deviation 1, over sqrt of num weights connected to same neuron
// init biases with gaussian distribution, mean 0, standard deviation 1
void default_weight_init(Network& net);

// init weights with gaussian distribution, mean 0, standard deviation 1
// init biases with gaussian distribution, mean 0, standard deviation 1
// only to be used as reference
void large_weight_init(Network& net);

// set velocities to 0
void reset_vel(Network& net);
} // namespace NeuralNet
