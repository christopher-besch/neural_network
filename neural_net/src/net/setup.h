#pragma once
#include "net/costs.h"
#include "net/net.h"

namespace NeuralNet {
// sizes of layers, first is input, last is output
// beware of memory leaks
void create_network(Network& net, const std::vector<size_t>& sizes, bool post_process = false);

// laod from json
// beware of memory leaks
void load_json_network(Network& net, const std::string& json_path);

void save_json(const Network& net, const std::string& path);

// set vectors to correct size
void null_weight_init(Network& net);

// doesn't effect post process layer
// requires null_weight_init to be used prior
// init weights with gaussian distribution, mean 0, standard deviation 1, over sqrt of num weights connected to same neuron
// init biases with gaussian distribution, mean 0, standard deviation 1
void default_weight_reset(Network& net);

// doesn't effect post process layer
// requires null_weight_init to be used prior
// init weights with gaussian distribution, mean 0, standard deviation 1
// init biases with gaussian distribution, mean 0, standard deviation 1
// only to be used as reference
void large_weight_reset(Network& net);
} // namespace NeuralNet
