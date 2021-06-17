#include "pch.h"

#include "setup.h"
Network* create_network(const std::vector<size_t>& sizes, std::shared_ptr<Cost> cost)
{
    Network* net = new Network();
    // todo: make multi threaded
    arma::arma_rng::set_seed_random();

    net->num_layers = sizes.size();
    net->sizes      = sizes;

    default_weight_init(net);
    reset_vel(net);

    net->cost = cost;
    return net;
}

Network* load_json_network(const std::string& json_path)
{
    Network*      net = new Network();
    std::ifstream file(json_path);
    if (!file)
        raise_error("Can't open input json file!");
    json json_net;
    file >> json_net;
    file.close();

    net->sizes      = static_cast<std::vector<size_t>>(json_net["sizes"]);
    net->num_layers = net->sizes.size();

    // loop over layer-by-layer weight sets
    for (const std::vector<std::vector<float>>& w : static_cast<std::vector<std::vector<std::vector<float>>>>(json_net["weights"]))
    {
        // create matrix of weights for this layer-by-layer part
        arma::fmat weight(w[0].size(), w.size());
        // loop over columns
        for (unsigned int x = 0; x < w.size(); ++x)
            // loop over rows
            for (unsigned int y = 0; y < w[0].size(); ++y)
                weight.at(y, x) = w[x][y];
        net->weights.push_back(weight);
    }

    for (const std::vector<float>& b : static_cast<std::vector<std::vector<float>>>(json_net["biases"]))
        net->biases.push_back(b);

    net->cost = Cost::get(json_net["cost"]);

    reset_vel(net);
    return net;
}

void save_json(const Network* net, const std::string& path)
{
    // loop over layer-by-layer weight sets
    std::vector<std::vector<std::vector<float>>> serialized_weights;
    for (const arma::fmat& w : net->weights)
    {
        std::vector<std::vector<float>> serialized_weight;
        // add column by column
        for (unsigned int i = 0; i < w.n_cols; ++i)
            serialized_weight.push_back(arma::conv_to<std::vector<float>>::from(w.col(i)));
        serialized_weights.push_back(serialized_weight);
    }
    std::vector<std::vector<float>> serialized_biases;
    for (const arma::fvec& b : net->biases)
        serialized_biases.push_back(arma::conv_to<std::vector<float>>::from(b));

    json json_net = {
        { "sizes", net->sizes },
        { "weights", serialized_weights },
        { "biases", serialized_biases },
        { "cost", net->cost->to_str() }
    };

    std::ofstream file(path);
    if (!file)
        raise_error("Can't open output json file!");
    file << std::setw(4) << json_net;
    file.close();
};

void default_weight_init(Network* net)
{
    net->biases.resize(0);
    // one for each layer except input layer
    net->biases.reserve(net->num_layers - 1);
    for (size_t layer_idx = 1; layer_idx < net->num_layers; ++layer_idx)
    {
        net->biases.emplace_back(net->sizes[layer_idx], 1, arma::fill::randn);
    }

    net->weights.resize(0);
    // one for each layer except input layer
    // one for each space between layers
    net->weights.reserve(net->num_layers - 1);
    for (size_t left_layer_idx = 0; left_layer_idx < net->num_layers - 1; ++left_layer_idx)
    {
        // from next layer to current layer
        net->weights.emplace_back(net->sizes[left_layer_idx + 1], net->sizes[left_layer_idx], arma::fill::randn);
        net->weights[left_layer_idx] /= sqrt(net->sizes[left_layer_idx + 1]);
    }
}

void large_weight_init(Network* net)
{
    net->biases.resize(0);
    // one for each layer except input layer
    net->biases.reserve(net->num_layers - 1);
    for (size_t layer_idx = 1; layer_idx < net->num_layers; ++layer_idx)
    {
        net->biases.emplace_back(net->sizes[layer_idx], 1, arma::fill::randn);
    }

    net->weights.resize(0);
    // one for each layer except input layer
    // one for each space between layers
    net->weights.reserve(net->num_layers - 1);
    for (size_t left_layer_idx = 0; left_layer_idx < net->num_layers - 1; ++left_layer_idx)
    {
        // from next layer to current layer
        net->weights.emplace_back(net->sizes[left_layer_idx + 1], net->sizes[left_layer_idx], arma::fill::randn);
    }
}

void reset_vel(Network* net)
{
    net->vel_biases.resize(net->biases.size());
    net->vel_weights.resize(net->weights.size());
    // set to size of weights and biases
    // start at all 0
    for (size_t i = 0; i < net->vel_biases.size(); ++i)
    {
        net->vel_biases[i]  = arma::fvec(net->biases[i].n_rows, arma::fill::zeros);
        net->vel_weights[i] = arma::fmat(net->weights[i].n_rows, net->weights[i].n_cols, arma::fill::zeros);
    }
}
