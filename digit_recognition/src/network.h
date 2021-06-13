#pragma once
#include "costs.h"
#include "learn_cfg.h"
#include "pch.h"
#include "read_mnist.h"

class Network
{
private:
    size_t m_num_layers;
    // one element per layer; amount of neurons
    std::vector<size_t> m_sizes;
    // column vector for each layer, except input layer; represented as single-column matrix/
    // m_biases[i] are for i+1-th layer
    std::vector<arma::fvec> m_biases;
    // matrix for each space between layers
    // m_weights[i] are between i-th and i+1-th layer
    // w_(j,k) = weight from k-th in first layer to j-th in second layer
    std::vector<arma::fmat> m_weights;

    std::shared_ptr<Cost> m_cost;

public:
    // sizes of layers, first is input, last is output
    // todo: is default possible
    Network(const std::vector<size_t>& sizes, std::shared_ptr<Cost> cost = std::make_shared<CrossEntropyCost>());

    // laod from json
    Network(const std::string& json_path);

    void save_json(const std::string& path);

    std::string to_str() const;

    // return number of correct results of neural network
    // neuron in final layer with highest activation determines result
    size_t total_accuracy(const Data* data) const;

    // return summed and regularized cost of all data sets in <data>
    float total_cost(const Data* data, float lambda_l1, float lambda_l2) const;

    // init weights with gaussian distribution, mean 0, standard deviation 1, over sqrt of num weights connected to same neuron
    // init biases with gaussian distribution, mean 0, standard deviation 1
    void default_weight_init();

    // init weights with gaussian distribution, mean 0, standard deviation 1
    // init biases with gaussian distribution, mean 0, standard deviation 1
    // only to be used as reference
    void large_weight_init();

    // return output of network with input a
    // input is vector as matrix
    // a gets changed
    arma::fmat feedforward(arma::fmat a) const;

    // stochastic gradient descent
    // eta = learning rate
    // lambda = regularization parameter
    void sgd(const Data* training_data,
             size_t      epochs,
             size_t      mini_batch_size,
             float       eta,
             float       lambda_l1 = 0.0f,
             float       lambda_l2 = 0.0f,
             // monitoring -> slow
             const Data* eval_data = nullptr,
             // set and store monitoring
             LearnCFG* learn_cfg = nullptr);

private:
    // update weights and biases
    // lambda = regularization parameter
    void update_mini_batch(const arma::subview<float> x,
                           const arma::subview<float> y,
                           float                      eta,
                           float                      lambda_l1,
                           float                      lambda_l2,
                           size_t                     n);

    // set nabla_b and nabla_w to sum of delta_nabla_b and delta_nabla_w representing gradient of cost function for all data sets in batch
    // layer-by-layer, congruent to m_biases and m_weights
    void backprop(const arma::subview<float> x, const arma::subview<float> y, std::vector<arma::fvec>& nabla_b, std::vector<arma::fmat>& nabla_w) const;
};
