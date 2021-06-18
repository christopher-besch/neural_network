#pragma once
#include "costs.h"
#include "data.h"
#include "pch.h"

struct Network
{
    size_t num_layers;
    // one element per layer; amount of neurons
    std::vector<size_t> sizes;
    // column vector for each layer, except input layer; represented as single-column matrix/
    // biases[i] are for i+1-th layer
    std::vector<arma::fvec> biases;
    // matrix for each space between layers
    // weights[i] are between i-th and i+1-th layer
    // w_(j,k) = weight from k-th in first layer to j-th in second layer
    std::vector<arma::fmat> weights;

    // used for momentum-based gradient descent
    std::vector<arma::fmat> vel_biases;
    std::vector<arma::fmat> vel_weights;

    std::shared_ptr<Cost> cost;
};

inline std::ostream& operator<<(std::ostream& out, const Network& net)
{
    out << "<Network: sizes: ";
    for (size_t size : net.sizes)
        out << size << " ";
    out << std::endl
        << "biases:" << std::endl;
    for (const arma::fvec& bias : net.biases)
        out << bias.n_rows << " " << bias.n_cols << std::endl;
    out << "weights:" << std::endl;
    for (const arma::fmat& weight : net.weights)
        out << weight.n_rows << " " << weight.n_cols << std::endl;
    out << ">";
    return out;
}

// data in hyper-space
struct HyperParameter
{
    // required
    size_t mini_batch_size = 0;
    // start learning rate
    float init_eta = 0.0f;
    // stop after these epchs regardless of any schedule
    size_t max_epochs = 0;

    // optional
    // -1 -> disabled
    // ignored if learning rate schedule disabled
    float stop_eta_fraction = -1.0f;
    // half eta after not improving in that many epochs
    // must be at least 2
    // 0 -> disable
    size_t no_improvement_in = 0;
    // momentum co-efficient
    float mu = 0.0f;
    // regularization parameter
    float lambda_l1 = 0.0f;
    float lambda_l2 = 0.0f;

    const Data* training_data = nullptr;
    const Data* test_data     = nullptr;

    // run time
    float eta                    = -1.0f;
    bool  monitor_test_cost      = false;
    bool  monitor_test_accuracy  = false;
    bool  monitor_train_cost     = false;
    bool  monitor_train_accuracy = false;

    // results
    std::vector<float>
        test_costs, test_accuracies,
        train_costs, train_accuracies;

    // check if required parameters are given
    bool is_valid()
    {
        return mini_batch_size && init_eta && max_epochs && training_data != nullptr;
    }
};

inline std::ostream& operator<<(std::ostream& out, const HyperParameter& hy)
{
    out << "\tmini batch size: " << hy.mini_batch_size << std::endl;
    out << "\tmu: " << hy.mu << std::endl;
    out << "\ttraining set size: " << hy.training_data->get_x().n_cols << std::endl;

    if (hy.no_improvement_in)
        out << "Using learning rate schedule with starting eta: " << hy.init_eta << std::endl;
    if (hy.no_improvement_in && hy.stop_eta_fraction)
        out << "Stopping early when eta drops below: 1/" << hy.stop_eta_fraction << std::endl;
    else
        out << "\tepochs: " << hy.max_epochs << std::endl;

    if (hy.test_data != nullptr)
        out << "\tusing test data of size: " << hy.test_data->get_x().n_cols << std::endl;
    else
        out << "\tusing no test data" << std::endl;

    if (hy.lambda_l1)
        out << "\tusing L1 regularization with lambda: " << hy.lambda_l1 << std::endl;
    if (hy.lambda_l2)
        out << "\tusing L2 regularization with lambda: " << hy.lambda_l2 << std::endl;
    return out;
}
