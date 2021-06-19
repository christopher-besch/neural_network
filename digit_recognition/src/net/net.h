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

    // optional
    // stop after these epchs regardless of any schedule
    size_t max_epochs = 200;
    // half eta after not improving in that many epochs
    // must be at least 2
    // 0 -> disable
    size_t no_improvement_in = 0;
    // 0 -> disabled
    // ignored if learning rate schedule disabled
    float stop_eta_fraction = 0.0f;
    // momentum co-efficient
    // 0 -> full friction, like without momentum
    float mu = 0.0f;
    // regularization parameter
    float lambda_l1 = 0.0f;
    float lambda_l2 = 0.0f;

    const Data* training_data = nullptr;
    const Data* test_data     = nullptr;
    const Data* eval_data     = nullptr;

    // run time
    bool      monitor_test_cost      = false;
    bool      monitor_test_accuracy  = false;
    bool      monitor_eval_cost      = false;
    bool      monitor_eval_accuracy  = false;
    bool      monitor_train_cost     = false;
    bool      monitor_train_accuracy = false;
    long long learn_time             = 0;

    // results
    std::vector<float>
        test_costs, test_accuracies,
        eval_costs, eval_accuracies,
        train_costs, train_accuracies;

    void reset_results()
    {
        test_costs.resize(0);
        test_accuracies.resize(0);
        train_costs.resize(0);
        train_accuracies.resize(0);
    }

    void reset_monitor()
    {
        monitor_test_cost      = false;
        monitor_test_accuracy  = false;
        monitor_eval_cost      = false;
        monitor_eval_accuracy  = false;
        monitor_train_cost     = false;
        monitor_train_accuracy = false;
    }


    // check if required parameters are given
    void is_valid() const
    {
        if ((monitor_test_cost || monitor_test_accuracy) && test_data == nullptr)
            raise_error("Test data is required for requested monitoring.");
        if ((monitor_eval_cost || monitor_eval_accuracy) && eval_data == nullptr)
            raise_error("Evaluation data is required for requested monitoring.");

        if (no_improvement_in)
        {
            if (test_data == nullptr)
                raise_error("Test data is required for early stopping and learning rate schedule.");
            if (!monitor_test_accuracy)
                raise_error("The test data accuracy has to be monitored for early stopping and learning rate schedule.");
            if (no_improvement_in < 2)
                raise_error("no_improvement_in has to be at least two or not given.");
        }

        if (!mini_batch_size)
            raise_error("mini_batch_size needs to be defined");
        if (!init_eta)
            raise_error("init_eta needs to be defined");
        if (training_data == nullptr)
            raise_error("training_data needs to be given");
    }
};

inline std::ostream& operator<<(std::ostream& out, const HyperParameter& hy)
{
    out << "\ttraining set size: " << hy.training_data->get_x().n_cols << std::endl;
    if (hy.test_data != nullptr)
        out << "\tusing test data of size: " << hy.test_data->get_x().n_cols << std::endl;
    else
        out << "\tusing no test data" << std::endl;
    if (hy.eval_data != nullptr)
        out << "\tusing evaluation data of size: " << hy.eval_data->get_x().n_cols << std::endl;
    else
        out << "\tusing no evaluation data" << std::endl;

    out << "\tmini batch size: " << hy.mini_batch_size << std::endl;

    if (hy.no_improvement_in)
    {
        out << "\tUsing learning rate schedule with starting eta: " << hy.init_eta << std::endl;
        out << "\thalf eta when test accuracy didn't improve in the last " << hy.no_improvement_in << " epochs" << std::endl;
        if (hy.stop_eta_fraction)
            out << "\tStopping early when eta drops below: 1/" << hy.stop_eta_fraction << std::endl;
    }
    else
        out << "\tusing constant eta: " << hy.init_eta << std::endl;
    out << "\tstop at max epochs: " << hy.max_epochs << std::endl;

    out << "\tmu: " << hy.mu << std::endl;

    if (hy.lambda_l1)
        out << "\tusing L1 regularization with lambda: " << hy.lambda_l1 << std::endl;
    if (hy.lambda_l2)
        out << "\tusing L2 regularization with lambda: " << hy.lambda_l2 << std::endl;
    return out;
}
