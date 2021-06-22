#pragma once
#include "costs.h"
#include "data.h"

namespace NeuralNet {
struct Network {
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

inline std::ostream& operator<<(std::ostream& out, const Network& net) {
    out << "<Network: sizes: ";
    for(size_t size: net.sizes)
        out << size << " ";
    out << std::endl << "biases:" << std::endl;
    for(const arma::fvec& bias: net.biases)
        out << bias.n_rows << " " << bias.n_cols << std::endl;
    out << "weights:" << std::endl;
    for(const arma::fmat& weight: net.weights)
        out << weight.n_rows << " " << weight.n_cols << std::endl;
    out << ">";
    return out;
}

enum class LearningScheduleType : uint8_t { None = 0, TestAccuracy, EvalAccuracy };

// data in hyper-space
struct HyperParameter {
    // required
    size_t mini_batch_size = 0;
    // start learning rate
    float init_eta = 0.0f;

    // optional
    // stop after these epchs regardless of any schedule
    size_t max_epochs = 200;

    // momentum co-efficient
    // 0 -> full friction, like without momentum
    float mu = 0.0f;
    // regularization parameter
    float lambda_l1 = 0.0f;
    float lambda_l2 = 0.0f;

    LearningScheduleType learning_schedule_type = LearningScheduleType::None;
    // half eta after not improving in that many epochs
    // must be at least 2
    // 0 -> disable
    size_t no_improvement_in = 0;
    // 0 -> disabled
    // ignored if learning rate schedule disabled
    float stop_eta_fraction = 0.0f;

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
    std::vector<float> test_costs, test_accuracies, eval_costs, eval_accuracies, train_costs, train_accuracies;

    void reset_results() {
        test_costs.resize(0);
        test_accuracies.resize(0);
        train_costs.resize(0);
        train_accuracies.resize(0);
    }

    void reset_monitor() {
        monitor_test_cost      = false;
        monitor_test_accuracy  = false;
        monitor_eval_cost      = false;
        monitor_eval_accuracy  = false;
        monitor_train_cost     = false;
        monitor_train_accuracy = false;
    }

    // check if required parameters are given
    void is_valid() const {
        if((monitor_test_cost || monitor_test_accuracy) && test_data == nullptr)
            raise_critical("Test data is required for requested monitoring.");
        if((monitor_eval_cost || monitor_eval_accuracy) && eval_data == nullptr)
            raise_critical("Evaluation data is required for requested monitoring.");

        switch(learning_schedule_type) {
        case LearningScheduleType::TestAccuracy:
            if(test_data == nullptr)
                raise_critical("Test data is required for early stopping and learning rate schedule.");
            if(!monitor_test_accuracy)
                raise_critical(
                    "The test data accuracy has to be monitored for early stopping and learning rate schedule.");
            break;
        case LearningScheduleType::EvalAccuracy:
            if(eval_data == nullptr)
                raise_critical("Evaluation data is required for early stopping and learning rate schedule.");
            if(!monitor_eval_accuracy)
                raise_critical(
                    "The evaluation data accuracy has to be monitored for early stopping and learning rate schedule.");
            break;
        case LearningScheduleType::None:
            break;
        }
        if(learning_schedule_type != LearningScheduleType::None) {
            if(no_improvement_in < 2)
                raise_critical("no_improvement_in has to be at least two.");
        }

        if(!mini_batch_size)
            raise_critical("mini_batch_size needs to be defined");
        if(!init_eta)
            raise_critical("init_eta needs to be defined");
        if(training_data == nullptr)
            raise_critical("training_data needs to be given");
    }

    std::string to_str() const {
        std::stringstream out;
        out << "\ttraining set size: " << training_data->get_x().n_cols << std::endl;
        if(test_data != nullptr)
            out << "\tusing test data of size: " << test_data->get_x().n_cols << std::endl;
        else
            out << "\tusing no test data" << std::endl;
        if(eval_data != nullptr)
            out << "\tusing evaluation data of size: " << eval_data->get_x().n_cols << std::endl;
        else
            out << "\tusing no evaluation data" << std::endl;

        out << "\tmini batch size: " << mini_batch_size << std::endl;

        switch(learning_schedule_type) {
        case LearningScheduleType::TestAccuracy:
            out << "\tUsing learning rate schedule on test data with starting eta: " << init_eta << std::endl;
            out << "\thalf eta when test accuracy didn't improve in the last " << no_improvement_in << " epochs"
                << std::endl;
            if(stop_eta_fraction)
                out << "\tStopping early when eta drops below: 1/" << stop_eta_fraction << std::endl;
            break;
        case LearningScheduleType::EvalAccuracy:
            out << "\tUsing learning rate schedule on evaluation data with starting eta: " << init_eta << std::endl;
            out << "\thalf eta when evaluation accuracy didn't improve in the last " << no_improvement_in << " epochs"
                << std::endl;
            if(stop_eta_fraction)
                out << "\tStopping early when eta drops below: 1/" << stop_eta_fraction << std::endl;
            break;
        case LearningScheduleType::None:
            out << "\tusing constant eta: " << init_eta << std::endl;
            break;
        }
        out << "\tstop at max epochs: " << max_epochs << std::endl;

        out << "\tmu: " << mu << std::endl;

        if(lambda_l1)
            out << "\tusing L1 regularization with lambda: " << lambda_l1 << std::endl;
        if(lambda_l2)
            out << "\tusing L2 regularization with lambda: " << lambda_l2;
        return out.str();
    }
};
} // namespace NeuralNet
