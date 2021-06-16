#include "pch.h"

#include "network.h"

#include <random>

Network::Network(const std::vector<size_t>& sizes, std::shared_ptr<Cost> cost)
{
    // todo: make multi threaded
    arma::arma_rng::set_seed_random();

    m_num_layers = sizes.size();
    m_sizes      = sizes;

    default_weight_init();
    reset_vel();

    m_cost = cost;
}

Network::Network(const std::string& json_path)
{
    std::ifstream file(json_path);
    if (!file)
        raise_error("Can't open input json file!");
    json json_net;
    file >> json_net;
    file.close();

    m_sizes      = static_cast<std::vector<size_t>>(json_net["sizes"]);
    m_num_layers = m_sizes.size();

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
        m_weights.push_back(weight);
    }

    for (const std::vector<float>& b : static_cast<std::vector<std::vector<float>>>(json_net["biases"]))
        m_biases.push_back(b);

    m_cost = Cost::get(json_net["cost"]);

    reset_vel();
}

void Network::save_json(const std::string& path)
{
    // loop over layer-by-layer weight sets
    std::vector<std::vector<std::vector<float>>> serialized_weights;
    for (const arma::fmat& w : m_weights)
    {
        std::vector<std::vector<float>> serialized_weight;
        // add column by column
        for (unsigned int i = 0; i < w.n_cols; ++i)
            serialized_weight.push_back(arma::conv_to<std::vector<float>>::from(w.col(i)));
        serialized_weights.push_back(serialized_weight);
    }
    std::vector<std::vector<float>> serialized_biases;
    for (const arma::fvec& b : m_biases)
        serialized_biases.push_back(arma::conv_to<std::vector<float>>::from(b));

    json json_net = {
        { "sizes", m_sizes },
        { "weights", serialized_weights },
        { "biases", serialized_biases },
        { "cost", m_cost->to_str() }
    };

    std::ofstream file(path);
    if (!file)
        raise_error("Can't open output json file!");
    file << std::setw(4) << json_net;
    file.close();
};

std::string Network::to_str() const
{
    std::stringstream buffer;
    buffer << "<Network: sizes: ";
    for (size_t size : m_sizes)
        buffer << size << " ";
    buffer << std::endl
           << "biases:" << std::endl;
    for (const arma::fvec& bias : m_biases)
        buffer << bias.n_rows << " " << bias.n_cols << std::endl;
    buffer << "weights:" << std::endl;
    for (const arma::fmat& weight : m_weights)
        buffer << weight.n_rows << " " << weight.n_cols << std::endl;
    buffer << ">";
    return buffer.str();
}

size_t Network::total_accuracy(const Data* data) const
{
    size_t sum = 0;
    // go over all data sets
    for (size_t i = 0; i < data->get_x().n_cols; ++i)
    {
        arma::subview<float> x = data->get_x().col(i);
        arma::subview<float> y = data->get_y().col(i);
        arma::fvec           a = feedforward(x);

        // determine highest confidences
        float  highest_correct_confidence;
        size_t correct_number    = -1;
        bool   got_first_correct = false;
        float  highest_selected_confidence;
        size_t selected_number    = -1;
        bool   got_first_selected = false;

        for (size_t idx = 0; idx < a.n_rows; ++idx)
        {
            if (!got_first_correct || y[idx] > highest_correct_confidence)
            {
                highest_correct_confidence = y[idx];
                correct_number             = idx;
                got_first_correct          = true;
            }
            if (!got_first_selected || a[idx] > highest_selected_confidence)
            {
                highest_selected_confidence = a[idx];
                selected_number             = idx;
                got_first_selected          = true;
            }
        }
        sum += selected_number == correct_number;
    }
    return sum;
}

float Network::total_cost(const Data* data, float lambda_l1, float lambda_l2) const
{
    float cost = 0.0f;
    // go over all data sets
    for (size_t i = 0; i < data->get_x().n_cols; ++i)
    {
        arma::subview<float> x = data->get_x().col(i);
        arma::subview<float> y = data->get_y().col(i);
        arma::fvec           a = feedforward(x);

        cost += m_cost->fn(a, y);
    }
    // take average
    cost /= data->get_x().n_cols;

    // L1 regularization
    if (lambda_l1)
    {
        // sum of absolute of all weights
        float sum = 0.0f;
        for (const arma::fmat& w : m_weights)
        {
            sum += arma::accu(arma::abs(w));
        }
        cost += (lambda_l1 / data->get_x().n_cols) * sum;
    }
    // L2 regularization
    if (lambda_l2)
    {
        // sum of squares of all weights
        float sum = 0.0f;
        for (const arma::fmat& w : m_weights)
        {
            // euclidean norm:
            // || a b ||
            // || c d ||2 = sqrt(a**2 + b**2 + c**2 + d**2)
            float norm = arma::norm(w);
            // the square root has to be removed
            sum += norm * norm;
        }
        cost += 0.5f * (lambda_l2 / data->get_x().n_cols) * sum;
    }
    return cost;
}

void Network::default_weight_init()
{
    m_biases.resize(0);
    // one for each layer except input layer
    m_biases.reserve(m_num_layers - 1);
    for (size_t layer_idx = 1; layer_idx < m_num_layers; ++layer_idx)
    {
        m_biases.emplace_back(m_sizes[layer_idx], 1, arma::fill::randn);
    }

    m_weights.resize(0);
    // one for each layer except input layer
    // one for each space between layers
    m_weights.reserve(m_num_layers - 1);
    for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
    {
        // from next layer to current layer
        m_weights.emplace_back(m_sizes[left_layer_idx + 1], m_sizes[left_layer_idx], arma::fill::randn);
        m_weights[left_layer_idx] /= sqrt(m_sizes[left_layer_idx + 1]);
    }
}

void Network::large_weight_init()
{
    m_biases.resize(0);
    // one for each layer except input layer
    m_biases.reserve(m_num_layers - 1);
    for (size_t layer_idx = 1; layer_idx < m_num_layers; ++layer_idx)
    {
        m_biases.emplace_back(m_sizes[layer_idx], 1, arma::fill::randn);
    }

    m_weights.resize(0);
    // one for each layer except input layer
    // one for each space between layers
    m_weights.reserve(m_num_layers - 1);
    for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
    {
        // from next layer to current layer
        m_weights.emplace_back(m_sizes[left_layer_idx + 1], m_sizes[left_layer_idx], arma::fill::randn);
    }
}

arma::fmat Network::feedforward(arma::fmat a) const
{
    // loop over each layer
    for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
    {
        //                               <- actually of right layer
        arma::fmat biases_mat = m_biases[left_layer_idx] * arma::fmat(1, a.n_cols, arma::fill::ones);
        a                     = sigmoid(m_weights[left_layer_idx] * a + biases_mat);
    }
    return a;
}

void Network::sgd(const Data* training_data,
                  size_t      epochs,
                  size_t      mini_batch_size,
                  float       eta,
                  float       mu,
                  float       lambda_l1,
                  float       lambda_l2,
                  const Data* eval_data,
                  LearnCFG*   learn_cfg)
{
    size_t n = training_data->get_x().n_cols;
    // info block
    std::cout << "Using stochastic gradient descent:" << std::endl;
    std::cout << "\tepochs: " << epochs << std::endl;
    std::cout << "\tmini batch size: " << mini_batch_size << std::endl;
    std::cout << "\teta: " << eta << std::endl;
    std::cout << "\tmu: " << mu << std::endl;
    std::cout << "\ttraining set size: " << n << std::endl;
    if (eval_data != nullptr)
        std::cout << "\tusing evaluation data of size: " << eval_data->get_x().n_cols << std::endl;
    else
        std::cout << "\tusing no evalution data" << std::endl;
    if (lambda_l1)
        std::cout << "\tusing L1 regularization with lambda: " << lambda_l1 << std::endl;
    if (lambda_l2)
        std::cout << "\tusing L2 regularization with lambda: " << lambda_l2 << std::endl;

    // go over epochs
    for (size_t e = 0; e < epochs; ++e)
    {
        Data this_training_data = training_data->get_shuffled();

        // go over mini batches
        for (size_t offset = 0; offset < n; offset += mini_batch_size)
        {
            // make last batch smaller if necessary
            size_t length = offset + mini_batch_size >= n ? n - offset : mini_batch_size;
            update_mini_batch(this_training_data.get_mini_x(offset, length),
                              this_training_data.get_mini_y(offset, length),
                              eta,
                              mu,
                              lambda_l1,
                              lambda_l2,
                              n);
        }

        // print report
        std::cout << "Epoch " << e << " training complete";
        if (learn_cfg != nullptr)
        {
            if (learn_cfg->monitor_eval_cost)
            {
                // only when eval_data is given
                if (eval_data == nullptr)
                    raise_error("evaluation data is required for requested monitoring");
                float cost = total_cost(eval_data, lambda_l1, lambda_l2);
                learn_cfg->eval_costs.push_back(cost);
                std::cout << "  \tCost on evaluation data: " << cost;
            }
            if (learn_cfg->monitor_eval_accuracy)
            {
                if (eval_data == nullptr)
                    raise_error("evaluation data is required for requested monitoring");
                float accuracy = total_accuracy(eval_data);
                learn_cfg->eval_accuracies.push_back(accuracy);
                size_t n_eval = eval_data->get_y().n_cols;
                std::cout << "  \tAccuracy on evaluation data: " << accuracy << " / " << n_eval;
            }

            if (learn_cfg->monitor_train_cost)
            {
                float cost = total_cost(training_data, lambda_l1, lambda_l2);
                learn_cfg->train_costs.push_back(cost);
                std::cout << "  \tCost on training data: " << cost;
            }
            if (learn_cfg->monitor_train_accuracy)
            {
                float accuracy = total_accuracy(training_data);
                learn_cfg->train_accuracies.push_back(accuracy);
                std::cout << "  \tAccuracy on training data: " << accuracy << " / " << n;
            }
        }
        std::cout << std::endl;
    }
}

void Network::reset_vel()
{
    m_vel_biases.resize(m_biases.size());
    m_vel_weights.resize(m_weights.size());
    // set to size of weights and biases
    // start at all 0
    for (size_t i = 0; i < m_vel_biases.size(); ++i)
    {
        m_vel_biases[i]  = arma::fvec(m_biases[i].n_rows, arma::fill::zeros);
        m_vel_weights[i] = arma::fmat(m_weights[i].n_rows, m_weights[i].n_cols, arma::fill::zeros);
    }
}

void Network::update_mini_batch(const arma::subview<float> x,
                                const arma::subview<float> y,
                                float                      eta,
                                float                      mu,
                                float                      lambda_l1,
                                float                      lambda_l2,
                                size_t                     n)
{
    // sums of gradients <- how do certain weights and biases change the cost
    // layer-wise
    std::vector<arma::fvec> nabla_b(m_biases.size());
    std::vector<arma::fmat> nabla_w(m_weights.size());
    // set to size of weights and biases
    // start at all 0
    for (size_t i = 0; i < nabla_b.size(); ++i)
    {
        nabla_b[i] = arma::fvec(m_biases[i].n_rows, arma::fill::zeros);
        nabla_w[i] = arma::fmat(m_weights[i].n_rows, m_weights[i].n_cols, arma::fill::zeros);
    }

    // use backprop to calculate gradient -> de-/increase delta
    // use matrix or multiple vectors
#if 1
    backprop(x, y, nabla_b, nabla_w);
#else
    for (size_t idx = 0; idx < x.n_cols; ++idx)
    {
        backprop(x.col(idx), y.col(idx), nabla_b, nabla_w);
    }
#endif

    // optimization
    float eta_over_length  = eta / x.n_cols;
    float lambda_l1_over_n = lambda_l1 / n;
    float lambda_l2_over_n = lambda_l2 / n;
    // update weights and biases
    // move in opposite direction -> reduce cost
    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        // include weight decay
        // L1 regularization
        m_vel_weights[i] -= eta * lambda_l1_over_n * arma::sign(m_weights[i]);
        // L2 regularization
        m_vel_weights[i] *= 1.0f - eta * lambda_l2_over_n;

        // update velocity
        // apply momentum co-efficient
        m_vel_weights[i] *= mu;
        m_vel_biases[i] *= mu;
        // approx gradient with mini batch
        m_vel_weights[i] -= eta_over_length * nabla_w[i];
        m_vel_biases[i] -= eta_over_length * nabla_b[i];

        // apply velocity
        m_weights[i] += m_vel_weights[i];
        m_biases[i] += m_vel_biases[i];
    }
}

void Network::backprop(const arma::subview<float> x,
                       const arma::subview<float> y,
                       std::vector<arma::fvec>&   nabla_b,
                       std::vector<arma::fmat>&   nabla_w) const
{
    // activations layer by layer <- needed by backprop algorithm
    // one per layer
    std::vector<arma::fmat> activations { x };
    activations.reserve(m_num_layers);
    // list of inputs for sigmoid function
    // one for each layer, except input
    std::vector<arma::fmat> zs;
    zs.reserve(m_num_layers - 1);

    // feedforward
    for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
    {
        // extend biases with as many copied columns as there are data sets in the batch
        //                               <- actually right layer
        arma::fmat biases_mat = m_biases[left_layer_idx] * arma::fmat(1, x.n_cols, arma::fill::ones);
        // weighted input
        zs.emplace_back(m_weights[left_layer_idx] * activations[activations.size() - 1] + biases_mat);
        activations.emplace_back(sigmoid(zs[zs.size() - 1]));
    }

    // calculate error for last layer (BP1)
    arma::fmat error = m_cost->error(zs[zs.size() - 1], activations[activations.size() - 1], y);
    // get gradient with respect to biases (BP3)
    // sum errors from each data set together -> sum into single column
    nabla_b[nabla_b.size() - 1] += arma::sum(error, 1);
    // get gradient with respect to weights (BP4)
    nabla_w[nabla_w.size() - 1] += error * activations[activations.size() - 2].t();

    // for all other layers
    // start at penultimate element of zs and go back to first
    for (int64_t layer_idx = m_num_layers - 3; layer_idx >= 0; --layer_idx)
    {
        // get input for sigmoid function of current layer
        arma::fmat sp = sigmoid_prime(zs[layer_idx]);
        // calculate error for current layer with error from layer to the right (BP2)
        error = (m_weights[layer_idx + 1].t() * error) % sp;

        // update gradient like with last layer
        nabla_b[layer_idx] += arma::sum(error, 1);
        // activations is one longer than zs
        nabla_w[layer_idx] += error * activations[layer_idx].t();
    }
}
