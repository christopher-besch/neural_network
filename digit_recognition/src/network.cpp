#include "pch.h"

#include "network.h"

#include <random>

Network::Network(const std::vector<size_t>& sizes)
{
    // todo: make multi threaded
    arma::arma_rng::set_seed_random();

    m_num_layers = sizes.size();
    m_sizes      = sizes;

    // one for each layer except input layer
    m_biases.reserve(m_num_layers - 1);
    for (size_t layer_idx = 1; layer_idx < m_num_layers; ++layer_idx)
    {
        m_biases.emplace_back(m_sizes[layer_idx], 1, arma::fill::randn);
    }

    // one for each space between layers
    m_weights.reserve(m_num_layers - 1);
    for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
    {
        // from next layer to current layer
        m_weights.emplace_back(m_sizes[left_layer_idx + 1], m_sizes[left_layer_idx], arma::fill::randn);
    }
}

arma::fmat Network::feedforward(arma::fmat a)
{
    // loop over each layer
    for (size_t left_layer_idx = 0; left_layer_idx < m_num_layers - 1; ++left_layer_idx)
    {
        arma::fmat biases_mat = m_biases[left_layer_idx] * arma::fmat(1, a.n_cols, arma::fill::ones);
        // only care about first column
        //                                          <- actually of right layer
        a = sigmoid(m_weights[left_layer_idx] * a + biases_mat);
    }
    return a;
}

void Network::sgd(const Data& training_data,
                  size_t      epochs,
                  size_t      mini_batch_size,
                  float       eta,
                  const Data& test_data)
{
    size_t n_test = test_data.x.n_cols;
    size_t n      = training_data.x.n_cols;

    for (size_t e = 0; e < epochs; ++e)
    {
        // todo: make fast with threading

        std::shuffle(training_data.begin(), training_data.end(), std::mt19937_64 {});

        for (size_t offset = 0; offset < n; offset += mini_batch_size)
        {
            // make last batch smaller if necessary
            size_t length = offset + mini_batch_size >= n ? n - offset : mini_batch_size;
            update_mini_batch(training_data, offset, length, eta);
        }

        if (n_test)
            std::cout << "Epoch " << e << ": " << evaluate(test_data) << " / " << n_test << std::endl;
        else
            std::cout << "Epoch " << e << ": complete";
    }
}

void Network::update_mini_batch(const std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
                                size_t offset, size_t length, float eta)
{
    // sums of gradients <- how do certain weights and biases change the cost
    // layer wise
    std::vector<arma::fvec> nabla_b(m_biases.size());
    std::vector<arma::fmat> nabla_w(m_weights.size());
    // start at all 0
    for (size_t i = 0; i < nabla_b.size(); ++i)
    {
        nabla_b[i] = arma::fvec(m_biases[i].n_rows, arma::fill::zeros);
        nabla_w[i] = arma::fmat(m_weights[i].n_rows, m_weights[i].n_cols, arma::fill::zeros);
    }

    // loop over all training sets in mini batch
    for (size_t idx = offset; idx < offset + length; ++idx)
    {
        // x: input
        // y: desired output
        auto& [x, y] = training_data[idx];

        // use backprop to calculate gradient
        // de-/increase delta
        backprop(x, y, nabla_b, nabla_w);
    }

    // optimization
    float eta_over_length = eta / length;
    // update weights and biases
    for (size_t i = 0; i < m_weights.size(); ++i)
    {
        m_biases[i] -= eta_over_length * nabla_b[i];
        m_weights[i] -= eta_over_length * nabla_w[i];
    }
}

void Network::backprop(const arma::fmat& x, const arma::fmat& y, std::vector<arma::fvec>& nabla_b, std::vector<arma::fmat>& nabla_w)
{
    // activations layer by layer <- needed for backprop algorithm
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
    arma::fmat error = cost_derivative(activations[activations.size() - 1], y) % sigmoid_prime(zs[zs.size() - 1]);
    // sum errors from each data set together -> sum rows into single column
    // get gradient with respect to biases (BP3)
    nabla_b[nabla_b.size() - 1] += arma::sum(error, 1);
    // get gradient with respect to weights (BP4)
    nabla_w[nabla_w.size() - 1] += arma::sum(error, 1) * activations[activations.size() - 2].t();

    // for all other layers
    // start at penultimate element of zs and go back to first
    for (int64_t layer_idx = m_num_layers - 3; layer_idx >= 0; --layer_idx)
    {
        // get input for sigmoid function of current layer
        arma::fvec sp = sigmoid_prime(zs[layer_idx]);
        // calculate error for current layer with error from layer to the right (BP2)
        error = (m_weights[layer_idx + 1].t() * error) % sp;

        // update gradient like with last layer
        nabla_b[layer_idx] += arma::sum(error, 1);
        // activations is one longer than zs
        nabla_w[layer_idx] += arma::sum(error, 1) * activations[layer_idx].t();
    }
}

size_t Network::evaluate(const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data)
{
    size_t sum = 0;
    // go over all data sets
    for (const auto& [x, y] : test_data)
    {
        arma::fvec a = feedforward(x);

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

std::string Network::to_string()
{
    std::stringstream buffer;
    buffer << "<Network: sizes: ";
    for (size_t size : m_sizes)
        buffer << size << " ";
    buffer << std::endl
           << "biases:" << std::endl;
    for (const arma::fvec& bias : m_biases)
        buffer << bias << std::endl;
    buffer << "weights:" << std::endl;
    for (const arma::fmat& weight : m_weights)
        buffer << weight << std::endl;
    buffer << ">";
    return buffer.str();
}
