#include "pch.h"

#include "network.h"

Network::Network(const std::vector<size_t>& sizes)
{
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

arma::fmat& Network::feedforward(arma::fmat& a)
{
    // loop over each layer
    for (size_t right_layer_idx = 0; right_layer_idx < m_num_layers - 1; ++right_layer_idx)
    {
        // only care about first column
        a = sigmoid(m_weights[right_layer_idx] * a + m_biases[right_layer_idx]);
    }
    return a;
}

void Network::sgd(std::vector<std::pair<arma::fvec, arma::fvec>>& training_data,
                  size_t epochs, size_t mini_batch_size, float eta,
                  const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data)
{
    size_t n_test = test_data.size();
    size_t n      = training_data.size();

    for (size_t e = 0; e < epochs; ++e)
    {
        // todo: make fast with threading
        std::shuffle(training_data.begin(), training_data.end(), std::mt19937_64 {});

        for (size_t offset = 0; offset < n; offset += mini_batch_size)
        {
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
                                size_t offset, size_t length, size_t eta)
{
    // sums of gradients
    // start at all 0
    std::vector<arma::fmat> nabla_b(m_biases.size());
    std::vector<arma::fmat> nabla_w(m_weights.size());
    for (size_t i = 0; i < nabla_b.size(); ++i)
    {
        nabla_b[i] = arma::fmat(m_biases[i].n_rows, m_biases[i].n_cols, arma::fill::zeros);
        nabla_w[i] = arma::fmat(m_weights[i].n_rows, m_weights[i].n_cols, arma::fill::zeros);
    }

    // loop ever all training sets in mini batch
    for (size_t idx = offset; idx < offset + length; ++idx)
    {
        // x: input
        // y: desired output
        auto& [x, y] = training_data[idx];

        // use backprop to calculate gradient
        std::vector<arma::fmat> delta_nabla_b, delta_nabla_w;
        backprop(x, y, delta_nabla_b, delta_nabla_w);

        // de-/in-crease delta
        for (size_t i = 0; i < nabla_b.size(); ++i)
        {
            nabla_b[i] += delta_nabla_b[i];
            nabla_w[i] += delta_nabla_w[i];
        }
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

float Network::evaluate(const std::vector<std::pair<arma::fvec, arma::fvec>>& test_data)
{
}

void Network::backprop(const arma::fvec& x, const arma::fvec& y, std::vector<arma::fmat>& delta_nabla_b, std::vector<arma::fmat>& delta_nabla_w)
{
}

std::string Network::to_string()
{
    std::stringstream buffer;
    buffer << "<Network: sizes: ";
    for (size_t size : m_sizes)
        buffer << size << " ";
    buffer << std::endl
           << "biases:" << std::endl;
    for (const arma::fmat& bias : m_biases)
        buffer << bias << std::endl;
    buffer << "weights:" << std::endl;
    for (const arma::fmat& weight : m_weights)
        buffer << weight << std::endl;
    buffer << ">";
    return buffer.str();
}
