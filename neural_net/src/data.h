#pragma once
#include "log.h"

#include <armadillo>
#include <stddef.h>

namespace NeuralNet {
class Data {
private:
    // one column per data set
    // input above desired output
    arma::fmat m_data;
    size_t     m_x_size;
    size_t     m_y_size;

public:
    Data(size_t n_cols, size_t x_size, size_t y_size)
        : m_data(x_size + y_size, n_cols, arma::fill::zeros), m_x_size(x_size), m_y_size(y_size) {}

    Data(arma::fmat data, size_t x_size, size_t y_size): m_data(data), m_x_size(x_size), m_y_size(y_size) {}

    // input
    const arma::subview<float> get_x() const { return m_data.rows(0, m_x_size - 1); }
    arma::subview<float>       get_x() { return m_data.rows(0, m_x_size - 1); }
    // desired output
    const arma::subview<float> get_y() const { return m_data.rows(m_x_size, m_x_size + m_y_size - 1); }
    arma::subview<float>       get_y() { return m_data.rows(m_x_size, m_x_size + m_y_size - 1); }

    // no bounds checking
    const arma::subview<float> get_mini_x(size_t offset, size_t length) const {
        return m_data.submat(0, offset, m_x_size - 1, offset + length - 1);
    }
    // no bounds checking
    arma::subview<float> get_mini_x(size_t offset, size_t length) {
        return m_data.submat(0, offset, m_x_size - 1, offset + length - 1);
    }
    // no bounds checking
    const arma::subview<float> get_mini_y(size_t offset, size_t length) const {
        return m_data.submat(m_x_size, offset, m_x_size + m_y_size - 1, offset + length - 1);
    }
    // no bounds checking
    arma::subview<float> get_mini_y(size_t offset, size_t length) {
        return m_data.submat(m_x_size, offset, m_x_size + m_y_size - 1, offset + length - 1);
    }

    Data get_shuffled() const {
        // todo: better random
        arma::arma_rng::set_seed_random();
        return {arma::shuffle(m_data, 1), m_x_size, m_y_size};
    }
    void shuffle() {
        // todo: better random
        arma::arma_rng::set_seed_random();
        m_data = arma::shuffle(m_data, 1);
    }

    // switch input and desired output
    Data get_switched() {
        Data switched_data(m_data.n_cols, m_y_size, m_x_size);
        switched_data.get_x() = get_y();
        switched_data.get_y() = get_x();
        return switched_data;
    }

    // reduce size of data
    Data get_sub(size_t offset, size_t length) const {
        if(offset + length > m_data.n_cols)
            raise_critical("Requested sub data is invlaid.");
        return {m_data.cols(offset, offset + length - 1), m_x_size, m_y_size};
    }
    void sub(size_t offset, size_t length) {
        if(offset + length > m_data.n_cols)
            raise_critical("Requested sub data is invlaid.");
        m_data = m_data.cols(offset, offset + length - 1);
    }
};
} // namespace NeuralNet
