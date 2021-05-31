#pragma once
#include "pch.h"

#include "utils.h"

class Data
{
private:
    // one column per data set
    // input above desired output
    arma::fmat m_data;
    size_t     m_x_size;
    size_t     m_y_size;

public:
    Data(size_t n_cols, size_t x_size, size_t y_size)
        : m_data(x_size + y_size, n_cols, arma::fill::zeros),
          m_x_size(x_size),
          m_y_size(y_size) {}

    Data(arma::fmat data, size_t x_size, size_t y_size)
        : m_data(data), m_x_size(x_size), m_y_size(y_size) {}

    // input
    const arma::subview<float> get_x() const
    {
        return m_data.rows(0, m_x_size - 1);
    }
    arma::subview<float> get_x()
    {
        return m_data.rows(0, m_x_size - 1);
    }
    // desired output
    const arma::subview<float> get_y() const
    {
        return m_data.rows(m_x_size, m_x_size + m_y_size - 1);
    }
    arma::subview<float> get_y()
    {
        return m_data.rows(m_x_size, m_x_size + m_y_size - 1);
    }

    // no bounds checking
    const arma::subview<float> get_mini_x(size_t offset, size_t length) const
    {
        return m_data.submat(0, offset,
                             m_x_size - 1, offset + length - 1);
    }
    // no bounds checking
    arma::subview<float> get_mini_x(size_t offset, size_t length)
    {
        return m_data.submat(0, offset,
                             m_x_size - 1, offset + length - 1);
    }
    // no bounds checking
    const arma::subview<float> get_mini_y(size_t offset, size_t length) const
    {
        return m_data.submat(m_x_size, offset,
                             m_x_size + m_y_size - 1, offset + length - 1);
    }
    // no bounds checking
    arma::subview<float> get_mini_y(size_t offset, size_t length)
    {
        return m_data.submat(m_x_size, offset,
                             m_x_size + m_y_size - 1, offset + length - 1);
    }

    Data get_shuffled() const
    {
        // todo: better random
        arma::arma_rng::set_seed_random();
        return Data(arma::shuffle(m_data, 1), m_x_size, m_y_size);
    }
    void shuffle()
    {
        // todo: better random
        arma::arma_rng::set_seed_random();
        m_data = arma::shuffle(m_data, 1);
    }

    // not fast, just to play around
    // todo: untested
    Data get_switched()
    {
        Data switched_data(m_data.n_cols, m_y_size, m_x_size);
        switched_data.get_x() = get_y();
        switched_data.get_y() = get_x();
        return switched_data;
    }
};

inline int32_t get_int32_t(std::ifstream& file)
{
    int32_t num = 0;
    file.read(reinterpret_cast<char*>(&num), 4);
    // number is in big endian
    if (is_little_endian())
        num = swap(num);
    return num;
}

// todo: not enough checks for production
inline Data load_data(std::string images_path, std::string labels_path)
{
    try
    {
        ////////////
        // images //
        ////////////
        std::ifstream images_file;
        images_file.open(images_path, std::ios::in | std::ios::binary);

        // magic number
        if (get_int32_t(images_file) != 2051)
            raise_error("images file '" << images_path << "' has an unsupported magic number");
        // data set dimensions
        int32_t images_amount = get_int32_t(images_file);
        int32_t n_rows        = get_int32_t(images_file);
        int32_t n_cols        = get_int32_t(images_file);

        ////////////
        // labels //
        ////////////
        std::ifstream labels_file;
        labels_file.open(labels_path, std::ios::in | std::ios::binary);

        // magic number
        if (get_int32_t(labels_file) != 2049)
            raise_error("labels file '" << labels_path << "' has an unsupported magic number");
        // data set dimensions
        int32_t labels_amount = get_int32_t(labels_file);
        if (images_amount != labels_amount)
            raise_error("images file '" << images_path << "' and labels file '" << labels_path << "' don't have same amount (" << images_amount << " and " << labels_amount << ") of data sets");

        //////////////////
        // read dataset //
        //////////////////
        // one column per data set
        // Data data(images_amount, n_rows * n_cols, 10);
        Data data(images_amount, 10, n_rows * n_cols);
        for (int i = 0; i < images_amount; ++i)
        {
            for (int pixel_idx = 0; pixel_idx < n_rows * n_cols; ++pixel_idx)
            {
                uint8_t pixel;
                images_file.read(reinterpret_cast<char*>(&pixel), 1);
                data.get_y().at(pixel_idx, i) = pixel / 255.0f;
            }
            uint8_t label;
            labels_file.read(reinterpret_cast<char*>(&label), 1);
            data.get_x().at(label, i) = 1.0f;
        }

        images_file.close();
        labels_file.close();
        // todo: bad copy?
        return data;
    }
    catch (const std::exception& ex)
    {
        raise_error(ex.what() << " error parsing MNIST file " << images_path << " or " << labels_path << "!");
    }
}
