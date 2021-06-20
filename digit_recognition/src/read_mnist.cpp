#include "read_mnist.h"

#include "neural_net.h"
#include "utils.h"

inline int32_t get_int32_t(std::ifstream& file)
{
    int32_t num = 0;
    file.read(reinterpret_cast<char*>(&num), 4);
    // number is in big endian
    if (is_little_endian())
        num = swap(num);
    return num;
}

NeuralNet::Data load_data(std::string images_path, std::string labels_path)
{
    try
    {
        ////////////
        // images //
        ////////////
        std::ifstream images_file;
        images_file.open(images_path, std::ios::in | std::ios::binary);
        if (!images_file.is_open())
            raise_error("Can't open images file: " << images_path);

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
        if (!images_file.is_open())
            raise_error("Can't open labels file: " << images_path);

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
        NeuralNet::Data data(images_amount, n_rows * n_cols, 10);
        for (int i = 0; i < images_amount; ++i)
        {
            for (int pixel_idx = 0; pixel_idx < n_rows * n_cols; ++pixel_idx)
            {
                uint8_t pixel;
                images_file.read(reinterpret_cast<char*>(&pixel), 1);
                data.get_x().at(pixel_idx, i) = pixel / 255.0f;
            }
            uint8_t label;
            labels_file.read(reinterpret_cast<char*>(&label), 1);
            data.get_y().at(label, i) = 1.0f;
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
