#include "pch.h"

#include "network.h"
#include "read_mnist.h"
#include "utils.h"

#include <iostream>

int main(int argc, const char* argv[])
{
    if (argc < 2)
        raise_error("Please specify the path to the data as the first parameter.");
    // load data
    std::stringstream root_data_path;
    root_data_path << argv[1] << file_slash << "mnist" << file_slash;

    std::vector<std::pair<arma::fvec, arma::fvec>> training_data;
    load_data(root_data_path.str() + std::string("training_images"), root_data_path.str() + std::string("training_labels"), training_data);
    std::vector<std::pair<arma::fvec, arma::fvec>> test_data;
    load_data(root_data_path.str() + std::string("test_images"), root_data_path.str() + std::string("test_labels"), test_data);

#if 0
    for (int x = 0; x < 28; ++x)
    {
        for (int y = 0; y < 28; ++y)
        {
            float pixel = training_data[50].first[28 * y + x];
            std::cout << (pixel > 0.5f ? '#' : ' ');
        }
        std::cout << std::endl;
    }
#endif

    // learn network
    Network net = Network({ 784, 30, 10 });
    net.sgd(training_data, 30, 10, 3.0f, test_data);

    return 0;
}
