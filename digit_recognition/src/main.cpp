#include "pch.h"

#include "network.h"
#include "read_mnist.h"
#include "utils.h"

#include <chrono>
#include <iostream>

int main(int argc, const char* argv[])
{
    auto begin = std::chrono::high_resolution_clock::now();
    if (argc < 2)
        raise_error("Please specify the path to the data as the first parameter.");
    // load data
    std::stringstream root_data_path;
    root_data_path << argv[1] << file_slash << "mnist" << file_slash;

    Data training_data = load_data(root_data_path.str() + std::string("training_images"), root_data_path.str() + std::string("training_labels"));
    Data test_data     = load_data(root_data_path.str() + std::string("test_images"), root_data_path.str() + std::string("test_labels"));

    begin = std::chrono::high_resolution_clock::now();

#if 0
    int idx = 1;
    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            float pixel = training_data.get_x().at(28 * y + x, idx);
            std::cout << (pixel > 0.5f ? '#' : ' ');
        }
        std::cout << std::endl;
    }
    std::cout << training_data.get_y().col(idx) << std::endl;

    training_data.shuffle();
    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            float pixel = training_data.get_x().at(28 * y + x, idx);
            std::cout << (pixel > 0.5f ? '#' : ' ');
        }
        std::cout << std::endl;
    }
    std::cout << training_data.get_y().col(idx) << std::endl;
#endif

    // learn network
    Network net = Network({ 784, 30, 10 });
    net.sgd(&training_data, 5, 10, 3.0f, &test_data);

    // report
    auto      end        = std::chrono::high_resolution_clock::now();
    long long delta_time = (end - begin).count();
    if (delta_time > 1e9)
        std::cout << std::to_string(delta_time / 1e9f) << " seconds";
    else if (delta_time > 1e6)
        std::cout << std::to_string(delta_time / 1e6f) << " milliseconds";
    else if (delta_time > 1e3)
        std::cout << std::to_string(delta_time / 1e3f) << " microseconds";
    else
        std::cout << std::to_string(delta_time) << " nanoseconds";
    std::cout << std::endl;

    return 0;
}
