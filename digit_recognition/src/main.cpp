#include "pch.h"

#include "costs.h"
#include "net/learn.h"
#include "net/net.h"
#include "net/setup.h"
#include "read_mnist.h"
#include "utils.h"

#include <chrono>
#include <iostream>

void print_img(const arma::fvec& img)
{
    for (int y = 0; y < 28; ++y)
    {
        for (int x = 0; x < 28; ++x)
        {
            float pixel = img[28 * y + x];
            std::cout << (pixel > 0.5f ? '#' : ' ');
        }
        std::cout << std::endl;
    }
}

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
    training_data.sub(50000);

    // x and y switched
    Data switched_training_data = training_data.get_switched();
    Data switched_test_data     = test_data.get_switched();

    begin = std::chrono::high_resolution_clock::now();

#if 1
    Network* net = create_network({ 784, 100, 10 }, Cost::get("cross_entropy"));

    // set monitoring
    LearnCFG learn_cfg;
    learn_cfg.monitor_test_cost      = true;
    learn_cfg.monitor_test_accuracy  = true;
    learn_cfg.monitor_train_cost     = true;
    learn_cfg.monitor_train_accuracy = true;

    // learn network
    sgd(net,
        &training_data,
        30,   // epochs
        10,   // mini_batch_size
        0.1f, // eta
        0,
        0.7f, // mu
        0.0f, // lambda for L1 regularization
        5.0f, // lambda for L2 regularization
        &test_data,
        &learn_cfg);
    save_json(net, "out_net.json");

    delete net;
#else
    // switched
    Network* net = create_network({ 10, 30, 784 }, Cost::get("cross_entropy"));
    sgd(net,
        &switched_training_data,
        5,    // epochs
        10,   // mini_batch_size
        0.5f, // eta
        0,
        0.0f, // mu
        5.0f, // lambda for L1 regularization
        0.0f, // lambda for L2 regularization
        nullptr,
        nullptr);

    arma::fmat input0 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    arma::fmat input1 = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
    arma::fmat input2 = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    arma::fmat input3 = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    arma::fmat input4 = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
    arma::fmat input5 = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    arma::fmat input6 = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
    arma::fmat input7 = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
    arma::fmat input8 = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
    arma::fmat input9 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    input0            = input0.t();
    input1            = input1.t();
    input2            = input2.t();
    input3            = input3.t();
    input4            = input4.t();
    input5            = input5.t();
    input6            = input6.t();
    input7            = input7.t();
    input8            = input8.t();
    input9            = input9.t();
    print_img(feedforward(net, input0));
    print_img(feedforward(net, input1));
    print_img(feedforward(net, input2));
    print_img(feedforward(net, input3));
    print_img(feedforward(net, input4));
    print_img(feedforward(net, input5));
    print_img(feedforward(net, input6));
    print_img(feedforward(net, input7));
    print_img(feedforward(net, input8));
    print_img(feedforward(net, input9));
    delete net;
#endif

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
