#include "pch.h"

#include "costs.h"
#include "net/hyper_surfer.h"
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
    if (argc < 2)
        raise_error("Please specify the path to the data as the first parameter.");
    // load data
    std::stringstream root_data_path;
    root_data_path << argv[1] << file_slash << "mnist" << file_slash;

    Data big_data      = load_data(root_data_path.str() + std::string("training_images"), root_data_path.str() + std::string("training_labels"));
    Data test_data     = load_data(root_data_path.str() + std::string("test_images"), root_data_path.str() + std::string("test_labels"));
    Data training_data = big_data.get_sub(0, 50000);
    Data eval_data     = big_data.get_sub(50000, 10000);

    // x and y switched
    Data switched_training_data = training_data.get_switched();
    Data switched_test_data     = test_data.get_switched();

    Data cp_training_data = training_data.get_sub(0, 3000);
    Data cp_test_data     = test_data.get_sub(0, 1000);
    Data cp_eval_data     = eval_data.get_sub(0, 1000);

#if 1
    Network net;
    create_network(net, { 784, 30, 10 }, Cost::get("cross_entropy"));

    // set monitoring
    // HyperParameter hy;
    // hy.mini_batch_size = 10;
    // hy.init_eta          = 0.1f;
    // hy.max_epochs        = 1000;
    // hy.stop_eta_fraction = 128.0f;
    // hy.no_improvement_in = 10;
    // hy.mu                = 0.0f;
    // hy.lambda_l1         = 0.0f;
    // hy.lambda_l2         = 5.0f;

    HyperParameter hy;
    hy.no_improvement_in = 100;
    hy.stop_eta_fraction = 512.0f;
    hy.lambda_l1         = 0.0f;
    hy.training_data     = &cp_training_data;
    hy.test_data         = &cp_test_data;
    hy.eval_data         = &cp_eval_data;

    hyper_surf(net, hy);
    // hy.init_eta        = 0.914062f;
    // hy.lambda_l2       = 15.2344f;
    // hy.mu              = 0.0f;
    // hy.mini_batch_size = 305;

    hy.training_data          = &training_data;
    hy.test_data              = &test_data;
    hy.eval_data              = &eval_data;
    hy.monitor_test_cost      = false;
    hy.monitor_test_accuracy  = true;
    hy.monitor_eval_cost      = false;
    hy.monitor_eval_accuracy  = false;
    hy.monitor_train_cost     = false;
    hy.monitor_train_accuracy = false;

    // learn network
    sgd(net, hy);
    save_json(net, "out_net.json");
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
    return 0;
}
