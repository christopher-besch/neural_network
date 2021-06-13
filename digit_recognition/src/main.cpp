#include "pch.h"

#include "network.h"
#include "read_mnist.h"
#include "utils.h"

#include <chrono>
#include <iostream>

#if 0
#include <cstdio>
#include <cstdlib>
#include <new>
// replacement of a minimal set of functions:
void* operator new(std::size_t sz) // no inline, required by [replacement.functions]/3
{
    std::printf("global op new called, size = %zu\n", sz);
    if (sz == 0)
        ++sz; // avoid std::malloc(0) which may return nullptr on success

    if (void* ptr = std::malloc(sz))
        return ptr;

    throw std::bad_alloc {}; // required by [new.delete.single]/3
}
void operator delete(void* ptr) noexcept
{
    std::puts("global op delete called");
    std::free(ptr);
}
#endif

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
    Data eval_data     = load_data(root_data_path.str() + std::string("test_images"), root_data_path.str() + std::string("test_labels"));

    // x and y switched
    Data switched_training_data = training_data.get_switched();
    Data switched_test_data     = eval_data.get_switched();

    begin = std::chrono::high_resolution_clock::now();

    std::shared_ptr<Cost> cross_entropy_cost = std::make_shared<CrossEntropyCost>();
    std::shared_ptr<Cost> quadratic_cost     = std::make_shared<QuadraticCost>();


#if 1
    training_data.sub(50000);
    Network net = Network({ 784, 100, 10 }, cross_entropy_cost);
    // set monitoring
    LearnCFG learn_cfg;
    learn_cfg.monitor_eval_cost      = false;
    learn_cfg.monitor_eval_accuracy  = true;
    learn_cfg.monitor_train_cost     = false;
    learn_cfg.monitor_train_accuracy = false;

    // learn network
    net.sgd(&training_data,
            60,   // epochs
            10,   // mini_batch_size
            0.1f, // eta
            0.0f, // lambda for L1 regularization
            5.0f, // lambda for L2 regularization
            &eval_data,
            &learn_cfg);

#else
    // switched
    Network net = Network({ 10, 30, 784 });
    net.sgd(&switched_training_data,
            1,    // epochs
            10,   // mini_batch_size
            0.5f, // eta
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
    print_img(net.feedforward(input0));
    print_img(net.feedforward(input1));
    print_img(net.feedforward(input2));
    print_img(net.feedforward(input3));
    print_img(net.feedforward(input4));
    print_img(net.feedforward(input5));
    print_img(net.feedforward(input6));
    print_img(net.feedforward(input7));
    print_img(net.feedforward(input8));
    print_img(net.feedforward(input9));
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
