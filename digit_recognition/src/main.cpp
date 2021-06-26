#include "neural_net.h"
#include "read_mnist.h"

#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#define file_slash '\\'
#else
#define file_slash '/'
#endif

void print_img(const arma::fvec& img) {
    for(int y = 0; y < 28; ++y) {
        for(int x = 0; x < 28; ++x) {
            float pixel = img[28 * y + x];
            std::cout << (pixel > 0.5f ? '#' : ' ');
        }
        std::cout << std::endl;
    }
}

int main(int argc, const char* argv[]) {
    NeuralNet::init();
    NeuralNet::Log::set_client_level(NeuralNet::LogLevel::Extra);
    if(argc < 2)
        raise_critical("Please specify the path to the data as the first parameter.");
    std::stringstream root_data_path;
    root_data_path << argv[1] << file_slash << "mnist" << file_slash;

    NeuralNet::Data big_data = load_data(root_data_path.str() + std::string("training_images"),
                                         root_data_path.str() + std::string("training_labels"));
    NeuralNet::Data test_data =
        load_data(root_data_path.str() + std::string("test_images"), root_data_path.str() + std::string("test_labels"));
    NeuralNet::Data training_data = big_data.get_sub(0, 50000);
    NeuralNet::Data eval_data     = big_data.get_sub(50000, 10000);

    // x and y switched
    NeuralNet::Data switched_training_data = training_data.get_switched();
    NeuralNet::Data switched_test_data     = test_data.get_switched();

    NeuralNet::Data cp_training_data = training_data.get_sub(0, 3000);
    NeuralNet::Data cp_test_data     = test_data.get_sub(0, 1000);
    NeuralNet::Data cp_eval_data     = eval_data.get_sub(0, 1000);

#if 1
    // create network
    NeuralNet::Network net;
    create_network(net, {784, 100, 10});

    // set some hyper parameters
    NeuralNet::HyperParameter hy;
    hy.training_data         = &training_data;
    hy.test_data             = &test_data;
    hy.eval_data             = &eval_data;
    hy.monitor_eval_accuracy = true;
    // hy.learning_schedule_type = NeuralNet::LearningScheduleType::None;
    // hy.max_epochs             = 60;
    // hy.mini_batch_size        = 10;
    // hy.init_eta               = 0.1f;
    // hy.lambda_l2              = 5.0f;

    hy.learning_schedule_type = NeuralNet::LearningScheduleType::EvalAccuracy;
    hy.max_epochs             = 500;
    hy.mini_batch_size        = 29;
    hy.init_eta               = 1.0686059f;
    hy.lambda_l2              = 0.17070314f;
    hy.mu                     = 0.0234375f;
    hy.no_improvement_in      = 10;
    hy.stop_eta_fraction      = 128;
    NeuralNet::sgd(net, hy);

    return 0;

    hy.lambda_l1     = 0.0f;
    hy.training_data = &cp_training_data;
    hy.test_data     = &cp_test_data;
    hy.eval_data     = &cp_eval_data;

    // coarse hyper surf
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Warn);
    NeuralNet::Log::set_hyper_level(NeuralNet::LogLevel::Extra);
    NeuralNet::coarse_hyper_surf(net, hy);

    // fine hyper surf
    hy.reset_monitor();
    hy.monitor_eval_accuracy  = true;
    hy.learning_schedule_type = NeuralNet::LearningScheduleType::EvalAccuracy;
    hy.max_epochs             = 200;
    hy.no_improvement_in      = 5;
    hy.stop_eta_fraction      = 128.0f;
    hy.training_data          = &training_data;
    hy.test_data              = &test_data;
    hy.eval_data              = &eval_data;
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Extra);
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Extra);
    NeuralNet::bounce_hyper_surf(net, hy, 2, 3);

    // learn
    hy.reset_monitor();
    hy.monitor_test_accuracy  = true;
    hy.max_epochs             = 1000;
    hy.no_improvement_in      = 10;
    hy.stop_eta_fraction      = 512.0f;
    hy.learning_schedule_type = NeuralNet::LearningScheduleType::TestAccuracy;
    sgd(net, hy);
    save_json(net, "out_net.json");
#else
    // switched
    Network* net = create_network({10, 30, 784}, Cost::get("cross_entropy"));
    sgd(net, &switched_training_data,
        5,    // epochs
        10,   // mini_batch_size
        0.5f, // eta
        0,
        0.0f, // mu
        5.0f, // lambda for L1 regularization
        0.0f, // lambda for L2 regularization
        nullptr, nullptr);

    arma::fmat input0 = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    arma::fmat input1 = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    arma::fmat input2 = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
    arma::fmat input3 = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    arma::fmat input4 = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    arma::fmat input5 = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    arma::fmat input6 = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};
    arma::fmat input7 = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
    arma::fmat input8 = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0};
    arma::fmat input9 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
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
