#include "neural_net.h"

#include <iostream>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#define file_slash '\\'
#else
#define file_slash '/'
#endif

NeuralNet::Data load_data(const std::string& matches_path) {
    // home_goals;guest_goals;home_guess_rel;draw_guess_rel;guest_guess_rel;home_guess;draw_guess;guest_guess;year;league;int_id
    try {
        std::ifstream file;
        file.open(matches_path, std::ios::in);
        if(!file.is_open())
            raise_critical("Can't open matches file: {}", matches_path);
        std::string buffer;

        std::getline(file, buffer);
        size_t lines = 0;
        while(std::getline(file, buffer))
            ++lines;
        // return to beginning
        file.clear();
        file.seekg(0);
        log_client_extra("found {} data points", lines);

        // skip header
        std::getline(file, buffer);
        NeuralNet::Data data(lines, 3, 12);
        for(size_t i = 0; std::getline(file, buffer); ++i) {
            std::stringstream buffer_ss(buffer);
            // home_goals
            std::getline(buffer_ss, buffer, ';');
            // anything bigger than 5 can't be expressed
            int home_goals = std::min(std::stoi(buffer), 5);
            // guest_goals
            std::getline(buffer_ss, buffer, ';');
            int guest_goals = std::min(std::stoi(buffer), 5);
            // home_guess_rel
            std::getline(buffer_ss, buffer, ';');
            float home_guess_rel = std::stof(buffer);
            // draw_guess_rel
            std::getline(buffer_ss, buffer, ';');
            float draw_guess_rel = std::stof(buffer);
            // guest_guess_rel
            std::getline(buffer_ss, buffer, ';');
            float guest_guess_rel = std::stof(buffer);
            // home_guess
            std::getline(buffer_ss, buffer, ';');
            // draw_guess
            std::getline(buffer_ss, buffer, ';');
            // guest_guess
            std::getline(buffer_ss, buffer, ';');
            // year
            std::getline(buffer_ss, buffer, ';');
            // league
            std::getline(buffer_ss, buffer, ';');
            // int_id
            std::getline(buffer_ss, buffer, ';');

            // input
            data.get_x().at(0, i) = home_guess_rel;
            data.get_x().at(1, i) = draw_guess_rel;
            data.get_x().at(2, i) = guest_guess_rel;

            // output
            //             data.get_y().at(0, i) = (home_goals >> 0) & 1;
            //             data.get_y().at(1, i) = (home_goals >> 1) & 1;
            //             data.get_y().at(2, i) = (home_goals >> 2) & 1;
            //             data.get_y().at(3, i) = (home_goals >> 3) & 1;
            //
            //             data.get_y().at(4, i) = (guest_goals >> 0) & 1;
            //             data.get_y().at(5, i) = (guest_goals >> 1) & 1;
            //             data.get_y().at(6, i) = (guest_goals >> 2) & 1;
            //             data.get_y().at(7, i) = (guest_goals >> 3) & 1;
            data.get_y().at(home_goals, i)      = true;
            data.get_y().at(guest_goals + 6, i) = true;
        }

        file.close();
        return data;
    } catch(const std::exception& ex) {
        raise_critical("{} error parsing matches file '{}'", ex.what(), matches_path);
    }
}

int main(int argc, char* argv[]) {
    NeuralNet::init();
    NeuralNet::Log::set_client_level(NeuralNet::LogLevel::Extra);
    if(argc < 2)
        raise_critical("Please specify the path to the data as the first parameter.");
    std::stringstream root_data_path;
    root_data_path << argv[1] << file_slash << "kickprophet" << file_slash;
    NeuralNet::Data data = load_data(root_data_path.str() + std::string("bundesliga.csv"));

    NeuralNet::Data train_data = data.get_sub(0, 3034);
    NeuralNet::Data test_data  = data.get_sub(3034, 100);
    NeuralNet::Data eval_data  = data.get_sub(3134, 100);

    NeuralNet::Network net;
    // create_network(net, {3, 100, 100, 12, 12}, true);
    create_network(net, {3, 100, 100, 12});
    net.evaluator = [](const arma::fvec& y, const arma::fvec& a) {
        // both home and guest have to be correct
        return NeuralNet::DefaultEvaluater::classifier(y.rows(0, 5), a.rows(0, 5)) &&
               NeuralNet::DefaultEvaluater::classifier(y.rows(6, 11), a.rows(6, 11));
    };
    // direct pass-through
    // net.weights[net.weights.size() - 1](0, 0)   = 1.0f;
    // net.weights[net.weights.size() - 1](1, 1)   = 1.0f;
    // net.weights[net.weights.size() - 1](2, 2)   = 1.0f;
    // net.weights[net.weights.size() - 1](3, 3)   = 1.0f;
    // net.weights[net.weights.size() - 1](4, 4)   = 1.0f;
    // net.weights[net.weights.size() - 1](5, 5)   = 1.0f;
    // net.weights[net.weights.size() - 1](6, 6)   = 1.0f;
    // net.weights[net.weights.size() - 1](7, 7)   = 1.0f;
    // net.weights[net.weights.size() - 1](8, 8)   = 1.0f;
    // net.weights[net.weights.size() - 1](9, 9)   = 1.0f;
    // net.weights[net.weights.size() - 1](10, 10) = 1.0f;
    // net.weights[net.weights.size() - 1](11, 11) = 1.0f;
    // shift -> smaller than 0.5 is negative <- compensate effect of sigmoid function
    // net.biases[net.biases.size() - 1](0)  = 0.5f;
    // net.biases[net.biases.size() - 1](1)  = 0.5f;
    // net.biases[net.biases.size() - 1](2)  = 0.5f;
    // net.biases[net.biases.size() - 1](3)  = 0.5f;
    // net.biases[net.biases.size() - 1](4)  = 0.5f;
    // net.biases[net.biases.size() - 1](5)  = 0.5f;
    // net.biases[net.biases.size() - 1](6)  = 0.5f;
    // net.biases[net.biases.size() - 1](6)  = 0.5f;
    // net.biases[net.biases.size() - 1](7)  = 0.5f;
    // net.biases[net.biases.size() - 1](8)  = 0.5f;
    // net.biases[net.biases.size() - 1](9)  = 0.5f;
    // net.biases[net.biases.size() - 1](10) = 0.5f;
    // net.biases[net.biases.size() - 1](11) = 0.5f;

    NeuralNet::HyperParameter hy;
    hy.training_data = &train_data;
    hy.test_data     = &test_data;
    hy.eval_data     = &eval_data;

    // coarse
    NeuralNet::Log::set_hyper_level(NeuralNet::LogLevel::Extra);
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Warn);
    // NeuralNet::coarse_hyper_surf(net, hy);

    // fine
    NeuralNet::Log::set_hyper_level(NeuralNet::LogLevel::Extra);
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Warn);
    hy.max_epochs             = 100;
    hy.learning_schedule_type = NeuralNet::LearningScheduleType::TestAccuracy;
    hy.no_improvement_in      = 10;
    hy.stop_eta_fraction      = 128;
    hy.reset_monitor();
    hy.monitor_test_accuracy = true;
    // NeuralNet::bounce_hyper_surf(net, hy, 3, 5);

    // learn
    hy.max_epochs             = 300;
    hy.monitor_test_accuracy  = true;
    hy.monitor_train_accuracy = true;
    hy.monitor_eval_accuracy  = true;
    hy.monitor_train_cost     = true;
    hy.monitor_test_cost      = true;
    hy.monitor_eval_cost      = true;
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Extra);

    NeuralNet::sgd(net, hy);

    arma::fmat at = {0.9f, 0.05f, 0.05f};
    arma::fmat a  = at.t();
    std::cout << feedforward(net, a);
}
