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
        NeuralNet::Data data(lines, 3, 8);
        for(size_t i = 0; std::getline(file, buffer); ++i) {
            std::stringstream buffer_ss(buffer);
            // home_goals
            std::getline(buffer_ss, buffer, ';');
            // anything bigger than 15 can't be expressed
            int home_goals = std::min(std::stoi(buffer), 15);
            // guest_goals
            std::getline(buffer_ss, buffer, ';');
            int guest_goals = std::min(std::stoi(buffer), 15);
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
            data.get_y().at(0, i) = home_goals & (1 << 0) >> 0;
            data.get_y().at(1, i) = home_goals & (1 << 1) >> 1;
            data.get_y().at(2, i) = home_goals & (1 << 2) >> 2;
            data.get_y().at(3, i) = home_goals & (1 << 3) >> 3;

            data.get_y().at(4, i) = guest_goals & (1 << 0) >> 0;
            data.get_y().at(5, i) = guest_goals & (1 << 1) >> 1;
            data.get_y().at(6, i) = guest_goals & (1 << 2) >> 2;
            data.get_y().at(7, i) = guest_goals & (1 << 3) >> 3;

            std::cout << home_goals << std::endl;
            std::cout << guest_goals << std::endl;
            std::cout << data.get_y().col(0) << std::endl;
            std::exit(EXIT_FAILURE);
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

    NeuralNet::Data train_data = data.get_sub(0, 2234);
    // todo: fix
    NeuralNet::Data test_data = data.get_sub(2234, 500);
    NeuralNet::Data eval_data = data.get_sub(2734, 500);

    NeuralNet::Network net;
    create_network(net, {3, 20, 20, 8}, NeuralNet::Cost::get("cross_entropy"));

    std::cout << train_data.get_x().col(0) << std::endl;
    std::cout << train_data.get_y().col(0) << std::endl;
    return 0;

    net.evaluator = [](const arma::fvec& y, const arma::fvec& a) {
        std::cout << y << std::endl;
        std::cout << a << std::endl;
        for(int i = 0; i < 8; ++i) {
            if(std::round(y[i]) != a[i])
                return false;
        }
        return true;
    };

    NeuralNet::HyperParameter hy;
    hy.training_data         = &train_data;
    hy.test_data             = &test_data;
    hy.eval_data             = &eval_data;
    hy.monitor_eval_accuracy = true;
    NeuralNet::Log::set_hyper_level(NeuralNet::LogLevel::Extra);
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Warn);
    // NeuralNet::coarse_hyper_surf(net, hy);
    NeuralNet::coarse_eta_surf(net, hy);
    NeuralNet::mini_batch_size_surf(net, hy);
    log_client_general(hy.to_str());
    return 0;

    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Extra);
    hy.learning_schedule_type = NeuralNet::LearningScheduleType::EvalAccuracy;
    hy.no_improvement_in      = 100;
    hy.max_epochs             = 10;
    hy.stop_eta_fraction      = 128;
    hy.monitor_eval_cost      = true;
    hy.monitor_train_cost     = true;
    hy.monitor_train_accuracy = true;
    NeuralNet::sgd(net, hy);
    arma::fmat at = {0.9f, 0.05f, 0.05f};
    arma::fmat a  = at.t();
    std::cout << NeuralNet::feedforward(net, a) << std::endl;
    return 0;

    hy.learning_schedule_type = NeuralNet::LearningScheduleType::EvalAccuracy;
    hy.max_epochs             = 500;
    hy.no_improvement_in      = 10;
    hy.stop_eta_fraction      = 128;
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Extra);
    NeuralNet::bounce_hyper_surf(net, hy, 2, 3);
    NeuralNet::sgd(net, hy);
}
