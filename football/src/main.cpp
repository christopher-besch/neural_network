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
    NeuralNet::Data train_data = load_data(root_data_path.str() + std::string("train_data.csv"));
    NeuralNet::Data test_data  = load_data(root_data_path.str() + std::string("test_data.csv"));

    NeuralNet::Network net;
    // create_network(net, {3, 100, 100, 12, 12}, true);
    create_network(net, {3, 100, 100, 12});

    // accuracy only
    net.evaluator = [](const arma::fvec& y, const arma::fvec& a) {
        // both home and guest goals have to be correct
        return NeuralNet::DefaultEvaluater::classifier(y.rows(0, 5), a.rows(0, 5)) &&
               NeuralNet::DefaultEvaluater::classifier(y.rows(6, 11), a.rows(6, 11));
    };

    // correct rules
    //     net.evaluator = [](const arma::fvec& y, const arma::fvec& a) {
    //         size_t correct_home;
    //         size_t selected_home;
    //         NeuralNet::get_highest_index(y.rows(0, 5), a.rows(0, 5), correct_home, selected_home);
    //         size_t correct_guest;
    //         size_t selected_guest;
    //         NeuralNet::get_highest_index(y.rows(6, 11), a.rows(6, 11), correct_guest, selected_guest);
    //
    //         float score = 0.0f;
    //         // everything correct
    //         if((correct_home == selected_home) && (correct_guest == selected_guest))
    //             score = 4.0f;
    //         // correct draw
    //         else if((correct_guest == correct_home) && (selected_guest == selected_home))
    //             score = 2.0f;
    //         // difference correct
    //         else if((selected_home - selected_guest) == (correct_home - correct_guest))
    //             score = 3.0f;
    //         // tendency correct
    //         else if((selected_home > selected_guest) == (correct_home > correct_guest))
    //             score = 2.0f;
    //
    //         // todo: test
    //         // std::cout << "--------------------------------------------------" << std::endl;
    //         // std::cout << correct_home << std::endl;
    //         // std::cout << correct_guest << std::endl;
    //         // std::cout << std::endl;
    //         // std::cout << selected_home << std::endl;
    //         // std::cout << selected_guest << std::endl;
    //         // std::cout << std::endl;
    //         // std::cout << score << std::endl;
    //
    //         return score;
    //     };

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
    // net.biases[net.biases.size() - 1](0)  = -0.5f;
    // net.biases[net.biases.size() - 1](1)  = -0.5f;
    // net.biases[net.biases.size() - 1](2)  = -0.5f;
    // net.biases[net.biases.size() - 1](3)  = -0.5f;
    // net.biases[net.biases.size() - 1](4)  = -0.5f;
    // net.biases[net.biases.size() - 1](5)  = -0.5f;
    // net.biases[net.biases.size() - 1](6)  = -0.5f;
    // net.biases[net.biases.size() - 1](6)  = -0.5f;
    // net.biases[net.biases.size() - 1](7)  = -0.5f;
    // net.biases[net.biases.size() - 1](8)  = -0.5f;
    // net.biases[net.biases.size() - 1](9)  = -0.5f;
    // net.biases[net.biases.size() - 1](10) = -0.5f;
    // net.biases[net.biases.size() - 1](11) = -0.5f;

    NeuralNet::HyperParameter hy;
    hy.training_data = &train_data;
    hy.test_data     = &test_data;
    hy.eval_data     = &test_data;

    // coarse
    NeuralNet::Log::set_hyper_level(NeuralNet::LogLevel::General);
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Warn);
    NeuralNet::coarse_hyper_surf(net, hy);

    // fine
    hy.max_epochs             = 100;
    hy.learning_schedule_type = NeuralNet::LearningScheduleType::TestAccuracy;
    hy.no_improvement_in      = 10;
    hy.stop_eta_fraction      = 32;
    hy.reset_monitor();
    hy.monitor_test_accuracy = true;
    NeuralNet::bounce_hyper_surf(net, hy, 3, 4);

    // learn
    hy.max_epochs        = 1000;
    hy.no_improvement_in = 100;
    hy.stop_eta_fraction = 32;
    hy.reset_monitor();
    hy.monitor_test_accuracy = true;
    // hy.monitor_train_accuracy = true;
    // hy.monitor_train_cost     = true;
    // hy.monitor_test_cost      = true;
    NeuralNet::Log::set_learn_level(NeuralNet::LogLevel::Extra);

    // todo: test
    //     hy.mini_batch_size = 30;
    //     hy.init_eta        = 1.69175f;
    //     hy.mu              = 0.166031f;
    //     hy.lambda_l2       = 0.270891f;
    //
    //     hy.mini_batch_size = 34;
    //     hy.init_eta        = 0.0021759f;
    //     hy.mu              = 0.998901f;
    //     hy.lambda_l2       = 0.000268912f;

    NeuralNet::sgd(net, hy);
    NeuralNet::save_json(net, "net.json");

    // NeuralNet::load_json_network(net, "net1.json");

    std::ofstream file("table.csv");
    if(!file)
        raise_critical("Can't open output csv file!");

    file << "home;guest;guess_home;guess_guest" << std::endl;
    arma::fmat in = {0.0f, 0.0f, 0.0f};
    for(size_t home = 0; home <= 100; ++home) {
        for(size_t guest = 0; guest + home <= 100; ++guest) {
            size_t draw = 100 - home - guest;
            if(home + draw + guest != 100) {
                std::cout << home << " " << draw << " " << guest << std::endl;
                std::cout << in << std::endl;
                raise_critical("broken");
            }
            in[0]           = home * 0.01f;
            in[1]           = draw * 0.01f;
            in[2]           = guest * 0.01f;
            arma::fmat t_in = in.t();
            arma::fvec res  = NeuralNet::feedforward(net, t_in);
            // std::cout << t_in << std::endl;
            // std::cout << res << std::endl;
            size_t guess_home, guess_guest;
            NeuralNet::get_highest_index(res.rows(0, 5), res.rows(0, 5), guess_home, guess_home);
            NeuralNet::get_highest_index(res.rows(6, 11), res.rows(6, 11), guess_guest, guess_guest);
            file << home << ";" << guest << ";" << guess_home << ";" << guess_guest << std::endl;
        }
    }
    file.close();

    // check result
    // arma::fmat at = {0.7f, 0.25f, 0.05f};
    // arma::fmat a  = at.t();
    // std::cout << feedforward(net, a);
    // NeuralNet::update_learn_status(net, hy);
}
