#pragma once
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace NeuralNet
{
enum class LogLevel
{
    Extra   = spdlog::level::trace,
    General = spdlog::level::debug,
    Warn    = spdlog::level::warn,
    Error   = spdlog::level::err
};

class Log
{
private:
    // std::cout and file
    static std::shared_ptr<spdlog::logger> s_learn_logger;
    static std::shared_ptr<spdlog::logger> s_hyper_logger;
    static std::shared_ptr<spdlog::logger> s_client_logger;
    // std::cerr
    static std::shared_ptr<spdlog::logger> s_error_logger;

public:
    static void init();

    static std::shared_ptr<spdlog::logger>& get_learn_logger()
    {
        return s_learn_logger;
    }
    static std::shared_ptr<spdlog::logger>& get_hyper_logger()
    {
        return s_hyper_logger;
    }
    static std::shared_ptr<spdlog::logger>& get_client_logger()
    {
        return s_client_logger;
    }
    static std::shared_ptr<spdlog::logger>& get_error_logger()
    {
        return s_error_logger;
    }

    static void set_learn_level(LogLevel log_level)
    {
        s_learn_logger->set_level(static_cast<spdlog::level::level_enum>(log_level));
    }
    static void set_hyper_level(LogLevel log_level)
    {
        s_hyper_logger->set_level(static_cast<spdlog::level::level_enum>(log_level));
    }
    static void set_client_level(LogLevel log_level)
    {
        s_client_logger->set_level(static_cast<spdlog::level::level_enum>(log_level));
    }
};

#ifndef NDEBUG
#define raise_critical(...)                                                                                \
    do                                                                                                     \
    {                                                                                                      \
        ::NeuralNet::Log::get_error_logger()->critical(__VA_ARGS__);                                       \
        ::NeuralNet::Log::get_error_logger()->critical("(in {}:{}; in function: {})", __FILE__, __func__); \
        std::exit(EXIT_FAILURE);                                                                           \
    } while (0)
#else
#define raise_critical(...)                                          \
    do                                                               \
    {                                                                \
        ::NeuralNet::Log::get_error_logger()->critical(__VA_ARGS__); \
        std::exit(EXIT_FAILURE);                                     \
    } while (0)
#endif

#define log_learn_extra(...)   Log::get_learn_logger()->trace(__VA_ARGS__)
#define log_learn_general(...) Log::get_learn_logger()->debug(__VA_ARGS__)
#define log_learn_warn(...)    Log::get_learn_logger()->warn(__VA_ARGS__)
#define log_learn_error(...)   Log::get_learn_logger()->error(__VA_ARGS__)

#define log_hyper_extra(...)   Log::get_hyper_logger()->trace(__VA_ARGS__)
#define log_hyper_general(...) Log::get_hyper_logger()->debug(__VA_ARGS__)
#define log_hyper_warn(...)    Log::get_hyper_logger()->warn(__VA_ARGS__)
#define log_hyper_error(...)   Log::get_hyper_logger()->error(__VA_ARGS__)

#define log_client_extra(...)   ::NeuralNet::Log::get_client_logger()->trace(__VA_ARGS__)
#define log_client_general(...) ::NeuralNet::Log::get_client_logger()->debug(__VA_ARGS__)
#define log_client_warn(...)    ::NeuralNet::Log::get_client_logger()->warn(__VA_ARGS__)
#define log_client_error(...)   ::NeuralNet::Log::get_client_logger()->error(__VA_ARGS__)
} // namespace NeuralNet
