#pragma once
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace NeuralNet
{
class Log
{
private:
    static std::shared_ptr<spdlog::logger> s_logger;

public:
    static void init();

    static std::shared_ptr<spdlog::logger>& get_logger()
    {
        return s_logger;
    }

    static void set_level(spdlog::level::level_enum log_level)
    {
        s_logger->set_level(log_level);
    }
};

#ifndef NDEBUG
#define raise_critical(...)                                                             \
    do                                                                                  \
    {                                                                                   \
        Log::get_logger()->critical(__VA_ARGS__);                                       \
        Log::get_logger()->critical("(in {}:{}; in function: {})", __FILE__, __func__); \
        std::exit(EXIT_FAILURE);                                                        \
    } while (0)
#else
#define raise_critical(...)                       \
    do                                            \
    {                                             \
        Log::get_logger()->critical(__VA_ARGS__); \
        std::exit(EXIT_FAILURE);                  \
    } while (0)
#endif
// used for too much log
#define log_extra(...) Log::get_logger()->trace(__VA_ARGS__)
// #define log_info(...)  Log::get_logger()->info(__VA_ARGS__)
// general infos
#define log_general(...) Log::get_logger()->debug(__VA_ARGS__)
// something isn't the way it should be but there aren't any immediate consequences
#define log_warn(...) Log::get_logger()->warn(__VA_ARGS__)
// errors that can be recovered from
#define log_error(...) Log::get_logger()->error(__VA_ARGS__)
} // namespace NeuralNet
