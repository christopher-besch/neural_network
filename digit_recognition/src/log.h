#pragma once
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

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
    {                                                                                   \
        Log::get_logger()->critical(__VA_ARGS__);                                       \
        Log::get_logger()->critical("(in {}:{}; in function: {})", __FILE__, __func__); \
        std::exit(EXIT_FAILURE);                                                        \
    }
#else
#define raise_critical(...)                       \
    {                                             \
        Log::get_logger()->critical(__VA_ARGS__); \
        std::exit(EXIT_FAILURE);                  \
    }
#endif
#define log_info(...)  Log::get_logger()->info(__VA_ARGS__)
#define log_debug(...) Log::get_logger()->debug(__VA_ARGS__)
