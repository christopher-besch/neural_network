#include "pch.h"

#include "log.h"

std::shared_ptr<spdlog::logger> Log::s_logger;

void Log::init()
{
    try
    {
        // log to console and file
        std::vector<spdlog::sink_ptr> log_sinks;
        log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("neural_net.log", true));

        log_sinks[0]->set_pattern("%^[%T] %n: %v%$");
        log_sinks[1]->set_pattern("[%T] [%l] %n: %v");

        s_logger = std::make_shared<spdlog::logger>("console", begin(log_sinks), end(log_sinks));
        spdlog::register_logger(s_logger);
        s_logger->set_level(spdlog::level::trace);
        s_logger->flush_on(spdlog::level::trace);
    }
    catch (const spdlog::spdlog_ex& ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
