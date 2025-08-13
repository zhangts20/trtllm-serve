#include "args_utils.h"
#include "cxxopts.hpp"
#include "log_utils.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

/// @brief Parse input args
/// @param argc The number of inputs
/// @param argv The input contents
/// @param envp The runtime environment variables
/// @return The struct of InputConfig
InputConfig parseInferArgs(int argc, char **argv, char **envp) {
    InputConfig input_config;
    // clang-format off
    cxxopts::Options options("MAIN", "A cpp inference based on TensorRT-LLM.");
    options.add_options()("help", "Print help");
    options.add_options()("model_dir", "The input engine directory.", cxxopts::value<std::string>());
    options.add_options()("input_text", "The input text for inference.", cxxopts::value<std::string>()->default_value("What is Deep Learning?"));
    options.add_options()("max_new_tokens", "The max generated tokens.", cxxopts::value<int>()->default_value("17"));
    options.add_options()("streaming", "Whether to use streaming inference.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("num_beams", "The number of return sequences.", cxxopts::value<int>()->default_value("1"));
    options.add_options()("top_k", "Samples from the k most probable words.", cxxopts::value<int>()->default_value("1"));
    options.add_options()("top_p", "Samples from words whose cumulative probability reaches p.", cxxopts::value<float>()->default_value("0.9"));
    // clang-format on
    cxxopts::ParseResult args = options.parse(argc, argv);
    // Check
    if (args.count("help")) {
        LOG_INFO(options.help());
        exit(1);
    }
    if (!args.count("model_dir")) {
        LOG_ERROR("The model dir is not given.\n");
    }
    std::optional<fs::path> model_dir = args["model_dir"].as<std::string>();
    if (!fs::exists(model_dir.value())) {
        LOG_ERROR("The model dir does not exist.\n");
    }
    input_config.model_dir = model_dir;
    input_config.input_text = args["input_text"].as<std::string>();
    input_config.sampling_parameters.max_new_tokens = args["max_new_tokens"].as<int>();
    input_config.sampling_parameters.is_streaming = args["streaming"].as<bool>();
    input_config.sampling_parameters.num_beams = args["num_beams"].as<int>();
    input_config.sampling_parameters.top_k = args["top_k"].as<int>();
    input_config.sampling_parameters.top_p = args["top_p"].as<float>();

    return input_config;
}

InputServerConfig parseServerArgs(int argc, char **argv, char **envp) {
    InputServerConfig input_server_config;
    cxxopts::Options options("main", "A cpp inference of TensorRT-LLM.");
    options.add_options()("help", "Print help");
    options.add_options()("model_dir", "The input model directory.", cxxopts::value<std::string>());
    options.add_options()("port", "The port of serving.", cxxopts::value<int>()->default_value("18001"));
    // TODO
    options.add_options()("capacity_scheduler_policy", "The policy used to select the subset of available requets.", cxxopts::value<std::string()>());
    cxxopts::ParseResult args = options.parse(argc, argv);

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        exit(1);
    }
    if (!args.count("model_dir")) {
        std::cerr << "The model dir is not given.\n";
        exit(1);
    }
    fs::path model_dir = args["model_dir"].as<std::string>();
    if (!fs::exists(model_dir)) {
        std::cerr << "The model dir does not exist.\n";
        exit(1);
    }
    input_server_config.model_dir = model_dir;
    input_server_config.port = args["port"].as<int>();

    return input_server_config;
}
