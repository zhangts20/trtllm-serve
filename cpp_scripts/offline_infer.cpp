#include "session.hpp"

int main(int argc, char **argv, char **envp) {
    InputConfig input_config = parseArgs(argc, argv, envp);
    // Initialize tensorrt_llm plugins
    initTrtLlmPlugins();
    // TP Launcher
    const char *world_rank_ = std::getenv("OMPI_COMM_WORLD_RANK");
    int world_rank = 0;
    if (world_rank_ != nullptr) {
        world_rank = std::stoi(world_rank_);
    }
    const char *world_size_ = std::getenv("OMPI_COMM_WORLD_SIZE");
    int world_size = 1;
    if (world_size_ != nullptr) {
        world_size = std::stoi(world_size_);
    }
    LOG_DEBUG("Process " + std::to_string(world_rank) + " of " +
              std::to_string(world_size));
    // Initialize InferenceSession
    InferenceSession inference_session;
    inference_session.initialize(input_config.engine_dir);
    inference_session.addRequests(input_config.input_text);
    inference_session.inferRequests();
}