#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <filesystem>
#include <fstream>
#include <memory>

#include "cxxopts.hpp"
#include "nlohmann/json.hpp"
#include "sentencepiece_processor.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

using json = nlohmann::json;

namespace fs = std::filesystem;
namespace sp = sentencepiece;
namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

struct InputConfig {
    std::string engine_dir;
    std::string input_text;
};

InputConfig parseArgs(int argc, char **argv, char **envp);

class TokenizerSession;

class InferenceSession {
  public:
    // The input directory contains engine file(s) and tokenizer about
    std::string engine_dir;
    // The Executor defined in executor.h
    std::unique_ptr<tle::Executor> executor;
    // The ExecutorConfig defined in executor.h
    std::unique_ptr<tle::ExecutorConfig> executor_config;
    // The session to encode input and decode output
    std::unique_ptr<TokenizerSession> tokenizer_session;
    // The requests and request_ids
    std::vector<tle::Request> requests;
    std::vector<tle::IdType> request_ids;

  public:
    InferenceSession()
        : executor_config(std::make_unique<tle::ExecutorConfig>()),
          tokenizer_session(std::make_unique<TokenizerSession>()) {}
    // Initialize inference session
    bool initialize(std::string engine_dir);
    // Initialize ExecutorConfig
    void initializeExecutorConfig();
    // Initialize requests
    void addRequests(std::optional<std::string> input_text = std::nullopt);
    // Do inference
    void inferRequests();
};

class TokenizerSession {
  public:
    // The SentencePieceProcessor to encode and decode
    std::unique_ptr<sp::SentencePieceProcessor> processor;

  public:
    TokenizerSession()
        : processor(std::make_unique<sp::SentencePieceProcessor>()) {}
    // Initialization
    bool initialize(fs::path model_dir);
    // Encode input text
    bool encode(std::string input_text, tle::VecTokens &input_ids);
    // Decode output text
    bool decode(std::string &output_text, tle::VecTokens &output_ids);
};

#endif // _CONFIG_H_