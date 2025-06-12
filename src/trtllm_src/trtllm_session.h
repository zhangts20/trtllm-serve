#ifndef _TRTLLM_SESSION_H_
#define _TRTLLM_SESSION_H_

#include "common_utils.h"
#include "log_utils.h"
#include "nlohmann/json.hpp"
#include "tensorrt_llm/executor/executor.h"
#include "tokenizers_cpp.h"
#include "types.h"

using json = nlohmann::json;

namespace fs = std::filesystem;
namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

struct OutputConfig {
    tle::IdType request_id;
    std::vector<std::vector<tle::TokenIdType>> output_tokens;
    std::vector<std::vector<tle::FloatType>> output_logprobs;
    std::vector<std::string> finish_reason;
    std::vector<std::string> generated_text;
};

static std::map<tle::FinishReason, std::string> FinishReasonMapping = {
    {tle::FinishReason::kEND_ID, "end_id"},
    {tle::FinishReason::kLENGTH, "length"},
    {tle::FinishReason::kNOT_FINISHED, "running"},
};

class TokenizerSession;

class InferenceSession {
  private:
    // The input directory contains model file(s) and tokenizer file(s)
    std::string model_dir;
    // The Executor defined in executor.h
    std::unique_ptr<tle::Executor> executor;
    // The ExecutorConfig defined in executor.h
    std::unique_ptr<tle::ExecutorConfig> executor_config;
    // The requests and request_ids
    std::vector<tle::Request> requests;
    std::vector<tle::IdType> request_ids;

  public:
    // The session to encode input and decode output
    std::unique_ptr<TokenizerSession> tokenizer_session;

  public:
    InferenceSession()
        : executor_config(std::make_unique<tle::ExecutorConfig>()),
          tokenizer_session(std::make_unique<TokenizerSession>()) {}
    // Initialize inference session
    bool initialize(const std::string &engine_dir);
    // Initialize Executor
    void initializeExecutor(bool enable_kv_reuse);
    // Initialize requests
    void addRequests(const InputConfig &input_config);
    // Do inference
    void infer();
    // Do inference server
    std::optional<OutputConfig> serve();
};

class TokenizerSession {
  private:
    // The tokenizers::Tokenizer to encode and decode
    std::unique_ptr<tokenizers::Tokenizer> processor;

  public:
    // Initialization
    bool initialize(fs::path model_dir);
    // Encode input text
    bool encode(const std::string &input_text, tle::VecTokens &input_ids);
    // Decode output text
    bool decode(std::string &output_text, const tle::VecTokens &output_ids);
};

#endif
