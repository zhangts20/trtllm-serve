#include <filesystem>
#include <fstream>
#include <mpi/mpi.h>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

void newRequests(std::vector<tle::Request> &requests) {
    tle::VecTokens vec_tokens = {1, 1724, 338, 21784, 29257, 29973};
    tle::SizeType32 max_new_tokens = 17;
    tle::OutputConfig output_config = tle::OutputConfig(
        /* returnLogProbs =*/false,
        /* returnContextLogits =*/false,
        /* returnGenerationLogits =*/false,
        /* excludeInputFromOutput =*/false,
        /* returnEncoderOutput =*/false);
    tle::SamplingConfig sampling_config = tle::SamplingConfig(
        /* beamWidth =*/1,
        /* topK =*/std::nullopt,
        /* topP =*/std::nullopt,
        /* topPMin =*/std::nullopt,
        /* topPResetIds =*/std::nullopt,
        /* topPDecay =*/std::nullopt,
        /* seed =*/std::nullopt,
        /* temperature =*/std::nullopt,
        /* minTokens =*/std::nullopt,
        /* beamSearchDiversityRate =*/std::nullopt,
        /* repetitionPenalty =*/std::nullopt,
        /* presencePenalty =*/std::nullopt,
        /* lengthPenalty =*/std::nullopt,
        /* earlyStopping =*/std::nullopt,
        /* noRepeatNgramSize =*/std::nullopt);

    for (size_t i = 0; i < 8; ++i) {
        requests.push_back(tle::Request(
            /* inputTokenIds =*/vec_tokens,
            /* maxTokens =*/max_new_tokens + i,
            /* streaming =*/true,
            /* samplingConfig =*/sampling_config,
            /* outputConfig =*/output_config,
            /* endId =*/std::nullopt,
            /* padId =*/std::nullopt,
            /* positionIds =*/std::nullopt,
            /* badWords =*/std::nullopt,
            /* stopWords =*/std::nullopt,
            /* embeddingBias =*/std::nullopt,
            /* externalDraftTokensConfig =*/std::nullopt,
            /* pTuningConfig =*/std::nullopt,
            /* loraConfig =*/std::nullopt,
            /* lookaheadConfig =*/std::nullopt,
            /* logitsPostProcessorName =*/std::nullopt,
            /* encoderInputTokenIds =*/std::nullopt,
            /* clientId =*/std::nullopt,
            /* returnAllGeneratedTokens =*/false,
            /* priority =*/tle::Request::kDefaultPriority,
            /* type =*/tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
            /* contextPhaseParams =*/std::nullopt,
            /* encoderInputFeatures =*/std::nullopt,
            /* encoderOutputLength =*/std::nullopt,
            /* numReturnSequences =*/1));
    }
}

std::vector<tle::IdType> addRequests(tle::Executor &executor,
                                     std::vector<tle::Request> &requests) {
    std::vector<tle::IdType> request_ids;
    for (size_t i = 0; i < requests.size(); ++i) {
        if (executor.canEnqueueRequests()) {
            request_ids.push_back(
                executor.enqueueRequest(std::move(requests[i])));
        }
    }

    return request_ids;
}

void setValue(tle::ExecutorConfig &executor_config,
              const fs::path &engine_dir) {
    fs::path config_path = engine_dir / "config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "open config file failed" << std::endl;
        exit(-1);
    }

    json config;
    config_file >> config;
    // read value from config
    tle::SizeType32 max_beam_width = config["build_config"]["max_beam_width"];
    tle::SizeType32 max_batch_size = config["build_config"]["max_batch_size"];
    tle::SizeType32 max_num_tokens = config["build_config"]["max_num_tokens"];
    executor_config.setMaxBeamWidth(max_beam_width);
    executor_config.setMaxBatchSize(max_batch_size);
    executor_config.setMaxNumTokens(max_num_tokens);
}

int main(int argc, char **argv, char **envp) {
    initTrtLlmPlugins();

#ifdef DEBUG_ENV
    for (char **env = envp; *env != 0; env++) {
        char *thisEnv = *env;
        printf("%s\n", thisEnv);
    }
#endif

    // print tensor parallel info
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
    std::cout << "Process " << world_rank << " of " << world_size << std::endl;

    static fs::path ENGINE_DIR =
        "/data/models/llama2-7b-tp4-fp-wcache";
    tle::ExecutorConfig executor_config;
    setValue(executor_config, ENGINE_DIR);

    tle::SchedulerConfig scheduler_config(
        /* capacitySchedulerPolicy =*/tle::CapacitySchedulerPolicy::
            kGUARANTEED_NO_EVICT,
        /* contextChunkingPolicy =*/tle::ContextChunkingPolicy::
            kFIRST_COME_FIRST_SERVED);
    executor_config.setSchedulerConfig(scheduler_config);

    tle::KvCacheConfig kv_cache_config(
        /* enableBlockReuse =*/false,
        /* maxTokens =*/std::nullopt,
        /* maxAttentionWindowVec =*/std::nullopt,
        /* sinkTokenLength =*/std::nullopt,
        /* freeGpuMemoryFraction =*/std::nullopt,
        /* hostCacheSize =*/std::nullopt,
        /* onboardBlocks =*/true);
    executor_config.setKvCacheConfig(kv_cache_config);

    executor_config.setEnableChunkedContext(/* enableChunkedContext =*/false);
    executor_config.setNormalizeLogProbs(/* normalizeLogProbs =*/false);
    executor_config.setIterStatsMaxIterations(
        /* iterStatsMaxIterations =*/tle::kDefaultIterStatsMaxIterations);
    executor_config.setRequestStatsMaxIterations(
        /* requestStatsMaxIterations =*/tle::kDefaultRequestStatsMaxIterations);
    executor_config.setBatchingType(
        /* batchingType =*/tle::BatchingType::kINFLIGHT);

    tle::ParallelConfig parallel_config(
        /* commType =*/tle::CommunicationType::kMPI,
        /* commMode =*/tle::CommunicationMode::kLEADER,
        /* deviceIds =*/std::nullopt,
        /* participantIds =*/std::nullopt,
        /* orchestratorConfig =*/std::nullopt);
    executor_config.setParallelConfig(parallel_config);

    tle::PeftCacheConfig peft_cache_config(
        /* numHostModuleLayer =*/0,
        /* numDeviceModuleLayer =*/0,
        /* optimalAdapterSize =*/8,
        /* maxAdapterSize =*/64,
        /* numPutWorkers =*/1,
        /* numEnsureWorkers =*/1,
        /* numCopyStreams =*/1,
        /* maxPagesPerBlockHost =*/24,
        /* maxPagesPerBlockDevice =*/8,
        /* deviceCachePercent =*/std::nullopt,
        /* hostCacheSize =*/std::nullopt);
    executor_config.setPeftCacheConfig(peft_cache_config);

    tle::LogitsPostProcessorConfig logits_post_processor_config(
        /* processorMap =*/std::nullopt,
        /* processorBatched =*/std::nullopt,
        /* replicate =*/true);
    executor_config.setLogitsPostProcessorConfig(logits_post_processor_config);

    tle::DecodingConfig decoding_config(
        /* decodingMode =*/std::nullopt,
        /* lookaheadDecodingConfig =*/std::nullopt,
        /* medusaChoices =*/std::nullopt);
    executor_config.setDecodingConfig(decoding_config);

    executor_config.setGpuWeightsPercent(/* gpuWeightsPercent =*/1);
    executor_config.setMaxQueueSize(/* maxQueueSize =*/std::nullopt);

    tle::ExtendedRuntimePerfKnobConfig extended_runtime_perf_knob_config(
        /* multiBlockMode =*/true,
        /* enableContextFMHAFP32Acc =*/false);
    executor_config.setExtendedRuntimePerfKnobConfig(
        extended_runtime_perf_knob_config);

#ifdef DEBUG_TLLM
    tle::DebugConfig debug_config(
        /* dumpInputTensors =*/false,
        /* dumpOuputTensors =*/false,
        /* debugTensorNames =*/{});
    executor_config.setDebugConfig(debug_config);
#endif

    executor_config.setRecvPollPeriodMs(/* recvPollPeriodMs =*/0);
    executor_config.setMaxSeqIdleMicroseconds(
        /* maxSeqIdleMicroseconds =*/180000000);

    tle::Executor executor = tle::Executor(
        /* modelPath =*/ENGINE_DIR,
        /* modelType =*/tle::ModelType::kDECODER_ONLY,
        /* executorConfig =*/executor_config);

    // initialize requests
    std::vector<tle::Request> requests;
    newRequests(requests);
    std::vector<tle::IdType> request_ids = addRequests(executor, requests);

    std::chrono::milliseconds ms(5000);
    tle::SizeType32 numFinished{0};
    while (numFinished < request_ids.size()) {
        // get results
        std::vector<tle::Response> responses =
            executor.awaitResponses(/* timeout =*/ms);
        // loop for each response, if response is finished, print
        for (tle::Response response : responses) {
            if (response.hasError()) {
                printf("Error: %s\n",
                       std::to_string(response.getRequestId()).c_str());
            } else {
                tle::Result result = response.getResult();
                // beam width is 0
                tle::VecTokens output_tokens = result.outputTokenIds.at(0);
                printf("Output tokens: %s\n",
                       tlc::vec2str(output_tokens).c_str());
                tle::FinishReason finish_reason = result.finishReasons.at(0);
                printf("Finish reason: %d\n", finish_reason);
                if (result.isFinal) {
                    printf("Finish: %lu\n", response.getRequestId());
                    numFinished++;
                }
            }
        }
    }
    return 0;
}