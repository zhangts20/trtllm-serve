#include "trtllm_session.h"

#include <fstream>

/// @brief Initialize InferenceSession, including loading weights and initializing executor
/// @param model_dir The directory contains model file(s) and tokenizer file(s)
/// @return Return true if initialization is successful; otherwise, return false
bool InferenceSession::initialize(const std::string &model_dir) {
    this->model_dir = model_dir;
    if (!tokenizer_session->initialize(model_dir)) {
        return false;
    }
    // Set value according to config.json
    fs::path config_path = fs::path(model_dir) / "config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        LOG_ERROR("Open config file failed.");
        return false;
    }
    // Deserialize config.json
    json config;
    config_file >> config;
    config_file.close();
    // Read value from config.json
    tle::SizeType32 max_beam_width = config["build_config"]["max_beam_width"];
    tle::SizeType32 max_batch_size = config["build_config"]["max_batch_size"];
    tle::SizeType32 max_num_tokens = config["build_config"]["max_num_tokens"];
    executor_config->setMaxBeamWidth(max_beam_width);
    executor_config->setMaxBatchSize(max_batch_size);
    executor_config->setMaxNumTokens(max_num_tokens);

    bool enable_kv_reuse = true;
    initializeExecutor(enable_kv_reuse);

    return true;
}

/// @brief Initialize SchedulerConfig defined in executor.h
/// @return The struct SchedulerConfig
static tle::SchedulerConfig getSchedulerConfig() {
    tle::DynamicBatchConfig dynamic_batch_config = tle::DynamicBatchConfig(
        /* enableBatchSizeTuning =*/false,
        /* enableMaxNumTokensTuning =*/false,
        /* dynamicBatchMovingAverageWindow =*/128,
        /* batchSizeTable =*/tle::DynamicBatchConfig::kDefaultBatchSizeTable);

    tle::SchedulerConfig scheduler_config(
        // - kMAX_UTILIZATION, this is expected to maximum GPU throught, it might require that some requests be paused
        // and restarted
        // - kGUARANTEED_NO_EVICT, uses KV cache more conservatively guaranteeing that a request, once started, will run
        // to completion without eviction
        // - kSTATIC_BATCH, does not schedule new requests until all requests in current batch are completed
        /* capacitySchedulerPolicy =*/tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        // - kFIRST_COME_FIRST_SERVED, sequential chunking, complete the unfinished context phase first
        // - kEQUAL_PROGRESS, Iterate through each context request in sequence and attempt to increase its chunk count
        // until the constraint is exceeded
        /* contextChunkingPolicy =*/tle::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED,
        /* dynamicBatchConfig =*/dynamic_batch_config);

    return scheduler_config;
}

/// @brief Initialize KvCacheConfig defined in executor.h
/// @return The struct KvCacheConfig
static tle::KvCacheConfig getKvCacheConfig(bool enable_kv_reuse) {
    // 1. The maxTokens and freeGpuMemoryFraction can define the memory allocated for KV, and minimum will be adopted
    // 2. hostCacheSize define the second memory (not GPU) to be allocated for KV
    tle::KvCacheConfig kv_cache_config(
        /* enableBlockReuse =*/enable_kv_reuse,
        /* maxTokens =*/std::nullopt,
        /* maxAttentionWindowVec =*/std::nullopt,
        /* sinkTokenLength =*/std::nullopt,
        /* freeGpuMemoryFraction =*/std::nullopt,
        /* hostCacheSize =*/std::nullopt,
        /* onboardBlocks =*/true,
        /* crossKvCacheFraction =*/std::nullopt,
        /* secondaryOffloadMinPriority =*/std::nullopt,
        /* eventBufferMaxSize =*/0,
        /* runtimeDefaults =*/std::nullopt,
        /* enablePartialReuse =*/true,
        /* copyOnPartialReuse =*/true);

    return kv_cache_config;
}

/// @brief Initialize executor
void InferenceSession::initializeExecutor(bool enable_kv_reuse) {
    executor_config->setSchedulerConfig(getSchedulerConfig());
    executor_config->setKvCacheConfig(getKvCacheConfig(enable_kv_reuse));
    // This feature splits the context into serveral chunks, the size of the chunk needs to be an integer multiple of
    // the kv-cache block size
    executor_config->setEnableChunkedContext(/* enableChunkedContext =*/false);
    executor_config->setNormalizeLogProbs(/* normalizeLogProbs =*/false);
    executor_config->setIterStatsMaxIterations(
        /* iterStatsMaxIterations =*/tle::ExecutorConfig::kDefaultIterStatsMaxIterations);
    executor_config->setRequestStatsMaxIterations(
        /* requestStatsMaxIterations =*/tle::ExecutorConfig::kDefaultRequestStatsMaxIterations);
    // - kSTATIC, the traditional batching schema with a batch of requets running in lockstep until all other requests
    // are completed
    // - kINFLIGHT, the requests are returned as soon as the end condition is met without any padding
    executor_config->setBatchingType(
        /* batchingType =*/tle::BatchingType::kINFLIGHT);

    tle::ParallelConfig parallel_config(
        /* commType =*/tle::CommunicationType::kMPI,
        /* commMode =*/tle::CommunicationMode::kLEADER,
        /* deviceIds =*/std::nullopt,
        /* participantIds =*/std::nullopt,
        /* orchestratorConfig =*/std::nullopt,
        /* numNodes =*/std::nullopt);
    executor_config->setParallelConfig(parallel_config);

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
        /* hostCacheSize =*/std::nullopt,
        /* loraPrefetchDir =*/std::nullopt);
    executor_config->setPeftCacheConfig(peft_cache_config);

    tle::LogitsPostProcessorConfig logits_post_processor_config(
        /* processorMap =*/std::nullopt,
        /* processorBatched =*/std::nullopt,
        /* replicate =*/true);
    executor_config->setLogitsPostProcessorConfig(logits_post_processor_config);

    tle::DecodingConfig decoding_config(
        /* decodingMode =*/std::nullopt,
        /* lookaheadDecodingConfig =*/std::nullopt,
        /* medusaChoices =*/std::nullopt,
        /* eagleConfig =*/std::nullopt);
    executor_config->setDecodingConfig(decoding_config);

    executor_config->setUseGpuDirectStorage(/* useGpuDirectStorage =*/false);
    // Set the GPU weights percent for weight streaming
    executor_config->setGpuWeightsPercent(/* gpuWeightsPercent =*/1.0);
    // The maximum number of requests allowed in queue before rejecting new requests
    executor_config->setMaxQueueSize(/* maxQueueSize =*/std::nullopt);

    tle::ExtendedRuntimePerfKnobConfig extended_runtime_perf_knob_config(
        /* multiBlockMode =*/true,
        /* enableContextFMHAFP32Acc =*/false,
        /* cudaGraphMode */ false,
        /* cudaGraphCacheSize =*/0);
    executor_config->setExtendedRuntimePerfKnobConfig(extended_runtime_perf_knob_config);

    // executor_config->setDebugConfig(/* debugConfig =*/std::nullopt);
    executor_config->setRecvPollPeriodMs(/* recvPollPeriodMs =*/0);
    // The maximum time in microseconds a scheduled request can remain idle before getting terminated, default is 3
    // minutes
    executor_config->setMaxSeqIdleMicroseconds(
        /* maxSeqIdleMicroseconds =*/180000000);

    tle::SpeculativeDecodingConfig speculative_decoding_config(
        /* fastLogits =*/false);
    executor_config->setSpecDecConfig(speculative_decoding_config);

    // tle::GuidedDecodingConfig guided_decoding_config(
    //     /* backend =*/tle::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR,
    //     /* encodedVocab =*/std::nullopt,
    //     /* tokenizerStr =*/std::nullopt,
    //     /* stopTokenIds =*/std::nullopt);
    // executor_config->setGuidedDecodingConfig(guided_decoding_config);

    std::vector<tle::AdditionalModelOutput> additional_model_outputs;
    executor_config->setAdditionalModelOutputs(/* additionalModelOutputs =*/additional_model_outputs);

    executor_config->setGatherGenerationLogits(/* gatherGenerationLogits =*/false);
    executor_config->setPromptTableOffloading(/* promptTableOffloading =*/false);

    tle::CacheTransceiverConfig cache_transceiver_config(/* maxNumTokens =*/std::nullopt);
    executor_config->setCacheTransceiverConfig(cache_transceiver_config);

    executor_config->setEnableTrtOverlap(/* enableTrtOverlap =*/false);

    executor = std::make_unique<tle::Executor>(
        /* modelPath =*/model_dir,
        /* modelType =*/tle::ModelType::kDECODER_ONLY,
        /* executorConfig =*/*executor_config);
}

/// @brief Add a new request
/// @param input_text The input text
/// @param streaming Whether to do streaming inference
/// @param max_new_tokens The max generated tokens
/// @param num_beams The max return sequences
void InferenceSession::addRequests(const InputConfig &input_config) {
    // Encode input text
    tle::VecTokens vec_tokens;
    tokenizer_session->encode(input_config.input_text, vec_tokens);
    tle::OutputConfig output_config = tle::OutputConfig(
        /* returnLogProbs =*/true,
        /* returnContextLogits =*/false,
        /* returnGenerationLogits =*/false,
        /* excludeInputFromOutput =*/true,
        /* returnEncoderOutput =*/false,
        /* returnPerfMetrics =*/false,
        /* additionalModelOutputs =*/std::nullopt);
    tle::SamplingConfig sampling_config = tle::SamplingConfig(
        /* beamWidth =*/input_config.sampling_parameters.num_beams,
        /* topK =*/input_config.sampling_parameters.top_k,
        /* topP =*/input_config.sampling_parameters.top_p,
        /* topPMin =*/std::nullopt,
        /* topPResetIds =*/std::nullopt,
        /* topPDecay =*/std::nullopt,
        /* seed =*/std::nullopt,
        /* temperature =*/std::nullopt,
        /* minTokens =*/std::nullopt,
        /* beamSearchDiversityRate =*/std::nullopt,
        /* repetitionPenalty =*/std::nullopt,
        /* presencePenalty =*/std::nullopt,
        /* frequencyPenalty =*/std::nullopt,
        /* lengthPenalty =*/std::nullopt,
        /* earlyStopping =*/std::nullopt,
        /* noRepeatNgramSize =*/std::nullopt,
        /* numReturnSequences =*/std::nullopt,
        /* minP =*/std::nullopt,
        /* beamWidthArray =*/std::nullopt);
    tle::Request request = tle::Request(
        /* inputTokenIds =*/vec_tokens,
        /* maxTokens =*/input_config.sampling_parameters.max_new_tokens,
        /* streaming =*/input_config.sampling_parameters.is_streaming,
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
        /* multimodalEmbedding =*/std::nullopt,
        /* mRopeConfig =*/std::nullopt,
        /* loraConfig =*/std::nullopt,
        /* lookaheadConfig =*/std::nullopt,
        /* kvCacheRetentionConfig =*/std::nullopt,
        /* logitsPostProcessorName =*/std::nullopt,
        /* logitsPostProcessor =*/std::nullopt,
        /* encoderInputTokenIds =*/std::nullopt,
        /* clientId =*/std::nullopt,
        /* returnAllGeneratedTokens =*/true,
        /* priority =*/tle::Request::kDefaultPriority,
        /* type =*/tle::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION,
        /* contextPhaseParams =*/std::nullopt,
        /* encoderInputFeatures =*/std::nullopt,
        /* encoderOutputLength =*/std::nullopt,
        /* crossAttentionMask =*/std::nullopt,
        /* numReturnSequences =*/1,
        /* eagleConfig =*/std::nullopt,
        /* skipCrossAttnBlocks =*/std::nullopt,
        /* guideDecodingParams =*/std::nullopt,
        /* languageAdapterUid =*/std::nullopt,
        /* allottedTimeMs =*/std::nullopt);

    if (executor->canEnqueueRequests()) {
        request_ids.push_back(executor->enqueueRequest(std::move(request)));
    }
}

/// @brief Do inference and print streaming or no-streaming outputs
void InferenceSession::infer() {
    std::chrono::milliseconds ms(5000);
    tle::SizeType32 numFinished{0};
    // To store output texts
    std::map<tle::IdType, std::vector<std::string>> output_texts_mapping;
    while (numFinished < request_ids.size()) {
        // Get results
        std::vector<tle::Response> responses = executor->awaitResponses(/* timeout =*/ms);
        // Loop for each response
        for (tle::Response response : responses) {
            if (response.hasError()) {
                LOG_ERROR("Response error: " + std::to_string(response.getRequestId()));
            } else {
                tle::Result result = response.getResult();
                if (result.isFinal) {
                    numFinished++;
                    // Loop for each beam
                    for (int b = 0; b < result.outputTokenIds.size(); ++b) {
                        std::string output_text;
                        tokenizer_session->decode(output_text, result.outputTokenIds.at(b));
                        output_texts_mapping.try_emplace(response.getRequestId())
                            .first->second.emplace_back(output_text);
                    }
                }
            }
        }
    }
    for (const auto &[request_id, output_texts] : output_texts_mapping) {
        printf("Finish request id: %lu\n", request_id);
        for (int i = 0; i < output_texts.size(); ++i) {
            printf("[Beam %d] %s\n", i, output_texts[i].c_str());
        }
    }
}

std::optional<OutputConfig> InferenceSession::serve() {
    bool is_final = false;
    while (!is_final) {
        std::vector<tle::Response> responses = executor->awaitResponses();
        for (tle::Response response : responses) {
            if (response.hasError()) {
                LOG_ERROR("Response error: " + std::to_string(response.getRequestId()));
                continue;
            }

            tle::Result result = response.getResult();

            std::vector<std::string> finish_reasons;
            for (auto finish_reason : result.finishReasons) {
                finish_reasons.emplace_back(FinishReasonMapping[finish_reason]);
                if (finish_reason != tle::FinishReason::kNOT_FINISHED) {
                    is_final = true;
                }
            }

            std::vector<std::vector<float>> output_logprobs;
            output_logprobs.reserve(result.logProbs.value().size());
            for (const std::vector<tle::FloatType> &vec_logprobs : result.logProbs.value()) {
                std::vector<float> inner_logprobs(vec_logprobs.size());
                std::transform(vec_logprobs.begin(), vec_logprobs.end(), inner_logprobs.begin(),
                               [](float v) { return std::round(v * 10000.0f) / 10000.0f; });
                output_logprobs.push_back(std::move(inner_logprobs));
            }

            return OutputConfig{
                /* request_id =*/response.getRequestId(),
                /* output_tokens =*/result.outputTokenIds,
                /* output_logprobs =*/output_logprobs,
                /* finish_reason =*/finish_reasons,
            };
        }
    }
    return std::nullopt;
}

/// @brief Initialize session of tokenizer
/// @param model_dir The directory contains tokenizer.model
/// @return Return true if initialization is successful; otherwise, return false
bool TokenizerSession::initialize(fs::path model_dir) {
    try {
        std::string blob = common_utils::LoadBytesFromFile((model_dir / "tokenizer.json").string());
        processor = tokenizers::Tokenizer::FromBlobJSON(blob);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to load tokenizer!");
        return false;
    }

    return true;
}

/// @brief Encode input text
/// @param input_text The input text to be encoded
/// @param input_ids The result of encoded input text
/// @return Return true if initialization is successful; otherwise, return false
bool TokenizerSession::encode(const std::string &input_text, tle::VecTokens &input_ids) {
    try {
        input_ids = processor->Encode(input_text);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to encode input!");
        return false;
    }

    return true;
}

/// @brief Decode output text
/// @param output_text The output text to be decoed
/// @param output_ids The result out decoded output text
/// @return Return true if initialization is successful; otherwise, return false
bool TokenizerSession::decode(std::string &output_text, const tle::VecTokens &output_ids) {
    try {
        output_text = processor->Decode(output_ids);
    } catch (const std::exception &e) {
        LOG_ERROR("Failed to decode output!");
        return false;
    }

    return true;
}
