#include "session.hpp"

InputConfig parseArgs(int argc, char **argv, char **envp) {
    InputConfig input_config;
    // clang-format off
    cxxopts::Options options("MAIN", "A cpp inference of TensorRT-LLM.");
    options.add_options()("help", "Print help");
    options.add_options()("model_dir", "The input engine directory.", cxxopts::value<std::string>());
    options.add_options()("input_text", "The input text for inference.", cxxopts::value<std::string>()->default_value("What is Deep Learning?"));
    options.add_options()("log_level", "The log level.", cxxopts::value<std::string>()->default_value("info"));
    // clang-format on
    cxxopts::ParseResult args = options.parse(argc, argv);
    // Check
    if (args.count("help")) {
        std::cerr << options.help() << std::endl;
        exit(1);
    }
    if (!args.count("model_dir")) {
        std::cerr << "The model dir is not given." << std::endl;
        exit(1);
    }
    fs::path engine_dir = args["model_dir"].as<std::string>();
    if (!fs::exists(engine_dir)) {
        std::cerr << "The model dir does not exist." << std::endl;
        exit(1);
    }
    input_config.engine_dir = engine_dir;
    input_config.input_text = args["input_text"].as<std::string>();
    // Set log level
    auto logger = tlc::Logger::getLogger();
    auto const log_level = args["log_level"].as<std::string>();
    if (log_level == "trace") {
        logger->setLevel(tlc::Logger::TRACE);
    } else if (log_level == "debug") {
        logger->setLevel(tlc::Logger::DEBUG);
    } else if (log_level == "info") {
        logger->setLevel(tlc::Logger::INFO);
    } else if (log_level == "warning") {
        logger->setLevel(tlc::Logger::WARNING);
    } else if (log_level == "error") {
        logger->setLevel(tlc::Logger::ERROR);
    } else {
        std::cerr << "Unexpected log level: " + log_level << std::endl;
        exit(1);
    }

#ifdef DEBUG_TLLM
    // Whether to print env info
    for (char **env = envp; *env != 0; env++) {
        char *thisEnv = *env;
        printf("%s\n", thisEnv);
    }
#endif

    return input_config;
}

bool InferenceSession::initialize(std::string engine_dir) {
    this->engine_dir = engine_dir;
    if (!tokenizer_session->initialize(engine_dir)) {
        std::cerr << "Failed to initialize tokenizer session." << std::endl;
        return false;
    }
    // Set value according to config.json
    fs::path config_path = fs::path(engine_dir) / "config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "open config file failed" << std::endl;
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

    initializeExecutorConfig();

    return true;
}

static tle::SchedulerConfig getSchedulerConfig() {
    tle::DynamicBatchConfig dynamic_batch_config = tle::DynamicBatchConfig(
        /* enableBatchSizeTuning =*/false,
        /* enableMaxNumTokensTuning =*/false,
        /* dynamicBatchMovingAverageWindow =*/
        tle::DynamicBatchConfig::kDefaultDynamicBatchMovingAverageWindow,
        /* batchSizeTable =*/tle::DynamicBatchConfig::kDefaultBatchSizeTable);

    // clang-format off
    tle::SchedulerConfig scheduler_config(
    // kMAX_UTILIZATION / kGUARANTEED_NO_EVICT / kSTATIC_BATCH
    /* capacitySchedulerPolicy =*/tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
    // kFIRST_COME_FIRST_SERVED / kEQUAL_PROGRESS
    /* contextChunkingPolicy =*/tle::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED,
    /* dynamicBatchConfig =*/dynamic_batch_config);
    // clang-format on

    return scheduler_config;
}

static tle::KvCacheConfig getKvCacheConfig() {
    tle::KvCacheConfig kv_cache_config(
        /* enableBlockReuse =*/false,
        /* maxTokens =*/std::nullopt,
        /* maxAttentionWindowVec =*/std::nullopt,
        /* sinkTokenLength =*/std::nullopt,
        /* freeGpuMemoryFraction =*/std::nullopt,
        /* hostCacheSize =*/std::nullopt,
        /* onboardBlocks =*/true,
        /* crossKvCacheFraction =*/std::nullopt,
        /* secondaryOffloadMinPriority =*/std::nullopt,
        /* eventBufferMaxSize =*/0,
        /* runtimeDefaults =*/std::nullopt);

    return kv_cache_config;
}

void InferenceSession::initializeExecutorConfig() {
    executor_config->setSchedulerConfig(getSchedulerConfig());
    executor_config->setKvCacheConfig(getKvCacheConfig());
    executor_config->setEnableChunkedContext(/* enableChunkedContext =*/false);
    executor_config->setNormalizeLogProbs(/* normalizeLogProbs =*/false);
    executor_config->setIterStatsMaxIterations(
        /* iterStatsMaxIterations =*/tle::ExecutorConfig::
            kDefaultIterStatsMaxIterations);
    executor_config->setRequestStatsMaxIterations(
        /* requestStatsMaxIterations =*/tle::ExecutorConfig::
            kDefaultRequestStatsMaxIterations);
    executor_config->setBatchingType(
        /* batchingType =*/tle::BatchingType::kINFLIGHT);

    tle::ParallelConfig parallel_config(
        /* commType =*/tle::CommunicationType::kMPI,
        /* commMode =*/tle::CommunicationMode::kLEADER,
        /* deviceIds =*/std::nullopt,
        /* participantIds =*/std::nullopt,
        /* orchestratorConfig =*/std::nullopt);
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
        /* hostCacheSize =*/std::nullopt);
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
        /* EagleConfig =*/std::nullopt);
    executor_config->setDecodingConfig(decoding_config);

    executor_config->setGpuWeightsPercent(/* gpuWeightsPercent =*/1);
    executor_config->setMaxQueueSize(/* maxQueueSize =*/std::nullopt);

    tle::ExtendedRuntimePerfKnobConfig extended_runtime_perf_knob_config(
        /* multiBlockMode =*/true,
        /* enableContextFMHAFP32Acc =*/false,
        /* cudaGraphMode */ false,
        /* cudaGraphCacheSize =*/0);
    executor_config->setExtendedRuntimePerfKnobConfig(
        extended_runtime_perf_knob_config);

#ifdef DEBUG_TLLM
    tle::DebugConfig debug_config(
        /* dumpInputTensors =*/false,
        /* dumpOuputTensors =*/false,
        /* debugTensorNames =*/{});
    executor_config->setDebugConfig(debug_config);
#endif

    executor_config->setRecvPollPeriodMs(/* recvPollPeriodMs =*/0);
    executor_config->setMaxSeqIdleMicroseconds(
        /* maxSeqIdleMicroseconds =*/180000000);

    tle::SpeculativeDecodingConfig speculative_decoding_config(
        /* fastLogits =*/false);
    executor_config->setSpecDecConfig(speculative_decoding_config);

    tle::GuidedDecodingConfig guided_decoding_config(
        /* backend =*/tle::GuidedDecodingConfig::GuidedDecodingBackend::
            kXGRAMMAR,
        /* encodedVocab =*/std::nullopt,
        /* tokenizerStr =*/std::nullopt,
        /* stopTokenIds =*/std::nullopt);
    // Initialize executor
    executor = std::make_unique<tle::Executor>(
        /* modelPath =*/engine_dir,
        /* modelType =*/tle::ModelType::kDECODER_ONLY,
        /* executorConfig =*/*executor_config);
}

void InferenceSession::addRequests(std::optional<std::string> input_text) {
    // Set `What is Deep Learning?` to the default input text
    tle::VecTokens vec_tokens;
    if (input_text) {
        tokenizer_session->encode(input_text.value(), vec_tokens);
    } else {
        vec_tokens = {1, 1724, 338, 21784, 29257, 29973};
    }
    // Set the default max generated tokens is 17
    tle::SizeType32 max_new_tokens = 17;
    tle::OutputConfig output_config = tle::OutputConfig(
        /* returnLogProbs =*/false,
        /* returnContextLogits =*/false,
        /* returnGenerationLogits =*/false,
        /* excludeInputFromOutput =*/false,
        /* returnEncoderOutput =*/false,
        /* returnPerfMetrics =*/false);
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
        /* frequencyPenalty =*/std::nullopt,
        /* lengthPenalty =*/std::nullopt,
        /* earlyStopping =*/std::nullopt,
        /* noRepeatNgramSize =*/std::nullopt,
        /* numReturnSequences =*/std::nullopt);
    // Append 8 requests
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
            /* mRopeConfig =*/std::nullopt,
            /* loraConfig =*/std::nullopt,
            /* lookaheadConfig =*/std::nullopt,
            /* kvCacheRetentionConfig =*/std::nullopt,
            /* logitsPostProcessorName =*/std::nullopt,
            /* encoderInputTokenIds =*/std::nullopt,
            /* clientId =*/std::nullopt,
            /* returnAllGeneratedTokens =*/false,
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
            /* allottedTimeMs =*/std::nullopt));
    }
    // Add requests
    for (size_t i = 0; i < requests.size(); ++i) {
        if (executor->canEnqueueRequests()) {
            request_ids.push_back(
                executor->enqueueRequest(std::move(requests[i])));
        }
    }
}

void InferenceSession::inferRequests() {
    std::chrono::milliseconds ms(5000);
    tle::SizeType32 numFinished{0};
    // To store output token ids and output texts
    std::map<tle::IdType, tle::VecTokens> output_tokens_mapping;
    std::vector<std::string> output_texts;
    while (numFinished < request_ids.size()) {
        // Get results
        std::vector<tle::Response> responses =
            executor->awaitResponses(/* timeout =*/ms);
        // Loop for each response, if response is finished, print
        for (tle::Response response : responses) {
            if (response.hasError()) {
                printf("Error: %s\n",
                       std::to_string(response.getRequestId()).c_str());
            } else {
                tle::Result result = response.getResult();
                // Set beam width to 0
                tle::VecTokens output_tokens = result.outputTokenIds.at(0);
                output_tokens_mapping.try_emplace(response.getRequestId())
                    .first->second.push_back(output_tokens.at(0));
                tle::FinishReason finish_reason = result.finishReasons.at(0);
                if (result.isFinal) {
                    printf("Finish: %lu\n", response.getRequestId());
                    printf("Finish reason: %d\n", finish_reason);
                    numFinished++;

                    std::string output_text;
                    tokenizer_session->decode(
                        output_text,
                        output_tokens_mapping.at(response.getRequestId()));
                    output_texts.emplace_back(output_text);
                }
            }
        }
    }
    for (const std::string output_text : output_texts) {
        printf("%s", output_text.c_str());
    }
}

bool TokenizerSession::initialize(fs::path model_dir) {
    const sp::util::Status status =
        processor->Load((model_dir / "tokenizer.model").c_str());
    if (!status.ok()) {
        std::cerr << "Failed to load tokenizer.model." << std::endl;
        return false;
    }
    return true;
}

bool TokenizerSession::encode(std::string input_text,
                              tle::VecTokens &input_ids) {
    const sp::util::Status status = processor->Encode(input_text, &input_ids);
    if (!status.ok()) {
        std::cerr << "Failed to encode input (" << input_text << ")."
                  << std::endl;
        return false;
    }
    return true;
}

bool TokenizerSession::decode(std::string &output_text,
                              tle::VecTokens &output_ids) {
    const sp::util::Status status = processor->Decode(output_ids, &output_text);
    if (!status.ok()) {
        std::cerr << "Failed to decode output." << std::endl;
        return false;
    }
    return true;
}
