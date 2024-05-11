#include <iostream>
#include <fstream>
#include <string>

#include "rapidjson/document.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"


namespace fs = std::filesystem;
namespace tr = tensorrt_llm::runtime;
namespace tbk = tensorrt_llm::batch_manager::kv_cache_manager;

int main() {
    const std::string engine_dir = "/data/models/llama2-7b_tp1_fp16_8_1024_1024";

    /// 1. Initialize GPT Inference Session
    const std::string config_path_str = engine_dir + "/" + "config.json";
    std::filesystem::path config_path(config_path_str);
    if (!fs::exists(config_path)) {
        std::cout << engine_dir << " doesn't cotain a config.json" << std::endl;
        return -1;
    }
    tr::GptJsonConfig json_config = tr::GptJsonConfig::parse(config_path);
    tr::GptModelConfig model_config = json_config.getModelConfig();
    const bool use_packed = model_config.usePackedInput();

    // TP and PP
    tr::SizeType tp_size = json_config.getTensorParallelism();
    tr::SizeType pp_size = json_config.getPipelineParallelism();
    // GPUs per node
    tr::WorldConfig world_config = tr::WorldConfig::mpi(8, tp_size, pp_size);

    tr::SizeType max_batch_size = model_config.getMaxBatchSize();
    tr::SizeType max_beam_width = model_config.getMaxBeamWidth();
    tr::SizeType max_seq_len = model_config.getMaxSequenceLen();
    tr::GptSession::Config session_config = tr::GptSession::Config(
        max_batch_size, max_beam_width, max_seq_len);
    session_config.kvCacheConfig = tbk::KvCacheConfig();

    // Initialize Plugin
    auto logger = std::make_shared<tr::TllmLogger>();
    initTrtLlmPlugins(logger.get());

    // Engine path
    const std::string engine_name = json_config.engineFilename(world_config);
    const std::string engine_path = engine_dir + "/" + engine_name;

    tr::GptSession session = tr::GptSession(
        session_config, model_config, world_config, engine_path);

    /// 2. Prepare GenerationInput and GenerationOutput
    auto constexpr end_id = 2;
    auto constexpr pad_id = 2;
    const tr::BufferManager& buffer_manager = session.getBufferManager();

    const int batch_size = 1, input_len = 10;
    tr::GenerationInput::TensorPtr input_ids;
    // Initialize Random Input
    std::vector<int32_t> input_ids_host(batch_size * input_len);
    srand(time(0));
    std::cout << "input ids:" << std::endl;
    for (int i = 0; i < input_ids_host.size(); ++i) {
        input_ids_host[i] = rand() % model_config.getVocabSizePadded(world_config.getSize());
        std::cout << input_ids_host[i] << " ";
    }
    std::cout << std::endl;
    if (use_packed) {
        input_ids = buffer_manager.copyFrom(
            input_ids_host, tr::ITensor::makeShape({batch_size * input_len}), tr::MemoryType::kGPU); 
    } else {
        input_ids = buffer_manager.copyFrom(
            input_ids_host, tr::ITensor::makeShape({batch_size, input_len}), tr::MemoryType::kGPU);  
    }
    
    std::vector<tr::SizeType> input_lengths_host(batch_size, input_len);
    tr::BufferManager::ITensorPtr input_lengths
        = buffer_manager.copyFrom(input_lengths_host, tr::ITensor::makeShape({batch_size}), tr::MemoryType::kGPU);

    // Initialize Input of Generation
    tr::GenerationInput generation_input{end_id, pad_id, std::move(input_ids), std::move(input_lengths), use_packed};

    tr::GenerationOutput generation_output{
        buffer_manager.emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kINT32),
        buffer_manager.emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kINT32)};
    if (session.getModelConfig().computeContextLogits()) {
        generation_output.contextLogits
            = buffer_manager.emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    }
    if (session.getModelConfig().computeGenerationLogits()) {
        generation_output.generationLogits
            = buffer_manager.emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    }

    /// 3. Generate
    tr::SamplingConfig sampling_config{1};  // beam_width=1
    tr::SizeType num_steps = 0;
    const int max_new_tokens = 5;

    generation_output.onTokenGenerated
        = [&num_steps, max_new_tokens](tr::GenerationOutput::TensorPtr const& output_ids, 
            tr::SizeType step, bool finished) { ++num_steps; };
    session.generate(generation_output, generation_input, sampling_config);
    buffer_manager.getStream().synchronize();
    cudaDeviceSynchronize();

    /// 4. Print Output
    tr::BufferManager::ITensorPtr output_ids_host = buffer_manager.copyFrom(
        *generation_output.ids, tr::MemoryType::kCPU);
    int32_t* output_ids = tr::bufferCast<std::int32_t>(*output_ids_host);
    std::cout << "output ids" << std::endl;
    for (int b = 0; b < batch_size; ++b) {
        for (int beam = 0; beam < 1; ++beam) {
            for (int s = 0; s < input_len + max_new_tokens; ++s) {
                std::cout << output_ids[s] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
