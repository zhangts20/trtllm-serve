#ifndef _SESSION_HPP_
#define _SESSION_HPP_

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/customAllReduceUtils.h"

namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace bk = tensorrt_llm::batch_manager::kv_cache_manager;
using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

class GenerationSession
{
public:
    class Config
    {
    public:
        // The maximum number of batch
        tr::SizeType max_batch_size;
        // The maximum width of beam
        tr::SizeType max_beam_width;
        // The length of the longest input sequence
        tr::SizeType max_seq_length;
        // The config of KV Cache
        bk::KvCacheConfig kv_cache_config{};
    };

public:
    // Constructor
    GenerationSession(Config const& session_config, tr::GptModelConfig const& model_config,
        tr::WorldConfig const& world_config, std::string const& engine_path, 
        LoggerPtr logger = nullptr);

    // Setup inference context
    void setup(Config const& session_config);

    // Create contexts of runtime
    void createContexts();

    // Create buffers of runtime
    void createBuffers(tr::SizeType num_micro_batches);

    void createCustomAllReduceWorkspace(
        tr::SizeType max_batch_size, tr::SizeType max_beam_width, tr::SizeType max_seq_length);
    void createKvCacheManager(tr::SizeType batchSize, tr::SizeType beamWidth, tr::SizeType maxAttentionWindow,
        tr::SizeType sinkTokenLength, tr::SizeType maxSequenceLength, bk::KvCacheConfig const& kvCacheConfig);

private:
    // Private members
    tr::GptModelConfig const m_model_config;
    tr::WorldConfig const m_world_config;
    int m_device{-1};
    LoggerPtr m_logger;

    tr::ITensor::SharedPtr m_comm_ptres;
    std::vector<std::shared_ptr<tr::IpcMemory>> m_ipc_memory_handles;

    std::shared_ptr<tr::TllmRuntime> m_runtime;
    std::shared_ptr<bk::KVCacheManager> m_kv_cache_manager;
    std::vector<std::shared_ptr<tr::RuntimeBuffers>> m_buffers;
};

#endif