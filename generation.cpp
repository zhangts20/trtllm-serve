#include "generation.hpp"

GenerationSession::GenerationSession(Config const &session_config, 
    tr::GptModelConfig const &model_config, tr::WorldConfig const &world_config, 
    std::string const &engine_path, LoggerPtr logger)
    : m_model_config(model_config)
    , m_world_config(world_config)
    , m_device{tr::utils::initDevice(world_config)}
    , m_logger{std::move(logger)}
{
    std::vector<uint8_t> engine_buffer = tr::utils::loadEngine(engine_path);
    m_runtime = std::make_shared<tr::TllmRuntime>(
        engine_buffer.data(), engine_buffer.size(), *m_logger);
    // TODO: support PipelineParallel

    // TODO: check expected and runtime tensor names

    setup(session_config);
}

void GenerationSession::setup(Config const &session_config)
{
    tr::SizeType const max_batch_size = session_config.max_batch_size;
    tr::SizeType const max_beam_width = session_config.max_beam_width;
    tr::SizeType const max_seq_length = session_config.max_seq_length;

    // Check window size of attention
    if (session_config.kv_cache_config.maxAttentionWindow.has_value()
        && session_config.kv_cache_config.maxAttentionWindow.value() > max_seq_length) {
        TLLM_LOG_WARNING(
            "The value of maxAttentionWindow cannot exceed maxSequenceLength. "
            "Therefore, it has been adjusted to match the value of maxSequenceLength.");
    }
    tr::SizeType const max_attention_window = session_config.kv_cache_config.maxAttentionWindow.has_value()
        ? std::min(session_config.kv_cache_config.maxAttentionWindow.value(), max_seq_length)
        : max_seq_length;
    // Check sink token length
    tr::SizeType const sink_token_length = session_config.kv_cache_config.sinkTokenLength.has_value()
        ? session_config.kv_cache_config.sinkTokenLength.value()
        : 0;

    // TODO: support CUDAGraph

    createContexts();

    // TODO: support micro batch
    tr::SizeType num_gen_batches = m_world_config.getPipelineParallelism();
    createBuffers(num_gen_batches);

    tr::SizeType gen_batch_size = tc::ceilDiv(max_batch_size, num_gen_batches);
    if (m_world_config.isTensorParallel() && m_model_config.useCustomAllReduce()) {
        createCustomAllReduceWorkspace(gen_batch_size, max_beam_width, max_seq_length);
    }
    if (m_model_config.isTransformerBased() && m_model_config.usePagedKvCache()) {
        createKvCacheManager(max_batch_size, max_beam_width, max_attention_window,
            sink_token_length, max_seq_length, session_config.kv_cache_config);
    }

    bk::KVCacheManager* kv_cache_manager = m_model_config.usePagedKvCache() ? m_kv_cache_manager.get() : nullptr;
    for (std::shared_ptr<tr::RuntimeBuffers>& buffers : m_buffers) {
        buffers->generationConfig = tr::GenerationConfig{
            gen_batch_size, max_beam_width, 0, max_attention_window, sink_token_length, max_seq_length};
    }
}

void GenerationSession::createContexts()
{
    m_runtime->clearContexts();

    tr::SizeType const num_profiles = m_runtime->getNbProfiles();
    // Instantiate 1 execution context for each profile
    for (int context_id = 0; context_id < num_profiles; ++context_id) {
        m_runtime->addContext(context_id);
    }
}

void GenerationSession::createBuffers(tr::SizeType num_micro_batches)
{
    m_buffers.clear();

    for (int i = 0; i < num_micro_batches; ++i) {
        m_buffers.emplace_back(std::make_shared<tr::RuntimeBuffers>());
        m_buffers.back()->create(*m_runtime, m_model_config, m_world_config);
    }
}

void GenerationSession::createCustomAllReduceWorkspace(
    tr::SizeType max_batch_size, tr::SizeType max_beam_width, tr::SizeType max_seq_length)
{
    tr::setPeerAccess(m_world_config, true);

    m_ipc_memory_handles.clear();
    const std::size_t buffer_size = std::min(static_cast<std::size_t>(max_batch_size) * max_beam_width * max_seq_length
        * m_model_config.getHiddenSize() * m_world_config.getTensorParallelism() * sizeof(float),
        ::tensorrt_llm::utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(m_world_config.getTensorParallelism()));
    m_ipc_memory_handles.emplace_back(std::make_shared<tr::IpcMemory>(m_world_config, buffer_size));
    m_ipc_memory_handles.emplace_back(std::make_shared<tr::IpcMemory>(m_world_config, buffer_size));
    m_ipc_memory_handles.emplace_back(std::make_shared<tr::IpcMemory>(m_world_config, tr::IpcMemory::FLAGS_SIZE * sizeof(int32_t)));
    m_ipc_memory_handles.emplace_back(std::make_shared<tr::IpcMemory>(m_world_config, tr::IpcMemory::FLAGS_SIZE * sizeof(int32_t)));

    m_comm_ptres = tr::BufferManager::cpu(
        tr::ITensor::makeShape({static_cast<tr::SizeType>(m_ipc_memory_handles.size()) * m_world_config.getTensorParallelism()}),
        nvinfer1::DataType::kINT64);
    auto* const comm_ptrs_data = tr::bufferCast<void*>(*m_comm_ptres);

    for (size_t mem_idx = 0; mem_idx < m_ipc_memory_handles.size(); mem_idx++) {
        auto const& mem_comm_ptrs = m_ipc_memory_handles[mem_idx]->getCommPtrsTensor();
        for (int tpIdx = 0; tpIdx < m_world_config.getTensorParallelism(); tpIdx++) {
            comm_ptrs_data[mem_idx * m_world_config.getTensorParallelism() + tpIdx] = mem_comm_ptrs[tpIdx];
        }
    }
}

void GenerationSession::createKvCacheManager(tr::SizeType batch_size, tr::SizeType beam_width, tr::SizeType max_attention_window,
    tr::SizeType sink_token_length, tr::SizeType max_seq_length, bk::KvCacheConfig const& kv_cache_config)
{
    auto const tokensPerBlock = m_model_config.getTokensPerBlock();

    auto const kvDtype = m_model_config.getKvDataType();

    auto const [blocksInPrimaryPool, blocksInSecondaryPool] = bk::KVCacheManager::calculateMaxNumBlocks(
        kv_cache_config, kvDtype, m_model_config, m_world_config, m_runtime->getBufferManager());

    // If beamWidth > 1, use one more block for each sequence in the paged kv cache to avoid dropping the needed
    // tokens, when enabling cyclic kv cache.
    auto const useOneMoreBlock = beam_width > 1 && max_seq_length > max_attention_window;

    auto const localNbLayers = m_model_config.getNbLayers(m_world_config.getPipelineParallelism());
    auto const nbKvHeads = m_model_config.getNbKvHeads();
    auto const sizePerHead = m_model_config.getSizePerHead();
    bool constexpr enableBlockReuse{false};
    m_kv_cache_manager = std::make_shared<bk::KVCacheManager>(localNbLayers, nbKvHeads, sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, batch_size, beam_width, max_attention_window, sink_token_length,
        useOneMoreBlock, kvDtype, m_runtime->getStreamPtr(), enableBlockReuse, kv_cache_config.useUvm,
        kv_cache_config.onboardBlocks);
}
