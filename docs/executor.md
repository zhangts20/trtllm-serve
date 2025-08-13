# 1. SchedulerConfig
```cpp
SchedulerConfig(
    CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
    std::optional<ContextChunkingPolicy> contextChunkingPolicy = std::nullopt,
    std::optional<DynamicBatchConfig> dynamicBatchConfig = std::nullopt);
```
# 1.1 CapacitySchedulerPolicy
在推理迭代中，控制选择新请求的策略。共有三个可选项，在 `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp` 的构造函数中对应初始化不同的类，三个类都重载了圆括号运算符。
## 1.1.1 kMAX_UTILIZATION
遍历所有新近请求，存在变量 `activateRequests` 中。依次判断在 PD 分离中请求是否可以被调度；判断当前请求是否可以重复使用其他请求的 KV；判断调度中的请求数是否大于设置的最大值以及一些其他条件，如果请求不可调度请求则判断是否可以将其暂停，并存放到变量 `pausedRequests` 中。

总的来说，这种调度策略会尽可能调度请求，但同时也受到设置的最大请求数限制以及暂停某些请求。

## 1.1.2 kGUARANTEED_NO_EVICT
遍历所有新近请求，存在变量 `activateRequests` 中。依次判断在 PD 分离中请求是否可以被调度；判断调度中的请求数是否大于设置的最大值以及一些其他条件，该过程可能将请求加入待调度队列。

总的来说，这种调度策略和上一种相比，少了暂停请求的操作。

## 1.1.3 kSTATIC_BATCH
接在上一个调度策略的后续，只有当前没有请求时才判断是否调度待调度请求，否则仅判断是否可以加入待调度队列。

总的来说，这种调度策略和上一种相比，只有当所有请求运行完毕后才去待调度队列调度请求。

> 在实际运行时如果测试最大吞吐，可以设置 kMAX_UTILIZATION 选项，但在实际使用时为了体验一般选用 kGUARANTEED_NO_EVICT。

## 对比实验

# 1.2 ContextChunkingPolicy
在 Prefill 阶段，将长请求分解为多个短请求。共有两个可选项，在 `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp` 的 `setCtxRequestsChunkSize` 函数中根据传入值调用相应函数。

## 1.2.1 kEQUAL

## 1.2.2 kFIRST_COME_FIRST_SERVED

# 1.3 DynamicBatchConfig

# 2. KvCacheConfig 
```cpp
KvCacheConfig(bool enableBlockReuse = true, std::optional<SizeType32> const& maxTokens = std::nullopt,
    std::optional<std::vector<SizeType32>> const& maxAttentionWindowVec = std::nullopt,
    std::optional<SizeType32> const& sinkTokenLength = std::nullopt,
    std::optional<FloatType> const& freeGpuMemoryFraction = std::nullopt,
    std::optional<size_t> const& hostCacheSize = std::nullopt, bool onboardBlocks = true,
    std::optional<FloatType> const& crossKvCacheFraction = std::nullopt,
    std::optional<RetentionPriority> secondaryOffloadMinPriority = std::nullopt, size_t eventBufferMaxSize = 0,
    std::optional<tensorrt_llm::runtime::RuntimeDefaults> const& runtimeDefaults = std::nullopt,
    bool enablePartialReuse = true, bool copyOnPartialReuse = true);
```

## 2.1 enableBlockReuse


## 2.2 maxTokens

## 2.3 maxAttentionWindowVec

## 2.4 sinkTokenLength

## 2.5 freeGpuMemoryFraction

## 2.6 hostCacheSize

## 2.7 onboardBlocks

## 2.8 crossKvCacheFraction

## 2.9 secondaryOffloadMinPriority

## 2.10 eventBufferMaxSize

## 2.11 runtimeDefaults

## 2.12 enablePartialReuse

## 2.13 copyOnPartialReuse
