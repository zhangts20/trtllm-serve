import os
import math
import json
import torch
import argparse
import tensorrt as trt
import tensorrt_llm

from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.runtime.generation import (_Runtime, KVCacheManager,
                                             GenerationSequence, RuntimeTensor,
                                             Mapping)
from tensorrt_llm._utils import trt_dtype_to_torch, str_dtype_to_torch
from transformers import AutoTokenizer
from typing import List, Tuple, Dict


class GenerationSession(object):

    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer: trt.IHostMemory,
        mapping: Mapping,
    ) -> None:
        self.model_config = model_config
        self.engine_buffer = engine_buffer
        self.mapping = mapping
        self.runtime = _Runtime(engine_buffer, mapping)

        self.device = torch.device(f"cuda:{self.runtime.runtime_rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        self.vocab_size_padded = int(
            math.ceil(self.model_config.vocab_size / self.mapping.tp_size) *
            self.mapping.tp_size)

        self.layer_types = ["attention"] * self.model_config.num_layers
        self.num_attn_layers = self.model_config.num_layers

    def _tensor_dtype(self, name):
        return trt_dtype_to_torch(self.runtime.engine.get_tensor_dtype(name))

    def _get_num_paged_blocks(
        self,
        max_attention_window_size: int,
        sink_token_length: int,
        use_one_more_block: bool,
    ) -> Tuple[int, int]:
        bubble_len = 0
        if sink_token_length % self.model_config.tokens_per_block > 0:
            bubble_len += (
                self.model_config.tokens_per_block -
                sink_token_length % self.model_config.tokens_per_block)
        max_blocks_per_seq = math.ceil(
            (max_attention_window_size + bubble_len) /
            self.model_config.tokens_per_block)
        if use_one_more_block:
            max_blocks_per_seq += 1
        num_blocks = self.batch_size * self.beam_width * max_blocks_per_seq

        return num_blocks, max_blocks_per_seq

    def _prepare_context_inputs(
        self,
        batch_size: int,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        last_token_ids = context_lengths.detach().clone()
        last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
        ret = {'last_token_ids': last_token_ids}

        position_ids = torch.concat([
            torch.arange(0,
                         host_context_lengths[i],
                         dtype=torch.int32,
                         device='cuda') for i in range(batch_size)
        ])

        if self.model_config.has_position_embedding:
            ret['position_ids'] = position_ids

        return ret

    def _prepare_generation_inputs(
        self,
        context_lengths: torch.Tensor,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        last_token_ids = torch.ones_like(context_lengths)
        last_token_ids = torch.cumsum(last_token_ids, dim=0).int()
        ret = {"last_token_ids": last_token_ids}

        position_ids = context_lengths + step

        if self.model_config.has_position_embedding:
            ret["position_ids"] = position_ids

        return ret

    def setup(
        self,
        batch_size: int,
        max_context_length: int,
        max_new_tokens: int,
        beam_width: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_context_length + max_new_tokens
        self.beam_width = beam_width

        self.max_attention_window_size = self.max_seq_length
        self.host_max_attention_window_sizes = torch.ones(
            (self.num_attn_layers, ),
            dtype=torch.int32) * self.max_attention_window_size

        self.sink_token_length = 0
        self.host_sink_token_length = torch.zeros((1, ), dtype=torch.int32)

        self.use_one_more_block = False

        self.buffer = dict()
        # logits
        self.buffer["logits"] = torch.empty(
            (batch_size, self.vocab_size_padded),
            dtype=self._tensor_dtype("logits"),
            device=self.device)
        # quant of kv cache
        if self.model_config.quant_mode.has_kv_cache_quant():
            kv_cache_type = torch.int8
        else:
            kv_cache_type = str_dtype_to_torch(self.model_config.dtype)
        # paged kv cache
        num_blocks, _ = self._get_num_paged_blocks(
            max_attention_window_size=self.max_attention_window_size,
            sink_token_length=self.sink_token_length,
            use_one_more_block=self.use_one_more_block)
        cache_shape = (num_blocks, self.num_attn_layers, 2,
                       self.model_config.num_kv_heads,
                       self.model_config.tokens_per_block,
                       self.model_config.head_size)
        self.kv_cache_pool = torch.empty(cache_shape,
                                         dtype=kv_cache_type,
                                         device=self.device)
        # use gpt attention
        self.sequence_length_buffer = torch.ones((batch_size, ),
                                                 dtype=torch.int32,
                                                 device=self.device)

        self.buffer_allocated = True

    @property
    def use_gpt_attention_plugin(self):
        return self.model_config.gpt_attention_plugin

    @property
    def has_position_embedding(self):
        return self.model_config.has_position_embedding

    @property
    def removing_input_padding(self):
        return self.model_config.remove_input_padding

    @property
    def paged_kv_cache(self):
        return self.model_config.paged_kv_cache

    @property
    def hidden_size(self):
        return self.model_config.hidden_size

    def get_shape_buffer(
        self,
        batch_size: int,
        input_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        max_context_length: int,
        position_ids: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_indirection: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_block_offsets: torch.Tensor,
        step: int,
        is_prefill: bool,
    ) -> Dict[str, RuntimeTensor]:

        def sym(x, name):
            return RuntimeTensor.from_torch(name, x)

        def add_tensor(x, name):
            return tensors.update({name: sym(x, name)})

        def add_tensor_with_shape(x, name, shape):
            return tensors.update({
                name:
                RuntimeTensor.from_torch(name, x, override_shape=shape)
            })

        if self.use_gpt_attention_plugin:
            assert attention_mask is None, "attention mask is none when use gpt attention plugin"

        tensors = {}
        if self.use_gpt_attention_plugin:
            add_tensor(context_lengths, "context_lengths")
        add_tensor(cache_indirection, "cache_indirection")
        if self.has_position_embedding:
            add_tensor(position_ids, "position_ids")

        add_tensor(self.buffer["logits"], "logits")
        add_tensor(last_token_ids, "last_token_ids")

        if is_prefill:
            add_tensor(input_ids, "input_ids")
        else:
            input_ids_shape = (
                batch_size, ) if self.removing_input_padding else (batch_size,
                                                                   1)
            add_tensor_with_shape(input_ids, "input_ids", input_ids_shape)

        if self.paged_kv_cache:
            shape = kv_cache_block_offsets.shape
            shape = [shape[0] * shape[1], *shape[2:]]
            add_tensor_with_shape(kv_cache_block_offsets,
                                  "kv_cache_block_offsets", shape)
            add_tensor_with_shape(host_kv_cache_block_offsets,
                                  "host_kv_cache_block_offsets", shape)
            add_tensor(self.buffer["host_kv_cache_pool_pointers"],
                       "host_kv_cache_pool_pointers")

        if self.use_gpt_attention_plugin:
            if is_prefill:
                self.sequence_length_buffer = context_lengths.detach().clone()
                add_tensor_with_shape(self.sequence_length_buffer,
                                      "sequence_length", (batch_size, ))
                add_tensor_with_shape(host_context_lengths,
                                      "host_past_key_value_lengths",
                                      (batch_size, ))
                host_request_types = torch.zeros_like(
                    host_context_lengths).int()
            else:
                sequence_length = self.sequence_length_buffer
                add_tensor_with_shape(sequence_length, "sequence_length",
                                      (batch_size, ))
                host_past_key_value_lengths = torch.tensor(
                    [max_context_length + step] * (batch_size, ),
                    dtype=torch.int32)
                add_tensor(host_past_key_value_lengths,
                           "host_past_key_value_lengths")
                host_request_types = torch.ones_like(
                    host_kv_cache_block_offsets).int()
            add_tensor(host_request_types, "host_request_types")
            add_tensor_with_shape(self.host_sink_token_length,
                                  "host_sink_token_length", (1, ))
            add_tensor_with_shape(self.host_max_attention_window_sizes,
                                  "host_max_attention_window_sizes",
                                  (self.num_attn_layers, ))
            if self.removing_input_padding:
                add_tensor(host_context_lengths, "host_context_lengths")

        return tensors

    def get_context_shape_buffer(
        self,
        input_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_indirection: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_block_offsets: torch.Tensor,
    ) -> Dict[str, RuntimeTensor]:

        def sym(x, name):
            return RuntimeTensor.from_torch(name, x)

        def add_tensor(x, name):
            return tensors.update({name: sym(x, name)})

        def add_tensor_with_shape(x, name, shape):
            return tensors.update({
                name:
                RuntimeTensor.from_torch(name, x, override_shape=shape)
            })

        tensors = dict()
        # gpt attention plugin
        add_tensor(context_lengths, "context_lengths")
        add_tensor(cache_indirection, "cache_indirection")
        add_tensor(position_ids, "position_ids")
        # assert attention_mask is None, "attention mask is None when using gpt attention plugin"

        add_tensor(self.buffer["logits"], "logits")
        add_tensor(last_token_ids, "last_token_ids")

        add_tensor(input_ids, "input_ids")
        # paged kv cache
        buffer = kv_cache_block_offsets.contiguous()
        shape = kv_cache_block_offsets.shape
        shape = [shape[0] * shape[1], *shape[2:]]
        add_tensor_with_shape(buffer, "kv_cache_block_offsets", shape)
        add_tensor_with_shape(host_kv_cache_block_offsets,
                              "host_kv_cache_block_offsets", shape)
        add_tensor(self.buffer["host_kv_cache_pool_pointers"],
                   "host_kv_cache_pool_pointers")

        batch_size = context_lengths.shape[0]
        host_request_types = torch.zeros_like(context_lengths,
                                              device="cpu").int()
        self.sequence_length_buffer = context_lengths.detach().clone()
        add_tensor_with_shape(self.sequence_length_buffer, "sequence_length",
                              (batch_size, ))
        add_tensor_with_shape(host_context_lengths,
                              "host_past_key_value_lengths", (batch_size, ))
        add_tensor_with_shape(self.host_sink_token_length,
                              "host_sink_token_length", (1, ))
        add_tensor(host_request_types, "host_request_types")
        add_tensor_with_shape(self.host_max_attention_window_sizes,
                              "host_max_attention_window_sizes",
                              (self.model_config.num_layers, ))
        # remove input padding
        add_tensor(host_context_lengths, "host_context_lengths")

        return tensors

    def get_next_step_shape_buffer(
        self,
        batch_size: int,
        max_context_length: int,
        step: int,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_indirection: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_block_offsets: torch.Tensor,
    ) -> Dict[str, RuntimeTensor]:

        def sym(x, name):
            return RuntimeTensor.from_torch(name, x)

        def add_tensor(x, name):
            return tensors.update({name: sym(x, name)})

        def add_tensor_with_shape(x, name, shape):
            return tensors.update({
                name:
                RuntimeTensor.from_torch(name, x, override_shape=shape)
            })

        tensors = dict()
        # use gpt attention
        add_tensor(context_lengths, "context_lengths")
        add_tensor(cache_indirection, "cache_indirection")
        add_tensor(position_ids, "position_ids")
        # assert attention_mask is None, "attention mask is None when using gpt attention plugin"

        add_tensor(self.buffer["logits"], "logits")
        add_tensor(last_token_ids, "last_token_ids")

        add_tensor_with_shape(self.new_tokens, "input_ids", (batch_size, ))
        # paged kv cache
        shape = kv_cache_block_offsets.shape
        shape = [shape[0] * shape[1], *shape[2:]]
        add_tensor_with_shape(kv_cache_block_offsets, "kv_cache_block_offsets",
                              shape)
        add_tensor_with_shape(host_kv_cache_block_offsets,
                              "host_kv_cache_block_offsets", shape)
        add_tensor(self.buffer["host_kv_cache_pool_pointers"],
                   "host_kv_cache_pool_pointers")
        # use gpt attention plugin
        host_request_types = torch.ones_like(context_lengths,
                                             device="cpu").int()
        host_past_key_value_lengths = torch.tensor(
            [max_context_length + step] * batch_size,
            dtype=torch.int32,
            device="cpu")
        add_tensor(host_past_key_value_lengths, "host_past_key_value_lengths")
        add_tensor(host_request_types, "host_request_types")
        sequence_length = self.sequence_length_buffer + step
        add_tensor_with_shape(sequence_length, "sequence_length",
                              (batch_size, ))
        add_tensor_with_shape(self.host_sink_token_length,
                              "host_sink_token_length", (1, ))
        add_tensor_with_shape(self.host_max_attention_window_sizes,
                              "host_max_attention_window_sizes",
                              (self.model_config.num_layers, ))
        add_tensor(host_context_lengths, "host_context_lengths")

        return tensors

    @staticmethod
    def get_next_tokens(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.int32)

    def decode_regular(
        self,
        batch_size: int,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        max_context_length: int,
        cache_indirections: List[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        for step in range(self.max_new_tokens):
            if step % 2:
                context = self.runtime.context_0
                curr_cache_indirection = cache_indirections[1]
                next_cache_indirection = cache_indirections[0]
            else:
                context = self.runtime.context_1
                curr_cache_indirection = cache_indirections[0]
                next_cache_indirection = cache_indirections[1]
            # prefill
            if step == 0:
                model_inputs = self._prepare_context_inputs(
                    batch_size=batch_size,
                    context_lengths=context_lengths,
                    host_context_lengths=host_context_lengths)
                position_ids = model_inputs.get("position_ids", None)
                last_token_ids = model_inputs.get("last_token_ids")
                attention_mask = model_inputs.get("attention_mask", None)
                # paged kv cache
                host_kv_cache_block_offsets = self.kv_cache_manager.get_block_offsets(
                    1)
                kv_cache_block_offsets = host_kv_cache_block_offsets.to("cuda")

                batch_size = context_lengths.size(0)
                ctx_tensors = self.get_shape_buffer(
                    batch_size=batch_size,
                    input_ids=input_ids,
                    context_lengths=context_lengths,
                    host_context_lengths=host_context_lengths,
                    max_context_length=max_context_length,
                    position_ids=position_ids,
                    last_token_ids=last_token_ids,
                    attention_mask=attention_mask,
                    cache_indirection=curr_cache_indirection,
                    kv_cache_block_offsets=kv_cache_block_offsets,
                    host_kv_cache_block_offsets=host_kv_cache_block_offsets,
                    step=step,
                    is_prefill=True)

                # ctx_tensors = self.get_context_shape_buffer(
                #     input_ids, context_lengths, host_context_lengths,
                #     position_ids, last_token_ids, attention_mask,
                #     curr_cache_indirection, kv_cache_block_offsets,
                #     host_kv_cache_block_offsets)
                context = self.runtime.ctx_context
                import pdb; pdb.set_trace()
                self.runtime._set_tensors(context, ctx_tensors)

            stream = torch.cuda.current_stream().cuda_stream
            ok = self.runtime._run(context, stream)
            assert ok, "run engine failed"

            # handle next step
            if not step == self.max_new_tokens - 1:
                model_inputs = self._prepare_generation_inputs(
                    context_lengths=context_lengths, step=step)
                position_ids = model_inputs.get("position_ids", None)
                last_token_ids = model_inputs.get("last_token_ids")
                attention_mask = model_inputs.get("attention_mask", None)

                self.kv_cache_manager.step([False] * batch_size)
                host_kv_cache_block_offsets = self.kv_cache_manager.get_block_offsets(
                    1)
                kv_cache_block_offsets = host_kv_cache_block_offsets.to("cuda")

                self.new_tokens = torch.tensor([[0]],
                                               device=self.device,
                                               dtype=torch.int32)
                next_context = self.runtime.context_1 if step % 2 else self.runtime.context_0
                next_step_tensors = self.get_next_step_shape_buffer(
                    batch_size, max_context_length, step, context_lengths,
                    host_context_lengths, position_ids, last_token_ids,
                    attention_mask, next_cache_indirection,
                    kv_cache_block_offsets, host_kv_cache_block_offsets)
                print(">>>>>> step=", step)
                for k, v in next_step_tensors.items():
                    print(f"{k}={v.to_torch()}")
                    print(f"{k}.shape={v.to_torch().shape}")
                self.runtime._set_tensors(next_context, next_step_tensors)

            # update input ids
            self.new_tokens = self.get_next_tokens(self.buffer["logits"])

    def decode(
        self,
        batch_input_ids: List[torch.Tensor],
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_lengths.size(0)
        max_context_length = torch.max(input_lengths).item()
        host_context_length = input_lengths.cpu()

        cache_indirections = [
            torch.full((batch_size, 1, self.max_attention_window_size),
                       0,
                       dtype=torch.int32,
                       device=self.device),
            torch.full((batch_size, 1, self.max_attention_window_size),
                       0,
                       dtype=torch.int32,
                       device=self.device),
        ]
        # paged kv cache
        num_blocks, max_blocks_per_seq = self._get_num_paged_blocks(
            self.max_attention_window_size, self.sink_token_length,
            self.use_one_more_block)
        self.buffer["host_kv_cache_pool_pointers"] = torch.tensor(
            [self.kv_cache_pool.data_ptr(), 0], dtype=torch.int64)
        # manager of kv cache
        block_size = self.model_config.num_kv_heads * \
            self.model_config.tokens_per_block * self.model_config.head_size
        self.kv_cache_manager = KVCacheManager(
            num_layers=self.num_attn_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            tokens_per_block=self.model_config.tokens_per_block,
            max_blocks_per_seq=max_blocks_per_seq,
            max_attention_window_size=self.max_attention_window_size,
            sink_token_len=self.sink_token_length,
            beam_width=1,
            use_one_more_block=self.use_one_more_block)
        # add sequences to the manager
        for b in range(batch_size):
            generation_seq = GenerationSequence(b, b)
            self.kv_cache_manager.add_sequence(generation_seq,
                                               max_context_length)

        return self.decode_regular(batch_size, input_lengths,
                                   host_context_length, max_context_length,
                                   cache_indirections, batch_input_ids)


class EngineConfig(object):

    def __init__(
        self,
        pretrained_config: PretrainedConfig,
        build_config: BuildConfig,
    ) -> None:
        self.pretrained_config = pretrained_config
        self.build_config = build_config

    @classmethod
    def from_json_file(cls, config_file):
        with open(config_file) as f:
            config = json.load(f)

        return cls(PretrainedConfig.from_dict(config["pretrained_config"]),
                   BuildConfig.from_dict(config["build_config"]))


class Engine(object):

    def __init__(self, config: EngineConfig, engine: trt.IHostMemory) -> None:
        self.config = config
        self.engine = engine

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int):
        engine_name = f"rank{rank}.engine"
        with open(os.path.join(engine_dir, engine_name), "rb") as f:
            engine_buffer = f.read()

        config = EngineConfig.from_json_file(
            os.path.join(engine_dir, "config.json"))
        config.pretrained_config.set_rank(rank)

        return cls(config, engine_buffer)


class ModelRunner(object):

    def __init__(
        self,
        session: GenerationSession,
        max_batch_size: int,
        max_input_len: int,
        max_seq_len: int,
        max_beam_width: int,
    ) -> None:
        self.session = session
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width

    def prepare_inputs(self, batch_input_ids: List[torch.Tensor]):
        input_lengths = [x.size(0) for x in batch_input_ids]
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
        # remove input padding
        batch_input_ids = torch.concat(batch_input_ids)

        return batch_input_ids, input_lengths

    def generate(self, batch_input_ids: torch.Tensor):
        batch_size = len(batch_input_ids)
        batch_input_ids, input_lengths = self.prepare_inputs(batch_input_ids)

        self.session.setup(batch_size=batch_size,
                           max_context_length=input_lengths.max().item(),
                           max_new_tokens=17,
                           beam_width=1)
        batch_input_ids = batch_input_ids.cuda()
        input_lengths = input_lengths.cuda()

        return self.session.decode(batch_input_ids, input_lengths)

    @classmethod
    def from_engine(cls, engine: Engine, rank: int) -> "ModelRunner":
        pretrained_config = engine.config.pretrained_config
        build_config = engine.config.build_config

        tp_size = pretrained_config.mapping.tp_size
        num_heads = pretrained_config.num_attention_heads // tp_size
        num_kv_heads = pretrained_config.num_key_value_heads
        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
        hidden_size = pretrained_config.hidden_size // tp_size
        head_size = pretrained_config.head_size

        max_batch_size = build_config.max_batch_size
        max_input_len = build_config.max_input_len
        max_output_len = build_config.max_output_len
        max_beam_width = build_config.max_beam_width
        model_config = ModelConfig(
            max_batch_size=build_config.max_batch_size,
            max_beam_width=build_config.max_beam_width,
            vocab_size=pretrained_config.vocab_size,
            num_layers=pretrained_config.num_hidden_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            gpt_attention_plugin=bool(
                build_config.plugin_config.gpt_attention_plugin),
            remove_input_padding=build_config.plugin_config.
            remove_input_padding,
            paged_kv_cache=build_config.plugin_config.paged_kv_cache,
            head_size=head_size,
            tokens_per_block=build_config.plugin_config.tokens_per_block,
            quant_mode=pretrained_config.quant_mode,
            dtype=pretrained_config.dtype)

        torch.cuda.set_device(rank % pretrained_config.mapping.gpus_per_node)

        session = GenerationSession(model_config, engine.engine,
                                    pretrained_config.mapping)

        return cls(session=session,
                   max_batch_size=max_batch_size,
                   max_input_len=max_input_len,
                   max_seq_len=max_input_len + max_output_len,
                   max_beam_width=max_beam_width)

    @classmethod
    def from_dir(cls, engine_dir: str, rank: int):
        engine = Engine.from_dir(engine_dir, rank)
        runner = ModelRunner.from_engine(engine, rank)

        return runner


def load_tokenizer(tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              legacy=False,
                                              padding_side="left",
                                              truncation_side="left",
                                              trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def prepare_input(tokenizer, input_text, max_input_length=923):
    batch_input_ids = []
    for curr_text in input_text:
        input_ids = tokenizer.encode(curr_text,
                                     truncation=True,
                                     max_length=max_input_length)
        batch_input_ids.append(input_ids)
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]

    return batch_input_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir",
                        type=str,
                        required=True,
                        default="The directory of input engine.",
                        help="The input prompt for inference.")
    parser.add_argument("--max-new-tokens",
                        type=int,
                        default=17,
                        help="The max output tokens.")
    parser.add_argument("--input-text",
                        type=str,
                        nargs="+",
                        default=["What is Deep Learning?"],
                        help="The input prompt for inference.")
    parser.add_argument("--no-add-special-tokens",
                        action="store_true",
                        help="Whether or not add special tokens.")
    parser.add_argument("--max-input-length",
                        type=int,
                        default=923,
                        help="The max length of input.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 0. load tokenizer
    tokenizer = load_tokenizer(args.engine_dir)
    # 1. prepare input
    batch_input_ids = prepare_input(tokenizer, args.input_text)
    # 2. init runner
    runtime_rank = tensorrt_llm.mpi_rank()
    runner = ModelRunner.from_dir(args.engine_dir, rank=runtime_rank)
    # 3. generate
    with torch.no_grad():
        runner.generate(batch_input_ids)
        torch.cuda.synchronize()
