import os
import torch
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from tensorrt_llm.builder import BuildConfig, Engine, build
from tensorrt_llm.models.modeling_utils import PretrainedConfig, load_model

from config import BuildParam


def build_model(model_dir: str, output_dir: str, build_config: BuildConfig,
                model_config: PretrainedConfig, rank: int) -> Engine:
    torch.cuda.set_device(rank)

    model_config = PretrainedConfig.from_json_file(
        os.path.join(model_dir, "config.json"))
    model = load_model(model_config, model_dir)

    engine = build(model, build_config)
    assert engine is not None
    engine.save(output_dir)


def export(p: BuildParam, model_dir: str, output_dir: str) -> None:
    build_config = BuildConfig.from_dict({
        "max_input_len": p.max_input_length,
        "max_output_len": p.max_output_length,
        "max_batch_size": p.max_batch_size,
        "max_beam_width": p.max_beam_width,
        "max_num_tokens": -1,
        "max_opt_tokens": -1,
    })
    model_config = PretrainedConfig.from_json_file(
        os.path.join(model_dir, "config.json"))

    # tp_size * pp_size threads to export engine
    max_workers = min(torch.cuda.device_count(), p.tp_size * p.pp_size)
    if max_workers == 1:
        for rank in range(p.tp_size * p.pp_size):
            build_model(model_dir, output_dir, build_config, model_config,
                        rank)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as e:
            futures = [
                e.submit(build_model, model_dir, output_dir, build_config,
                         model_config, rank) for rank in p.tp_size * p.pp_size
            ]
            expections = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    expections.append(e)
            assert len(
                expections
            ) == 0, "Checkpoint conversion failed, please check error log."
