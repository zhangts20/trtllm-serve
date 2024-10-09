import os
import shutil
import subprocess
import tensorrt_llm

tllm_version = tensorrt_llm.__version__

from tllm.args import BuildConfig

TLLM_ROOT = os.getenv("TLLM_ROOT", "/usr/src/TensorRT-LLM/examples")

DTYPE_MAPPING = {
    "bfloat16": "bf",
    "float16": "fp",
    "float8": "f8",
    "w4a16": "w4",
    "w8a16": "w8",
    "w8a8": "a8"
}


def get_temp_dir(temp_dir: str, model_dir: str):
    return temp_dir if temp_dir is "None" else f"{model_dir}-temp"


def convert_checkpoint(build_config: BuildConfig):
    command = [
        "python",
        os.path.join(TLLM_ROOT, str(build_config.model_type),
                     "convert_checkpoint.py")
    ]
    command.extend(["--model_dir", build_config.model_dir])
    command.extend(["--tp_size", str(build_config.tp_size), "--pp_size", "1"])
    command.extend(["--dtype", build_config.dtype])
    temp_dir = get_temp_dir(build_config.temp_dir, build_config.model_dir)
    build_config.temp_dir = temp_dir
    command.extend(["--output_dir", temp_dir])
    command.extend(["--workers", str(min(16, os.cpu_count() // 2))])

    command = " ".join(command)
    print(f"The command of converting checkpoint: {command}")
    ret = subprocess.run(command, shell=True).returncode
    if ret != 0:
        raise RuntimeError("Error when converting checkpoint")


def get_output_dir(build_config: BuildConfig):
    if build_config.output_dir != "None":
        print(f"generated engines will be saved to {build_config.output_dir}")
        return build_config.output_dir

    input_file_name = os.path.basename(build_config.model_dir)
    # name-tpN-fp/f8/w4/w8/a8-wcache/ocache
    cache_str = "wcache" if build_config.use_prompt_cache else "ocache"
    output_file_name = f"{input_file_name}-tp{build_config.tp_size}-{DTYPE_MAPPING[build_config.dtype]}-{cache_str}"

    return os.path.join(os.path.dirname(build_config.model_dir),
                        output_file_name)


def export_engine(build_config: BuildConfig):
    command = ["trtllm-build"]
    command.extend([
        "--checkpoint_dir",
        get_temp_dir(build_config.temp_dir, build_config.model_dir)
    ])
    output_dir = get_output_dir(build_config)
    build_config.output_dir = output_dir
    command.extend(["--output_dir", output_dir])
    command.extend(["--gemm_plugin", build_config.dtype])
    command.extend(["--gpt_attention_plugin", build_config.dtype])

    command.extend(["--max_batch_size", str(build_config.max_batch_size)])
    command.extend(["--max_input_len", str(build_config.max_input_len)])
    if tllm_version == "0.10.0":
        command.extend(["--max_output_len", str(build_config.max_output_len)])
    elif tllm_version == "0.13.0":
        command.extend([
            "--max_seq_len",
            str(build_config.max_input_len + build_config.max_output_len)
        ])
    else:
        raise RuntimeError(f"unsupported tensorrt_llm version: {tllm_version}")
    if build_config.max_num_tokens == -1:
        build_config.max_num_tokens = build_config.max_batch_size * build_config.max_input_len
    command.extend(["--max_num_tokens", str(build_config.max_num_tokens)])
    if build_config.use_prompt_cache:
        command.extend(["--use_paged_context_fmha", "enable"])

    command = " ".join(command)
    print(f"The command of converting engine: {command}")
    ret = subprocess.run(command, shell=True).returncode
    if ret != 0:
        raise RuntimeError("Error when converting engine")

    if build_config.remove_temp_dir:
        shutil.rmtree(build_config.temp_dir)


def copy_tokenizer(src_dir: str, dst_dir: str):
    possible_files = [
        "special_tokens_map.json", "tokenization_internlm2.py",
        "tokenization_internlm2_fast.py", "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json"
    ]
    for possible_file in possible_files:
        src_file = os.path.join(src_dir, possible_file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_dir)


def export(build_config: BuildConfig):
    convert_checkpoint(build_config)
    export_engine(build_config)
    copy_tokenizer(build_config.model_dir, build_config.output_dir)
