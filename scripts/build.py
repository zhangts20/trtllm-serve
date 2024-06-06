import os
import subprocess
from typing import List
from dataclasses import dataclass

# The root directory of models
root_dir = "/data/zhangtaoshan/models"


@dataclass
class BuildParam:
    # The os.path.join(root_dir, model_type) refers to the origin model
    model_type: str
    # The build type, support fp16, w8a16, w4a16, w8a8 and all can be
    # combined with int8kv
    build_type: str
    # Tensor Parallelism
    tp_size: int
    # Pipeline Parallelism
    pp_size: int
    # The max batch size of generated model
    max_batch_size: int
    # The max input length of generated model
    max_input_length: int
    # The max output length of generated model
    max_output_length: int


def export(p: BuildParam, input_dir: str) -> List[str]:
    cmd = ["trtllm-build"]

    cmd.extend(["--checkpoint_dir", os.path.join(root_dir, input_dir)])
    output_name = "_".join(p.build_type.split(","))
    output_dir = os.path.join(
        root_dir,
        f"{p.model_type}_{output_name}_{p.max_batch_size}_{p.max_input_length}_{p.max_output_length}"
    )
    cmd.extend(["--output_dir", output_dir])

    cmd.extend(["--max-batch-size", p.max_batch_size])
    cmd.extend(["--max_input_len", p.max_input_length])
    cmd.extend(["--max_output_len", p.max_output_length])

    cmd.extend(["--tp_size", p.tp_size])
    cmd.extend(["--pp_size", p.pp_size])

    cmd = " ".join(cmd)
    print(f"The cmd of exporting engine: {cmd}")
    ret = subprocess.run(cmd, shell=True).returncode
    if ret != 0:
        raise RuntimeError("Error when exporting engine.")
    print(f"Export engine to {output_dir} successfully.")


def main(param: BuildParam) -> None:
    pass


if __name__ == "__main__":
    # The list of params to build model
    build_params = [{
        "model_type": "llama-7b",
        "build_type": "fp16",
        "tp_size": 1,
        "pp_size": 1,
        "max_batch_size": 8,
        "max_input_length": 1024,
        "max_output_length": 512
    }]
    for build_param in build_params:
        main(**build_param)
