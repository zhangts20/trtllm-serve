import os
import shutil
import tensorrt_llm.models.convert_utils as cu

from typing import Optional
from datasets import load_from_disk


def load_calib_dataset_re(dataset_name_or_dir: str,
                          config_name: Optional[str] = None,
                          split: Optional[str] = None,
                          key: Optional[str] = None,
                          **kwargs):
    dataset = load_from_disk(dataset_name_or_dir)

    return dataset["text"]


def replace_function():
    cu.load_calib_dataset = load_calib_dataset_re


def copy_tokenizer(input_dir: str, output_dir: str):
    # copy tokenizer*
    for filename in os.listdir(input_dir):
        if filename.startswith("tokenizer"):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy(src_path, dst_path)
    # copy special_tokens_map.json
    src_path = os.path.join(input_dir, "special_tokens_map.json")
    dst_path = os.path.join(output_dir, "special_tokens_map.json")
    shutil.copy(src_path, dst_path)
