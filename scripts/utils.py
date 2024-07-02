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
