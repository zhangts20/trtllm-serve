from dataclasses import dataclass


@dataclass
class BuildParam:
    model_name: str
    build_type: str
    tp_size: int
    pp_size: int
    max_batch_size: int
    max_beam_width: int
    max_input_length: int
    max_output_length: int
