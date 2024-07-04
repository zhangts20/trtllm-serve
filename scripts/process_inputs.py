import torch

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              legacy=False,
                                              padding_side="left",
                                              truncation_side="left",
                                              trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def parse_inputs(tokenizer, input_text, add_special_tokens: bool,
                 max_input_length: int):
    batch_input_ids = []
    for curr_text in input_text:
        input_ids = tokenizer.encode(curr_text,
                                     add_special_tokens=add_special_tokens,
                                     truncation=True,
                                     max_length=max_input_length)
        batch_input_ids.append(input_ids)

    return [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
