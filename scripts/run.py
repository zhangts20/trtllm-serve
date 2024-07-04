import torch
import argparse
import tensorrt_llm

from process_inputs import load_tokenizer, parse_inputs
from tensorrt_llm.runtime import ModelRunner


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

    tokenizer = load_tokenizer(args.engine_dir)
    batch_input_ids = parse_inputs(tokenizer, args.input_text,
                                   not args.no_add_special_tokens,
                                   args.max_input_length)
    input_lengths = [x.size(0) for x in batch_input_ids]

    runtime_rank = tensorrt_llm.mpi_rank()
    runner_kwargs = dict(engine_dir=args.engine_dir, rank=runtime_rank)
    runner = ModelRunner.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(batch_input_ids,
                                  max_new_tokens=args.max_new_tokens,
                                  end_id=tokenizer.eos_token_id,
                                  pad_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()

    if runtime_rank == 0:
        # shape=(batch_size, num_beams, input+output)
        output_ids = outputs
        batch_size, num_beams, _ = output_ids.size()
        for b in range(batch_size):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f"Input [Text {b}]: '{input_text}'")
            for n in range(num_beams):
                outputs = output_ids[b][n][input_lengths[b]:].tolist()
                output_text = tokenizer.decode(outputs)
                print(f"Output [Text {b} Beam {n}]: '{output_text}'")
