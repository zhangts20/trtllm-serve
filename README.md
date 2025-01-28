# Repo
Ref: https://github.com/NVIDIA/TensorRT-LLM/tree/main

# Usage Guide
## Build Engine
```shell
# copy your own yml first, please refer to ## Yaml Parser
copy base.py llama2-7b.yml
# use convert_checkpoint.py and trtllm-build to export engine
cd py_scripts
python export.py llama2-7b.yml 
```
## Execute Engine
### Offline Inference
```shell
# build
cd cpp_scripts
cmake .. -DTRT_ROOT=/usr/local/tensorrt && make -j 32
# run --help to get the optional input args, and use mpirun -n N (N=tp*pp) to run
offline_infer --help
```
| args | type | default | notes |
| :---- | :---- | :---- | :---- |
| model_dir | string | None | The input engine directory |
| input_text | string | What is Deep Learning? | The input text for inference |
| max_new_tokens | int | 17 | The max generated tokens |
| streaming | bool | False | Whether to use streaming inference |
| num_beams | int | 1 | The number of return sequences |
| log_level | string | info | The log level, choices=['debug', 'info', 'warning', 'error'] |
### Online inference
#### Start Server
```shell
# build
cd cpp_scripts
cmake .. -DTRT_ROOT=/usr/local/tensorrt && make -j 32
# run --help to get the optional input args
online_infer --help
```
| args | type | default | notes |
| :---- | :---- | :---- | :---- |
| model_dir | string | None | The input engine directory |
| port | int | 18001 | The port of online server |
| log_level | string | info | The log level, choices=['debug', 'info', 'warning', 'error'] |
#### Start Client
```shell
curl 127.0.0.1:18001/generate_stream \
    -X POST \
    -d '{
        "inputs": "What is AI?",
        "parameters": {
            "max_new_tokens": 17,
            "num_beams": 2
        }
    }'
```
The return format is as follows:
```json
{
    "finish_reason": ["running", "running"],
    "generated_text": ["What is", "A"],
    "output_logprobs": [[-2.4906, -1.2093], [-1.2093, -2.4906]],
    "output_tokens": [[1724, 338], [13, 29909]],
    "request_id": 2
}
```
As the number of iterations increases, the values of the `generated_text`, `output_logprobs`, and `output_tokens` gradually grow.
## Yaml Parser
```yaml
# The input directory
model_dir: "/data/llama2-7b"
# The output directory (default: None). Using None is recommended
output_dir: None
# To decide where the `convert_checkpoint.py` should be used
model_type: "llama"
# Whether to remove the output directory of `convert_checkpoint.py`
remove_temp_dir: true
# The tensor parallel size
tp_size: 4
# The pipeline parallel size
pp_size: 1
# The data type, choices=[fp, bf, f8, w4, w8, a8]
dtype: "bf"
# The max batch size of generated engine
max_batch_size: 128
# The max input length of generated engine
max_input_len: 2048
# The max beam width of generated engine
max_beam_width: 1
# The max output length of generated engine
max_output_len: 1024
# The max num tokens of generated engine
max_num_tokens: 20000
# Whether to use kv cache reuse
use_prompt_cache: true
```
Other settings such as paged KV cache and remove input padding, along with other configurations not explicitly mentioned, will follow the default values set by TensorRT-LLM.
## Other Settings
1. When building engine, the env `TRTLLM_ROOT` is set to a default value `../../TensorRT-LLM/examples`.