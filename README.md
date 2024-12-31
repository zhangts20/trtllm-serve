# Repo
Ref: https://github.com/NVIDIA/TensorRT-LLM/tree/main

# Usage Guide
## Build Engine
```shell
# copy your own yml first
copy base.py llama2-7b.yml
# use convert_checkpoint.py and trtllm-build to export engine
cd py_scripts
python export.py llama2-7b.yml 
```
## Execute Engine
```shell
# build
cd cpp_scripts
cmake .. -DTRT_ROOT=/usr/local/tensorrt
make -j 64
# run
mpirun -n 4 build/llm --model_dir /models/llama2-7b-tp2-pp2-float16-wcache
```
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