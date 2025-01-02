#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7

cd build && cmake .. -DTRT_DIR=/data/zhangtaoshan/projects/dockers/hub/tensorrt/TensorRT-10.7.0.23 && make -j32 && cd ..

mpirun -n 4 build/online_infer \
    --model_dir "/data/zhangtaoshan/models/llm/trtllm_0.16.0/llama2-7b-tp2-pp2-float16-wcache" \
    --input_text "Why the color of sky is blue?"
