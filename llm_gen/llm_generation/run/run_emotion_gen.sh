#!/usr/bin/bash

# Need to configure
# Model related
model_name="llama3"
model_log_name="llama3-8b"
model="meta-llama/Meta-Llama-3-8B-Instruct"
cuda_devices="2, 3"
n_gpu=2
batch_size=8

paraphrase_gen_output_dir="./data_output/${model_log_name}_emotion_gen"

source /home/dyx723/local/anaconda3/bin/activate pipp_datagen # TODO: change to your environment

# Run direct inference stage
echo "[start] Run emotion generation stage with model ${model_name}"
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python llm_generation/main_hf.py \
    --output_dir ${paraphrase_gen_output_dir} \
    --model ${model} \
    --model_name ${model_name} \
    --batch_size ${batch_size} \
    --n_gpu ${n_gpu} \
    --generation_type emotion_gen
test $? -eq 0 || { echo "Inference failed"; exit 1; }
echo "[end] Run emotion generation stage with model ${model_name}"