#!/usr/bin/bash

# Need to configure
# Data related
input_dir="process_data"
input_file_name="rejected_top_30_ngrams"
# Model related
model_name="llama2"
model_log_name="llama2-70B"
model="meta-llama/Llama-2-70b-chat-hf"
cuda_devices="2,4,5,6"
n_gpu=4
batch_size=8

gen_file="${input_dir}/${input_file_name}.json"
n_gram_centered_gen_output_dir="./data_output/${input_file_name}_${model_log_name}_n_gram_centered_gen"

source /home/dyx723/local/anaconda3/bin/activate pipp_datagen # TODO: change to your environment

# Run direct inference stage
echo "[start] Run n-gram centered generation stage with model ${model}, gen file ${gen_file}"
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python llm_generation/main_hf.py \
    --gen_file ${gen_file} \
    --output_dir ${n_gram_centered_gen_output_dir} \
    --model ${model} \
    --model_name ${model_name} \
    --batch_size ${batch_size} \
    --n_gpu ${n_gpu} \
    --generation_type n_gram_centered_gen
test $? -eq 0 || { echo "Direct inference failed"; exit 1; }
echo "[end] Run n-gram centered generation stage with model ${model_name}, gen file ${gen_file}"