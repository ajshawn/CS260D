#!/usr/bin/bash

# Need to configure
# Data related
input_dir="/local2/shawn/reward_model/data"
input_file_name="clean_test"
# Model related
model_name="gemma"
model_log_name="gemma-2-9b"
# model="meta-llama/Meta-Llama-3-70B-Instruct"
# model="mistralai/Mistral-7B-Instruct-v0.1"
# model="mistralai/Mixtral-8x7B-Instruct-v0.1"
# model="google/gemma-7b-it"
model="google/gemma-2-9b-it"
cuda_devices="1, 2, 3, 4"
n_gpu=4
batch_size=4

gen_file="${input_dir}/${input_file_name}".csv
paraphrase_gen_output_dir="./data_output/${input_file_name}_${model_log_name}_paraphrase_gen"

source /home/dyx723/local/anaconda3/bin/activate pipp_datagen # TODO: change to your environment

# Run direct inference stage
echo "[start] Run paraphrase generation stage with model ${model_name}, gen file ${gen_file}"
CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python llm_generation/main_hf.py \
    --gen_file ${gen_file} \
    --output_dir ${paraphrase_gen_output_dir} \
    --model ${model} \
    --model_name ${model_name} \
    --batch_size ${batch_size} \
    --n_gpu ${n_gpu} \
    --generation_type paraphrase
test $? -eq 0 || { echo "Direct inference failed"; exit 1; }
echo "[end] Run paraphrase generation stage with model ${model_name}, gen file ${gen_file}"