import torch, random, time
import numpy as np
import json

from transformers import AutoTokenizer
from vllm import LLM

PROMPT = {
    "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",  # meta-llama/Meta-Llama-3-8B-Instruct
    "llama2": "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]",      # meta-llama/Llama-2-13b-chat-hf
    "mixtral": "<s>[INST] \n{system_prompt}\n{user_prompt}\n [/INST]",                       # mistralai/Mixtral-8x7B-Instruct-v0.1
    "gemma": "<bos><start_of_turn>user\n{system_prompt}\n{user_prompt}\n<end_of_turn>\n<start_of_turn>model" # google/gemma-7b-it
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

def load_model(model_name_or_path, n_gpu=1, seed=0, gpu_memory_utilization=0.9, dtype="auto", lora_finetuned_dir=None):

    # Load the FP16 model
    start_time = time.time()
    if lora_finetuned_dir:
        with open(lora_finetuned_dir + "adapter_config.json", "r") as f:
            lora_config = json.load(f)
        print ("Loading %s in %s... with Low-Rank Adapter" % (lora_config['base_model_name_or_path'], dtype))
        model = LLM(model=lora_config['base_model_name_or_path'], 
                    dtype=dtype, 
                    tensor_parallel_size=n_gpu,
                    seed=seed,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enable_lora=True,
                    max_lora_rank=lora_config['r'],
                    )
    else:
        print ("Loading %s in %s..." % (model_name_or_path, dtype))
        model = LLM(model=model_name_or_path, 
                    dtype=dtype, 
                    tensor_parallel_size=n_gpu,
                    seed=seed,
                    gpu_memory_utilization=gpu_memory_utilization,
                    )
    print ("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"

    return model, tokenizer

def flatten_prompt(model_name_or_path, dialogs):
    base_prompt = ""
    for m in PROMPT:
        if m.lower() in model_name_or_path.lower():
            base_prompt = PROMPT[m]
            break
    if base_prompt == "":
        assert False, "Model name '%s' not found in base prompt" % model_name_or_path
    
    prompts = []
    for dialog in dialogs:
        assert len(dialog) == 2 and dialog[0]['role'] == "system" and dialog[1]['role'] == "user", ("Dialog not formatted correctly", dialog)
        prompt = base_prompt.format(system_prompt=dialog[0]['content'], user_prompt=dialog[1]['content'])
        prompts.append(prompt)
    
    return prompts