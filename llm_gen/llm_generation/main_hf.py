# CUDA_VISIBLE_DEVICES=5,6 PYTHONPATH=. python llm-refinement/refine_hf.py --events_dir ./event_data_genia --triggers_file ./trigger_output/trigger_genia_llama2_70b_chat_0shot --gen_file ./data_output/genia/llama2_70b_chat_0shot/gen.txt --output_dir ./data_output/genia/llama2_70b_chat_0shot/dv/ --model meta-llama/Llama-2-70b-chat-hf --model_name llama2 --batch_size 32
import os, random, json, math
from typing import Optional, List
from argparse import ArgumentParser
from tqdm import tqdm
from vllm import SamplingParams

from generation_types import generation_types
from extract_clean_gen import extract_clean_gen_types
from utils.vllm import set_seed, load_model, flatten_prompt

def main():

    # configuration
    parser = ArgumentParser()
    parser.add_argument('--gen_file', required=False, default="./data_output/llama2_13b_chat_hf_0shot/gen.txt")
    parser.add_argument('--output_dir', required=False, default="./data_output/llama2_13b_chat_hf_0shot/refinement/")
    parser.add_argument('--model', required=False, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument('--model_name', required=False, default="llama2_13b_chat_hf")
    parser.add_argument('--n_gpu', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_tokens', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--generation_type', type=str, default="n_gram_centered_gen")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load model
    model, tokenizer = load_model(args.model, n_gpu=args.n_gpu, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_f = open(args.output_dir+"/gen.txt", "w")
    output_gen_json_f = open(args.output_dir+"/gen.json", "w")
    input_prompts_f = open(args.output_dir+"/prompts.txt", "w")
    output_generations_f = open(args.output_dir+"/raw_gen.txt", "w")
    
    if args.generation_type in generation_types:
        generation_tool = generation_types[args.generation_type](
            args.gen_file, 
            args.batch_size, 
        )
    else:
        raise ValueError(f"Generation type {args.generation_type} not found")
    
    n_queries = generation_tool.get_length()
    print ("Total number of queries to make: %d" % n_queries)
    n_total_batches = math.ceil(n_queries / args.batch_size)
    for prompt_query_tuple in tqdm(generation_tool.batched_prompt_generator(), total=n_total_batches):
        prompts = prompt_query_tuple[0]
        queries = prompt_query_tuple[1]

        flattened_prompts = flatten_prompt(args.model_name, prompts)
        try:
            results = model.generate(flattened_prompts, sampling_params, use_tqdm=False)
        except Exception as e:
            print ("Error in generating: ", e)
            continue
        generations = [ r.outputs[0].text for r in results ]

        extract_gen = extract_clean_gen_types[args.generation_type]
        clean_generations = [ extract_gen(gen, query) for gen, query in zip(generations, queries) ]

        for p, g in zip(prompts, generations):
            input_prompts_f.write(json.dumps(p)+"\n")
            output_generations_f.write(json.dumps(g)+"\n")
    
        for idx, generation in enumerate(clean_generations):
            if not generation:
                generation = "no generation"
            try:
                output_f.write(generation+"\n")
                output_f.flush()
            except Exception as e:
                print ("Error in writing to file: ", e)
                output_f.write("Error in writing to file\n")
                output_f.flush()
        
        for query in queries:
            output_gen_json_f.write(json.dumps(query)+"\n")
            output_gen_json_f.flush()

if __name__ == "__main__":
    main()