import re
import json
import argparse
import random
import pandas as pd

random.seed(42)

def post_process_emotion_gen(input_file):
    with open(input_file, 'r') as f:
       lines = f.readlines()
    
    data = [json.loads(line) for line in lines]

    processed = []

    emotion2label = {
        'anger': '3',
        'fear': '4',
        'joy': '1',
        'love': '2',
        'sadness': '0',
        'surprise': '5'
    }

    for d in data:
       if d['emotion'] in emotion2label:
           for sent in d['gen_sents'][1:]:
               processed.append([
                   sent,
                   emotion2label[d['emotion']]
               ])
    # Shuffle the data
    random.shuffle(processed)
    
    df = pd.DataFrame(processed, columns=['text', 'labels'])

    df.to_csv(input_file.replace('.json', '.csv'), index=False)

input_file = '/local2/shawn/reward_model/data_output/llama3-8b_emotion_gen/gen.json'
post_process_emotion_gen(input_file)