import openai
import json
import os
from tqdm import tqdm
import pandas as pd
import random

random.seed(42)

emotion2label = {
        'anger': '3',
        'fear': '4',
        'joy': '1',
        'love': '2',
        'sadness': '0',
        'surprise': '5'
    }

class EmotionSimilarGen:
    """
    A class for emotion sentence generation using OpenAI API.
    """

    def __init__(self, gen_file: str, batch_size: int, api_key: str, **kwargs):
        """
        Initialize with gen_file and batch size. Accepts additional parameters.
        """
        openai.api_key = api_key
        self.gen_file = gen_file
        self.batch_size = batch_size
        self.preamble = ""
        self.n_gen = kwargs.get("n_gen", 6000)
        self.n_gen_per_inference = kwargs.get("n_gen_per_inference", 5)
        self.emotions = kwargs.get("emotions", ["anger", "fear", "joy", "love", "sadness", "surprise"])
        self.queries = []
        self._generate_base_queries()

    def _generate_base_queries(self):
        """
        Generate base queries for the generation task based on input data.
        """
        label2coreset = json.load(open(self.gen_file, 'r'))
        n_queries_per_emotion = self.n_gen // len(self.emotions) // self.n_gen_per_inference
        self.queries = [
            {"emotion": emotion, "coreset": label2coreset[emotion2label[emotion]]}
            for emotion in self.emotions
            for _ in range(n_queries_per_emotion)
        ]

    def _formulate_prompt(self, query: dict):
        """
        Formulate a prompt for OpenAI API based on the given query.
        """
        emotion = query["emotion"]
        coreset = query["coreset"]
        prompt = f"""
## Role:
You are a natural language processing expert, skilled at generating multiple sentences with the emotion {emotion} in Twitter style that are similar to the coreset reference.

## Task:
Given the emotion {emotion}, and coreset reference:
{coreset}

Now generate {self.n_gen_per_inference} sentences that are similar to the coreset reference.

## Note:
1. The sentences should be in Twitter style and as diverse as possible.
2. Just generate the {self.n_gen_per_inference} sentences, one sentence per line. No explanations or analysis.
"""
        return prompt

    def generate_sentences(self):
        """
        Use OpenAI API to generate sentences for all queries.
        """
        results = []
        for query in tqdm(self.queries, desc="Generating sentences"):
            prompt = self._formulate_prompt(query)
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.preamble},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=250,
                )
                # Extract generated text
                generation = response.choices[0].message.content
                results.append({"query": query, "generation": generation})
            except Exception as e:
                print(f"Error generating for query {query}: {e}")
                results.append({"query": query, "generation": None})
        return results

def open_ai_gen(
    gen_file,
    output_dir,
    batch_size=8,
    api_key=None,
    **kwargs
):
    if not api_key:
        raise ValueError("An OpenAI API key is required!")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize EmotionSimilarGen
    generator = EmotionSimilarGen(gen_file=gen_file, batch_size=batch_size, api_key=api_key, **kwargs)

    # Generate sentences
    print("Generating sentences...")
    results = generator.generate_sentences()

    # Write outputs to files
    with open(os.path.join(output_dir, "gen.json"), "w") as gen_json:
        for result in results:
            gen_json.write(json.dumps(result) + "\n")
    
    # Organize into new dataset
    processed = []
    for res in results:
        label = res["query"]["emotion"]
        for new_sent in res["generation"].split("\n"):
            if not new_sent.strip():
                continue
            processed.append([new_sent, emotion2label[label]])
    
    random.shuffle(processed)
    
    df = pd.DataFrame(processed, columns=['text', 'labels'])

    df.to_csv(os.path.join(output_dir, "gen.csv"), index=False)