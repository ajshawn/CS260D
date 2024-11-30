import openai
import json
import os
import asyncio
from tqdm import tqdm
import pandas as pd
import random
from aiohttp import ClientSession

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
    A class for emotion sentence generation using OpenAI API with async batching, worker limits, and sleep intervals.
    """

    def __init__(self, gen_file: str, batch_size: int, api_key: str, max_workers: int = 10, sleep_interval: int = 50, time_sleep: int = 10, **kwargs):
        """
        Initialize with gen_file, batch size, max workers, sleep interval, and sleep duration. Accepts additional parameters.
        """
        openai.api_key = api_key
        self.gen_file = gen_file
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.sleep_interval = sleep_interval
        self.time_sleep = time_sleep
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
You are a natural language processing expert, skilled at generating more diverse sentences with the emotion {emotion} in Twitter style that are different to the coreset reference.

## Task:
Given the emotion {emotion}, and coreset reference:
{coreset}

Now generate {self.n_gen_per_inference} sentences that are different to the coreset reference to increase the diversity of the dataset.

## Note:
1. The sentences should be in Twitter style and as diverse as possible.
2. Just generate the {self.n_gen_per_inference} sentences, one sentence per line. No explanations or analysis.
"""
        return prompt

    async def _generate_query(self, session, semaphore, query):
        """
        Generate sentences for a single query with semaphore control.
        """
        async with semaphore:
            prompt = self._formulate_prompt(query)
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {openai.api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": self.preamble},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 250,
                    },
                ) as response:
                    data = await response.json()
                    generation = data["choices"][0]["message"]["content"]
                    return {"query": query, "generation": generation}
            except Exception as e:
                print(f"Error generating for query: {e} with data: {data if 'data' in locals() else None}")
                return {"query": query, "generation": None}

    async def generate_sentences(self):
        """
        Generate sentences for all queries in chunks with sleep intervals.
        """
        semaphore = asyncio.Semaphore(self.max_workers)
        results = []
        async with ClientSession() as session:
            for i in tqdm(range(0, len(self.queries), self.sleep_interval), desc="Generating chunks"):
                chunk = self.queries[i: i + self.sleep_interval]
                tasks = [self._generate_query(session, semaphore, query) for query in chunk]
                
                # Process chunk
                chunk_results = await asyncio.gather(*tasks)
                results.extend(chunk_results)

                # Pause between chunks
                if i + self.sleep_interval < len(self.queries):
                    print(f"Pausing for {self.time_sleep} seconds...")
                    await asyncio.sleep(self.time_sleep)
                                
        return results


def open_ai_gen(
    gen_file,
    output_dir,
    batch_size=16,
    api_key=None,
    max_workers=5,
    sleep_interval=50,
    time_sleep=10,
    **kwargs
):
    if not api_key:
        raise ValueError("An OpenAI API key is required!")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize EmotionSimilarGen
    generator = EmotionSimilarGen(
        gen_file=gen_file,
        batch_size=batch_size,
        api_key=api_key,
        max_workers=max_workers,
        sleep_interval=sleep_interval,
        time_sleep=time_sleep,
        **kwargs,
    )

    # Generate sentences
    print("Generating sentences...")
    results = asyncio.run(generator.generate_sentences())

    # Write outputs to files
    with open(os.path.join(output_dir, "gen.json"), "w") as gen_json:
        for result in results:
            gen_json.write(json.dumps(result) + "\n")
    
    # Organize into new dataset
    processed = []
    for res in results:
        label = res["query"]["emotion"]
        if res["generation"]:
            for new_sent in res["generation"].split("\n"):
                if not new_sent.strip():
                    continue
                processed.append([new_sent, emotion2label[label]])
    
    random.shuffle(processed)
    
    df = pd.DataFrame(processed, columns=['text', 'labels'])

    df.to_csv(os.path.join(output_dir, "gen.csv"), index=False)
