import json
from tqdm import tqdm
from abc import abstractmethod
from typing import List
import pandas as pd

from aux_files.default_prompts import *

class PromptGeneration:
    '''
    A base class that generates prompts for a given generation task.
    Note:
        1. This abstract class should not be instantiated.
        2. The child class should implement the following methods:
            1) _load_extra_info()
                This is where you prepare the ingredients for generating prompts.
            2) _formulate_prompt_lists()
                This is where you specify the prompts
    '''
    def __init__(self, gen_file:str, batch_size:int, **kwargs):
        self.gen_file = gen_file
        self.batch_size = batch_size
        self.load_examples = False if "load_examples" not in kwargs else kwargs["load_examples"]
        self.prompt_path = None if "prompt_path" not in kwargs else kwargs["prompt_path"]
        self.sys_prompt_list = []
        self.usr_prompt_list = []
    
    @abstractmethod
    def _generate_base_queries(self):
        '''
        Generate a list of base queries for the generation task.
        
        Each query contains dynamic information needed for the prompt
        '''

    @abstractmethod
    def _formulate_prompt_lists(query:dict):
        '''
        Determine the content of sys_prompt_list and usr_prompt_list for a given query.
        '''
        pass
    
    @staticmethod
    def _load_examples(example_file:str):
        '''
        Load external examples
        '''
        #TODO: not sure how this will be used
        with open(example_file, 'r') as file:
            examples = file.readlines()
        return examples
    
    @staticmethod
    def _create_dialog(sys_list: List[str], usr_list: List[str]):
        sys_dialog = {"role": "system", "content": "\n".join(sys_list)}
        usr_dialog = {"role": "user", "content": "\n".join(usr_list)}
        return [sys_dialog, usr_dialog]
    
    def _generate_final_prompt(self, query:dict):
        '''
        Generate the final refinement prompt for a given query.
        '''
        self._formulate_prompt_lists(query)
        return self._create_dialog(self.sys_prompt_list, self.usr_prompt_list)

    def batched_prompt_generator(self):
        '''
        return a generator that yields a two tuple, with the following elements:
            1) a list of final prompts
            2) a list of queries
        '''
        for i in range(0, len(self.queries), self.batch_size):
            yield ([self._generate_final_prompt(query) for query in self.queries[i:i+self.batch_size]], [query for query in self.queries[i:i+self.batch_size]])

    def get_length(self):
        return len(self.queries)
    
class N_Gram_Centered_Gen(PromptGeneration):
    '''
    A class for generating prompts for n-gram centered generation task.
    '''
    
    def __init__(self, gen_file:str, batch_size:int, **kwargs):
        '''
        gen_file should be a json file containing a list of base n_grams
        '''
        DEFAULT_N_GEN = 3
        super().__init__(gen_file, batch_size, **kwargs)
        self.preamble = N_Gram_Centered_Gen_PREAMBLE
        self.n_gen = DEFAULT_N_GEN if "n_gen" not in kwargs else kwargs["n_gen"]
        self._generate_base_queries()

    def _generate_base_queries(self):
        '''
        Generate a list of base queries for the generation task.

        Each query contains dynamic information needed for the prompt
        '''
        
        n_grams = json.load(open(self.gen_file, 'r'))
        n_grams = list(n_grams.keys())
        self.queries = [{"n_gram": n_gram.lower(), 'id': id } for id, n_gram in enumerate(n_grams)]

    def _formulate_prompt_lists(self, query:dict):
        '''
        Determine the content of sys_prompt_list and usr_prompt_list for a given query.
        '''
        n_gram = query["n_gram"]
        n = len(n_gram.split())
        instruction = f'''
## Role:
I'm a natural langauge processing expert, and I'm good at generating multiple independent sentences based on a single {n}-gram.

## Task
Given the {n}-gram "{n_gram}", please generate {self.n_gen} sentences that contains the {n}-gram and is meaningful.

## Note
1. The sentences should be meaningful and contains the {n}-gram as it is.
2. Just generate the {self.n_gen} sentences, do not include explanations or analysis. One sentence per line.
3. Make sure the {n}-gram is in the sentence and is not altered.
'''
        self.sys_prompt_list = [self.preamble]
        self.usr_prompt_list = [instruction]

class Paraphrase(PromptGeneration):
    '''
    A class for paraphrase generation task.
    '''
    
    def __init__(self, gen_file:str, batch_size:int, **kwargs):
        '''
        gen_file should be a json file containing a list of base n_grams
        '''
        DEFAULT_N_GEN = 3
        ASSISTANT_PREFIX = 'Assistant: '
        super().__init__(gen_file, batch_size, **kwargs)
        self.preamble = ""
        self.n_gen = DEFAULT_N_GEN if "n_gen" not in kwargs else kwargs["n_gen"]
        self.assistant_prefix = ASSISTANT_PREFIX
        self._generate_base_queries()

    def _generate_base_queries(self):
        '''
        Generate a list of base queries for the generation task.

        Each query contains dynamic information needed for the prompt
        '''
        
        df = pd.read_csv(self.gen_file)
        # Extract the chosen sentences
        sents = df["chosen"].tolist()
        # Extract the last response from the assistant as the seed sentence
        last_responses = [sent.split(self.assistant_prefix)[-1] for sent in sents]
        self.queries = [{"sent": last_responses, 'context': sent, 'id': id } for id, last_responses, sent in zip(range(len(last_responses)), last_responses, sents)]

    def _formulate_prompt_lists(self, query:dict):
        '''
        Determine the content of sys_prompt_list and usr_prompt_list for a given query.
        '''
        seed_sent = query["sent"]
        instruction = f'''
## Role:
You are a natural langauge processing expert, good at generating multiple paraphrases based on a single sentence.

## Task
Given the seed sentence "{seed_sent}", please generate {self.n_gen} paraphrases.

## Note
1. Make sure to keep the meaning of the seed sentence intact.
2. Just generate the {self.n_gen} sentences, do not include explanations or analysis. One sentence per line.
'''
        self.sys_prompt_list = [self.preamble]
        self.usr_prompt_list = [instruction]

class Emotion_gen(PromptGeneration):
    '''
    A class for emotion sentence generation task.
    '''
    
    def __init__(self, gen_file:str, batch_size:int, **kwargs):
        '''
        gen_file should be a json file containing a list of base n_grams
        '''
        DEFAULT_N_GEN = 18000
        N_GEN_PER_INFERENCE = 5
        ASSISTANT_PREFIX = 'Assistant: '
        super().__init__(gen_file, batch_size, **kwargs)
        self.preamble = ""
        self.n_gen = DEFAULT_N_GEN if "n_gen" not in kwargs else kwargs["n_gen"]
        self.n_gen_per_inference = N_GEN_PER_INFERENCE if "n_gen_per_inference" not in kwargs else kwargs["n_gen_per_inference"]
        self.emotions = ["anger", "fear", "joy", "love", "sadness", "surprise"] if "emotions" not in kwargs else kwargs["emotions"]
        self._generate_base_queries()

    def _generate_base_queries(self):
        '''
        Generate a list of base queries for the generation task.

        Each query contains dynamic information needed for the prompt
        '''
        
        n_queries_per_emotion = self.n_gen // len(self.emotions) // self.n_gen_per_inference

        self.queries = [{"emotion": emotion} for emotion in self.emotions for _ in range(n_queries_per_emotion)]

    def _formulate_prompt_lists(self, query:dict):
        '''
        Determine the content of sys_prompt_list and usr_prompt_list for a given query.
        '''
        emotion = query["emotion"]
        instruction = f'''
## Role:
You are a natural langauge processing expert, good at generating multiple sentences with the emotion {emotion} in twitter style.

## Task
Given the emotion {emotion}, please generate {self.n_gen_per_inference} sentences.

## Note
1. Make sure the sentences convey the emotion {emotion}.
2. The sentences should be in twitter style and as diverse as possible.
3. Just generate the {self.n_gen_per_inference} sentences, do not include explanations or analysis. One sentence per line.
'''
        self.sys_prompt_list = [self.preamble]
        self.usr_prompt_list = [instruction]


generation_types = {
    "n_gram_centered_gen": N_Gram_Centered_Gen,
    "paraphrase": Paraphrase,
    "emotion_gen": Emotion_gen,
}

if __name__ == "__main__":
    import math
    generation_type = "paraphrase"
    gen_file = "/local2/shawn/reward_model/data/clean_test.csv"
    batch_size = 2
    prompt_generator = generation_types[generation_type](gen_file, batch_size)
    total_batches = math.ceil(prompt_generator.get_length() / batch_size)
    for prompt_query_tuple in tqdm(prompt_generator.batched_prompt_generator(), total=total_batches):
        print(prompt_query_tuple)
        break