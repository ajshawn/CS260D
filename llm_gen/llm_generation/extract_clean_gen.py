import re

def extract_gen_for_n_gram_centered_gen(raw_gen: str, query: dict) -> str:
    extra_prefix = 'Sure,'
    # Split based on new line
    eles = raw_gen.split("\n")
    # Filter out empty strings
    eles = list(filter(lambda x: x != "", eles))
    res = []
    for ele in eles:
        ele = ele.strip()
        # Skip sentences that starts with extra_prefix
        if ele.startswith(extra_prefix):
            continue
        # Remove leading non-alpha characters
        while len(ele) > 0 and not ele[0].isalpha():
            ele = ele[1:]
        if ele:
            res.append(ele)
    query["gen_sents"] = res
    
    return str(res)

def extract_gen_for_paraphrase(raw_gen: str, query: dict) -> str:
    extra_prefix = 'Sure,'
    # Split based on new line
    eles = raw_gen.split("\n")
    # Filter out empty strings
    eles = list(filter(lambda x: x != "", eles))
    res = []
    for ele in eles:
        ele = ele.strip()
        # Skip sentences that starts with extra_prefix
        if ele.startswith(extra_prefix):
            continue
        # Remove leading non-alpha characters
        while len(ele) > 0 and not ele[0].isalpha():
            ele = ele[1:]
        if ele:
            res.append(ele)

    if len(res) == 4:
        query["gen_sents"] = res[1:]
    elif len(res) == 3:
        query["gen_sents"] = res
    
    return str(res)

def extract_gen_for_emotion_gen(raw_gen: str, query: dict) -> str:
    extra_prefix = 'Sure,'
    # Split based on new line
    eles = raw_gen.split("\n")
    # Filter out empty strings
    eles = list(filter(lambda x: x != "", eles))
    res = []
    for ele in eles:
        ele = ele.strip()
        # Skip sentences that starts with extra_prefix
        if ele.startswith(extra_prefix):
            continue
        # Remove leading non-alpha characters
        while len(ele) > 0 and not ele[0].isalpha():
            ele = ele[1:]
        if ele:
            res.append(ele)
    query["gen_sents"] = res
    
    return str(res)

extract_clean_gen_types = {
    "n_gram_centered_gen": extract_gen_for_n_gram_centered_gen,
    "paraphrase": extract_gen_for_paraphrase,
    "emotion_gen": extract_gen_for_emotion_gen,
}
