import json
from transformers import AutoTokenizer
import os
import numpy as np

model = "Qwen3-8B"
languages = ["en", "it", "ms", "zh", "ru", "de", "iw", "bg", "da", "no", "sv",
             "es", "tl", "oc", "fr", "mix"]
temperature = 0.6
data = "novelty"
output_dir = "evaluation_results"
result_table = {}

for lang in languages:
    if lang == 'mix':
        path =  f"{output_dir}/{model}_multilingual_{temperature}_20_0_similarity.jsonl"
    else:
        path = f"{output_dir}/{model}_{lang}_{temperature}_20_0_similarity.jsonl"

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        instances = [json.loads(l) for l in f]

    if len(instances) == 0:
        raise FileNotFoundError(path)

    res_lst = []
    less_cot = 0

    for item in instances:
        score = item.get("score", [])

        if len(score) < 15:
            less_cot += 1
            continue
        score = score[:15]
        all_score = []
        for s in score:
            all_score.extend(s)
        avg = np.mean(score).item()
        res_lst.append(avg)
    if len(res_lst) == 0:
        avg_res = 0.0
    else:
        avg_res = sum(res_lst) / len(res_lst)
    result_table[lang] = round(avg_res*100, 3)

print(result_table)