import json
import os

model = "Qwen3-8B"
languages = ["en", "it", "ms", "zh", "ru", "de", "iw", "bg", "da", "no", "sv",
             "es", "tl", "oc", "fr", "mix"]
temperature = 0.6
data = "novelty"
output_dir = "evaluation_results"
result_table = {}

for lang in languages:
    if lang == 'mix':
        path =  f"{output_dir}/{model}_multilingual_{temperature}_20_0_distinct.jsonl"
    else:
        path = f"{output_dir}/{model}_{lang}_{temperature}_20_0_distinct.jsonl"

    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            instances = [json.loads(l) for l in f]

        if len(instances) == 0:
            raise FileNotFoundError(path)

        res_lst = []
        less_cot = 0

        for item in instances:
            partition = item.get("partition", [])

            if len(partition) < 15:
                less_cot += 1
                continue

            partition = partition[:15]
            distinct = max(partition)
            num = distinct + 1
            res_lst.append(num / len(partition))

        # 防止除 0
        if len(res_lst) == 0:
            avg_res = 0.0
        else:
            avg_res = sum(res_lst) / len(res_lst)

        result_table[lang] = round(avg_res*100, 3)

    except Exception as e:
        result_table[lang] = None

print(result_table)