import argparse
import functools
import json
import os

import datasets
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
import vllm
from vllm import LLM


def process_instances(instances, output_file):
    if os.path.exists(output_file):
        exit("exists")
    model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
    all_input_texts = []
    sample_num_lst = []
    for instance in instances:
        input_texts = instance['responses']
        all_input_texts.extend(input_texts)
        sample_num_lst.append(len(input_texts))
    outputs = model.embed(all_input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    
    results = []
    start_id = 0
    for i, num in enumerate(sample_num_lst):
        sample_embeddings = embeddings[start_id: start_id + num]
        sample_scores = sample_embeddings @ sample_embeddings.T
        start_id += num 
        results.append({
            "idx": instances[i]['idx'],
            "question": instances[i]['question'],
            "score": sample_scores.tolist()
            }
        )
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory to save evaluation results", required=True
    )

    parser.add_argument(
        "--output-dir", type=str, required=True
    )

    parser.add_argument(
        "--multilingual", action="store_true",
    )

    parser.add_argument(
        "--language", type=str, default="en"
    )

    parser.add_argument(
        "--temperature", type=float
    )

    parser.add_argument(
        "--model", type=str, default="Qwen3-8B"
    )

    parser.add_argument(
        "--flag", type=int, default=0
    )

    parser.add_argument(
        "--sample_num", type=int, default=20
    )

    args = parser.parse_args()

    eval_dir = args.eval_dir
    model = args.model

    temperature = args.temperature
    flag = args.flag
    sample_num = args.sample_num
    if args.multilingual:
        all_data = {}
        languages = ['en', 'it', 'ms', 'zh', 'ru', 'de', 'iw', 'bg', 'da', 'no', 'sv', 'es', 'tl', 'oc', 'fr']
        for language in languages:
            file_name = f"{eval_dir}/{model}_{language}_{temperature}_{sample_num}_{flag}_dp_rank_0.jsonl"
            with open(file_name, 'r') as f:
                lang_data = [json.loads(l) for l in f]
            all_data[language] = lang_data
        item_num = len(all_data[languages[0]])
        tmp_instances = [{"idx":i, "question": None, "responses": []} for i in range(item_num)]

        for language, lang_data in all_data.items():
            assert len(lang_data) == item_num
            for idx, item in enumerate(lang_data):
                question = item['question']
                if not tmp_instances[idx]['question']:
                    tmp_instances[idx]['question'] = question
                else:
                    assert tmp_instances[idx]['question'] == question
                if "final_answer" in item:
                    responses = item['final_answer']
                    tmp_instances[idx]['responses'].append(responses[0].strip())
        instances = []
        for tmp_instance in tmp_instances:
            if len(tmp_instance['responses']) > 2:
                instances.append(tmp_instance)
        for instance in instances:
            print(len(instance['responses']))
    else:
        language = args.language
        file_name = f"{eval_dir}/{model}_{language}_{temperature}_{sample_num}_{flag}_dp_rank_0.jsonl"
        with open(file_name, 'r') as f:
            data = [json.loads(l) for l in f]
        item_num = len(data)
        instances = []
        for idx, item in enumerate(data):
            if "final_answer" not in item:
                continue
            responses = [ans.strip() for ans in item['final_answer']]
            item_ = {
                'idx': idx,
                'question': item['question'],
                'responses': responses
            }
            if len(responses) < 2:
                continue
            instances.append(item_)
    
    if not args.multilingual:
        output_file = f"{args.output_dir}/{model}_{language}_{temperature}_{sample_num}_{flag}_similarity.jsonl"
    else:
        output_file = f"{args.output_dir}/{model}_multilingual_{temperature}_{sample_num}_{flag}_similarity.jsonl"
    print(output_file)
    process_instances(instances, output_file)
    print(output_file)


if __name__ == "__main__":
    main()