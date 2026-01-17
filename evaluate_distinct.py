import argparse
import asyncio
import functools
import json
import os

import datasets
import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CONCURRENT_REQUESTS = 1

@functools.cache
def load_deberta_tokenizer_and_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("deberta-v3")
    model = AutoModelForSequenceClassification.from_pretrained(
        "deberta-v3"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


async def bleu(prompt: str, s1: str, s2: str):
    return (
        sacrebleu.corpus_bleu([s1], [[s2]]).score
        + sacrebleu.corpus_bleu([s2], [[s1]]).score
    ) / 200


async def rouge1(prompt: str, s1: str, s2: str):
    rouge_eval = rouge_scorer.score(s1, s2)
    return rouge_eval["rouge1"].fmeasure


async def bertscore(prompt: str, s1: str, s2: str):
    return bertscorer.compute(
        predictions=[s1],
        references=[s2],
        model_type="microsoft/deberta-large",
    )["f1"][0]


@torch.inference_mode()
async def classifier_score(prompt: str, s1: str, s2: str):
    tokenizer, model = load_deberta_tokenizer_and_model()
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()


async def equivalence_check_gpt4(prompt: str, response_0: str, response_1: str) -> bool:
    class Equivalence(BaseModel):
        equivalent: bool

    """Asynchronously checks equivalence between two responses."""
    messages = [
        {
            "role": "system",
            "content": "For a given prompt, determine whether the two responses are semantically equivalent.",
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Prompt: " + prompt,
                    "Response A: " + response_0,
                    "Response B: " + response_1,
                ],
            ),
        },
    ]

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0,
            response_format=Equivalence,
        )
        return response.choices[0].message.parsed.equivalent
    except Exception as e:
        print(f"Error in equivalence check: {e}")
        return False


async def equivalence_check_unigram(
    prompt: str, response_0: str, response_1: str
) -> bool:
    return await rouge1(prompt, response_0, response_1) > 0.458


async def equivalence_check_bertscore(
    prompt: str,
    response_0: str,
    response_1: str,
) -> bool:
    scores = await bertscore(prompt, response_0, response_1)
    return scores["f1"][0] > 0.719


async def equivalence_check_classifier(
    prompt: str,
    response_0: str,
    response_1: str,
) -> bool:
    score = await classifier_score(prompt, response_0, response_1)
    return score > 0.102, score


async def partition_responses(
    prompt: str,
    responses: list[str],
    equivalence_alg,
) -> list[int]:
    """Partitions responses into equivalence classes."""

    equivalence_classes = []
    
    ids = [i for i in range(len(responses))]
    score_lst = []
    partition = [-1] * len(responses)

    for i in range(len(responses)):
        if partition[i] >= 0:
            continue

        current_class = [responses[i]]
        partition[i] = len(equivalence_classes)

        for j in range(i + 1, len(responses)):
            if partition[j] == -1:
                result, score = await equivalence_alg(
                prompt,
                current_class[0],
                responses[j],
                )

                score_lst.append(((i, j), score))

                if result:
                    current_class.append(responses[j])
                    partition[j] = len(equivalence_classes)

        equivalence_classes.append(current_class)

    assert all(p >= 0 for p in partition)
    return partition, ids, score_lst


EQUIVALENCE_ALGS = {
    "gpt4": equivalence_check_gpt4,
    "unigram": equivalence_check_unigram,
    "bertscore": equivalence_check_bertscore,
    "classifier": equivalence_check_classifier,
}


async def process_instances(instances, output_file, equivalence_alg):
    """Processes all instances concurrently and writes results to a file."""
    # Check if file exists and has matching keys
    if os.path.exists(output_file):
        print("exist")
        return None

    async with aio_open(output_file, "w", buffering=1) as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                partition, ids, score_list = await partition_responses(
                    instance["question"],
                    instance["responses"],
                    equivalence_alg,
                )
                return {**instance, "partition": partition, "distinct": max(partition), "ids": ids, "score_list": score_list}

        tasks = [process_single_instance(instance) for instance in instances]

        for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg",
        default="classifier",
        help="Equivalence testing method",
        choices=EQUIVALENCE_ALGS,
    )
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
    equivalence_alg = EQUIVALENCE_ALGS[args.alg]

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
        instances = [{"idx":i, "question": None, "responses": []} for i in range(item_num)]

        for language, lang_data in all_data.items():
            assert len(lang_data) == item_num
            for idx, item in enumerate(lang_data):
                question = item['question']
                if not instances[idx]['question']:
                    instances[idx]['question'] = question
                else:
                    assert instances[idx]['question'] == question
                if "final_answer" in item:
                    responses = item['final_answer']
                    instances[idx]['responses'].append(responses[0].strip())
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
        output_file = f"{args.output_dir}/{model}_{language}_{temperature}_{sample_num}_{flag}_distinct.jsonl"
    else:
        output_file = f"{args.output_dir}/{model}_multilingual_{temperature}_{sample_num}_{flag}_distinct.jsonl"
    print(output_file)
    await process_instances(instances, output_file, equivalence_alg)
    print(output_file)


if __name__ == "__main__":
    asyncio.run(main())