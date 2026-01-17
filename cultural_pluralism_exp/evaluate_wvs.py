import json
from collections import defaultdict
from utils import normalize_latex_text, normalized_entropy, extract_boxed_answer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

def main():
    data_name = "wvs"
    output_dir = f"{data_name}_output"
    language = "en"
    sample_num = 40
    baseline = ["none", "high_temperature", "request_diversity"][0]
    model = "Qwen/Qwen3-8B"
    temperature = 1.0 if baseline == "high_temperature" else 0.6
    target_sample_num = 15
    if baseline in ["none", "high_temperature"]:
        output_path = f"{output_dir}/{model.split('/')[-1]}_{language}_{temperature}_{sample_num}_0_dp_rank_0.jsonl"
    elif baseline == "request_diversity":
        output_path = f"{output_dir}/{model.split('/')[-1]}_{baseline}_{language}_{temperature}_{sample_num}_0_dp_rank_0.jsonl"

    path = "../data/wvs.json"
    with open(path, "r", encoding="utf-8") as f:
        references = json.load(f)

    with open(output_path, 'r') as f:
        data = [json.loads(l) for l in f]

    per_question_entropies = []
    not_in = 0
    sample_num_all_question = []
    for id_, item in enumerate(data):
        sample_num_this_question = 0
        reference = references[id_]

        responses = item["responses"]
        options = reference["option"]

        per_question_counts = defaultdict(int)

        for resp in responses:
            if sample_num_this_question >= target_sample_num:
                break
            if "</think>" not in resp:
                continue

            think, answer = resp.rsplit("</think>", 1)
            ans = normalize_latex_text(extract_boxed_answer(answer).lower())
            if ans.isdigit():
                per_question_counts[ans] += 1
                sample_num_this_question += 1
        
        sample_num_all_question.append(sample_num_this_question)
        try:
            H = normalized_entropy(per_question_counts)
            per_question_entropies.append(H)
        except:
            import pdb
            pdb.set_trace()

    final_score = sum(per_question_entropies) / len(per_question_entropies)
    print("avg:", sum(sample_num_all_question) / len(sample_num_all_question))
    print("Final Diversity Score:", final_score)
    print("Not matched:", not_in)

if __name__ == "__main__":
    main()