import json
from collections import defaultdict
from utils import normalize_latex_text, normalized_entropy, extract_boxed_answer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 


def main():
    data_name = "blend"
    output_dir = f"{data_name}_output"
    baseline = ["none", "multilingual_prompting"][0] # "none" represents Mixed-Language Sampling
    model = "Qwen/Qwen3-8B"
    temperature = 0.6
    target_sample_num = 15
    path = "../data/blend.json"
    with open(path, "r", encoding="utf-8") as f:
        references = json.load(f)

    multilingual_data = {}
    languages = ["en", "it", "ms", "zh", "ru", "de", "iw", "bg", "da", "no", "sv", "es", "tl", "oc", "fr"]
    for language in languages:
        if language == "en": sample_num = 40
        else: sample_num = 6
        if baseline == "none":
            output_path = f"{output_dir}/{model.split('/')[-1]}_{language}_{temperature}_{sample_num}_0_dp_rank_0.jsonl"
        else:
            if language != "en":
                output_path = f"{output_dir}/{model.split('/')[-1]}_{baseline}_{language}_{temperature}_{sample_num}_0_dp_rank_0.jsonl"
            else:
                output_path = f"{output_dir}/{model.split('/')[-1]}_{language}_{temperature}_{sample_num}_0_dp_rank_0.jsonl"
        with open(output_path, 'r') as f:
            data = [json.loads(l) for l in f]
        multilingual_data[language] = data

    per_question_entropies = []
    not_in = 0

    en_data = multilingual_data["en"]
    sample_num_all_question = []
    for id_, _ in enumerate(en_data):
        reference = references[id_]
        per_question_counts = defaultdict(int)

        stop_flag = 0
        sample_num_this_question = 0
        for language in languages:
            if sample_num_this_question >= target_sample_num:
                break
            lang_data = multilingual_data[language]
            item = lang_data[id_]
            assert item["idx"] == reference["idx"]
            responses = item["responses"]
            options = reference["options"]
            lang_cot = 0
            lang_num = 1 if target_sample_num == 15 else 2
            for resp in responses:
                if lang_cot == lang_num:
                    break
                if "</think>" not in resp:
                    continue
                
                try:
                    think, answer = resp.rsplit("</think>", 1)
                    ans = normalize_latex_text(extract_boxed_answer(answer).lower())
                except:
                    import pdb
                    pdb.set_trace()
                if ans in options:
                    countries = options[ans]
                    for c in countries:
                        per_question_counts[c] += 1
                    lang_cot += 1
                    sample_num_this_question += 1
                else:
                    not_in += 1
        if sample_num_this_question < target_sample_num:
            print(f"Warning: question {id_} only has {sample_num_this_question} samples.")
            continue
        sample_num_all_question.append(sample_num_this_question)
        try:
            H = normalized_entropy(per_question_counts)
            per_question_entropies.append(H)
        except:
            import pdb
            pdb.set_trace()

    final_score = sum(per_question_entropies) / len(per_question_entropies)
    print("avg:", sum(sample_num_all_question) / len(sample_num_all_question))
    print(sample_num_all_question)
    print("Final Diversity Score:", final_score)
    print("Not matched:", not_in)

if __name__ == "__main__":
    main()