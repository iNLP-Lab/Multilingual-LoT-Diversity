import json

data_name = "novelty"
models = [
    "Qwen3-8B"
]

languages = [
    'en', 'it', 'ms', 'zh', 'ru', 'de', 'iw',
    'bg', 'da', 'no', 'sv', 'es', 'tl', 'oc', 'fr'
]

for model in models:
    print(f"Model: {model}")
    json_path = f"gpt_eval_single_{model}_{data_name}.jsonl"

    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    language_scores = {lang: [] for lang in languages}

    for item in data:
        lang = item["language"]
        total_score = item.get("scores", {}).get("Total Score", 0)
        if lang in language_scores:
            language_scores[lang].append(total_score)
    avg_scores = {}
    for lang, scores in language_scores.items():
        if scores:
            avg_scores[lang] = sum(scores) / len(scores)
        else:
            avg_scores[lang] = 0.0

    # 输出
    for lang in languages:
        print(f"{lang}: {avg_scores[lang]:.6f}")