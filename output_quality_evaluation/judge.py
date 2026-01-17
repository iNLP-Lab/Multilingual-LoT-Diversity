import os
import json
import random
import re
import time
from tqdm.auto import tqdm
from openai import OpenAI

# =========================
# Config
# =========================
MODEL_NAME_ = "Qwen/Qwen3-8B"
DATA_NAME = "novelty"
TEMPERATURE = 0.6
SAMPLE_NUM = 20
FLAG = 0
SAMPLE_K = 3
EVAL_MODEL = "gpt-4o-mini"

LANGUAGES = [
    'en', 'it', 'ms', 'zh', 'ru', 'de', 'iw',
    'bg', 'da', 'no', 'sv', 'es', 'tl', 'oc', 'fr'
]
os.environ["OPENAI_API_KEY"] = "xxx"

MODEL_NAME = MODEL_NAME_.split("/")[-1]
EVAL_DIR = f"../{DATA_NAME}_output_english_answer"
SAVE_PATH = f"gpt_eval_single_{MODEL_NAME}_{DATA_NAME}.jsonl"
BATCH_INPUT_PATH = f"batch_eval_input_{MODEL_NAME}_{DATA_NAME}.jsonl"

# =========================
# GPT Evaluation Instruction
# =========================
INSTRUCTION = """
You are an evaluator assessing the quality of a single response to a task instruction.

You will be given:
(1) A task instruction
(2) A response

Evaluate the response along the following two dimensions:

1. Instruction Adherence (0–50)
To what extent does the response follow the task instruction?
Note that if the response explicitly refuses to perform the task, this should NOT be penalized.
You only need to judge the degree to which the response is relevant to the task instruction.

2. Response Quality (0–50)
Assess the overall quality of the response in terms of clarity, fluency, and grammatical correctness.

Scoring:
- Each dimension should be scored from 0 to 50 (integer only).
- Total Score = sum of the two dimensions (0–100).

Output format (strict JSON only):
{
  "Instruction Adherence": <score>,
  "Response Quality": <score>,
  "Total Score": <score>
}
"""

# =========================
# Utils
# =========================
def select_random_k(candidates, k=3, seed=None):
    if not candidates:
        return []
    if seed is not None:
        random.seed(seed)
    if len(candidates) <= k:
        return [c.strip() for c in candidates]
    idxs = random.sample(range(len(candidates)), k)
    return [candidates[i].strip() for i in idxs]


def build_prompt(instruction, unit):
    return (
        instruction.strip()
        + "\n\n- Instruction:\n"
        + unit["question"]
        + "\n\n- Response:\n"
        + unit["response"]
    )


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def unit_uid(unit):
    return f"{unit['idx']}_{unit['language']}_{unit['sample_id']}"

# =========================
# Main
# =========================
def main():
    client = OpenAI()

    # -------- Load multilingual outputs --------
    all_data = {}
    for lang in LANGUAGES:
        path = f"{EVAL_DIR}/{MODEL_NAME}_{lang}_{TEMPERATURE}_{SAMPLE_NUM}_{FLAG}_dp_rank_0.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            all_data[lang] = [json.loads(l) for l in f]

    # -------- Build atomic eval units --------
    eval_units = []
    for lang, data in all_data.items():
        for idx, item in enumerate(data):
            if "final_answer" not in item:
                continue

            question = item["question"]
            responses = select_random_k(item["final_answer"], k=SAMPLE_K)

            for sid, resp in enumerate(responses):
                eval_units.append({
                    "idx": idx,
                    "language": lang,
                    "sample_id": sid,
                    "question": question,
                    "response": resp
                })

    print(f"🧮 Total evaluation units: {len(eval_units)}")

    # -------- Resume logic --------
    completed = set()
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(unit_uid(r))
                except:
                    pass
        print(f"🔁 Loaded {len(completed)} completed units")

    remaining = [u for u in eval_units if unit_uid(u) not in completed]
    print(f"⏳ Remaining units: {len(remaining)}")

    if not remaining:
        print("✅ Nothing to evaluate.")
        return

    # =========================
    # Build Batch Input JSONL
    # =========================
    with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
        for unit in remaining:
            prompt = build_prompt(INSTRUCTION, unit)
            f.write(json.dumps({
                "custom_id": unit_uid(unit),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": EVAL_MODEL,
                    "input": prompt,
                    "temperature": 0.0
                }
            }, ensure_ascii=False) + "\n")

    print(f"📦 Batch input written to {BATCH_INPUT_PATH}")

    # =========================
    # Submit Batch
    # =========================
    batch_file = client.files.create(
        file=open(BATCH_INPUT_PATH, "rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"job": "gpt_eval_single"}
    )

    print(f"🚀 Batch submitted: {batch.id}")
    batch_id = batch.id
    # =========================
    # Poll Batch Status
    # =========================
    print("⏳ Polling batch status...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"🕒 Batch status: {status}")

        if status == "completed":
            print("✅ Batch completed.")
            break

        if status in ["failed", "expired", "cancelled"]:
            raise RuntimeError(f"❌ Batch ended with status: {status}")

        time.sleep(10)
    # =========================
    # Retrieve Results
    # =========================
    output = client.files.content(batch.output_file_id)
    lines = output.text.splitlines()

    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    with open(SAVE_PATH, "a", encoding="utf-8") as fout:
        for line in lines:
            obj = json.loads(line)

            if obj.get("error"):
                print(f"❌ Error in {obj['custom_id']}: {obj['error']}")
                continue

            unit_id = obj["custom_id"]
            scores = parse_json(obj["response"]["body"]["output"][0]["content"][0]["text"])
            idx, lang, sid = unit_id.split("_", 2)

            fout.write(json.dumps({
                "idx": int(idx),
                "language": lang,
                "sample_id": int(sid),
                "scores": scores
            }, ensure_ascii=False) + "\n")

    print(f"✅ Evaluation saved to {SAVE_PATH}")


# =========================
if __name__ == "__main__":
    main()