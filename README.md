# Multilingual-LoT-Diversity

The official code of "Language of Thought Shapes Output Diversity in Large Language Model".

## Get up

Python 3.12.0, Python packages are printed in `pip_packages.txt`.

All the data used in experiments, like noveltybench, blend or wvs with Google Translations, are in `data` folder.

## Preliminary Study

The script analyzes multilingual thinking representations, returning the Figure 1 in the paper.

```bash
CUDA_VISIBLE_DEVICES=0 python representation_analysis.py \
    --eval-dir novelty_output_english_answer \
    --output-dir output_representations \
    --temperature 0.6 \
    --sample_num 20 \
    --flag 0 \
    --model Qwen/Qwen3-8B
```

## Multilingual Repeated Sampling

For each language, the script first conducts Thinking Language Control to generate samples in `{data_name}_output` folder. Then, it regenerate English final outputs with Output Language Control, generating samples in `{data_name}_english_answer` folder. `{data_name}` can be *novelty* or *infinity*. Generated samples from *noveltybench* are provided.

```bash
languages=("en" "it" "ms" "zh" "ru" "de" "iw" "bg" "da" "no" "sv" "es" "tl" "oc" "fr")

for lang in "${languages[@]}"; do

    VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python multilingual_repeated_sampling.py \
        --model Qwen/Qwen3-8B \
        --data_name novelty \
        --dp-size 1 \
        --output_path novelty_output \
        --language "$lang" \
        --sample_num 20 \
        --temperature 0.6 \
        --flag 0
done
```

## Output Diversity Evaluation

The scripts evaluate diversity on English output generated from the last step. *Distinct Score* and *Similarity Score* are computed for *Single-Language Sampling* and *Mixed-Language Sampling* strategies. Evaluation results on Qwen3-8B are provided, you can just run `show_distinct.py` and `show_similarity.py` to output these results.

```bash
# Distinct Score of Single-Language Sampling
languages=("en" "it" "ms" "zh" "ru" "de" "iw" "bg" "da" "no" "sv" "es" "tl" "oc" "fr")
for lang in "${languages[@]}"; do
    echo "Evaluating language: $lang at temperature: 0.6"
    CUDA_VISIBLE_DEVICES=0 python evaluate_distinct.py \
        --output-dir evaluation_results \
        --eval-dir novelty_output_english_answer \
        --language "$lang" \
        --temperature 0.6 \
        --sample_num 20 \
        --flag 0 \
        --model Qwen3-8B
done
# Distinct Score of Mixed-Language Sampling
CUDA_VISIBLE_DEVICES=0 python evaluate_distinct.py \
    --output-dir evaluation_results \
    --eval-dir novelty_output_english_answer \
    --multilingual \
    --temperature 0.6 \
    --sample_num 20 \
    --flag 0 \
    --model Qwen3-8B

# Similarity Score of Single-Language Sampling
languages=("en" "it" "ms" "zh" "ru" "de" "iw" "bg" "da" "no" "sv" "es" "tl" "oc" "fr")
for lang in "${languages[@]}"; do
    echo "Evaluating language: $lang at temperature: 0.6"
    CUDA_VISIBLE_DEVICES=0 python evaluate_similarity.py \
        --output-dir evaluation_results \
        --eval-dir novelty_output_english_answer \
        --language "$lang" \
        --temperature 0.6 \
        --sample_num 20 \
        --flag 0 \
        --model Qwen3-8B
done
# Similarity Score of Mixed-Language Sampling
CUDA_VISIBLE_DEVICES=0 python evaluate_similarity.py \
    --output-dir evaluation_results \
    --eval-dir novelty_output_english_answer \
    --multilingual \
    --temperature 0.6 \
    --sample_num 20 \
    --flag 0 \
    --model Qwen3-8B
```

## Cultural Pluralism

The codes for cultural pluralism experiments (Section 6 in paper) are in `cultural_pluralism_exp` folder. 
- multilingual_repeated_sampling_culture.py
- evaluate_blend.py: summarize results of English Sampling, High Temperature and Request Diversity baselines on blend.
- evaluate_blend_multilingual.py: summarize results of Multilingual Prompting baseline and Mixed-Language Sampling on blend.
- evaluate_wvs.py: same, but on wvs.
- evaluate_wvs_multilingual.py: same, but on wvs.


# Others

Scripts for evaluating quality on generated outputs are in `output_quality_evaluation` folder.
