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

CUDA_VISIBLE_DEVICES=0 python evaluate_similarity.py \
    --output-dir evaluation_results \
    --eval-dir novelty_output_english_answer \
    --multilingual \
    --temperature 0.6 \
    --sample_num 20 \
    --flag 0 \
    --model Qwen3-8B