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