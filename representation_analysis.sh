CUDA_VISIBLE_DEVICES=0 python representation_analysis.py \
    --eval-dir novelty_output_english_answer \
    --output-dir output_representations \
    --temperature 0.6 \
    --sample_num 20 \
    --flag 0 \
    --model Qwen/Qwen3-8B