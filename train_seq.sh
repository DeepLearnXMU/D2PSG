#!/usr/bin/env bash

DATA_DIR='saved_data/data_seq'
OUTPUT_DIR='saved_data'

cd transformers
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google/t5-v1_1-large \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/t5-v1_1-large-seq" \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --gradient_accumulation_steps 128 \
    --predict_with_generate \
    --num_train_epochs 15 \
    --text_column="context" \
    --summary_column="relation" \
    --save_steps=500 \
    --max_target_length 512 \
    --overwrite_output_dir
