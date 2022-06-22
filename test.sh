#!/usr/bin/env bash

DATA_DIR='saved_data/data'
MODEL_DIR='saved_data/t5-v1_1-large'

cd transformers
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path $MODEL_DIR \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/valid.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir "$MODEL_DIR" \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --predict_with_generate \
    --learning_rate 5e-5 \
    --num_train_epochs 15 \
    --text_column="context" \
    --summary_column="relation" \
    --save_steps=500 \
    --max_target_length 512
