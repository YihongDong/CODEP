#!/bin/bash

test_file="data/conala/test.bin"
model_name="conala.parse.lr0.001.hs512.es256.aes256.NAME_TOKEN_NUM15"

python train.py \
    --cuda \
    --mode test \
    --load_model saved_models/conala/${model_name}.bin \
    --beam_size 15 \
    --test_file ${test_file} \
    --evaluator default_evaluator \
    --save_decode_to decodes/conala/${model_name}.test.decode \
    --decode_max_time_step 100
