#!/bin/bash

test_file="data/juice/test.bin"
model_name="juice.parse.d0.3.bs64.lr0.001.hs512.es256.aes256.NAME_TOKEN_NUM5.me600.nodfa"

python train.py \
    --cuda \
    --mode test \
    --load_model saved_models/juice/${model_name}.bin \
    --beam_size 5 \
    --test_file ${test_file} \
    --evaluator default_evaluator \
    --save_decode_to decodes/juice/${model_name}.test.decode \
    --decode_max_time_step 100
