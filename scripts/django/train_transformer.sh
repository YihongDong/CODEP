#!/bin/bash
set -e

seed=0 #${1:-0}
alpha=${4:-1} 
vocab="data/django/vocab.freq15.bin"
train_file="data/django/train.bin"
dev_file="data/django/dev.bin"
test_file="data/django/test.bin"
dropout=${3:-0.3}
hidden_size=512 #256
embed_size=256 #128
action_embed_size=256 #128
lr=${1:-0.0001}
lr_decay=0.99 #0.5
batch_size=${2:-32}
max_epoch=600 #80
beam_size=15
NAME_TOKEN_NUM=${beam_size}
clip_grad=5
lr_decay_after_epoch=600 #40
vaildate_begin_epoch=50
valid_every_epoch=10
patience=5
warm_up_step=10
warm_up_change_every=2
model_name=django.transformer.parse.a${alpha}.d${dropout}.bs${batch_size}.lr${lr}.hs${hidden_size}.es${embed_size}.aes${action_embed_size}.NAME_TOKEN_NUM${NAME_TOKEN_NUM}.me${max_epoch}.un.pwn.fd
# model_name=django.parse.lr${lr}.hs${hidden_size}.es${embed_size}.aes${action_embed_size}.NAME_TOKEN_NUM${NAME_TOKEN_NUM}.un.pwn.nopda
#    --use_nonterminal\
#    --predict_with_nonterminal\
#    --feed_nonterminal\

echo "**** Writing results to logs/django/${model_name}.log ****"
mkdir -p logs/django
echo commit hash: `git rev-parse HEAD` > logs/django/${model_name}.log

python -u train.py \
    --cuda \
    --alpha ${alpha}\
    --NAME_TOKEN_NUM ${NAME_TOKEN_NUM}\
    --parser transformer_parser\
    --predict_with_nonterminal\
    --feed_nonterminal\
    --use_nonterminal\
    --grammar_version 2.7\
    --warm_up_step ${warm_up_step}\
    --warm_up_change_every ${warm_up_change_every}\
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --evaluator django_evaluator\
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --vocab ${vocab} \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --vaildate_begin_epoch ${vaildate_begin_epoch}\
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --valid_every_epoch ${valid_every_epoch}\
    --save_decode_to decodes/django/$model_name.test.decode \
    --decode_max_time_step 60\
    --save_to saved_models/django/${model_name} 2>&1 | tee logs/django/${model_name}.log
