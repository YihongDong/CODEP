#!/bin/bash
set -e

seed=0 #${1:-0}
freq=2
alpha=${4:-1} 
vocab="data/conala/vocab.src_freq${freq}.code_freq${freq}.bin"
train_file="data/conala/train.bin"
dev_file="data/conala/dev.bin"
test_file="data/conala/test.bin"
dropout=${3:-0.5}
hidden_size=512 #256
embed_size=256 #128
action_embed_size=256 #128
lr=${1:-0.001}
lr_decay=0.5
batch_size=${2:-32}
max_epoch=200 #80
beam_size=15
NAME_TOKEN_NUM=${beam_size}
lstm='lstm'  # lstm
lr_decay_after_epoch=60 #40
vaildate_begin_epoch=20
valid_every_epoch=1
patience=5
model_name=conala.parse.a${alpha}.d${dropout}.bs${batch_size}.lr${lr}.hs${hidden_size}.es${embed_size}.aes${action_embed_size}.NAME_TOKEN_NUM${NAME_TOKEN_NUM}.me${max_epoch}.un.pwn.fd
# model_name=conala.parse.lr${lr}.hs${hidden_size}.es${embed_size}.aes${action_embed_size}.NAME_TOKEN_NUM${NAME_TOKEN_NUM}.un.pwn.nodfa
#    --use_nonterminal\
#    --predict_with_nonterminal\
#    --feed_nonterminal\
#    --parser lstm_nopda_parser\

echo "**** Writing results to logs/conala/${model_name}.log ****"
mkdir -p logs/conala
echo commit hash: `git rev-parse HEAD` > logs/conala/${model_name}.log

python -u train.py \
    --cuda \
    --alpha ${alpha}\
    --NAME_TOKEN_NUM ${NAME_TOKEN_NUM}\
    --parser lstm_parser\
    --predict_with_nonterminal\
    --feed_nonterminal\
    --use_nonterminal\
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --evaluator default_evaluator\
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --test_file ${test_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
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
    --save_decode_to decodes/conala/$model_name.test.decode \
    --decode_max_time_step 50\
    --save_to saved_models/conala/${model_name} 2>&1 | tee logs/conala/${model_name}.log
