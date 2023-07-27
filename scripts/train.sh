#!/bin/bash
set -e

seed=0 #${1:-0}
freq=2
alpha=${4:-1} 
vocab="data/juice/vocab.src_freq${freq}.code_freq${freq}.bin"
train_file="data/juice/train.bin"
dev_file="data/juice/dev.bin"
test_file="data/juice/test.bin"
dropout=${3:-0.3}
hidden_size=512 #256
embed_size=256 #128
action_embed_size=256 #128
lr=${1:-0.001}
lr_decay=0.5
batch_size=${2:-64}
max_epoch=200 #80
beam_size=5
NAME_TOKEN_NUM=${beam_size}
lstm='lstm'  # lstm
lr_decay_after_epoch=200 #40
vaildate_begin_epoch=50
valid_every_epoch=50
patience=5
model_name=juice.parse.a${alpha}.d${dropout}.bs${batch_size}.lr${lr}.hs${hidden_size}.es${embed_size}.aes${action_embed_size}.NAME_TOKEN_NUM${NAME_TOKEN_NUM}.me${max_epoch}.un.pwm.fd
# model_name=juice.parse.lr${lr}.hs${hidden_size}.es${embed_size}.aes${action_embed_size}.NAME_TOKEN_NUM${NAME_TOKEN_NUM}.un.pwn.nodfa
#    --use_nonterminal\
#    --predict_with_nonterminal\
#    --parser lstm_nopda_parser\

echo "**** Writing results to logs/juice/${model_name}.log ****"
mkdir -p logs/juice
echo commit hash: `git rev-parse HEAD` > logs/juice/${model_name}.log

python -u train.py \
    --cuda \
    --alpha ${alpha}\
    --NAME_TOKEN_NUM ${NAME_TOKEN_NUM}\
    --use_nonterminal\
    --feed_nonterminal\
    --predict_with_nonterminal\
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
    --save_decode_to decodes/juice/$model_name.test.decode \
    --decode_max_time_step 125\
    --save_to saved_models/juice/${model_name} 2>&1 | tee logs/juice/${model_name}.log
