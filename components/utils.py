# coding=utf-8
import argparse
import numpy as np

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Run mode')

    #### Modularized configuration ####
    arg_parser.add_argument('--parser', type=str, default='lstm_parser', required=False, help='name of parser class to load')
    arg_parser.add_argument('--evaluator', type=str, default='default_evaluator', required=False, help='name of evaluator class to use')

    #### Model configuration ####
    arg_parser.add_argument('--lstm', choices=['lstm'], default='lstm', help='Type of LSTM used, currently only standard LSTM cell is supported')

    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--state_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')

    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')
    arg_parser.add_argument('--ptrnet_hidden_dim', default=32, type=int, help='Hidden dimension used in pointer network')
    arg_parser.add_argument('--att_vec_size', default=256, type=int, help='size of attentional vector')

    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true',
                            help='Do not use additional linear layer to transform the attentional vector for computing action probabilities')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'],
                            help='Type of activation if using additional linear layer')
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true',
                            help='Use different linear mapping ')

    # supervised attention
    arg_parser.add_argument('--sup_attention', default=False, action='store_true', help='Use supervised attention')

    arg_parser.add_argument('--no_input_feed', default=False, action='store_true', help='Do not use input feeding in decoder LSTM')
    arg_parser.add_argument('--no_copy', default=False, action='store_true', help='Do not use copy mechanism')

    arg_parser.add_argument('--use_nonterminal', default=False, action='store_true', help='Do not use nonterminal')
    arg_parser.add_argument('--feed_nonterminal', default=False, action='store_true', help='Do not feed nonterminal')
    arg_parser.add_argument('--predict_with_nonterminal', default=False, action='store_true', help='predict with nonterminal')

    # Model configuration parameters specific for wikisql
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine', help='How to perform attention over table columns')
    arg_parser.add_argument('--answer_prune', dest='answer_prune', action='store_true', help='Whether to use answer pruning [default: True]')
    arg_parser.set_defaults(answer_prune=True)
    arg_parser.add_argument('--no_answer_prune', dest='answer_prune', action='store_false', help='Do not use answer prunning')

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--glove_embed_path', default=None, type=str, help='Path to pretrained Glove mebedding')

    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')
    arg_parser.add_argument('--pretrain', type=str, help='path to the pretrained model file')

    arg_parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    arg_parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    arg_parser.add_argument('--word_dropout', default=0., type=float, help='Word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='Word dropout rate on decoder')
    arg_parser.add_argument('--code_token_label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing when predicting primitive tokens')
    arg_parser.add_argument('--src_token_label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing in reconstruction model when predicting source tokens')

    arg_parser.add_argument('--negative_sample_type', default='best', type=str, choices=['best', 'sample', 'all'])

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int, help='Perform validation every x epoch')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')

    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true', help='Save all intermediate checkpoints')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='If specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true', help='Use glorot initialization')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--decay_lr_every_epoch', action='store_true', default=False, help='force to decay learning rate after each epoch')
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False, help='Whether to reset optimizer when loading the best checkpoint')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=False,
                            help='Only evaluate the top prediction in validation')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    arg_parser.add_argument('--decode_max_time_step', default=100, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--sample_size', default=5, type=int, help='Sample size')
    arg_parser.add_argument('--test_file', type=str, help='Path to the test file')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')

    #### reranking ####
    arg_parser.add_argument('--features', nargs='+')
    arg_parser.add_argument('--load_reconstruction_model', type=str, help='Load reconstruction model')
    arg_parser.add_argument('--load_paraphrase_model', type=str, help='Load paraphrase model')
    arg_parser.add_argument('--load_reranker', type=str, help='Load reranking model')
    arg_parser.add_argument('--tie_embed', action='store_true', help='tie source and target embedding in training paraphrasing model')
    arg_parser.add_argument('--train_decode_file', default=None, type=str, help='Decoding results on training set')
    arg_parser.add_argument('--test_decode_file', default=None, type=str, help='Decoding results on test set')
    arg_parser.add_argument('--dev_decode_file', default=None, type=str, help='Decoding results on dev set')
    arg_parser.add_argument('--metric', default='accuracy', choices=['bleu', 'accuracy'])
    arg_parser.add_argument('--num_workers', default=1, type=int, help='number of multiprocess workers')

    #### self-training ####
    arg_parser.add_argument('--load_decode_results', default=None, type=str)
    arg_parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')
    arg_parser.add_argument('--unlabeled_file', type=str, help='Path to the training source file used in semi-supervised self-training')

    #### interactive mode ####
    arg_parser.add_argument('--example_preprocessor', default=None, type=str, help='name of the class that is used to pre-process raw input examples')

    #### dataset specific config ####
    arg_parser.add_argument('--sql_db_file', default=None, type=str, help='path to WikiSQL database file for evaluation (SQLite)')

    arg_parser.add_argument('--grammar_version', default='3.7', type=str, help='python version')
    arg_parser.add_argument('--NAME_TOKEN_NUM', default=15, type=int, help='NAME_TOKEN_NUM')
    arg_parser.add_argument('--vaildate_begin_epoch', default=10, type=int, help='vaildate_begin_epoch')

    #### transformer ####
    arg_parser.add_argument('--encoder_layers', default='2', type=int, help='encoder_layers')
    arg_parser.add_argument('--decoder_layers', default='2', type=int, help='decoder_layers')
    arg_parser.add_argument('--attn_heads', default='4', type=int, help='attn_heads')

    #### warmup ####
    arg_parser.add_argument('--warm_up_step', default='0', type=int, help='warm_up_step')
    arg_parser.add_argument('--warm_up_change_every', default='1', type=int, help='warm_up_change_every')

    ### hyperparameter ###
    arg_parser.add_argument('--alpha', default='1', type=float, help='alpha')

    
    return arg_parser


def update_args(args, arg_parser):
    for action in arg_parser._actions:
        if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreTrueAction) \
                or isinstance(action, argparse._StoreFalseAction):
            if not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)
