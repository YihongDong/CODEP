# coding=utf-8
from __future__ import print_function
from lib2to3.pgen2 import grammar
from ntpath import join

import time
import os
import sys
import torch
import numpy as np
import random
import six
import six.moves.cPickle as pickle
from six.moves import input
from six.moves import xrange as range
from torch.autograd import Variable


import components.evaluation as evaluation
from components.registerable import Registrable
from components.utils import update_args, init_arg_parser
from components.nn_utils import GloveHelper
from components.dataset import Dataset
import components.nn_utils as nn_utils
import components.model
import components.transformer

from pyparser import load_grammar
import wandb
import warnings
warnings.filterwarnings("ignore")
# from postnet import PostNet
os.environ["WANDB_MODE"] = 'disabled'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.chdir(sys.path[0])

#assert astor.__version__ == "0.7.1"
if six.PY3:
    # import additional packages for wikisql dataset (works only under Python 3)
    pass


def init_config():
    args = arg_parser.parse_args()
    # seed the RNG
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(int(args.seed * 13 / 7))
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    return args


def train(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])

    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = load_grammar(version=args.grammar_version)

    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    if args.pretrain:
        print('Finetune with: ', args.pretrain, file=sys.stderr)
        model = parser_cls.load(model_path=args.pretrain, cuda=args.cuda)
    else:
        model = parser_cls(args, vocab, grammar)

    model.train()
    evaluator = Registrable.by_name(args.evaluator)(grammar, args=args)
    if args.cuda: model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    if not args.pretrain:
        if args.uniform_init:
            print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
        elif args.glorot_init:
            print('use glorot initialization', file=sys.stderr)
            nn_utils.glorot_init(model.parameters())

        # load pre-trained word embedding (optional)
        if args.glove_embed_path:
            print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
            glove_embedding = GloveHelper(args.glove_embed_path)
            glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    wandb.watch(model)
    while True:
        epoch += 1
        epoch_begin = time.time()
            
        wandb.log({"{}/epoch".format(args.mode):epoch})

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_code) <= args.decode_max_time_step]
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.score(batch_examples)
            loss = -ret_val[0]
            # # if args.use_nonterminal and args.parser == 'transformer_parser':
            # #     loss = -torch.mean(ret_val[0]) - torch.mean(ret_val[1])

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)
            # report_loss += loss.data.item()
            # report_examples += 1

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data[0]
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            wandb.log({"{}/train_iter".format(args.mode):train_iter,"{}/loss".format(args.mode):loss.item()})

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        is_better = False
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0 and epoch >= args.vaildate_begin_epoch:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=False, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                wandb.log({"dev/{}".format(k):v for k,v in eval_results.items()})

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
        else:
            is_better = True

        if epoch < args.warm_up_step*args.warm_up_change_every:
            # warm up lr
            lr_scale = 0.5 ** int(args.warm_up_step-epoch/args.warm_up_change_every)
            lr = optimizer.param_groups[0]['lr'] * lr_scale
            print('decay learning rate to %f' % lr, file=sys.stderr)

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience and epoch >= args.lr_decay_after_epoch and epoch >= args.vaildate_begin_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            test_wandb(args)
            exit(0)

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch and epoch >= args.vaildate_begin_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                test_wandb(args)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    grammar = load_grammar(version=saved_args.grammar_version)
    saved_args.cuda = args.cuda
    print(saved_args, file=sys.stderr)
    # set the correct domain from saved arg

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(grammar, args=saved_args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    
    # wandb.log({"test/{}".format(k):v for k,v in eval_results.items()})

    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))

def test_wandb(args):
    test_set = Dataset.from_bin_file(args.test_file)
    args.load_model=args.save_to + '.bin'
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    grammar = load_grammar(version=args.grammar_version)
    # set the correct domain from saved arg

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(grammar, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    

    wandb.log({"test/{}".format(k):v for k,v in eval_results.items()})

    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))



if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    if args.mode == 'train':
        print(args, file=sys.stderr)
        wandb.init(config=args,project='parse-'+args.vocab.strip().split('/')[1],name="{}_lr{}bs{}d{}a{}{}{}{}".format(args.parser[:-7],
           args.lr,args.batch_size,args.dropout,args.alpha,'_un' if args.use_nonterminal else '','_pwn'if args.predict_with_nonterminal else '','_fd'if args.feed_nonterminal else ''))
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise RuntimeError('unknown mode')
