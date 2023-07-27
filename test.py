# coding=utf-8
from __future__ import print_function
from lib2to3.pgen2 import grammar
from ntpath import join

import time
import os
import sys
import torch
import numpy as np
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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.chdir(sys.path[0])

import pickle

file_path = "saved_models/juice/juice.transformer.parse.a0.4.d0.1.bs64.lr0.0001.hs1024.es256.aes256.NAME_TOKEN_NUM5.me600.un.pwm.fd.bin"
test_set = Dataset.from_bin_file('data/juice/test.bin')

params = torch.load(file_path, map_location=lambda storage, loc: storage)
saved_args = params['args']
grammar = load_grammar(version=saved_args.grammar_version)

# saved_args.parser = 'lstm_nopda_parser'
# saved_args.predict_with_nonterminal = False
parser_cls = Registrable.by_name(saved_args.parser)
parser = parser_cls.load(model_path=file_path, cuda=saved_args.cuda)
parser.eval()
evaluator = Registrable.by_name(saved_args.evaluator)(grammar, args=saved_args)
eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, saved_args,
                                                    verbose=saved_args.verbose, return_decode_result=True)

print(eval_results, file=sys.stderr)
pickle.dump(decode_results, open(saved_args.save_decode_to, 'wb'))

