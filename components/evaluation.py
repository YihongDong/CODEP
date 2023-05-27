# coding=utf-8
from __future__ import print_function
import imp

import sys
import traceback
from tqdm import tqdm
from . import evaluator
from parso.parser import Stack, StackNode
from pyparser import PyParser
import torch
import re
from .evaluator import decanonicalize_split_code

def decode(examples, model, evaluator, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        with torch.no_grad():
            hyps = model.parse(example, context=None, beam_size=args.beam_size)
        # print('Intent: %s\nTarget Code:\n%s\n' % (' '.join(example.src_sent),' '.join([t[0] for t in example.tgt_code])))
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                got_code = True
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                             ' '.join(example.src_sent),
                                                                                             ' '.join([t[0] for t in example.tgt_code]),
                                                                                             hyp_id,
                                                                                             ' '.join([t[0] for t in hyp.token])), file=sys.stdout)
                    if got_code:
                        print()
                        print(' '.join([t[0] for t in hyp.token]))
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        if len(decoded_hyps) == 0:
            first_dfa = evaluator.grammar._pgen_grammar.nonterminal_to_dfas[evaluator.grammar._start_nonterminal][0]
            zero_hyp = PyParser(evaluator.grammar._pgen_grammar, error_recovery=True, start_nonterminal=evaluator.grammar._start_nonterminal)
            zero_hyp.stack = Stack([StackNode(first_dfa)])
            zero_hyp.token = [['','ENDMARKER']]
            decoded_hyps.append(zero_hyp)

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    decode_results = decode(examples, parser, evaluator, args, verbose=verbose)

    eval_result = []
    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only, args=args)

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
