import csv

from components.dataset import Dataset
from data.conala.gendata import is_enumerable_str, replace_identifiers_in_ast
from components.bleu_score import compute_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from components.registerable import Registrable
import sys, traceback
import numpy as np
import ast
import astor
import re

def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens

def decanonicalize_split_code(code, slot_map):
    for slot_name, slot_val in slot_map.items():
        for i,token in enumerate(code):
            code[i] = code[i].replace(slot_name, slot_val)
    return code

@Registrable.register('default_evaluator')
class Evaluator(object):
    def __init__(self, grammar=None, args=None):
        self.grammar = grammar
        self.default_metric = 'corpus_bleu'

    def is_hyp_correct(self, example, hyp):
        ref_code_tokens = decanonicalize_split_code([t[0] for t in example.tgt_code], example.meta['slot_map'])
        hyp_code_tokens = decanonicalize_split_code([t[0] for t in hyp.token], example.meta['slot_map'])

        return ref_code_tokens == hyp_code_tokens

    def get_sentence_bleu(self, example, hyp):
        return sentence_bleu(tokenize_for_bleu_eval(' '.join(decanonicalize_split_code([t[0] for t in example.tgt_code], example.meta['slot_map']))),
                             tokenize_for_bleu_eval(' '.join(decanonicalize_split_code([t[0] for t in hyp.token], example.meta['slot_map']))),
                             smoothing_function=SmoothingFunction().method3)

    def evaluate_dataset(self, dataset, decode_results, fast_mode=False, args=None):
        output_plaintext_file = None
        if args and args.save_decode_to:
            output_plaintext_file = open(args.save_decode_to + '.txt', 'w', encoding='utf-8')
        examples = dataset.examples if isinstance(dataset, Dataset) else dataset
        assert len(examples) == len(decode_results)

        # speed up, cache tokenization results
        if not hasattr(examples[0], 'reference_code_tokens'):
            for example in examples:
                setattr(example, 'reference_code_tokens', tokenize_for_bleu_eval(' '.join(decanonicalize_split_code([t[0] for t in example.tgt_code], example.meta['slot_map']))))

        if not hasattr(decode_results[0][0], 'decanonical_code_tokens'):
            for i, example in enumerate(examples):
                hyp_list = decode_results[i]
                # here we prune any hypothesis that throws an error when converting back to the decanonical code!
                # This modifies the decode_results in-place!
                filtered_hyp_list = []
                for hyp in hyp_list:
                    if not hasattr(hyp, 'decanonical_code'):
                        try:
                            hyp.decanonical_code = decanonicalize_split_code([t[0] for t in hyp.token], example.meta['slot_map'])
                            if hyp.decanonical_code:
                                hyp.decanonical_code_tokens = tokenize_for_bleu_eval(' '.join(hyp.decanonical_code))
                                filtered_hyp_list.append(hyp)
                        except: 
                            pass 

                decode_results[i] = filtered_hyp_list

        if fast_mode:
            references = [e.reference_code_tokens for e in examples]
            hypotheses = [hyp_list[0].decanonical_code_tokens if hyp_list else [] for hyp_list in decode_results]

            bleu_tup = compute_bleu([[x] for x in references], hypotheses, smooth=False)
            bleu = bleu_tup[0]

            return bleu
        else:
            tokenized_ref_snippets = []
            hyp_code_tokens = []
            best_hyp_code_tokens = []
            sm_func = SmoothingFunction().method3
            sent_bleu_scores = []
            oracle_bleu_scores = []
            oracle_exact_match = []
            for example, hyp_list in zip(examples, decode_results):
                tokenized_ref_snippets.append(example.reference_code_tokens)
                example_hyp_bleu_scores = []
                if hyp_list:
                    for i, hyp in enumerate(hyp_list):
                        hyp.bleu_score = sentence_bleu([example.reference_code_tokens],
                                                       hyp.decanonical_code_tokens,
                                                       smoothing_function=sm_func)
                        hyp.is_correct = self.is_hyp_correct(example, hyp)

                        example_hyp_bleu_scores.append(hyp.bleu_score)

                    top_decanonical_code_tokens = hyp_list[0].decanonical_code_tokens
                    sent_bleu_score = hyp_list[0].bleu_score
                    best_hyp_idx = np.argmax(example_hyp_bleu_scores)
                    oracle_sent_bleu = example_hyp_bleu_scores[best_hyp_idx]
                    _best_hyp_code_tokens = hyp_list[best_hyp_idx].decanonical_code_tokens
                else:
                    top_decanonical_code_tokens = []
                    sent_bleu_score = 0.
                    oracle_sent_bleu = 0.
                    _best_hyp_code_tokens = []
                
                # write results to file
                if output_plaintext_file:
                    output_plaintext_file.write(" ".join(top_decanonical_code_tokens) + '\n')
                oracle_exact_match.append(any(hyp.is_correct for hyp in hyp_list))
                hyp_code_tokens.append(top_decanonical_code_tokens)
                sent_bleu_scores.append(sent_bleu_score)
                oracle_bleu_scores.append(oracle_sent_bleu)
                best_hyp_code_tokens.append(_best_hyp_code_tokens)

            bleu_tup = compute_bleu([[x] for x in tokenized_ref_snippets], hyp_code_tokens, smooth=False)
            corpus_bleu = bleu_tup[0]
            codebleu = compute_codebleu([[' '.join(x) for x in tokenized_ref_snippets]], [' '.join(x) for x in hyp_code_tokens])

            bleu_tup = compute_bleu([[x] for x in tokenized_ref_snippets], best_hyp_code_tokens, smooth=False)
            oracle_corpus_bleu = bleu_tup[0]

            avg_sent_bleu = np.average(sent_bleu_scores)
            oracle_avg_sent_bleu = np.average(oracle_bleu_scores)
            exact = sum([1 if h == r else 0 for h, r in zip(hyp_code_tokens, tokenized_ref_snippets)]) / float(
                len(examples))
            oracle_exact_match = np.average(oracle_exact_match)

            return {'corpus_bleu': corpus_bleu,
                    'oracle_corpus_bleu': oracle_corpus_bleu,
                    'avg_sent_bleu': avg_sent_bleu,
                    'oracle_avg_sent_bleu': oracle_avg_sent_bleu,
                    'exact_match': exact,
                    'oracle_exact_match': oracle_exact_match,
                    'codebleu':codebleu}


import argparse
import CodeBLEU.bleu as bleu
import CodeBLEU.weighted_ngram_match as weighted_ngram_match
import CodeBLEU.syntax_match as syntax_match
import CodeBLEU.dataflow_match as dataflow_match

parser = argparse.ArgumentParser()

lang = 'python'
alpha,beta,gamma,theta = 0.25,0.25,0.25,0.25

def compute_codebleu(reference_corpus, translation_corpus):
    # preprocess inputs
    pre_references = [[x.strip() for x in ref] for ref in reference_corpus]
    hypothesis = [x.strip() for x in translation_corpus]

    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('CodeBLEU/keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
                        format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

    return code_bleu_score

@Registrable.register('django_evaluator')
class DjangoEvaluator(object):
    def __init__(self, grammar=None, args=None):
        self.grammar = grammar
        self.default_metric = 'accuracy'

    def is_hyp_correct(self, example, hyp):
        ref_code_tokens = [t[1] for t in example.tgt_code]
        hyp_code_tokens = [t[1] for t in hyp.token]

        return ref_code_tokens == hyp_code_tokens

    def evaluate_dataset(self, examples, decode_results, fast_mode=False, args=None):
        correct_array = []
        oracle_array = []
        for example, hyp_list in zip(examples, decode_results):
            if fast_mode:
                hyp_list = hyp_list[:1]

            if hyp_list:
                for hyp_id, hyp in enumerate(hyp_list):
                    try:
                        is_correct = self.is_hyp_correct(example, hyp)
                    except:
                        is_correct = False

                        print('-' * 60, file=sys.stdout)
                        print('Error in evaluating Example %s, hyp %d {{ %s }}' % (example.idx, hyp_id, hyp.tgt_code),
                              file=sys.stdout)

                        print('example id: %s, hypothesis id: %d' % (example.idx, hyp_id), file=sys.stdout)
                        traceback.print_exc(file=sys.stdout)
                        print('-' * 60, file=sys.stdout)

                    hyp.is_correct = is_correct

                correct_array.append(hyp_list[0].is_correct)
                oracle_array.append(any(hyp.is_correct for hyp in hyp_list))
            else:
                correct_array.append(False)
                oracle_array.append(False)

        acc = np.average(correct_array)

        oracle_acc = np.average(oracle_array)
        eval_results = dict(accuracy=acc,
                            oracle_accuracy=oracle_acc)

        return eval_results
