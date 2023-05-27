import argparse
import json
import os
import pickle
import sys
import re
import ast
import astor
import nltk

path_list = os.path.split(sys.path[0])
sys.path.append(path_list[0])
path_list = os.path.split(path_list[0])
sys.path.append(path_list[0])
os.chdir(path_list[0])

import numpy as np
from copy import deepcopy

from parso.utils import split_lines, python_bytes_to_unicode
from parso.grammar import PythonGrammar
from parso.python.token import PythonTokenTypes
from parso.python.tokenize import PythonToken
from parso.parser import Stack, StackNode, BaseParser, _token_to_transition
from pyparser import load_grammar, PyParser, space
from components.dataset import Example
from components.vocab import Vocab, VocabEntry, VocabgrammarEntry

#assert astor.__version__ == '0.7.1'

QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")

def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'

def canonicalize_intent(intent):
    marked_token_matches = QUOTED_TOKEN_RE.findall(intent)

    slot_map = dict()
    var_id = 0
    str_id = 0
    already_match = []
    for match in marked_token_matches:
        if match in already_match:
            continue
        quote = match[0]
        value = match[1]
        quoted_value = quote + value + quote

        slot_type = infer_slot_type(quote, value)

        if slot_type == 'var':
            slot_name = 'var_%d' % var_id
            var_id += 1
            slot_type = 'var'
        else:
            slot_name = 'str_%d' % str_id
            str_id += 1
            slot_type = 'str'

        intent = intent.replace(quoted_value, slot_name)
        slot_map[slot_name] = value.strip().encode().decode('unicode_escape', 'ignore')
        already_match.append(match)

    return intent, slot_map

def tokenize_intent(intent):
    lower_intent = intent.lower()
    tokens = nltk.word_tokenize(lower_intent)

    return tokens

def replace_identifiers_in_ast(py_ast, identifier2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v!=' ':
                    if v in identifier2slot:
                        slot_name = identifier2slot[v]
                        # Python 3
                        # if isinstance(slot_name, unicode):
                        #     try: slot_name = slot_name.encode('ascii')
                        #     except: pass

                        setattr(node, k, slot_name)
                else:
                    if '' in identifier2slot:
                        slot_name = identifier2slot['']
                        setattr(node, k, slot_name)


def is_enumerable_str(identifier_value):
    """
    Test if the quoted identifier value is a list
    """

    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in ('}', ']', ')')


def canonicalize_code(code, slot_map, name, grammar):
    string2slot = {x: slot_name for slot_name, x in list(slot_map.items())}

    py_ast = ast.parse(code)
    replace_identifiers_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast).strip()

    first_dfa = grammar._pgen_grammar.nonterminal_to_dfas[grammar._start_nonterminal][0]
    p = PyParser(grammar._pgen_grammar, error_recovery=True, start_nonterminal=grammar._start_nonterminal)
    p.stack = Stack([StackNode(first_dfa)])
    tokens = grammar._tokenize(canonical_code+'\n')

    for token in tokens:
        type_, value, start_pos, prefix = token
        if type_.name in ('NAME', 'STRING', 'NUMBER'):
            if value in string2slot and value != 'for':
                token = PythonToken(PythonTokenTypes.NAME, string2slot[value],start_pos,prefix)
            elif value in space:
                token = PythonToken(PythonTokenTypes.STRING, "' '",start_pos,prefix)
        p._add_token(token)

    assert p.stack[-1].dfa.is_final and len(p.stack) == 1
    canonical_code = ''.join([i.get_code() for i in p.stack[-1].nodes])

    entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val)]
    if entries_that_are_lists:
        for slot_name in entries_that_are_lists:
            canonical_code = canonical_code.replace(slot_map[slot_name], slot_name)


    for slot_name, x in list(slot_map.items()):
            
        canonical_code = canonical_code.replace('\''+slot_name+'\'', slot_name)
        canonical_code = canonical_code.replace("\"\"\""+slot_name+"\"\"\"", slot_name)
    canonical_code = canonical_code.replace('\r\n', '\n')
    canonical_code = canonical_code.replace('\r', '\n')
    if canonical_code.endswith('\n'):
        return canonical_code
    return canonical_code+'\n'


def preprocess_conala_dataset(train_file, test_file, grammar_version, src_freq=2, code_freq=2,
                              mined_data_file=None, api_data_file=None,
                              vocab_size=20000, num_mined=0, out_dir=os.path.dirname(__file__)):
    np.random.seed(1234)

    grammar = load_grammar(version=grammar_version)

    print('process gold training data...')
    train_examples = preprocess_dataset(train_file, name='train', grammar=grammar)

    # held out 200 examples for development
    full_train_examples = train_examples[:]
    np.random.shuffle(train_examples)
    dev_examples = train_examples[:200]
    train_examples = train_examples[200:]

    mined_examples = []
    api_examples = []
    if mined_data_file and num_mined > 0:
        print("use mined data: ", num_mined)
        print("from file: ", mined_data_file)
        mined_examples = preprocess_dataset(mined_data_file, name='mined', grammar=grammar,
                                            firstk=num_mined)
        pickle.dump(mined_examples, open(os.path.join(out_dir, 'mined_{}.bin'.format(num_mined)), 'wb'))

    if api_data_file:
        print("use api docs from file: ", api_data_file)
        name = os.path.splitext(os.path.basename(api_data_file))[0]
        api_examples = preprocess_dataset(api_data_file, name='api', grammar=grammar)
        pickle.dump(api_examples, open(os.path.join(out_dir, name + '.bin'), 'wb'))

    if mined_examples and api_examples:
        pickle.dump(mined_examples + api_examples, open(os.path.join(out_dir, 'pre_{}_{}.bin'.format(num_mined, name)), 'wb'))

    # combine to make vocab
    train_examples += mined_examples
    train_examples += api_examples
    print(f'{len(train_examples)} training instances', file=sys.stderr)
    print(f'{len(dev_examples)} dev instances', file=sys.stderr)

    print('process testing data...')
    test_examples = preprocess_dataset(test_file, name='test', grammar=grammar)
    print(f'{len(test_examples)} testing instances', file=sys.stderr)

    src_vocab = VocabEntry.from_corpus([e.src_sent for e in full_train_examples], size=vocab_size,
                                       freq_cutoff=src_freq)
    
    # grammar_vocab = VocabEntry.from_corpus([[e] for e in grammar._pgen_grammar.nonterminal_to_dfas.keys()], size=vocab_size)

    # generate vocabulary for the code tokens!
    code_vocab = VocabgrammarEntry.from_corpus([e.tgt_code for e in full_train_examples], size=vocab_size, freq_cutoff=code_freq)

    # vocab = Vocab(source=src_vocab, grammar=grammar_vocab, code=code_vocab)
    vocab = Vocab(source=src_vocab, code=code_vocab)
    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    code_lens = [len(e.tgt_code) for e in full_train_examples]
    print('Max code len: %d' % max(code_lens), file=sys.stderr)
    print('Avg code len: %d' % np.average(code_lens), file=sys.stderr)
    print('Code larger than 100: %d' % len(list(filter(lambda x: x > 100, code_lens))), file=sys.stderr)

    pickle.dump(train_examples, open(os.path.join(out_dir, 'train.all_{}.bin'.format(num_mined) if mined_data_file and num_mined > 0 else 'train.bin'), 'wb'))
    pickle.dump(dev_examples, open(os.path.join(out_dir, 'dev.bin'), 'wb'))
    pickle.dump(test_examples, open(os.path.join(out_dir, 'test.bin'), 'wb'))
    if mined_examples and api_examples:
        vocab_name = 'vocab.src_freq%d.code_freq%d.mined_%s.%s.bin' % (src_freq, code_freq, num_mined, name)
    elif mined_examples:
        vocab_name = 'vocab.src_freq%d.code_freq%d.mined_%s.bin' % (src_freq, code_freq, num_mined)
    elif api_examples:
        vocab_name = 'vocab.src_freq%d.code_freq%d.%s.bin' % (src_freq, code_freq, name)
    else:
        vocab_name = 'vocab.src_freq%d.code_freq%d.bin' % (src_freq, code_freq)
    pickle.dump(vocab, open(os.path.join(out_dir, vocab_name), 'wb'))


def preprocess_dataset(file_path, grammar, name='train', firstk=None):
    try:
        dataset = json.load(open(file_path))
    except:
        dataset = [json.loads(jline) for jline in open(file_path).readlines()]
    if firstk:
        dataset = dataset[:firstk]
    examples = []
    first_dfa = grammar._pgen_grammar.nonterminal_to_dfas[grammar._start_nonterminal][0]
    skipped_list = []
    for i, example_json in enumerate(dataset):
        try:
            example_dict = preprocess_example(example_json, name, grammar)
            canonical_snippet = example_dict['canonical_snippet']
            tgt_code = []
            tgt_grammar = []
            p = PyParser(grammar._pgen_grammar, error_recovery=True, start_nonterminal=grammar._start_nonterminal)
            p.stack = Stack([StackNode(first_dfa)])
            tokens = grammar._tokenize(canonical_snippet)
            # sanity check
            for token in tokens:
                type_, value, start_pos, prefix = token
                transition = _token_to_transition(grammar._pgen_grammar, type_, value)
                if type(transition.value) == str:
                    tgt_code.append((token.string,transition.value))
                else:
                    tgt_code.append((token.string,transition.name))
                p._add_token(token)
                tgt_grammar.append(p.stack[-1].nonterminal)
            assert p.stack[-1].dfa.is_final and len(p.stack) == 1
            
        except (AssertionError, SyntaxError, ValueError, OverflowError, NotImplementedError) as e:
            skipped_list.append(example_json['question_id'])
            continue
        example = Example(idx=f'{i}-{example_json["question_id"]}',
                          src_sent=example_dict['intent_tokens'],
                          tgt_grammar=tgt_grammar,
                          tgt_code=tgt_code,
                          tgt_dfa=p.stack[-1],
                          meta=dict(example_dict=example_json,
                                    slot_map=example_dict['slot_map']))

        examples.append(example)
    print('Skipped due to exceptions: %d' % len(skipped_list), file=sys.stderr)
    return examples


def preprocess_example(example_json, name, grammar):
    intent = example_json['intent']
    if 'rewritten_intent' in example_json:
        rewritten_intent = example_json['rewritten_intent']
    else:
        rewritten_intent = None

    if rewritten_intent is None:
        rewritten_intent = intent
    snippet = example_json['snippet']
    # rewritten_intent = replace_special_character(rewritten_intent)
    canonical_intent, slot_map = canonicalize_intent(rewritten_intent)
    canonical_snippet = canonicalize_code(snippet, slot_map, name, grammar)
    intent_tokens = tokenize_intent(canonical_intent)

    return {'canonical_intent': canonical_intent,
            'intent_tokens': intent_tokens,
            'slot_map': slot_map,
            'canonical_snippet': canonical_snippet}

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    #### General configuration ####
    arg_parser.add_argument('--pretrain', type=str, help='Path to pretrain file')
    arg_parser.add_argument('--out_dir', type=str, default='data/conala', help='Path to output file')
    arg_parser.add_argument('--topk', type=int, default=0, help='First k number from mined file')
    arg_parser.add_argument('--freq', type=int, default=2, help='minimum frequency of tokens')
    arg_parser.add_argument('--vocabsize', type=int, default=1000, help='First k number from pretrain file')
    arg_parser.add_argument('--include_api', type=str, help='Path to apidocs file')
    args = arg_parser.parse_args()

    # the json files can be downloaded from http://conala-corpus.github.io
    preprocess_conala_dataset(train_file='conala-train.json',
                              test_file='conala-test.json',
                              grammar_version='3.7',
                              mined_data_file=args.pretrain,
                              api_data_file=args.include_api,
                              src_freq=args.freq, code_freq=args.freq,
                              vocab_size=args.vocabsize,
                              num_mined=args.topk,
                              out_dir=args.out_dir)
