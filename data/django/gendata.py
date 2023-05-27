# coding=utf-8

from __future__ import print_function

import re
import pickle
import ast
import astor
import nltk
import sys
import os

import numpy as np

path_list = os.path.split(sys.path[0])
sys.path.append(path_list[0])
path_list = os.path.split(path_list[0])
sys.path.append(path_list[0])
os.chdir(path_list[0])

from pyparser import load_grammar
from parso.parser import Stack, StackNode, BaseParser, _token_to_transition
from components.vocab import Vocab, VocabEntry, VocabgrammarEntry

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

def replace_string_ast_nodes(py_ast, str_map):
    for node in ast.walk(py_ast):
        if isinstance(node, ast.Str):
            str_val = node.s

            if str_val in str_map:
                node.s = str_map[str_val]
            else:
                # handle cases like `\n\t` in string literals
                for key, val in str_map.items():
                    str_literal_decoded = key.decode('string_escape')
                    if str_literal_decoded == str_val:
                        node.s = val


class Django(object):
    @staticmethod
    def canonicalize_code(code):
        if p_elif.match(code):
            code = 'if True: pass\n' + code

        if p_else.match(code):
            code = 'if True: pass\n' + code

        if p_try.match(code):
            code = code + 'pass\nexcept: pass'
        elif p_except.match(code):
            code = 'try: pass\n' + code
        elif p_finally.match(code):
            code = 'try: pass\n' + code

        if p_decorator.match(code):
            code = code + '\ndef dummy(): pass'

        if code[-1] == ':':
            code = code + 'pass'

        return code

    @staticmethod
    def canonicalize_str_nodes(py_ast, str_map):
        for node in ast.walk(py_ast):
            if isinstance(node, ast.Str):
                str_val = node.s

                if str_val in str_map:
                    node.s = str_map[str_val]
                else:
                    # handle cases like `\n\t` in string literals
                    for str_literal, slot_id in str_map.items():
                        str_literal_decoded = str_literal.decode('string_escape')
                        if str_literal_decoded == str_val:
                            node.s = slot_id

    @staticmethod
    def canonicalize_query(query):
        """
        canonicalize the query, replace strings to a special place holder
        """
        str_count = 0
        str_map = dict()

        matches = QUOTED_STRING_RE.findall(query)
        # de-duplicate
        cur_replaced_strs = set()
        for match in matches:
            # If one or more groups are present in the pattern,
            # it returns a list of groups
            quote = match[0]
            str_literal = match[1]
            quoted_str_literal = quote + str_literal + quote

            if str_literal in cur_replaced_strs:
                # replace the string with new quote with slot id
                query = query.replace(quoted_str_literal, str_map[str_literal])
                continue

            # FIXME: substitute the ' % s ' with
            if str_literal in ['%s']:
                continue

            str_repr = 'STR_%d' % str_count
            str_map[str_literal] = str_repr

            query = query.replace(quoted_str_literal, str_repr)

            str_count += 1
            cur_replaced_strs.add(str_literal)

        # tokenize
        query_tokens = nltk.word_tokenize(query)

        new_query_tokens = []
        # break up function calls like foo.bar.func
        for token in query_tokens:
            new_query_tokens.append(token)
            i = token.find('.')
            if 0 < i < len(token) - 1:
                new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
                new_query_tokens.extend(new_tokens)

        query = ' '.join(new_query_tokens)
        query = query.replace('\' % s \'', '%s').replace('\" %s \"', '%s')

        return query, str_map

    @staticmethod
    def canonicalize_example(query, code):

        canonical_query, str_map = Django.canonicalize_query(query)
        query_tokens = canonical_query.split(' ')

        canonical_code = Django.canonicalize_code(code)
        ast_tree = ast.parse(canonical_code)

        Django.canonicalize_str_nodes(ast_tree, str_map)
        canonical_code = astor.to_source(ast_tree)

        return query_tokens, canonical_code, str_map

    @staticmethod
    def parse_django_dataset(annot_file, code_file, grammar, max_query_len=70, vocab_freq_cutoff=10):
        loaded_examples = []

        from components.vocab import Vocab, VocabEntry
        from components.dataset import Example
        first_dfa = grammar._pgen_grammar.nonterminal_to_dfas[grammar._start_nonterminal][0]
        skipped_list = []
        for idx, (src_query, original_code) in enumerate(zip(open(annot_file), open(code_file))):
            src_query = src_query.strip()
            original_code = original_code.strip()

            src_query_tokens, tgt_canonical_code, str_map = Django.canonicalize_example(src_query, original_code)
            python_ast = ast.parse(tgt_canonical_code).body[0]
            gold_source = astor.to_source(python_ast).strip()
            for x, slot_name in list(str_map.items()):
                gold_source = gold_source.replace('\''+slot_name+'\'', slot_name)
            if not gold_source.endswith('\n'):
                gold_source = gold_source+'\n'
            # try:
            tgt_code = []
            tgt_grammar = []
            p = BaseParser(grammar._pgen_grammar, error_recovery=True, start_nonterminal=grammar._start_nonterminal)
            p.stack = Stack([StackNode(first_dfa)])
            tokens = grammar._tokenize(gold_source)
            # sanity check
            for token in tokens:
                type_, value, start_pos, prefix = token
                transition = _token_to_transition(grammar._pgen_grammar, type_, value)
                if hasattr(transition, 'value'):
                    tgt_code.append((token.string,transition.value))
                else:
                    tgt_code.append((token.string,transition.name))
                p._add_token(token)
                tgt_grammar.append(p.stack[-1].nonterminal)
            assert p.stack[-1].dfa.is_final and len(p.stack) == 1
            # except (AssertionError, SyntaxError, ValueError, OverflowError, NotImplementedError) as e:
            #     skipped_list.append(idx)
            #     continue

            loaded_examples.append({'src_query_tokens': src_query_tokens,
                                    'tgt_canonical_code': tgt_code,
                                    'tgt_grammar':tgt_grammar,
                                    'tgt_dfa':p.stack[-1],
                                    'raw_code': original_code, 'str_map': str_map})

        print('Skipped due to exceptions: %d' % len(skipped_list), file=sys.stderr)

        train_examples = []
        dev_examples = []
        test_examples = []

        action_len = []

        for idx, e in enumerate(loaded_examples):
            src_query_tokens = e['src_query_tokens'][:max_query_len]

            example = Example(idx=idx,
                              src_sent=src_query_tokens,
                              tgt_code=e['tgt_canonical_code'],
                              tgt_grammar=e['tgt_grammar'],
                              tgt_dfa=e['tgt_dfa'],
                              meta={'raw_code': e['raw_code'], 'str_map': e['str_map']})

            # print('second pass, processed %d' % idx, file=sys.stderr)

            action_len.append(len(e['tgt_canonical_code']))

            # train, valid, test split
            if 0 <= idx < 16000:
                train_examples.append(example)
            elif 16900 <= idx < 17000:
                dev_examples.append(example)
            else:
                test_examples.append(example)

        print('Max action len: %d' % max(action_len), file=sys.stderr)
        print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
        print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

        src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_examples], size=10000, freq_cutoff=vocab_freq_cutoff)

        code_vocab = VocabgrammarEntry.from_corpus([e.tgt_code for e in train_examples], size=10000, freq_cutoff=vocab_freq_cutoff)

        vocab = Vocab(source=src_vocab, code=code_vocab)
        print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

        return (train_examples, dev_examples, test_examples), vocab

    @staticmethod
    def process_django_dataset():
        vocab_freq_cutoff = 15  # TODO: found the best cutoff threshold
        annot_file = 'data/django/all.anno'
        code_file = 'data/django/all.code'
        grammar = load_grammar(version='2.7')
        (train, dev, test), vocab = Django.parse_django_dataset(annot_file, code_file,
                                                                grammar,
                                                                vocab_freq_cutoff=vocab_freq_cutoff)

        pickle.dump(train, open('data/django/train.bin', 'w'),-1)
        pickle.dump(dev, open('data/django/dev.bin', 'w'),-1)
        pickle.dump(test, open('data/django/test.bin', 'w'),-1)
        pickle.dump(vocab, open('data/django/vocab.freq%d.bin' % vocab_freq_cutoff, 'w'),-1)


    @staticmethod
    def canonicalize_raw_django_oneliner(code):
        # use the astor-style code
        code = Django.canonicalize_code(code)
        py_ast = ast.parse(code).body[0]
        code = astor.to_source(py_ast).strip()

        return code


def generate_vocab_for_paraphrase_model(vocab_path, save_path):
    from components.vocab import VocabEntry, Vocab

    vocab = pickle.load(open(vocab_path))
    para_vocab = VocabEntry()
    for i in range(0, 10):
        para_vocab.add('<unk_%d>' % i)
    for word in vocab.source.word2id:
        para_vocab.add(word)
    for word in vocab.code.word2id:
        para_vocab.add(word)

    pickle.dump(para_vocab, open(save_path, 'w'))


if __name__ == '__main__':
    Django.process_django_dataset()