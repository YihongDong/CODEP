from re import S
import parso
from parso.grammar import PythonGrammar
from parso.utils import split_lines, python_bytes_to_unicode, \
    PythonVersionInfo, parse_version_string
from parso.parser import Stack, StackNode, BaseParser
import os
import copy

space = ("''", '""""""', "' '", '""" """')

def load_grammar(version= None, path= None):
    version_info = parse_version_string(version)
    file = os.path.join('grammar','grammar%s%s.txt' % (version_info.major, version_info.minor))
    path = os.path.join(os.path.dirname(__file__), file)
    with open(path) as f:
        bnf_text = f.read()
    grammar = PythonGrammar(version_info, bnf_text)
    return grammar


class PyParser(BaseParser):
    def __init__(self, pgen_grammar, start_nonterminal='file_input', error_recovery=False):
        super().__init__(pgen_grammar, start_nonterminal, error_recovery)
        self.score = 0
        self.token = []
    
    def copy(self):
        new_hyp = PyParser(self._pgen_grammar, self._start_nonterminal, self._error_recovery)
        tem = []
        for s in self.stack:
            temnode = StackNode(s.dfa)
            temnode.nodes = copy.deepcopy(s.nodes)
            tem.append(temnode)
        new_hyp.stack = Stack(tem)
        new_hyp.token = copy.deepcopy(self.token)

        # new_hyp.token = self.token
        new_hyp.score = self.score 
        new_hyp.node_map = dict(self.node_map)
        new_hyp.leaf_map = dict(self.leaf_map)

        return new_hyp
    
    def completed_copy(self, HasStack=True):
        new_hyp = completed_hyp()
        if HasStack:
            tem = []
            for s in self.stack:
                temnode = StackNode(s.dfa)
                temnode.nodes = copy.deepcopy(s.nodes)
                tem.append(temnode)
            new_hyp.stack = Stack(tem)
        new_hyp.token = copy.deepcopy(self.token)

        # new_hyp.token = self.token
        new_hyp.score = self.score 
        new_hyp.node_map = dict(self.node_map)
        new_hyp.leaf_map = dict(self.leaf_map)

        return new_hyp

class completed_hyp:
    def __init__(self):
        pass
# grammar = load_grammar(version='3.7')
#print(self.stack[-1].dfa.transitions.keys())
#print(self.stack[-1].nonterminal)