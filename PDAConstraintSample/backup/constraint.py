from abc import ABC, abstractmethod
import imp
from typing import List, Optional
from pyparser import load_grammar, PyParser
from parso.parser import Stack, StackNode, _token_to_transition
from parso.python.token import PythonTokenTypes
from parso.python.tokenize import PythonToken

class Constraint(ABC):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    """

    def __init__(self):
        # test for the above condition
        self.test()

    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        counter = 0
        completed = False
        while not completed:
            if counter == 1:
                self.reset()
            advance = self.advance()
            if not self.does_advance(advance):
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            stepped, completed, reset = self.update(advance)
            counter += 1

            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Return:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

# class PDAModule:
#     def __init__(self, token_ids: List[List[int]], tokenizer, grammarversion='3,7'):
#         r"""
#         A helper class that builds a PDA with the words represented in `token_ids`.
#         """
#         self.max_height = 512

#         self.grammar = load_grammar(grammarversion)
#         self.tokenizer = tokenizer

#         first_dfa = self.grammar._pgen_grammar.nonterminal_to_dfas[self.grammar._start_nonterminal][0]
#         self.PDA = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
#         self.PDA.stack = Stack([StackNode(first_dfa)])

#     def next_token_types(self, hyp):
#         """
#         The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
#         """
#         tem_hyp =hyp.copy()
#         action_types = set(tem_hyp.stack[-1].dfa.transitions.keys())
#         while tem_hyp.stack[-1].dfa.is_final:
#             tem_hyp._pop()
#             action_types = action_types | set(tem_hyp.stack[-1].dfa.transitions.keys())

#         return action_types

class PDAConstraint(Constraint):
    def __init__(self, tokenizer, vocab, id2type=None, grammarversion='3.7'):
        super(Constraint, self).__init__()
        
        self.grammar = load_grammar(grammarversion)
        self.tokenizer = tokenizer
        if hasattr(vocab, 'code'):
            self.vocab = vocab.code   
            self.initVocab()
        else:
            self.vocab = vocab
        
        if id2type is not None:
            self.id2type = id2type
        else:
            self.id2type = {} 
            for i in self.vocab.keys():
                for j in self.vocab[i]:
                    self.id2type[j] = i
        self.first_dfa = self.grammar._pgen_grammar.nonterminal_to_dfas[self.grammar._start_nonterminal][0]
        self.PDA = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
        self.PDA.stack = Stack([StackNode(self.first_dfa)])
        # self.PDA = PDAModule(tokenizer, grammarversion)

        self.seqlen = 128
        self.completed = False
        
    def advance(self):
        if not hasattr(self, 'next_tokens'):
            self.token_type_list = self.next_token_types(self.PDA)
            #convert_tokens_to_string convert_ids_to_tokens tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(203))

            if len(self.token_type_list) == 0:
                return None
            else:
                self.next_tokens = []
                for i in self.token_type_list:
                    if i in self.vocab.keys():
                        self.next_tokens.extend(self.vocab[i])
                # token_list = [self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(i)) for i in self.token_ids]
                # self.tokenizer(self.tokenizer.decode(i))['input_ids'][1:-1]
                return  self.next_tokens
        else:
            return  self.next_tokens

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        if not hasattr(self, 'next_tokens'):
            self.token_type_list = self.next_token_types(self.PDA)
            self.next_tokens = []
            for i in self.token_type_list:
                if i in self.vocab.keys():
                    self.next_tokens.extend(self.vocab[i])
            return token_id in self.next_tokens
        else:
            return token_id in self.next_tokens

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            if token_id == self.tokenizer.eos_token_id:
                token = PythonToken(PythonTokenTypes['ENDMARKER'], '',(1,0),'')
                self.PDA._add_token(token)
                self.PDA.token.append('ENDMARKER')
            # elif token_id==11:
            #     try:
            #         token = PythonToken(PythonTokenTypes['INDENT'], '',(1,0),'')
            #         self.PDA._add_token(token)
            #         self.PDA.token.append('INDENT')
            #     except:
            #         token = PythonToken(PythonTokenTypes['DEDENT'], '',(1,0),'')
            #         self.PDA._add_token(token)
            #         self.PDA.token.append('DEDENT')
            else:
                token_name = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token_id))
                try:
                    token = PythonToken(PythonTokenTypes[self.id2type[token_id]], token_name,(1,0),'')
                except:
                    token = list(self.grammar._tokenize_lines([token_name]))[0]
                self.PDA._add_token(token)
                self.PDA.token.append(token_name)
            del self.next_tokens
            stepped = True
        # else:
        #     reset = True
        #     self.reset()
        if not hasattr(self, 'next_tokens'):
            self.token_type_list = self.next_token_types(self.PDA)
            self.next_tokens = []
            for i in self.token_type_list:
                if i in self.vocab.keys():
                    self.next_tokens.extend(self.vocab[i])

        completed = 'ENDMARKER' in self.token_type_list #self.PDA.stack[-1].dfa.is_final and len(self.PDA.stack) == 1
        self.completed = completed

        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.PDA = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
        self.PDA.stack = Stack([StackNode(self.first_dfa)])
        del self.next_tokens

    def remaining(self):
        if self.completed:
            # since this can be completed without reaching max height
            return 0
        else:
            return self.seqlen - len(self.PDA.token)

    def copy(self, stateful=False):
        new_constraint = PDAConstraint(self.tokenizer, self.vocab, self.id2type)
        
        if stateful:
            new_constraint.seqlen = self.seqlen
            new_constraint.PDA = self.PDA.copy()
            new_constraint.completed = self.completed
            new_constraint.vocab = self.vocab
            new_constraint.tokenizer = self.tokenizer
            if not hasattr(self, 'next_tokens'):
                self.token_type_list = self.next_token_types(self.PDA)
                self.next_tokens = []
                for i in self.token_type_list:
                    if i in self.vocab.keys():
                        self.next_tokens.extend(self.vocab[i])
                new_constraint.next_tokens = self.next_tokens
            else:
                new_constraint.next_tokens = self.next_tokens

        return new_constraint

    def next_token_types(self, hyp):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        tem_hyp =hyp.copy()
        action_types = set(tem_hyp.stack[-1].dfa.transitions.keys())
        while tem_hyp.stack[-1].dfa.is_final:
            tem_hyp._pop()
            action_types = action_types | set(tem_hyp.stack[-1].dfa.transitions.keys())

        action_types = [i.value if type(i.value) == str else i.name for i in action_types]
        if hyp.stack[-1].nonterminal == self.first_dfa.from_rule and not hyp.stack[-1].dfa.is_final:
            action_types=['def']
        return action_types

    def initVocab(self):
        tem = {}
        for i in self.vocab.transition2id.keys():
            # if i != 'NAME':
            #     tem[i] = [self.tokenizer.encode(self.vocab.id2word[j][0])[1] for j in self.vocab.transition2id[i]]
            # else:
            #     # tem[i]=[]
            #     # for j in self.vocab.transition2id[i]:
            #     #     if self.vocab.id2word[j][0] not in self.grammar._pgen_grammar.reserved_syntax_strings.keys():
            #     #         tem[i].append(self.tokenizer.encode(self.vocab.id2word[j][0])[1])
            #     # not in self.grammar._pgen_grammar.reserved_syntax_strings.keys()

            # codet5
            # tem[i] = [self.tokenizer.encode(self.vocab.id2word[j][0])[1] for j in self.vocab.transition2id[i] \
            #     if len(self.tokenizer.encode(self.vocab.id2word[j][0])) == 3]
            # codegen
            tem[i] = [self.tokenizer.encode(self.vocab.id2word[j][0])[0] for j in self.vocab.transition2id[i] \
                if len(self.tokenizer.encode(self.vocab.id2word[j][0])) == 1]
            # incoder
            # tem[i] = [self.tokenizer.encode(self.vocab.id2word[j][0])[-1] for j in self.vocab.transition2id[i] \
            # if len(self.tokenizer.encode(self.vocab.id2word[j][0])) == 2]
        tem['STRING']=[]
        tem['ENDMARKER'] = [self.tokenizer.eos_token_id] #[2]
        self.vocab = tem

