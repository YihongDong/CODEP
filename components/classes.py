import numpy
import torch


class Example(object):
    def __init__(self, src_sent, tgt_grammar, tgt_code, tgt_dfa, idx=0, meta=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_dfa = tgt_dfa
        self.tgt_grammar = tgt_grammar

        self.idx = idx
        self.meta = meta

class GrammerExample(object):
    def __init__(self, example, vocab, grammar, copy=True):
        self.example = example
        self.grammar = grammar
        self.action_num = len(self.example.tgt_grammar)
        self.code_num = len(self.example.tgt_code)
        self.vocab = vocab
        self.src_sent = self.example.src_sent
        self.src_sent_idx = [self.vocab.source.word2id[token] 
                if (token in self.vocab.source.word2id) else self.vocab.source.unk_id 
                for token in self.src_sent]
        self.copy = copy
        self.init_index_tensors()

    def init_index_tensors(self):
        self.app_rule_idx_row = []
        self.app_rule_mask_row = []
        self.primitive_row = []
        self.primitive_gen_mask_row = []
        self.primitive_copy_mask_row = []
        self.primitive_copy_idx_mask = [[True for _ in self.src_sent] for _ in range(self.action_num)]

        e = self.example

        for t in range(self.action_num):
            app_rule_idx = token_idx = 0
            app_rule_mask = gen_token_mask = copy_mask = True

            app_rule_idx = self.grammar.nonterminal2id[e.tgt_grammar[t]]
            app_rule_mask = False

            token = e.tgt_code[t][0]
            token_idx = self.vocab.code[e.tgt_code[t]]
            token_can_copy = False
            if self.copy and token in self.src_sent:
                token_pos_list = [idx for idx, _token in enumerate(self.src_sent) if _token == token]
                for pos in token_pos_list:
                    self.primitive_copy_idx_mask[t][pos] = False
                copy_mask = False
                token_can_copy = True
            if token_can_copy is False or token_idx != self.vocab.code.unk_id:
                gen_token_mask = False

            self.app_rule_idx_row.append(app_rule_idx)
            self.app_rule_mask_row.append(app_rule_mask)
            self.primitive_row.append(token_idx)
            self.primitive_gen_mask_row.append(gen_token_mask)
            self.primitive_copy_mask_row.append(copy_mask)
            


class Batch(object):
    def __init__(self, examples, vocab, grammar, copy=True):
        self.examples = [GrammerExample(example, vocab, grammar) for example in examples]
        self.max_action_num = max(e.action_num for e in self.examples)
        self.max_code_num = max(e.code_num for e in self.examples)
        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.max_src_sents_len = max(self.src_sents_len)
        self.copy = copy
        self.init_index_tensors()

    def __len__(self):
        return len(self.examples)

    def list_to_longtensor(self, data):
        tensor_list = [torch.LongTensor(seq) for seq in data]
        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)

    def list_to_floattensor(self, data):
        tensor_list = [torch.FloatTensor(seq) for seq in data]
        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0.)

    def list_to_booltensor(self, data):
        tensor_list = [torch.BoolTensor(seq) for seq in data]
        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=True)

    def init_index_tensors(self):
        self.src_sents_idx_matrix = numpy.zeros((len(self.examples), self.max_src_sents_len), dtype='int')
        self.src_sents_mask = numpy.ones((len(self.examples), self.max_src_sents_len), dtype='bool')

        self.apply_rule_idx_matrix = []
        self.apply_rule_mask = []
        self.primitive_idx_matrix = []
        self.primitive_gen_mask = []
        self.primitive_copy_mask = []
        self.primitive_copy_idx_mask = numpy.ones((len(self.examples),self.max_action_num, self.max_src_sents_len), dtype='bool')

        for e_id, e in enumerate(self.examples):
            self.src_sents_idx_matrix[e_id,:len(e.src_sent_idx)] = e.src_sent_idx
            self.src_sents_mask[e_id,:len(e.src_sent_idx)] = False

            primitive_copy_idx_mask = numpy.array(e.primitive_copy_idx_mask)
            self.primitive_copy_idx_mask[e_id,:primitive_copy_idx_mask.shape[0],:primitive_copy_idx_mask.shape[1]] = primitive_copy_idx_mask

            self.apply_rule_idx_matrix.append(e.app_rule_idx_row)
            self.apply_rule_mask.append(e.app_rule_mask_row)
            self.primitive_idx_matrix.append(e.primitive_row)
            self.primitive_gen_mask.append(e.primitive_gen_mask_row)
            self.primitive_copy_mask.append(e.primitive_copy_mask_row)

        self.src_sents_idx_matrix = torch.LongTensor(self.src_sents_idx_matrix)
        self.src_sents_mask = torch.BoolTensor(self.src_sents_mask)
        self.apply_rule_idx_matrix = self.list_to_longtensor(self.apply_rule_idx_matrix)
        self.apply_rule_mask = self.list_to_booltensor(self.apply_rule_mask)
        self.primitive_idx_matrix = self.list_to_longtensor(self.primitive_idx_matrix)
        self.primitive_gen_mask = self.list_to_booltensor(self.primitive_gen_mask)
        self.primitive_copy_mask = self.list_to_booltensor(self.primitive_copy_mask)
        self.primitive_copy_idx_mask = torch.from_numpy(self.primitive_copy_idx_mask)
        pass

    def __len__(self) -> int:
        return len(self.examples)

    def pin_memory(self):
        self.src_sents_idx_matrix = self.src_sents_idx_matrix.pin_memory()
        self.src_sents_mask = self.src_sents_mask.pin_memory()
        self.apply_rule_idx_matrix = self.apply_rule_idx_matrix.pin_memory()
        self.apply_rule_mask = self.apply_rule_mask.pin_memory()
        self.primitive_idx_matrix = self.primitive_idx_matrix.pin_memory()
        self.primitive_gen_mask = self.primitive_gen_mask.pin_memory()
        self.primitive_copy_mask = self.primitive_copy_mask.pin_memory()
        self.primitive_copy_idx_mask = self.primitive_copy_idx_mask.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.src_sents_idx_matrix = self.src_sents_idx_matrix.to(device)
        self.src_sents_mask = self.src_sents_mask.to(device)
        self.apply_rule_idx_matrix = self.apply_rule_idx_matrix.to(device)
        self.apply_rule_mask = self.apply_rule_mask.to(device)
        self.primitive_idx_matrix = self.primitive_idx_matrix.to(device)
        self.primitive_gen_mask = self.primitive_gen_mask.to(device)
        self.primitive_copy_mask = self.primitive_copy_mask.to(device)
        self.primitive_copy_idx_mask = self.primitive_copy_idx_mask.to(device)
