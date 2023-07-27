# coding=utf-8
from __future__ import print_function
from ast import Str

import os
from six.moves import xrange as range
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from parso.parser import Stack, StackNode, _token_to_transition
from parso.python.token import PythonTokenTypes
from parso.python.tokenize import PythonToken
from components.registerable import Registrable
from components.classes import Batch
from components.utils import update_args, init_arg_parser
import components.nn_utils as nn_utils
from pyparser import PyParser
from pyparser import load_grammar
import copy

def PreprocessGrammar(grammar):
    grammar.nonterminal2id = {nonterminal: i+4 for i, nonterminal in enumerate(grammar._pgen_grammar.nonterminal_to_dfas.keys())}
    grammar.nonterminal2id['<pad>'] = 0
    grammar.nonterminal2id['<SOS>'] = 1
    grammar.nonterminal2id['<EOS>'] = 2 
    grammar.nonterminal2id['<unk>'] = 3
    grammar.id2nonterminal = {v: k for k, v in grammar.nonterminal2id.items()}
    return grammar

@Registrable.register('transformer_parser')
class TransformerParser(nn.Module):
    def __init__(self, args, vocab, grammar):
        super(TransformerParser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.grammar = PreprocessGrammar(grammar)

        # Embedding layers
        # source token embedding
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size, padding_idx=vocab.source.word2id['<pad>'])

        # embedding nonterminal
        if args.use_nonterminal or self.args.feed_nonterminal:
            self.state_embed = nn.Embedding(len(self.grammar.nonterminal2id), args.action_embed_size, padding_idx=self.grammar.nonterminal2id['<pad>'])
    
        # embedding table for code tokens
        self.code_embed = nn.Embedding(len(vocab.code), args.action_embed_size, padding_idx=vocab.code.word2id['<pad>'])

        nn.init.xavier_normal_(self.src_embed.weight.data)
        if args.use_nonterminal:
            nn.init.xavier_normal_(self.state_embed.weight.data)
        nn.init.xavier_normal_(self.code_embed.weight.data)

        self.encoder = SeqEncoder(args=self.args)

        self.input_dim = args.action_embed_size  # previous action
        self.input_dim += args.action_embed_size * (self.args.feed_nonterminal)

        if args.use_nonterminal:
            self.code_decoder = GrammarDecoder(args=self.args, feature_dim=self.input_dim, encoder_output_dim=args.embed_size,
            state_embed=self.state_embed,code_embed=self.code_embed)
        else:
            self.code_decoder = GrammarDecoder(args=self.args, feature_dim=self.input_dim, encoder_output_dim=args.embed_size,
            code_embed=self.code_embed)

        if args.code_token_label_smoothing:
            self.label_smoothing = nn_utils.LabelSmoothing(args.code_token_label_smoothing, len(self.vocab.code), ignore_indices=[0, 1, 2])

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def score(self, examples, return_encode_state=False):
        batch = Batch(examples, self.vocab, self.grammar)
        if self.args.cuda:
            batch.move_to_device('cuda')

        # [batch size; length]
        batch_idx_matrix = batch.src_sents_idx_matrix
        batch_input = self.src_embed(batch_idx_matrix)
        inputs_mask = batch.src_sents_mask
        # [batch size; length; hidden size]
        batched_features, attention_mask = self.encoder(batch_input, inputs_mask)

        if self.args.use_nonterminal:
            apply_rule_prob, primitive_gen_prob, primitive_copy_prob, primitive_predictor_prob = self.decode(batched_features, attention_mask, batch)
        else:
            primitive_gen_prob, primitive_copy_prob, primitive_predictor_prob = self.decode(batched_features, attention_mask, batch)
        
        primitive_gen = primitive_gen_prob* primitive_predictor_prob[:,:,:1]
        primitive_copy = primitive_copy_prob* primitive_predictor_prob[:,:,1:]
        if self.args.use_nonterminal:
            tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=-1,
                                           index=batch.apply_rule_idx_matrix.unsqueeze(dim=-1)).squeeze(-1)
        tgt_primitive_gen_from_vocab_prob = torch.gather(primitive_gen, dim=-1,
                                                         index=batch.primitive_idx_matrix.unsqueeze(dim=-1)).squeeze(-1)
        tgt_primitive_copy_prob = torch.sum(primitive_copy * ~batch.primitive_copy_idx_mask, dim=-1)
        action_prob = tgt_primitive_gen_from_vocab_prob * ~batch.primitive_gen_mask + \
                    tgt_primitive_copy_prob * ~batch.primitive_copy_mask
        action_mask = (batch.primitive_gen_mask & batch.primitive_copy_mask) #batch.apply_rule_mask

        mask_pad = 1.0-action_mask.float()
        action_prob.data.masked_fill_(action_mask.data, 1.e-7)
        action_prob = action_prob.log()*mask_pad

        if self.args.use_nonterminal:
            grammar_mask_pad = 1.0-batch.apply_rule_mask.float() 
            grammar_prob = tgt_apply_rule_prob * ~batch.apply_rule_mask
            grammar_prob.data.masked_fill_(batch.apply_rule_mask.data, 1.e-7)
            grammar_prob = grammar_prob.log() * grammar_mask_pad

        # scores = torch.sum(action_prob, dim=-1)
        # returns = [scores]
        # if self.args.use_nonterminal:
        #     returns.append(torch.sum(grammar_prob, dim=-1))
        scores = torch.sum(action_prob, dim=-1)
        if self.args.use_nonterminal:
            scores += self.args.alpha * torch.sum(grammar_prob, dim=-1)

        returns = [scores]

        return returns

    def gen_grammar_features(self, batch, batched_features):
        batch_size = batch.apply_rule_idx_matrix.shape[0]
        init_input = Variable(batched_features.new_zeros(self.input_dim), requires_grad=False)
        # [batch size; 1; ast_decoder_input_dim]
        init_input = init_input.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, -1, -1)
        # [batch size; max_ast_len; action_embed_size]
        input_gentoken = self.code_embed(batch.primitive_idx_matrix)
        gentoken_mask = batch.primitive_gen_mask.unsqueeze(dim=-1) & batch.primitive_copy_mask.unsqueeze(dim=-1)
        input_gentoken = input_gentoken*(~gentoken_mask)
        if self.args.feed_nonterminal:
            # [batch size; max_ast_len; action_embed_size]
            input_actions = self.state_embed(batch.apply_rule_idx_matrix)
            input_actions = input_actions*(~batch.apply_rule_mask.unsqueeze(dim=-1))
            # [batch size; max_ast_len; ast_decoder_input_dim]
            decoder_input = torch.cat([input_gentoken, input_actions], dim=-1)
        else:
            decoder_input = input_gentoken
        # [batch size; max_ast_len+1; ast_decoder_input_dim]
        decoder_input = torch.cat([init_input, decoder_input], dim=1)
        return decoder_input

    def decode(self, 
        batched_features, 
        batched_mask,
        batch):

        tgt_input = self.gen_grammar_features(batch, batched_features)
        if self.args.use_nonterminal:
            action_prob, primitive_gen_prob, primitive_copy_prob, predictor_prob = self.code_decoder(
                tgt_input, batched_features, batched_mask
            )

            action_prob = action_prob[:,:-1]
            primitive_gen_prob = primitive_gen_prob[:,:-1]
            primitive_copy_prob = primitive_copy_prob[:,:-1]
            predictor_prob = predictor_prob[:,:-1]
            # [batch size; target len; vocab size] 
            return action_prob, primitive_gen_prob, primitive_copy_prob, predictor_prob
        else:
            primitive_gen_prob, primitive_copy_prob, predictor_prob = self.code_decoder(
                tgt_input, batched_features, batched_mask
            )

            primitive_gen_prob = primitive_gen_prob[:,:-1]
            primitive_copy_prob = primitive_copy_prob[:,:-1]
            predictor_prob = predictor_prob[:,:-1]
            # [batch size; target len; vocab size] 
            return primitive_gen_prob, primitive_copy_prob, predictor_prob

    def parse(self, example, context=None, beam_size=5, debug=False):
        args = self.args
        primitive_vocab = self.vocab.code
        T = torch.cuda if args.cuda else torch
        batch = Batch([example], self.vocab, self.grammar)
        if self.args.cuda:
            batch.move_to_device('cuda')

        # [batch size; length]
        batch_idx_matrix = batch.src_sents_idx_matrix
        batch_input = self.src_embed(batch_idx_matrix)
        inputs_mask = batch.src_sents_mask
        # [batch size; length; hidden size]
        batched_features, attention_mask = self.encoder(batch_input, inputs_mask)
        src_idx, src_features, src_mask = batch_idx_matrix, batched_features, attention_mask

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(example.src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0
        first_dfa = self.grammar._pgen_grammar.nonterminal_to_dfas[self.grammar._start_nonterminal][0]
        p = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
        p.stack = Stack([StackNode(first_dfa)])
        hypotheses = [p]
        completed_hypotheses = []
        out = []
        with torch.no_grad():
            x = Variable(self.new_tensor(1, self.input_dim).zero_())
            out.append(x)

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            current_input=torch.stack(out)
            batch_size=current_input.size(0)
            if self.args.use_nonterminal:
                apply_rule_prob, primitive_prob, primitive_copy_prob, primitive_predictor_prob = self.code_decoder(
                current_input, src_features.expand(batch_size,-1,-1), src_mask.expand(batch_size,-1))
                apply_rule_prob = apply_rule_prob[:,-1]
                primitive_prob = primitive_prob[:,-1]
                primitive_copy_prob = primitive_copy_prob[:,-1]
                primitive_predictor_prob = primitive_predictor_prob[:,-1]
                primitive_prob = primitive_prob*primitive_predictor_prob[:,:1]
                primitive_copy_prob = primitive_copy_prob*primitive_predictor_prob[:,1:]
                apply_rule_log_prob = torch.log(apply_rule_prob)
            else:
                primitive_prob, primitive_copy_prob, primitive_predictor_prob = self.code_decoder(
                current_input, src_features.expand(batch_size,-1,-1), src_mask.expand(batch_size,-1))
                primitive_prob = primitive_prob[:,-1]
                primitive_copy_prob = primitive_copy_prob[:,-1]
                primitive_predictor_prob = primitive_predictor_prob[:,-1]
                primitive_prob = primitive_prob*primitive_predictor_prob[:,:1]
                primitive_copy_prob = primitive_copy_prob*primitive_predictor_prob[:,1:]

            # gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []

            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                tem_hyp =hyp.copy()
                action_types = set(tem_hyp.stack[-1].dfa.transitions.keys())
                while tem_hyp.stack[-1].dfa.is_final:
                    tem_hyp.stack.pop()
                    action_types = action_types | set(tem_hyp.stack[-1].dfa.transitions.keys())

                if sum([1 if type(action_type.value) != str and action_type.name in ('NUMBER', 'STRING', 'NAME') else 0 for action_type in action_types]) > 0:
                    hyp_unk_copy_info = []

                    if args.no_copy is False:
                        for token, token_pos_list in aggregated_primitive_tokens.items():
                            sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, Variable(T.LongTensor(token_pos_list))).sum()
                            gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                            if token in primitive_vocab.key2id.keys():
                                token_id = primitive_vocab.key2id[token]
                                primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob
                            else:
                                hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                            'copy_prob': gated_copy_prob.data.item()})

                    if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                        unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                        token = hyp_unk_copy_info[unk_i]['token']
                        primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']

                for action_type in action_types:
                    if type(action_type.value) != str and action_type.name in ('NUMBER', 'STRING', 'NAME'):
                        if tem_hyp.stack[-1].nonterminal == first_dfa.from_rule and action_type.name == 'NUMBER':
                            continue
                        prod_id = copy.deepcopy(self.vocab.code.transition2id[action_type.name])
                        
                        if action_type.name == 'NAME':
                            prod_id = prod_id + [self.vocab.code.unk_id]
                            maxn_id_list = torch.topk(primitive_prob[hyp_id,prod_id], k=min(len(prod_id),self.args.NAME_TOKEN_NUM))[1]
                        else:
                            maxn_id_list = torch.topk(primitive_prob[hyp_id,prod_id], k=min(len(prod_id),2))[1]
                        for i in maxn_id_list:
                            maxn_id = prod_id[i]
                            prod_score = torch.log(primitive_prob[hyp_id, maxn_id]).data.item() #gen_from_vocab_prob
                            new_hyp_score = hyp.score + prod_score

                            applyrule_new_hyp_scores.append(new_hyp_score)
                            applyrule_new_hyp_prod_ids.append(maxn_id)
                            applyrule_prev_hyp_ids.append(hyp_id)
                            if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                                gentoken_new_hyp_unks.append(token)
                            else:
                                gentoken_new_hyp_unks.append('')
                    else: 
                        try:
                            prod_id = self.vocab.code.transition2id[action_type.name] if type(action_type.value) != str else self.vocab.code.transition2id[action_type.value]
                            prod_id = prod_id[0]
                        except:
                            continue
                        # tem_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob
                        prod_score = torch.log(primitive_prob[hyp_id, prod_id]).data.item() #gen_from_vocab_prob
                        new_hyp_score = hyp.score + prod_score

                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_new_hyp_prod_ids.append(prod_id)
                        applyrule_prev_hyp_ids.append(hyp_id)
                        gentoken_new_hyp_unks.append('')

            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses)))
            #                                                 k=new_hyp_scores.size(0))

            live_hyp_ids = []
            new_hypotheses = []
            # i = 0 
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                # it's an ApplyRule or Reduce action
                prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                prev_hyp = hypotheses[prev_hyp_id]

                prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                if prod_id < primitive_vocab.unk_id:
                    continue
                # ApplyRule action
                if prod_id == primitive_vocab.unk_id:
                    production = gentoken_new_hyp_unks[new_hyp_pos]
                    if production != '' and production not in self.grammar._pgen_grammar.reserved_syntax_strings.keys():
                        production = (production, 'NAME')
                    else:
                        # print('continue')
                        continue
                else:
                    production = self.vocab.code.id2word[prod_id]
                try:
                    token = PythonToken(PythonTokenTypes[production[1]], production[0],(1,0),'')
                except:
                    token = list(self.grammar._tokenize_lines([production[0]]))[0]

                new_hyp = prev_hyp.copy()
                new_hyp._add_token(token)
                new_hyp.token.append(production)
                if self.args.use_nonterminal and self.args.predict_with_nonterminal:
                    nonterminal_id = self.grammar.nonterminal2id[new_hyp.stack[-1].nonterminal]
                    new_hyp.score = new_hyp_score + self.args.alpha * apply_rule_log_prob[prev_hyp_id, nonterminal_id]
                else:
                    new_hyp.score = new_hyp_score

                if new_hyp.stack[-1].dfa.is_final and len(new_hyp.stack) == 1:
                    # add length normalization
                    new_hyp.score /= (t+1)
                    completed_hypotheses.append(new_hyp.completed_copy())
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)
                    # if t==args.decode_max_time_step-1:
                    #     new_hyp.score /= (t+1)
                    #     completed_hypotheses.append(new_hyp.completed_copy())

            if live_hyp_ids:
                out_new=[]
                hypotheses = new_hypotheses
                actions_tm1 = [[hyp.token[-1], hyp.stack[-1].nonterminal] for hyp in hypotheses]
                for i,id in enumerate(live_hyp_ids):
                    out_i=out[id]
                    embeds=[]
                    a_tm1, a_tm2 = actions_tm1[i]
                    if a_tm1:
                        a_tm1_embed = self.code_embed.weight[self.vocab.code[a_tm1]]
                        embeds.append(a_tm1_embed)
                    else:
                        embeds.append(zero_action_embed)
                    if self.args.feed_nonterminal:
                        if a_tm2:
                            a_tm2_embed = self.state_embed.weight[self.grammar.nonterminal2id[a_tm2]]
                            embeds.append(a_tm2_embed)
                        else:
                            embeds.append(zero_action_embed)

                    embeds=torch.cat(embeds,dim=-1).unsqueeze(dim=0)
                    out_i=torch.cat((out_i,embeds),dim=0)
                    out_new.append(out_i)
                out=out_new
                t += 1
            else:
                break
        
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # print([' '.join([t[0] for t in hyp.token]) for hyp in completed_hypotheses])

        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        grammar = load_grammar(version=saved_args.grammar_version)
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = cls(saved_args, vocab, grammar)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser

from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.nn.modules.sparse import Embedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SeqEncoder(LightningModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self._negative_value = -1e9

        self.encoder_layers = args.encoder_layers
        self.attn_heads = args.attn_heads
        self.ffn_dim = args.hidden_size
        self.dropout = args.dropout
        self.hidden = args.embed_size

        self.src_position = Position(d_model=self.hidden)
        self.transformer_encoder_layer \
            = TransformerEncoderLayer(d_model=self.hidden
            , nhead=self.attn_heads, dim_feedforward=self.ffn_dim
            , dropout=self.dropout, batch_first=True)

        self.order_encoder \
            = TransformerEncoder(encoder_layer=self.transformer_encoder_layer
            , num_layers=self.encoder_layers)

    def forward(
        self, 
        inputs: torch.Tensor,
        inputs_mask: torch.Tensor
    ) -> torch.Tensor:
        inputs = self.src_position(inputs)
        # [max path length; totul paths, embedding_size]
        batched_order = self.order_encoder(src=inputs, src_key_padding_mask=inputs_mask)
        return batched_order, inputs_mask

import math

class Position(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(Position, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, d_model])
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x, pos_list = None, pos_mask = None):
        if pos_list == None:
            x = x + self.pe[:,:x.size(1), :]
        else:
            for batch in range(x.size(0)):
                l = pos_list[batch]+2
                pos = self.pe.index_select(dim=1,index=l)[0]
                x[batch] = x[batch] + pos
        return self.dropout(x)

from typing import Tuple
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.sparse import Embedding

class GrammarDecoder(LightningModule):
    _negative_value = -1e9

    def __init__(
        self,
        args,
        feature_dim,
        encoder_output_dim,
        code_embed,
        state_embed=None,
    ):
        super().__init__()
        self.args = args
        self.state_embed = state_embed
        self.code_embed = code_embed
        # self.project_in_dim = nn.Linear(feature_dim, encoder_output_dim)
        self.ast_dim_to_src_dim = nn.Linear(feature_dim, encoder_output_dim)
        self.transformer_decoder_layer = TransformerDecoderLayer(d_model=encoder_output_dim, nhead=args.attn_heads
                                            , dim_feedforward=args.hidden_size
                                            , batch_first=True
                                            , dropout=args.dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer=self.transformer_decoder_layer
                                            , num_layers=args.decoder_layers)
        self.norm = nn.LayerNorm(encoder_output_dim)

        self.ast_decoder_to_query_vec = nn.Linear(encoder_output_dim
                , args.action_embed_size) 
        
        self.query_vec_to_action_embed = nn.Linear(args.action_embed_size
                , encoder_output_dim)
        self.query_vec_to_code_embed = nn.Linear(args.action_embed_size
                , encoder_output_dim)

        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(self.code_embed.weight.shape[0]).zero_())

        if self.args.use_nonterminal:
            self.nonterminal_readout_b = nn.Parameter(torch.FloatTensor(self.state_embed.weight.shape[0]).zero_())
            self.production_readout \
                    = lambda q: F.linear(torch.tanh(self.query_vec_to_action_embed(q))
                    , self.state_embed.weight,self.nonterminal_readout_b)

        self.tgt_token_readout \
                = lambda q: F.linear(torch.tanh(self.query_vec_to_code_embed(q))
                , self.code_embed.weight, self.tgt_token_readout_b)

        self.primitive_predictor = nn.Linear(args.action_embed_size, 2)
        self.primitive_pointer_net = PointerNet(query_vec_size=args.action_embed_size
                , src_encoding_size=encoder_output_dim)

        self.ast_position = Position(d_model=encoder_output_dim)

    def buffered_future_mask(self, tensor: torch.Tensor, dim: int):
        self._future_mask = torch.triu((tensor.new_ones(dim, dim)*float('-inf')), 1)
        return self._future_mask

    def forward(
        self,
        incomplete_features: torch.Tensor,  # [batch size; step; decoder size]
        input_features: torch.Tensor,  # [batch size; context size; decoder size]
        input_mask: torch.Tensor,  # [batch size; context size]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # apply_rule_prob, primitive_gen_prob, primitive_copy_prob
        incomplete_features = self.ast_dim_to_src_dim(incomplete_features)
        incomplete_features = self.ast_position(incomplete_features)
        # hidden -- [n layers; batch size; decoder size]
        # output -- [batch size; 1; decoder size]
        current_output = self.transformer_decoder(tgt=incomplete_features
                        , tgt_mask=self.buffered_future_mask(incomplete_features,incomplete_features.size(1))
                        , memory=input_features
                        , memory_key_padding_mask=input_mask
                        )
        # [batch size; vocab size]
        # current_output = self.norm(trans_output)
        query_vectors = self.ast_decoder_to_query_vec(current_output)
        # query_vectors = torch.tanh(query_vectors)
        if self.args.use_nonterminal:
            apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)
        predictor_prob = F.softmax(self.primitive_predictor(query_vectors), dim=-1)
        # [batch_size; max_ast_len; src_sent_len]
        primitive_copy_prob = self.primitive_pointer_net(input_features, input_mask, query_vectors)
        primitive_gen_prob = gen_from_vocab_prob
        primitive_copy_prob = primitive_copy_prob
        if self.args.use_nonterminal:
            return apply_rule_prob, primitive_gen_prob, primitive_copy_prob, predictor_prob
        else:
            return primitive_gen_prob, primitive_copy_prob, predictor_prob

class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, src_feature_dim)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(batch_size, query_vec_len, tgt_feature_dim)
        :return: Variable(batch_size, tgt_action_num, src_sent_len)
        """
        if self.attention_type == 'affine':
            src_encodings = self.src_encoding_linear(src_encodings)
        # [batch_size; tgt_feature_dim; src_sent_len]
        src_encodings = src_encodings.transpose(1,2)

        # [batch_size; query_vec_len; src_sent_len]
        weights = torch.matmul(query_vec, src_encodings)

        # [query_vec_len; batch_size; src_sent_len]
        weights = weights.permute(1,0,2)
        if src_token_mask is not None:
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            weights.data.masked_fill_(src_token_mask, -float('inf'))
        # [batch_size; query_vec_len; src_sent_len]
        weights = weights.permute(1,0,2)
        ptr_weights = F.softmax(weights, dim=-1)
        return ptr_weights

@Registrable.register('transformer_nopda_parser')
class TransformerNopdaParser(TransformerParser):
    def __init__(self, args, vocab, grammar):
        super().__init__(args, vocab, grammar)

    def parse(self, example, context=None, beam_size=5, debug=False):
        args = self.args
        primitive_vocab = self.vocab.code
        T = torch.cuda if args.cuda else torch
        batch = Batch([example], self.vocab, self.grammar)
        if self.args.cuda:
            batch.move_to_device('cuda')

        # [batch size; length]
        batch_idx_matrix = batch.src_sents_idx_matrix
        batch_input = self.src_embed(batch_idx_matrix)
        inputs_mask = batch.src_sents_mask
        # [batch size; length; hidden size]
        batched_features, attention_mask = self.encoder(batch_input, inputs_mask)
        src_idx, src_features, src_mask = batch_idx_matrix, batched_features, attention_mask

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(example.src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]))
        t = 0
        first_dfa = self.grammar._pgen_grammar.nonterminal_to_dfas[self.grammar._start_nonterminal][0]
        p = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
        p.stack = Stack([StackNode(first_dfa)])
        hypotheses = [p]
        completed_hypotheses = []
        out = []
        with torch.no_grad():
            x = Variable(self.new_tensor(1, self.input_dim).zero_())
            out.append(x)

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            current_input=torch.stack(out)
            batch_size=current_input.size(0)
            if self.args.use_nonterminal:
                apply_rule_prob, primitive_prob, primitive_copy_prob, primitive_predictor_prob = self.code_decoder(
                current_input, src_features.expand(batch_size,-1,-1), src_mask.expand(batch_size,-1))
                apply_rule_prob = apply_rule_prob[:,-1]
                primitive_prob = primitive_prob[:,-1]
                primitive_copy_prob = primitive_copy_prob[:,-1]
                primitive_predictor_prob = primitive_predictor_prob[:,-1]
                primitive_prob = primitive_prob*primitive_predictor_prob[:,:1]
                primitive_copy_prob = primitive_copy_prob*primitive_predictor_prob[:,1:]
                apply_rule_log_prob = torch.log(apply_rule_prob)
            else:
                primitive_prob, primitive_copy_prob, primitive_predictor_prob = self.code_decoder(
                current_input, src_features.expand(batch_size,-1,-1), src_mask.expand(batch_size,-1))
                primitive_prob = primitive_prob[:,-1]
                primitive_copy_prob = primitive_copy_prob[:,-1]
                primitive_predictor_prob = primitive_predictor_prob[:,-1]
                primitive_prob = primitive_prob*primitive_predictor_prob[:,:1]
                primitive_copy_prob = primitive_copy_prob*primitive_predictor_prob[:,1:]

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            for hyp_id, hyp in enumerate(hypotheses):
                gentoken_prev_hyp_ids.append(hyp_id)
                hyp_unk_copy_info = []

                if args.no_copy is False:
                    for token, token_pos_list in aggregated_primitive_tokens.items():
                        sum_copy_prob = torch.gather(primitive_copy_prob[hyp_id], 0, Variable(T.LongTensor(token_pos_list))).sum()
                        gated_copy_prob = primitive_predictor_prob[hyp_id, 1] * sum_copy_prob

                        if token in primitive_vocab.key2id.keys():
                            token_id = primitive_vocab.key2id[token]
                            primitive_prob[hyp_id, token_id] = primitive_prob[hyp_id, token_id] + gated_copy_prob
                        else:
                            hyp_unk_copy_info.append({'token': token, 'token_pos_list': token_pos_list,
                                                        'copy_prob': gated_copy_prob.data.item()})

                if args.no_copy is False and len(hyp_unk_copy_info) > 0:
                    unk_i = np.array([x['copy_prob'] for x in hyp_unk_copy_info]).argmax()
                    token = hyp_unk_copy_info[unk_i]['token']
                    primitive_prob[hyp_id, primitive_vocab.unk_id] = hyp_unk_copy_info[unk_i]['copy_prob']
                    gentoken_new_hyp_unks.append(token)
                else:
                    gentoken_new_hyp_unks.append(primitive_vocab.id2word[primitive_vocab.unk_id])
                    
            new_hyp_scores = None
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]).view(-1)

                new_hyp_scores = gen_token_new_hyp_scores
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses)))


            live_hyp_ids = []
            new_hypotheses = []
            # i = 0 
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                # it's an ApplyRule or Reduce action
                token_id = new_hyp_pos % primitive_prob.size(1)

                k = new_hyp_pos // primitive_prob.size(1)
                prev_hyp_id = gentoken_prev_hyp_ids[k]
                prev_hyp = hypotheses[prev_hyp_id]

                if token_id == primitive_vocab.unk_id:
                    if gentoken_new_hyp_unks:
                        production = gentoken_new_hyp_unks[k]
                        production = (production, 'NAME')
                else:
                    if token_id < primitive_vocab.unk_id:
                        continue
                    production = primitive_vocab.id2word[token_id.item()]

                new_hyp = prev_hyp.copy()
                new_hyp.token.append(production)
                new_hyp.score = new_hyp_score

                if production[1] == 'ENDMARKER':
                    # add length normalization
                    new_hyp.score /= (t+1)
                    completed_hypotheses.append(new_hyp.completed_copy(HasStack=False))
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                out_new=[]
                hypotheses = new_hypotheses
                actions_tm1 = [[hyp.token[-1], hyp.stack[-1].nonterminal] for hyp in hypotheses]
                for i,id in enumerate(live_hyp_ids):
                    out_i=out[id]
                    embeds=[]
                    a_tm1, a_tm2 = actions_tm1[i]
                    if a_tm1:
                        a_tm1_embed = self.code_embed.weight[self.vocab.code[a_tm1]]
                        embeds.append(a_tm1_embed)
                    else:
                        embeds.append(zero_action_embed)
                    if self.args.feed_nonterminal:
                        if a_tm2:
                            a_tm2_embed = self.state_embed.weight[self.grammar.nonterminal2id[a_tm2]]
                            embeds.append(a_tm2_embed)
                        else:
                            embeds.append(zero_action_embed)

                    embeds=torch.cat(embeds,dim=-1).unsqueeze(dim=0)
                    out_i=torch.cat((out_i,embeds),dim=0)
                    out_new.append(out_i)
                out=out_new
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break
        
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # print([' '.join([t[0] for t in hyp.token]) for hyp in completed_hypotheses])

        return completed_hypotheses