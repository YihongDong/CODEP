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
from components.dataset import Batch
from components.utils import update_args, init_arg_parser
import components.nn_utils as nn_utils
from pyparser import PyParser
from pyparser import load_grammar
import copy


@Registrable.register('lstm_parser')
class LSTMParser(nn.Module):
    def __init__(self, args, vocab, grammar):
        super(LSTMParser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.grammar = grammar
        self.grammar.nonterminal2id = {nonterminal: i for i, nonterminal in enumerate(self.grammar._pgen_grammar.nonterminal_to_dfas.keys())}
        self.grammar.id2nonterminal = {i: nonterminal for i, nonterminal in enumerate(self.grammar._pgen_grammar.nonterminal_to_dfas.keys())}

        # Embedding layers
        # source token embedding
        self.src_embed = nn.Embedding(len(vocab.source), args.embed_size)

        # embedding nonterminal
        if args.use_nonterminal or self.args.feed_nonterminal:
            self.nonterminal_embed = nn.Embedding(len(self.grammar._pgen_grammar.nonterminal_to_dfas.keys()), args.action_embed_size)

        # embedding table for code tokens
        self.code_embed = nn.Embedding(len(vocab.code), args.action_embed_size)

        nn.init.xavier_normal_(self.src_embed.weight.data)
        if args.use_nonterminal:
            nn.init.xavier_normal_(self.nonterminal_embed.weight.data)
        nn.init.xavier_normal_(self.code_embed.weight.data)

        # LSTMs
        if args.lstm == 'lstm':
            self.encoder_lstm = nn.LSTM(args.embed_size, int(args.hidden_size / 2), bidirectional=True)

            input_dim = args.action_embed_size  # previous action
            input_dim += args.action_embed_size * (self.args.feed_nonterminal)

            input_dim += args.att_vec_size * (not args.no_input_feed)  # input feeding

            self.decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)
        else:
            raise ValueError('Unknown LSTM type %s' % args.lstm)

        if args.no_copy is False:
            # pointer net for copying tokens from source side
            self.src_pointer_net = nn_utils.PointerNet(query_vec_size=args.att_vec_size, src_encoding_size=args.hidden_size)

            # given the decoder's hidden state, predict whether to copy or generate a target primitive token
            # output: [p(gen(token)) | s_t, p(copy(token)) | s_t]

            self.primitive_predictor = nn.Linear(args.att_vec_size, 2)

        if args.code_token_label_smoothing:
            self.label_smoothing = nn_utils.LabelSmoothing(args.code_token_label_smoothing, len(self.vocab.code), ignore_indices=[0, 1, 2])

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's hidden space

        self.att_src_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)

        self.att_vec_linear = nn.Linear(args.hidden_size + args.hidden_size, args.att_vec_size, bias=False)

        # bias for predicting ApplyConstructor and GenToken actions
        self.nonterminal_readout_b = nn.Parameter(torch.FloatTensor(len(self.grammar._pgen_grammar.nonterminal_to_dfas.keys())).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab.code)).zero_())

        if args.no_query_vec_to_action_map:
            # if there is no additional linear layer between the attentional vector (i.e., the query vector)
            # and the final softmax layer over target actions, we use the attentional vector to compute action
            # probabilities

            assert args.att_vec_size == args.action_embed_size
            self.nonterminal_readout = lambda q: F.linear(q, self.nonterminal_embed.weight, self.nonterminal_readout_b)
            self.tgt_token_readout = lambda q: F.linear(q, self.code_embed.weight, self.tgt_token_readout_b)
        else:
            # by default, we feed the attentional vector (i.e., the query vector) into a linear layer without bias, and
            # compute action probabilities by dot-producting the resulting vector and (GenToken, ApplyConstructor) action embeddings
            # i.e., p(action) = query_vec^T \cdot W \cdot embedding

            self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            if args.query_vec_to_action_diff_map:
                # use different linear transformations for GenToken and ApplyConstructor actions
                self.query_vec_to_primitive_embed = nn.Linear(args.att_vec_size, args.embed_size, bias=args.readout == 'non_linear')
            else:
                self.query_vec_to_primitive_embed = self.query_vec_to_action_embed

            self.read_out_act = torch.tanh if args.readout == 'non_linear' else nn_utils.identity

            self.nonterminal_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                         self.nonterminal_embed.weight, self.nonterminal_readout_b)
            self.tgt_token_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_primitive_embed(q)),
                                                        self.code_embed.weight, self.tgt_token_readout_b)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def encode(self, src_sents_var, src_sents_len):
        # (tgt_query_len, batch_size, embed_size)
        # apply word dropout
        if self.training and self.args.word_dropout:
            mask = Variable(self.new_tensor(src_sents_var.size()).fill_(1. - self.args.word_dropout).bernoulli().long())
            src_sents_var = src_sents_var * mask + (1 - mask) * self.vocab.source.unk_id

        src_token_embed = self.src_embed(src_sents_var)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len)

        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # src_encodings: (batch_size, tgt_query_len, hidden_size)
        src_encodings = src_encodings.permute(1, 0, 2)

        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, Variable(self.new_tensor(h_0.size()).zero_())

    def score(self, examples, return_encode_state=False):
        batch = Batch(examples, self.grammar, self.vocab, copy=self.args.no_copy is False, cuda=self.args.cuda)

        # src_encodings: (batch_size, src_sent_len, hidden_size * 2)
        # (last_state, last_cell, dec_init_vec): (batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encode(batch.src_sents_var, batch.src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)

        # query vectors are sufficient statistics used to compute action probabilities
        # query_vectors: (tgt_action_len, batch_size, hidden_size)

        # if use supervised attention
        if self.args.sup_attention:
            query_vectors, att_prob = self.decode(batch, src_encodings, dec_init_vec)
        else:
            query_vectors = self.decode(batch, src_encodings, dec_init_vec)


        if self.args.use_nonterminal:
            # (tgt_action_len, batch_size, grammar_size)
            apply_rule_prob = F.softmax(self.nonterminal_readout(query_vectors), dim=-1)

            # probabilities of target (gold-standard) ApplyRule actions
            # (tgt_action_len, batch_size)
            tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
                                            index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)

        #### compute generation and copying probabilities

        # (tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
                                                         index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)

        if self.args.no_copy:
            # mask positions in action_prob that are not used

            if self.training and self.args.code_token_label_smoothing:
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

                tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                    gen_from_vocab_prob.log(),
                    batch.primitive_idx_matrix)
            else:
                tgt_primitive_gen_from_vocab_log_prob = tgt_primitive_gen_from_vocab_prob.log()

            # (tgt_action_len, batch_size)
            action_prob = tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask
            if self.args.use_nonterminal:
                grammar_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask
        else:
            # binary gating probabilities between generating or copying a primitive token
            # (tgt_action_len, batch_size, 2)
            primitive_predictor = F.softmax(self.primitive_predictor(query_vectors), dim=-1)

            # pointer network copying scores over source tokens
            # (tgt_action_len, batch_size, src_sent_len)
            primitive_copy_prob = self.src_pointer_net(src_encodings, batch.src_token_mask, query_vectors)

            # marginalize over the copy probabilities of tokens that are same
            # (tgt_action_len, batch_size)
            tgt_primitive_copy_prob = torch.sum(primitive_copy_prob * batch.primitive_copy_token_idx_mask, dim=-1)

            # mask positions in action_prob that are not used
            # (tgt_action_len, batch_size)
            action_mask_pad = torch.eq(batch.gen_token_mask + batch.primitive_copy_mask, 0.)
            action_mask = 1. - action_mask_pad.float()

            # (tgt_action_len, batch_size)
            action_prob = primitive_predictor[:, :, 0] * tgt_primitive_gen_from_vocab_prob * batch.gen_token_mask + \
                          primitive_predictor[:, :, 1] * tgt_primitive_copy_prob * batch.primitive_copy_mask

            # avoid nan in log
            action_prob.data.masked_fill_(action_mask_pad.data, 1.e-7)

            action_prob = action_prob.log() * action_mask
            if self.args.use_nonterminal:
                grammar_mask_pad = torch.eq(batch.apply_rule_mask, 0.)
                grammar_mask = 1. - grammar_mask_pad.float()
                grammar_prob = tgt_apply_rule_prob * batch.apply_rule_mask
                grammar_prob.data.masked_fill_(grammar_mask_pad.data, 1.e-7)
                grammar_prob = grammar_prob.log() * grammar_mask

        scores = torch.sum(action_prob, dim=0)
        if self.args.use_nonterminal:
            scores += self.args.alpha * torch.sum(grammar_prob, dim=0)

        returns = [scores]

        if self.args.sup_attention:
            returns.append(att_prob)
        if return_encode_state: returns.append(last_state)

        return returns

    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_token_mask=None, return_att_weight=False):
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_encodings, src_encodings_att_linear,
                                                     mask=src_token_mask)

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else: return (h_t, cell_t), att_t

    def decode(self, batch, src_encodings, dec_init_vec):
        batch_size = len(batch)
        args = self.args

        h_tm1 = dec_init_vec

        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        att_vecs = []
        history_states = []
        att_probs = []
        att_weights = []

        for t in range(batch.max_code_num):
            # the input to the decoder LSTM is a concatenation of multiple signals
            # [
            #   embedding of previous action -> `a_tm1_embed`,
            #   previous attentional vector -> `att_tm1`,
            #   embedding of the current frontier (parent) constructor (rule) -> `parent_production_embed`,
            #   embedding of the frontier (parent) field -> `parent_field_embed`,
            #   embedding of the ASDL type of the frontier field -> `parent_field_type_embed`,
            #   LSTM state of the parent action -> `parent_states`
            # ]

            if t == 0:
                x = Variable(self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_(), requires_grad=False)
            else:
                a_tm1_embeds = []
                a_tm2_embeds = []
                for example in batch.examples:
                    # action t - 1
                    if t < len(example.tgt_code):
                        a_tm1 = example.tgt_code[t - 1]
                        a_tm1_embed = self.code_embed.weight[self.vocab.code[a_tm1]]
                        if self.args.feed_nonterminal:
                            a_tm2 = example.tgt_grammar[t - 1]
                            a_tm2_embed = self.nonterminal_embed.weight[self.grammar.nonterminal2id[a_tm2]]
                    else:
                        a_tm1_embed = zero_action_embed
                        if self.args.feed_nonterminal:
                           a_tm2_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)
                    if self.args.feed_nonterminal:
                        a_tm2_embeds.append(a_tm2_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]
                if self.args.feed_nonterminal:
                    a_tm2_embeds = torch.stack(a_tm2_embeds)
                    inputs.append(a_tm2_embeds)
                
                if args.no_input_feed is False:
                    inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, src_encodings,
                                                         src_encodings_att_linear,
                                                         src_token_mask=batch.src_token_mask,
                                                         return_att_weight=True)

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)
            att_weights.append(att_weight)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        att_vecs = torch.stack(att_vecs, dim=0)
        if args.sup_attention:
            return att_vecs, att_probs
        else: return att_vecs

    def parse(self, example, context=None, beam_size=5, debug=False):
        src_sent=example.src_sent
        args = self.args
        primitive_vocab = self.vocab.code
        T = torch.cuda if args.cuda else torch

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

        # Variable(1, src_sent_len, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(src_sent)])
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        h_tm1 = dec_init_vec

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]))

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0
        first_dfa = self.grammar._pgen_grammar.nonterminal_to_dfas[self.grammar._start_nonterminal][0]
        p = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
        p.stack = Stack([StackNode(first_dfa)])
        hypotheses = [p]
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).zero_())
            else:
                actions_tm1 = [[hyp.token[-1], hyp.stack[-1].nonterminal] for hyp in hypotheses]

                a_tm1_embeds = []
                a_tm2_embeds = []
                for a_tm1, a_tm2 in actions_tm1:
                    if a_tm1:
                        a_tm1_embed = self.code_embed.weight[self.vocab.code[a_tm1]]
                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                    if self.args.feed_nonterminal:
                        if a_tm2:
                            a_tm2_embed = self.nonterminal_embed.weight[self.grammar.nonterminal2id[a_tm2]]
                            a_tm2_embeds.append(a_tm2_embed)
                        else:
                            a_tm2_embeds.append(zero_action_embed)
                    
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]
                if self.args.feed_nonterminal:
                    a_tm2_embeds = torch.stack(a_tm2_embeds)
                    inputs.append(a_tm2_embeds)

                if args.no_input_feed is False:
                    inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)
            if self.args.use_nonterminal:
                # Variable(batch_size, grammar_size)
                # apply_rule_log_prob = torch.log(F.softmax(self.nonterminal_readout(att_t), dim=-1))
                apply_rule_log_prob = F.log_softmax(self.nonterminal_readout(att_t), dim=-1)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            if args.no_copy:
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                # Variable(batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

                # if src_unk_pos_list:
                #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

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
                # i+=1
                # if i >= beam_size:
                #     break

            if live_hyp_ids:
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                t += 1
            else:
                break
        
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # print([' '.join([t[0] for t in hyp.token]) for hyp in completed_hypotheses], ' '.join([t[0] for t in example.tgt_code]))

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

@Registrable.register('lstm_nopda_parser')
class LSTMNopdaParser(LSTMParser):
    def __init__(self, args, vocab, grammar):
        super().__init__(args, vocab, grammar)

    def parse(self, example, context=None, beam_size=5, debug=False):
        src_sent = example.src_sent
        args = self.args
        primitive_vocab = self.vocab.code
        T = torch.cuda if args.cuda else torch

        src_sent_var = nn_utils.to_input_variable([src_sent], self.vocab.source, cuda=args.cuda, training=False)

        # Variable(1, src_sent_len, hidden_size * 2)
        src_encodings, (last_state, last_cell) = self.encode(src_sent_var, [len(src_sent)])
        # (1, src_sent_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        h_tm1 = dec_init_vec

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        with torch.no_grad():
            hyp_scores = Variable(self.new_tensor([0.]))

        # For computing copy probabilities, we marginalize over tokens with the same surface form
        # `aggregated_primitive_tokens` stores the position of occurrence of each source token
        aggregated_primitive_tokens = OrderedDict()
        for token_pos, token in enumerate(src_sent):
            aggregated_primitive_tokens.setdefault(token, []).append(token_pos)

        t = 0
        first_dfa = self.grammar._pgen_grammar.nonterminal_to_dfas[self.grammar._start_nonterminal][0]
        p = PyParser(self.grammar._pgen_grammar, error_recovery=True, start_nonterminal=self.grammar._start_nonterminal)
        p.stack = Stack([StackNode(first_dfa)])
        hypotheses = [p]
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            hyp_num = len(hypotheses)

            # (hyp_num, src_sent_len, hidden_size * 2)
            exp_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))
            # (hyp_num, src_sent_len, hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, src_encodings_att_linear.size(1), src_encodings_att_linear.size(2))

            if t == 0:
                with torch.no_grad():
                    x = Variable(self.new_tensor(1, self.decoder_lstm.input_size).zero_())
            else:
                actions_tm1 = [hyp.token[-1] for hyp in hypotheses]

                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        a_tm1_embed = self.code_embed.weight[self.vocab.code[a_tm1]]
                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                    
                a_tm1_embeds = torch.stack(a_tm1_embeds)
                inputs = [a_tm1_embeds]

                if args.no_input_feed is False:
                    inputs.append(att_tm1)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_src_encodings,
                                             exp_src_encodings_att_linear,
                                             src_token_mask=None)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(att_t), dim=-1)

            if args.no_copy:
                primitive_prob = gen_from_vocab_prob
            else:
                # Variable(batch_size, src_sent_len)
                primitive_copy_prob = self.src_pointer_net(src_encodings, None, att_t.unsqueeze(0)).squeeze(0)

                # Variable(batch_size, 2)
                primitive_predictor_prob = F.softmax(self.primitive_predictor(att_t), dim=-1)

                # Variable(batch_size, primitive_vocab_size)
                primitive_prob = primitive_predictor_prob[:, 0].unsqueeze(1) * gen_from_vocab_prob

                # if src_unk_pos_list:
                #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

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
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                # it's a GenToken action
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
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break
        
        completed_hypotheses.sort(key=lambda hyp: -hyp.score)
        # print([' '.join([t[0] for t in hyp.token]) for hyp in completed_hypotheses])

        return completed_hypotheses