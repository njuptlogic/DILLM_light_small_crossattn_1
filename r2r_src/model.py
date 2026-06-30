import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import numpy as np

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout_ratio)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size

        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths, enforce_sorted=True):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=enforce_sorted)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim, use_tilde=True):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.use_tilde = use_tilde
        if use_tilde:
            self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
            self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde and self.use_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn

class FusionProjection(nn.Module):
    """Gate-based fusion of visual and object features.

    Projects both streams to ``out_dim``, then learns a per-element gate that
    decides how much of each stream to keep.  Output shape: (batch, views, out_dim).
    """

    def __init__(self, vis_dim=640, obj_dim=640, out_dim=384):
        super(FusionProjection, self).__init__()
        self.vis_proj = nn.Linear(vis_dim, out_dim, bias=False)
        self.obj_proj = nn.Linear(obj_dim, out_dim, bias=False)
        self.gate = nn.Linear(out_dim + out_dim, out_dim)

    def forward(self, vis, obj):
        # vis: (batch, views, vis_dim)   obj: (batch, views, obj_dim)
        v = self.vis_proj(vis)                                       # (batch, views, out_dim)
        o = self.obj_proj(obj)                                       # (batch, views, out_dim)
        g = torch.sigmoid(self.gate(torch.cat([v, o], dim=-1)))      # (batch, views, out_dim)
        return g * v + (1 - g) * o                                   # gated fusion


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.use_fusion = args.fusion_proj

        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)

        if self.use_fusion:
            # --- Gate-based FusionProjection mode ---
            vis_dim = feature_size - args.angle_feat_size       # 640
            obj_dim = args.obj_dim                               # 640
            fused_dim = 384
            self.fusion_proj = FusionProjection(vis_dim, obj_dim, fused_dim)
            effective_feat_size = fused_dim + args.angle_feat_size  # 512
        else:
            # --- Original mode: raw features directly into LSTM ---
            effective_feat_size = feature_size                   # 768 (640+128)
        self.proj_feature_size = effective_feat_size

        self.lstm = nn.LSTMCell(effective_feat_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, effective_feat_size, use_tilde=False)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size, use_tilde=False)
        self.attention_layer_sub = SoftDotAttention(hidden_size, hidden_size, use_tilde=False)
        self.candidate_att_layer = SoftDotAttention(hidden_size + hidden_size + hidden_size, effective_feat_size, use_tilde=False)

        self.lin_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sm = nn.Softmax(dim=1)

        # Language-Conditioned Visual Attention (LangCondVA)
        self.use_cross_attn = args.cross_attn
        if self.use_cross_attn:
            self.lang_pre_attn = SoftDotAttention(hidden_size, hidden_size, use_tilde=False)
            self.vis_query_gate = nn.Linear(hidden_size + hidden_size, hidden_size)

    def forward(self, feature, cand_feat,
                h_0, c_0, subgoal_ctx, subgoal_mask,
                ctx, ctx_mask=None,
                already_dropfeat=False,
                obj_feat=None,
                cand_obj_feat=None):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        obj_feat: batch x 36 x obj_dim  (object semantic features for 36 views)
        cand_obj_feat: batch x cand x obj_dim  (object features for candidates)
        '''
        if self.use_fusion:
            # --- FusionProjection path: fuse vis+obj, re-attach angle ---
            angle_size = args.angle_feat_size
            vis_feat = feature[..., :-angle_size]          # (batch, 36, 640)
            angle_feat = feature[..., -angle_size:]         # (batch, 36, 128)

            cand_vis = cand_feat[..., :-angle_size]         # (batch, cand, 640)
            cand_angle = cand_feat[..., -angle_size:]       # (batch, cand, 128)

            if not already_dropfeat:
                vis_feat = self.drop_env(vis_feat)
                cand_vis = self.drop_env(cand_vis)

            if obj_feat is not None:
                fused = self.fusion_proj(vis_feat, obj_feat)
            else:
                fused = self.fusion_proj.vis_proj(vis_feat)
            feature = torch.cat([fused, angle_feat], dim=-1)

            if cand_obj_feat is not None:
                cand_fused = self.fusion_proj(cand_vis, cand_obj_feat)
            else:
                cand_fused = self.fusion_proj.vis_proj(cand_vis)
            cand_feat = torch.cat([cand_fused, cand_angle], dim=-1)
        else:
            # --- Original path: raw features directly ---
            if not already_dropfeat:
                feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])

        prev_h0_drop = self.drop(h_0)
        if self.use_cross_attn:
            lang_summary, _ = self.lang_pre_attn(prev_h0_drop, subgoal_ctx, subgoal_mask, output_tilde=False)
            g = torch.sigmoid(self.vis_query_gate(torch.cat([prev_h0_drop, lang_summary], dim=-1)))
            vis_query = g * prev_h0_drop + (1 - g) * lang_summary
            attn_feat, _ = self.feat_att_layer(vis_query, feature, output_tilde=False)
        else:
            attn_feat, _ = self.feat_att_layer(prev_h0_drop, feature, output_tilde=False)

        h_1, c_1 = self.lstm(attn_feat, (h_0, c_0))

        h_1_drop = self.drop(h_1)

        weighted_context, _ = self.attention_layer(h_1_drop, ctx, ctx_mask, output_tilde=False)
        weighted_context_sub, _ = self.attention_layer_sub(h_1_drop, subgoal_ctx, subgoal_mask, output_tilde=False)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        concat_input = torch.cat((h_1_drop, weighted_context), 1)
        att_concat_input = torch.cat((concat_input, weighted_context_sub), 1)

        _, logit = self.candidate_att_layer(att_concat_input, cand_feat, output_tilde=False, output_prob=False)

        return h_1, c_1, logit, attn_feat, weighted_context_sub

class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.W_1 = nn.Linear(36 + 36 + args.angle_feat_size + 640, 64)
        self.W_2 = nn.Linear(64, 1)
        self.activation_relu = nn.ReLU()
        self.activation_sigmoid = nn.Sigmoid()
        # self.sm = nn.Softmax(dim=1)

    def forward(self, obj_score, view_score, input_a_t, text_f):
        concat_input = torch.cat((obj_score, view_score, input_a_t, text_f), 1)
        h_dis = self.W_1(concat_input)
        h_dis = self.activation_relu(h_dis)
        h_dis = self.W_2(h_dis)
        out = self.activation_sigmoid(h_dis)
        return out

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


