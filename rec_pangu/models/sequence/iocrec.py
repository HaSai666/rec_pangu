# -*- ecoding: utf-8 -*-
# @ModuleName: iocrec
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/7/23 15:04
from typing import Dict
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math
from rec_pangu.models.base_model import SequenceBaseModel


class IOCRec(SequenceBaseModel):
    def __init__(self, enc_dict, config):
        super(IOCRec, self).__init__(enc_dict, config)
        self.initializer_range = config.get('initializer_range', 0.02)
        self.aug_views = 2
        self.tao = config.get('tao', 2)
        self.all_hidden = config.get('all_hidden', True)
        self.lamda = config.get('lamda', 0.1)
        self.k_intention = config.get('K', 4)
        self.layer_norm_eps = self.config.get('layer_norm_eps', 1e-12)
        self.hidden_dropout = self.config.get('hidden_dropout', 0.5)
        self.ffn_hidden = self.config.get('ffn_hidden', 128)
        self.num_blocks = self.config.get('num_blocks', 3)
        self.num_heads = self.config.get('num_heads', 2)
        self.attn_dropout = self.config.get('attn_dropout', 0.5)

        self.position_embedding = nn.Embedding(self.max_length, self.embedding_dim)
        self.input_layer_norm = nn.LayerNorm(self.embedding_dim, eps=self.layer_norm_eps)
        self.input_dropout = nn.Dropout(self.hidden_dropout)
        self.local_encoder = Transformer(embed_size=self.embedding_dim,
                                         ffn_hidden=self.ffn_hidden,
                                         num_blocks=self.num_blocks,
                                         num_heads=self.num_heads,
                                         attn_dropout=self.attn_dropout,
                                         hidden_dropout=self.hidden_dropout,
                                         layer_norm_eps=self.layer_norm_eps)
        self.global_seq_encoder = GlobalSeqEncoder(embed_size=self.embedding_dim,
                                                   max_len=self.max_length,
                                                   dropout=self.hidden_dropout)
        self.disentangle_encoder = DisentangleEncoder(k_intention=self.k_intention,
                                                      embed_size=self.embedding_dim,
                                                      max_len=self.max_length)
        self.data_augmenter = DataAugmenter(num_items=self.enc_dict[self.config['item_col']]['vocab_size'] - 1)
        self.nce_loss = InfoNCELoss(temperature=self.tao,
                                    similarity_type='dot')
        self.cross_entropy = nn.CrossEntropyLoss()

        # self.apply(self._init_weights)
        self.reset_parameters()

    def forward(self, data: Dict[str, torch.tensor], is_training: bool = True):
        """
        This method initializes the forward step to compute the user embeddings which will then be used for
        recommendations.

        Args:
            data (dict): a dictionary with input features as keys and the corresponding tensors as values .
            is_training (bool): a flag variable to set the mode of the model; default is True.

        Returns:
            dict: a dictionary with the user embeddings and model loss (if training) as keys and the corresponding
            tensors as values.
        """
        item_seq = data['hist_item_list']
        mask = data['hist_mask_list']
        seq_len = torch.sum(mask, dim=-1)

        local_seq_emb = self.local_seq_encoding(item_seq, seq_len, return_all=True)  # [B, L, D]
        global_seq_emb = self.global_seq_encoding(item_seq, seq_len)
        disentangled_intention_emb = self.disentangle_encoder(local_seq_emb, global_seq_emb, seq_len)  # [B, K, L, D]

        gather_index = seq_len.view(-1, 1, 1, 1).repeat(1, self.k_intention, 1, self.embedding_dim)
        user_emb = disentangled_intention_emb.gather(2, gather_index - 1).squeeze()  # [B, K, D]

        if is_training:
            # rec task
            item = data['target_item'].squeeze()
            candidates = self.item_emb.weight.unsqueeze(0)  # [1, num_items, D]
            logits = user_emb @ candidates.permute(0, 2, 1)  # [B, K, num_items]
            max_logits, _ = torch.max(logits, 1)
            rec_loss = self.cross_entropy(max_logits, item)

            # cl task
            B = item.shape[0]
            aug_seq_1 = self.data_augmenter.augment(item_seq)
            aug_seq_2 = self.data_augmenter.augment(item_seq)

            aug_local_emb_1 = self.local_seq_encoding(aug_seq_1, seq_len, return_all=self.all_hidden)
            aug_global_emb_1 = self.global_seq_encoding(aug_seq_1, seq_len)
            disentangled_intention_1 = self.disentangle_encoder(aug_local_emb_1, aug_global_emb_1, seq_len)
            disentangled_intention_1 = disentangled_intention_1.view(B * self.k_intention, -1)  # [B * K, L * D]

            aug_local_emb_2 = self.local_seq_encoding(aug_seq_2, seq_len, return_all=self.all_hidden)
            aug_global_emb_2 = self.global_seq_encoding(aug_seq_2, seq_len)
            disentangled_intention_2 = self.disentangle_encoder(aug_local_emb_2, aug_global_emb_2, seq_len)
            disentangled_intention_2 = disentangled_intention_2.view(B * self.k_intention, -1)  # [B * K, L * D]

            cl_loss = self.nce_loss(disentangled_intention_1, disentangled_intention_2)

            loss = rec_loss + self.lamda * cl_loss

            output_dict = {
                'loss': loss
            }
        else:
            output_dict = {
                'user_emb': user_emb
            }
        return output_dict

    def position_encoding(self, item_input):
        seq_embedding = self.item_emb(item_input)
        position = torch.arange(self.max_length, device=item_input.device).unsqueeze(0)
        position = position.expand_as(item_input).long()
        pos_embedding = self.position_embedding(position)
        seq_embedding += pos_embedding
        seq_embedding = self.input_layer_norm(seq_embedding)
        seq_embedding = self.input_dropout(seq_embedding)

        return seq_embedding

    def local_seq_encoding(self, item_seq, seq_len, return_all=False):
        seq_embedding = self.position_encoding(item_seq)
        out_seq_embedding = self.local_encoder(item_seq, seq_embedding)
        if not return_all:
            out_seq_embedding = self.gather_indexes(out_seq_embedding, seq_len - 1)
        return out_seq_embedding

    def global_seq_encoding(self, item_seq, seq_len):
        return self.global_seq_encoder(item_seq, seq_len, self.item_emb)


class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden_view1, aug_hidden_view2, mask=None):
        """
        Args:
            aug_hidden_view1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden_view2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden_view1.ndim > 2:
            # flatten tensor
            aug_hidden_view1 = aug_hidden_view1.view(aug_hidden_view1.size(0), -1)
            aug_hidden_view2 = aug_hidden_view2.view(aug_hidden_view2.size(0), -1)

        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden_view1, aug_hidden_view1)
            sim22 = self.cosinesim(aug_hidden_view2, aug_hidden_view2)
            sim12 = self.cosinesim(aug_hidden_view1, aug_hidden_view2)
        elif self.sim_type == 'dot':
            # calc similarity
            sim11 = aug_hidden_view1 @ aug_hidden_view1.t()
            sim22 = aug_hidden_view2 @ aug_hidden_view2.t()
            sim12 = aug_hidden_view1 @ aug_hidden_view2.t()
        # mask non-calc value
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')

        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        if mask is not None:
            cl_logits = torch.masked_fill(cl_logits, mask, float('-inf'))
        target = torch.arange(cl_logits.size(0)).long().to(aug_hidden_view1.device)
        cl_loss = self.criterion(cl_logits, target)

        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)


class GlobalSeqEncoder(nn.Module):
    def __init__(self, embed_size, max_len, dropout=0.5):
        super(GlobalSeqEncoder, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        self.Q_s = nn.Parameter(torch.randn(max_len, embed_size))
        self.K_linear = nn.Linear(embed_size, embed_size)
        self.V_linear = nn.Linear(embed_size, embed_size)

    def forward(self, item_seq, seq_len, item_embeddings):
        """
        Args:
            item_seq (tensor): [B, L]
            seq_len (tensor): [B]
            item_embeddings (tensor): [num_items, D], item embedding table

        Returns:
            global_seq_emb: [B, L, D]
        """
        item_emb = item_embeddings(item_seq)  # [B, L, D]
        item_key = self.K_linear(item_emb)
        item_value = self.V_linear(item_emb)

        attn_logits = self.Q_s @ item_key.permute(0, 2, 1)  # [B, L, L]
        attn_score = F.softmax(attn_logits, -1)
        global_seq_emb = self.dropout(attn_score @ item_value)

        return global_seq_emb


class DisentangleEncoder(nn.Module):
    def __init__(self, k_intention, embed_size, max_len):
        super(DisentangleEncoder, self).__init__()
        self.embed_size = embed_size

        self.intentions = nn.Parameter(torch.randn(k_intention, embed_size))
        self.pos_fai = nn.Embedding(max_len, embed_size)
        self.rou = nn.Parameter(torch.randn(embed_size, ))
        self.W = nn.Linear(embed_size, embed_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.layer_norm_3 = nn.LayerNorm(embed_size)
        self.layer_norm_4 = nn.LayerNorm(embed_size)
        self.layer_norm_5 = nn.LayerNorm(embed_size)

    def forward(self, local_item_emb, global_item_emb, seq_len):
        """
        Args:
            local_item_emb: [B, L, D]
            global_item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            disentangled_intention_emb: [B, K, L, D]
        """
        local_disen_emb = self.intention_disentangling(local_item_emb, seq_len)
        global_siden_emb = self.intention_disentangling(global_item_emb, seq_len)
        disentangled_intention_emb = local_disen_emb + global_siden_emb

        return disentangled_intention_emb

    def item2IntentionScore(self, item_emb):
        """
        Args:
            item_emb: [B, L, D]
        Returns:
            score: [B, L, K]
        """
        item_emb_norm = self.layer_norm_1(item_emb)  # [B, L, D]
        intention_norm = self.layer_norm_2(self.intentions).unsqueeze(0)  # [1, K, D]

        logits = item_emb_norm @ intention_norm.permute(0, 2, 1)  # [B, L, K]
        score = F.softmax(logits / math.sqrt(self.embed_size), -1)

        return score

    def item2AttnWeight(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            score: [B, L]
        """
        B, L = item_emb.size(0), item_emb.size(1)
        dev = item_emb.device
        item_query_row = item_emb[torch.arange(B).to(dev), seq_len - 1]  # [B, D]
        item_query_row += self.pos_fai(seq_len - 1) + self.rou
        item_query = self.layer_norm_3(item_query_row).unsqueeze(1)  # [B, 1, D]

        pos_fai_tensor = self.pos_fai(torch.arange(L).to(dev)).unsqueeze(0)  # [1, L, D]
        item_key_hat = self.layer_norm_4(item_emb + pos_fai_tensor)
        item_key = item_key_hat + torch.relu(self.W(item_key_hat))

        logits = item_query @ item_key.permute(0, 2, 1)  # [B, 1, L]
        logits = logits.squeeze() / math.sqrt(self.embed_size)
        score = F.softmax(logits, -1)

        return score

    def intention_disentangling(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B. L, D]
            seq_len: [B]
        Returns:
            item_disentangled_emb: [B, K, L, D]
        """
        # get score
        item2intention_score = self.item2IntentionScore(item_emb)
        item_attn_weight = self.item2AttnWeight(item_emb, seq_len)

        # get disentangled embedding
        score_fuse = item2intention_score * item_attn_weight.unsqueeze(-1)  # [B, L, K]
        score_fuse = score_fuse.permute(0, 2, 1).unsqueeze(-1)  # [B, K, L, 1]
        item_emb_k = item_emb.unsqueeze(1)  # [B, 1, L, D]
        disentangled_item_emb = self.layer_norm_5(score_fuse * item_emb_k)
        return disentangled_item_emb


class DataAugmenter:

    def __init__(self, num_items, beta_a=3, beta_b=3):
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.num_items = num_items

    def reorder_op(self, seq):
        ratio = torch.distributions.beta.Beta(self.beta_a, self.beta_b).sample().item()
        select_len = int(len(seq) * ratio)
        start = torch.randint(0, len(seq) - select_len + 1, (1,)).item()
        idx_range = torch.arange(len(seq))
        idx_range[start: start + select_len] = idx_range[start: start + select_len][torch.randperm(select_len)]
        return seq[idx_range]

    def mask_op(self, seq):
        ratio = torch.distributions.beta.Beta(self.beta_a, self.beta_b).sample().item()
        selected_len = int(len(seq) * ratio)
        mask = torch.full((len(seq),), False, dtype=torch.bool)
        mask[:selected_len] = True
        mask = mask[torch.randperm(len(mask))]
        seq[mask] = self.num_items
        return seq

    def augment(self, seqs):
        seqs = seqs.clone()
        for i, seq in enumerate(seqs):
            if torch.rand(1) > 0.5:
                seqs[i] = self.mask_op(seq.clone())
            else:
                seqs[i] = self.reorder_op(seq.clone())
        return seqs


class Transformer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_blocks, num_heads, attn_dropout, hidden_dropout,
                 layer_norm_eps=0.02, bidirectional=False):
        super(Transformer, self).__init__()
        self.bidirectional = bidirectional
        encoder_layer = EncoderLayer(embed_size=embed_size,
                                     ffn_hidden=ffn_hidden,
                                     num_heads=num_heads,
                                     attn_dropout=attn_dropout,
                                     hidden_dropout=hidden_dropout,
                                     layer_norm_eps=layer_norm_eps)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_blocks)])

    def forward(self, item_input, seq_embedding):
        """
        Only output the sequence representations of the last layer in Transformer.
        out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        mask = self.create_mask(item_input)
        for layer in self.encoder_layers:
            seq_embedding = layer(seq_embedding, mask)
        return seq_embedding

    def create_mask(self, input_seq):
        """
        Parameters:
            input_seq: torch.LongTensor, [batch_size, max_len]
        Return:
            mask: torch.BoolTensor, [batch_size, 1, max_len, max_len]
        """
        mask = (input_seq != 0).bool().unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, max_len]
        mask = mask.expand(-1, -1, mask.size(-1), -1)
        if not self.bidirectional:
            mask = torch.tril(mask)
        return mask

    def set_attention_direction(self, bidirection=False):
        self.bidirectional = bidirection


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, ffn_hidden, num_heads, attn_dropout, hidden_dropout, layer_norm_eps):
        super(EncoderLayer, self).__init__()

        self.attn_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.pff_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)

        self.self_attention = MultiHeadAttentionLayer(embed_size, num_heads, attn_dropout)
        self.pff = PointWiseFeedForwardLayer(embed_size, ffn_hidden)

        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.pff_out_drop = nn.Dropout(hidden_dropout)

    def forward(self, input_seq, inputs_mask):
        """
        input:
            inputs: torch.FloatTensor, [batch_size, max_len, embed_size]
            inputs_mask: torch.BoolTensor, [batch_size, 1, 1, max_len]
        return:
            out_seq_embed: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        out_seq, att_matrix = self.self_attention(input_seq, input_seq, input_seq, inputs_mask)
        input_seq = self.attn_layer_norm(input_seq + self.hidden_dropout(out_seq))
        out_seq = self.pff(input_seq)
        out_seq = self.pff_layer_norm(input_seq + self.pff_out_drop(out_seq))
        return out_seq


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_size, nhead, attn_dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.nhead = nhead

        if self.embed_size % self.nhead != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.embed_size, self.nhead)
            )
        self.head_dim = self.embed_size // self.nhead

        # Q K V input linear layer
        self.fc_q = nn.Linear(self.embed_size, self.embed_size)
        self.fc_k = nn.Linear(self.embed_size, self.embed_size)
        self.fc_v = nn.Linear(self.embed_size, self.embed_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc_o = nn.Linear(self.embed_size, self.embed_size)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim).float()))

    def forward(self, query, key, value, inputs_mask=None):
        """
        :param query: [query_size, max_len, embed_size]
        :param key: [key_size, max_len, embed_size]
        :param value: [key_size, max_len, embed_size]
        :param inputs_mask: [N, 1, max_len, max_len]
        :return: [N, max_len, embed_size]
        """
        batch_size = query.size(0)
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # [batch_size, n_head, max_len, head_dim]
        Q = Q.view(query.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        K = K.view(key.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))
        V = V.view(value.size(0), -1, self.nhead, self.head_dim).permute((0, 2, 1, 3))

        # calculate attention score
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if inputs_mask is not None:
            energy = energy.masked_fill(inputs_mask == 0, -1.e10)

        attention_prob = F.softmax(energy, dim=-1)
        attention_prob = self.attn_dropout(attention_prob)

        out = torch.matmul(attention_prob, V)  # [batch_size, n_head, max_len, head_dim]
        out = out.permute((0, 2, 1, 3)).contiguous()  # memory layout
        out = out.view((batch_size, -1, self.embed_size))
        out = self.fc_o(out)
        return out, attention_prob


class PointWiseFeedForwardLayer(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(PointWiseFeedForwardLayer, self).__init__()

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, inputs):
        out = self.fc2(F.gelu(self.fc1(inputs)))
        return out
