# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils


class RelativeMultiheadAttention(nn.Module):
    """Relative Multi-headed attention.

    See "Self-Attention with Relative Position Representations" for more details.
    """

    def __init__(self, embed_dim, num_heads, max_relative_length, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, k_only=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_relative_length = max_relative_length
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.k_only = k_only
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.relative_position_keys = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.head_dim))
        if not self.k_only:
            self.relative_position_values = Parameter(torch.Tensor(2 * self.max_relative_length + 1, self.head_dim))

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        nn.init.xavier_uniform_(self.relative_position_keys)
        if not self.k_only:
            nn.init.xavier_uniform_(self.relative_position_values)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        relative_positions_matrix = self._generate_relative_positions_matrix(
            src_len, self.max_relative_length, incremental_state
        )

        if self.k_only:
            relation_keys = F.embedding(relative_positions_matrix.long().cuda(), self.relative_position_keys)
        else:
            relation_keys = F.embedding(relative_positions_matrix.long().cuda(), self.relative_position_keys)
            relation_values = F.embedding(relative_positions_matrix.long().cuda(), self.relative_position_values)

        relative_attn_weights = self._relative_attention_inner(q, k, relation_keys, transpose=True)
        assert list(relative_attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(relative_attn_weights.size(0), 1, 1)
            relative_attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            relative_attn_weights = relative_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                relative_attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    relative_attn_weights.float()
                ).type_as(relative_attn_weights)
            else:
                relative_attn_weights = relative_attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(relative_attn_weights)  # FP16 support: cast to float and back
                relative_attn_weights = relative_attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        relative_attn_weights = utils.softmax(
            relative_attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(relative_attn_weights)
        relative_attn_weights = F.dropout(relative_attn_weights, p=self.dropout, training=self.training)
        # key only mode
        if self.k_only:
            attn = torch.bmm(relative_attn_weights, v)
        # original implementation
        else:
            attn = self._relative_attention_inner(relative_attn_weights, v, relation_values, transpose=False)

        #attn = torch.bmm(relative_attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            relative_attn_weights = relative_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            relative_attn_weights = relative_attn_weights.sum(dim=1) / self.num_heads
        else:
            relative_attn_weights = None

        return attn, relative_attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def _generate_relative_positions_matrix(self, length, max_relative_length, incremental_state):
        if not incremental_state:
            # training process
            range_vec = torch.arange(length)
            range_mat = range_vec.repeat(length, 1)
            distance_mat = range_mat - range_mat.transpose(0, 1)
        else:
            distance_mat = torch.range(-length + 1, 0).view(1, -1)

        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_length, max_relative_length)

        # position difference.
        final_mat = distance_mat_clipped + max_relative_length

        return final_mat

    def _relative_attention_inner(self, x, y, z, transpose=True):
        """Relative position-aware dot-product attention inner calculation.

        This batches matrix multiply calculations to avoid unnecessary broadcasting.

        Args:
          x: Tensor with shape [batch_size*heads, length, length or depth].
          y: Tensor with shap e [batch_size*heads, length, depth].
          z: Tensor with shape [length, length, depth].
          transpose: Whether to tranpose inner matrices of y and z. Should be true if
              last dimension of x is depth, not length.

        Returns:
          A Tensor with shape [batch_size*heads, length, length or depth].

          wq: this function actually does 'X(Y+Z)', where Z is vector,
          but factor above formular as: 'XY + XZ'
        """
        batch_size_mul_head = x.size()[0]
        length = z.size()[0]
        # print(batch_size_mul_head, length)
        # xy_matmul is [batch_size*heads, length, length or depth]
        if transpose:
            y = y.transpose(1, 2)
        xy_matmul = torch.bmm(x, y)
        # x_t is [length, batch_size * heads, length or depth]
        x_t = x.transpose(0, 1)
        # x_tz_matmul is [length, batch_size * heads, length or depth]
        if transpose:
            z = z.transpose(1, 2)
        x_tz_matmul = torch.bmm(x_t, z).transpose(0, 1).view(batch_size_mul_head, length, -1)

        attn = xy_matmul + x_tz_matmul

        return attn