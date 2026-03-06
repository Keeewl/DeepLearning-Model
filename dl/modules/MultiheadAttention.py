import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    MultiheadAttention

    一个用 PyTorch 基础算子实现的多头注意力模块，
    输入输出风格尽量与 nn.MultiheadAttention 保持一致。

    参数:
        embed_dim:      输入/输出特征维度
        num_heads:      注意力头数
        dropout:        注意力权重上的 dropout
        bias:           线性层是否使用偏置
        batch_first:    是否使用 [B, T, C]，否则使用 [T, B, C]

    输入:
        query: [L, N, E] 或 [N, L, E]
        key:   [S, N, E] 或 [N, S, E]
        value: [S, N, E] 或 [N, S, E]

        attn_mask:
            shape 可为 [L, S] 或 [N*num_heads, L, S]
            True / 非0 表示该位置不可见

        key_padding_mask:
            shape 为 [N, S]
            True 表示该 key 位置是 padding，需要被 mask

        need_weights:
            是否返回注意力权重

        average_attn_weights:
            是否对多头注意力权重求平均
            True  -> [N, L, S]
            False -> [N, num_heads, L, S]

    输出:
        attn_output:
            与 query 形状对应
            [L, N, E] 或 [N, L, E]

        attn_weights:
            若 need_weights=True:
                average_attn_weights=True  -> [N, L, S]
                average_attn_weights=False -> [N, num_heads, L, S]
            否则为 None
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # 与官方风格接近：Q/K/V 分别做线性映射
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape_proj(self, x, seq_len, bsz):
        """
        将线性映射后的 [seq_len, batch_size, embed_dim]
        变形成 [batch_size, num_heads, seq_len, head_dim]
        """
        x = x.contiguous().view(seq_len, bsz, self.num_heads, self.head_dim)
        x = x.permute(1, 2, 0, 3).contiguous()
        return x

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        # 统一转成 [L, N, E] / [S, N, E]
        if self.batch_first:
            # [N, L, E] -> [L, N, E]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        src_len, bsz_k, embed_dim_k = key.shape
        src_len_v, bsz_v, embed_dim_v = value.shape

        if bsz != bsz_k or bsz != bsz_v:
            raise ValueError("query, key, and value must have the same batch size.")
        if embed_dim != self.embed_dim or embed_dim_k != self.embed_dim or embed_dim_v != self.embed_dim:
            raise ValueError("query, key, and value embedding dimensions must equal embed_dim.")
        if src_len != src_len_v:
            raise ValueError("key and value must have the same sequence length.")

        # 1) Q, K, V 线性映射
        q = self.q_proj(query)  # [L, N, E]
        k = self.k_proj(key)    # [S, N, E]
        v = self.v_proj(value)  # [S, N, E]

        # 2) 分头
        q = self._shape_proj(q, tgt_len, bsz)  # [N, H, L, D]
        k = self._shape_proj(k, src_len, bsz)  # [N, H, S, D]
        v = self._shape_proj(v, src_len, bsz)  # [N, H, S, D]

        # 3) 缩放点积注意力分数
        q = q * self.scaling
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [N, H, L, S]

        # 4) attn_mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [L, S] -> [1, 1, L, S]
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [N*H, L, S] -> [N, H, L, S]
                if attn_mask.shape[0] != bsz * self.num_heads:
                    raise ValueError(
                        f"attn_mask first dimension should be batch_size*num_heads={bsz * self.num_heads}, "
                        f"but got {attn_mask.shape[0]}."
                    )
                attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, src_len)
            else:
                raise ValueError("attn_mask must be 2D or 3D.")

            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
            else:
                attn_scores = attn_scores + attn_mask

        # 5) key_padding_mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, src_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(bsz, src_len)}, "
                    f"but got {key_padding_mask.shape}."
                )
            # [N, S] -> [N, 1, 1, S]
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask, float("-inf"))

        # 6) softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, H, L, S]
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 7) 加权求和
        attn_output = torch.matmul(attn_weights, v)  # [N, H, L, D]

        # 8) 拼接多头
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # [L, N, H, D]
        attn_output = attn_output.view(tgt_len, bsz, self.embed_dim)  # [L, N, E]

        # 9) 输出投影
        attn_output = self.out_proj(attn_output)  # [L, N, E]

        # 10) 恢复 batch_first
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # [N, L, E]

        # 11) 返回注意力权重
        if need_weights:
            if average_attn_weights:
                attn_weights_out = attn_weights.mean(dim=1)  # [N, L, S]
            else:
                attn_weights_out = attn_weights  # [N, H, L, S]
        else:
            attn_weights_out = None

        return attn_output, attn_weights_out