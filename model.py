"""
model.py — Transformer components built from scratch
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) @ V"""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask: (batch, 1, seq_len) or (batch, 1, 1, seq_len) for multi-head
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)
        return context, weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention built from scratch."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def split_heads(self, x):
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        batch, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.embed_dim)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast across heads

        context, weights = self.attention(Q, K, V, mask)
        context = self.combine_heads(context)
        out = self.W_o(context)
        return out, weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        return self.linear2(x)


class EncoderBlock(nn.Module):
    """One Transformer encoder block: MHA + FFN with residuals and LayerNorm."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        attn_out, weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, weights


class MiniTransformer(nn.Module):
    """
    Full mini Transformer encoder for binary classification.

    Pipeline:
      tokens → embedding → positional encoding
             → N × encoder blocks
             → mean pooling over valid tokens
             → linear classifier
             → binary prediction
    """

    def __init__(
        self, vocab_size, embed_dim, num_heads, ff_dim,
        num_layers, max_len, dropout=0.1,
        use_positional_encoding=True
    ):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens, mask):
        x = self.dropout(self.embedding(tokens))

        if self.use_positional_encoding:
            x = self.pos_encoding(x)

        for block in self.encoder_blocks:
            x, _ = block(x, mask)

        # Mean pooling over real tokens only
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x * mask_expanded
        pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)

        return self.classifier(pooled)