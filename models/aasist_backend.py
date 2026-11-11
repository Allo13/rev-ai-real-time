import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention over a fully-connected temporal graph."""

    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))

    def forward(self, x):
        # x: (B, N, in_dim)
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
        h = h.permute(0, 2, 1, 3)  # (B, H, N, D)

        # Compute attention coefficients via additive mechanism
        e_src = (h * self.a_src.view(1, self.num_heads, 1, self.head_dim)).sum(dim=-1, keepdim=True)
        e_dst = (h * self.a_dst.view(1, self.num_heads, 1, self.head_dim)).sum(dim=-1, keepdim=True)
        attn = self.leaky_relu(e_src + e_dst.transpose(-2, -1))  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, h)  # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, out_dim)
        return out


class HeteroGraphLayer(nn.Module):
    """Heterogeneous graph layer with spectral and temporal sub-graphs."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.gat_spectral = GraphAttentionLayer(dim, dim, num_heads)
        self.gat_temporal = GraphAttentionLayer(dim, dim, num_heads)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, dim)
        # Spectral sub-graph: operates on the full sequence
        h_s = self.norm_s(self.gat_spectral(x) + x)

        # Temporal sub-graph: operates on a strided view for efficiency
        B, N, D = x.shape
        stride = max(1, N // 64)
        if stride > 1:
            idx = torch.arange(0, N, stride, device=x.device)
            x_sub = x[:, idx]
            h_t_sub = self.gat_temporal(x_sub)
            h_t = torch.zeros_like(x)
            h_t[:, idx] = h_t_sub
            # Interpolate for non-sampled positions
            if N > len(idx):
                h_t = F.interpolate(
                    h_t.transpose(1, 2), size=N, mode="linear", align_corners=False
                ).transpose(1, 2)
        else:
            h_t = self.gat_temporal(x)
        h_t = self.norm_t(h_t + x)

        # Gated fusion of spectral and temporal
        g = self.gate(torch.cat([h_s, h_t], dim=-1))
        fused = g * h_s + (1 - g) * h_t

        out = self.norm_out(self.ffn(fused) + fused)
        return out


class MaxGraphOperation(nn.Module):
    """Graph-level pooling via max operation with learned projection."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, dim)
        h = self.proj(x)
        pooled, _ = h.max(dim=1)  # (B, dim)
        return pooled


class AASISTBackend(nn.Module):
    """
    AASIST-style backend for SSL-based anti-spoofing.
    Takes (Batch, Time, 1024) from an SSL frontend, outputs (Batch, 2) logits.
    """

    def __init__(self, input_dim=1024, hidden_dim=160, num_heads=4, num_classes=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.graph_layers = nn.ModuleList([
            HeteroGraphLayer(hidden_dim, num_heads) for _ in range(2)
        ])

        self.mgo_spectral = MaxGraphOperation(hidden_dim)
        self.mgo_temporal = MaxGraphOperation(hidden_dim)

        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, 1024)
        h = self.projection(x)  # (B, T, hidden_dim)

        for layer in self.graph_layers:
            h = layer(h)

        # Multi-view graph-level readout
        p_max_s = self.mgo_spectral(h)   # (B, hidden_dim)
        p_max_t = self.mgo_temporal(h)   # (B, hidden_dim)

        # Attention-weighted pooling
        attn_w = F.softmax(self.attn_pool(h), dim=1)  # (B, T, 1)
        p_attn = (h * attn_w).sum(dim=1)  # (B, hidden_dim)

        # Concatenate all graph-level representations
        readout_in = torch.cat([p_max_s, p_max_t, p_attn], dim=-1)  # (B, hidden_dim*3)
        logits = self.readout(readout_in)  # (B, num_classes)
        return logits


class SSLAntiSpoofModel(nn.Module):
    """Unified model combining an SSL front-end with an anti-spoofing backend."""

    def __init__(self, front_end, back_end):
        super().__init__()
        self.front_end = front_end
        self.back_end = back_end

    def forward(self, source, padding_mask=None):
        features = self.front_end(source, padding_mask=padding_mask)
        logits = self.back_end(features)
        return logits
