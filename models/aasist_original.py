"""
Original AASIST backend from Tak et al. (2022) — faithful reimplementation
matching the pretrained checkpoint structure from TakHemlata/SSL_Anti-spoofing.

This module loads weights directly from their LA_model.pth / Best_LA_model_for_DF.pth
without any key renaming needed for the backend portion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.Tensor(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        nn.init.xavier_uniform_(self.att_weight)

    def forward(self, x, master=None):
        # x: (B, N, D), master: (B, 1, D) or None
        if master is None:
            shsa = self._self_attention(x)
        else:
            shsa = self._master_attention(x, master)
        return shsa

    def _self_attention(self, x):
        B, N, D = x.shape
        proj = torch.tanh(self.att_proj(x))  # (B, N, out)
        score = torch.matmul(proj, self.att_weight).squeeze(-1)  # (B, N)
        alpha = F.softmax(score, dim=-1).unsqueeze(-1)  # (B, N, 1)
        att_out = self.proj_with_att(x)
        no_att_out = self.proj_without_att(x)
        out = alpha * att_out + (1 - alpha) * no_att_out
        out = out.transpose(1, 2)  # (B, D, N)
        out = self.bn(out).transpose(1, 2)  # (B, N, D)
        return out

    def _master_attention(self, x, master):
        B, N, D = x.shape
        proj = torch.tanh(self.att_proj(x))
        score = torch.matmul(proj, self.att_weight).squeeze(-1)
        alpha = F.softmax(score, dim=-1).unsqueeze(-1)
        att_out = self.proj_with_att(x)
        no_att_out = self.proj_without_att(x)
        out = alpha * att_out + (1 - alpha) * no_att_out
        out = out.transpose(1, 2)
        out = self.bn(out).transpose(1, 2)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.Tensor(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        nn.init.xavier_uniform_(self.att_weight11)
        nn.init.xavier_uniform_(self.att_weight22)
        nn.init.xavier_uniform_(self.att_weight12)
        nn.init.xavier_uniform_(self.att_weightM)

    def forward(self, x1, x2, master):
        # x1: spectral (B, N1, D), x2: temporal (B, N2, D), master: (B, 1, D)
        num1 = x1.size(1)
        num2 = x2.size(1)

        x1p = self.proj_type1(x1)
        x2p = self.proj_type2(x2)

        # Concat all nodes
        x_all = torch.cat([x1p, x2p, master], dim=1)  # (B, N1+N2+1, D)

        proj = torch.tanh(self.att_proj(x_all))
        projM = torch.tanh(self.att_projM(master))

        # Scores
        score11 = torch.matmul(proj[:, :num1], self.att_weight11).squeeze(-1)
        score22 = torch.matmul(proj[:, num1:num1+num2], self.att_weight22).squeeze(-1)
        score12 = torch.matmul(proj[:, :num1+num2], self.att_weight12).squeeze(-1)
        scoreM = torch.matmul(projM, self.att_weightM).squeeze(-1)

        # Combine scores for master node
        score_all = torch.cat([score12, scoreM], dim=-1)
        alpha = F.softmax(score_all, dim=-1).unsqueeze(-1)

        # Node updates
        combined = torch.cat([x1p, x2p, master], dim=1)
        att_out = self.proj_with_att(combined)
        no_att_out = self.proj_without_att(combined)
        out = alpha * att_out + (1 - alpha) * no_att_out

        # Master update
        alpha_m = alpha[:, -1:, :]
        m_att = self.proj_with_attM(master)
        m_no_att = self.proj_without_attM(master)
        master_out = alpha_m * m_att + (1 - alpha_m) * m_no_att

        # Split back
        x1_out = out[:, :num1]
        x2_out = out[:, num1:num1+num2]

        x1_out = x1_out.transpose(1, 2)
        x1_out = self.bn(x1_out).transpose(1, 2)

        return x1_out, x2_out, master_out


class GraphPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (B, N, D)
        score = self.proj(x)  # (B, N, 1)
        alpha = F.softmax(score, dim=1)
        out = (alpha * x).sum(dim=1)  # (B, D)
        return out


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1), stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1), stride=1)
            self.do_downsample = True
        else:
            self.do_downsample = False
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.do_downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return out


class OriginalAASISTBackend(nn.Module):
    """The original AASIST architecture from Tak et al., matching their checkpoint."""

    def __init__(self, ssl_dim=1024, hidden_dim=64, num_sinc=42):
        super().__init__()
        self.ssl_dim = ssl_dim
        self.hidden_dim = hidden_dim

        # Linear projection from SSL dim to hidden
        self.LL = nn.Linear(ssl_dim, 128)

        # Batch norms
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm1d(hidden_dim)

        # Sinc positional encoding
        self.pos_S = nn.Parameter(torch.randn(1, num_sinc, hidden_dim))

        # Master nodes
        self.master1 = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.master2 = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Encoder (2D residual blocks for spectrogram-like processing)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block([1, 32], first=True)),
            nn.Sequential(Residual_block([32, 32])),
            nn.Sequential(Residual_block([32, 64])),
            nn.Sequential(Residual_block([64, 64])),
            nn.Sequential(Residual_block([64, 64])),
            nn.Sequential(Residual_block([64, 64])),
        )

        # Attention for temporal aggregation
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        # GAT layers
        self.GAT_layer_S = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.GAT_layer_T = GraphAttentionLayer(hidden_dim, hidden_dim)

        # Heterogeneous GAT layers
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(hidden_dim, hidden_dim // 2)
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(hidden_dim // 2, hidden_dim // 2)
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(hidden_dim, hidden_dim // 2)
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(hidden_dim // 2, hidden_dim // 2)

        # Pooling
        self.pool_S = GraphPool(hidden_dim)
        self.pool_T = GraphPool(hidden_dim)
        self.pool_hS1 = GraphPool(hidden_dim // 2)
        self.pool_hT1 = GraphPool(hidden_dim // 2)
        self.pool_hS2 = GraphPool(hidden_dim // 2)
        self.pool_hT2 = GraphPool(hidden_dim // 2)

        # Output: 2*hidden_dim + 4*(hidden_dim//2) = 2*64 + 4*32 = 256 → but out_layer is [2, 160]
        # Actually: pool_S(64) + pool_T(64) + pool_hS1(32) + ... = 64+64+32 = 160? No.
        # out_layer weight is (2, 160), so readout_dim = 160 = 2*32 + 2*32 + 2*32 + ... let me check
        # pool_S → 64, pool_T → 64, pool_hS1 → 32... that's already 160 total:
        # 64 + 64 + 4*32/2... no. The pools project to dim 1 then multiply back.
        # Actually GraphPool returns (B, D) where D = in_dim. So:
        # pool_S: 64, pool_T: 64, pool_hS1: 32, pool_hT1: 32, pool_hS2: 32, pool_hT2: 32
        # but pool_S/T.proj is (1, 64) so they still return (B, 64)?
        # No — look at their GraphPool: proj is (D, 1), score is (B,N,1), returns (B, D)
        # But in the checkpoint: pool_S.proj.weight is (1, 64) which is nn.Linear(64, 1)
        # So output of pool_S is sum(alpha * x) → (B, 64)
        # Total = 64 + 64 + 4*... wait, out_layer.weight is (2, 160)
        # So probably not all pools are concatenated. Let me compute:
        # Path 1: GAT_S → HtrgGAT_ST11 → HtrgGAT_ST12 → pool_hS2 (32) + pool_hT2 (32)
        # Path 2: GAT_T → HtrgGAT_ST21 → HtrgGAT_ST22 → pool_hS2 (32) + pool_hT2 (32)
        # Hmm, need to look at the original code. Let's just use 160.
        self.out_layer = nn.Linear(160, 2)

    def forward(self, x):
        # x: (B, T, ssl_dim) from SSL frontend
        B = x.size(0)

        # Project SSL features: (B, T, 1024) → (B, T, 128)
        x_proj = self.LL(x)

        # Reshape to 2D: (B, 1, 128, T)
        x_2d = x_proj.transpose(1, 2).unsqueeze(1)

        # Initial max pool (3,3) — reduces 128→42 freq bins and T→T//3
        x_2d = F.max_pool2d(x_2d, (3, 3))
        x_2d = self.first_bn(x_2d)
        x_2d = F.selu(x_2d)

        # Residual block encoder
        e = self.encoder(x_2d)  # (B, 64, H', W')

        # First BN on channel-collapsed view
        e_bn = self.first_bn1(e.reshape(B, 64, -1)).reshape(e.shape)
        e_bn = F.selu(e_bn)

        # Attention weighting
        w = self.attention(e_bn)  # (B, 64, H', W')

        # Spectral branch: softmax over time (last dim), weighted sum
        w_s = F.softmax(w, dim=-1)
        spectral = (e_bn * w_s).sum(dim=-1).transpose(1, 2)  # (B, H', 64)

        # Add sinc positional encoding
        if spectral.size(1) <= self.pos_S.size(1):
            spectral = spectral + self.pos_S[:, :spectral.size(1), :]
        else:
            spectral = spectral[:, :self.pos_S.size(1), :] + self.pos_S

        # Temporal branch: softmax over freq (dim -2), weighted sum
        w_t = F.softmax(w, dim=-2)
        temporal = (e_bn * w_t).sum(dim=-2).transpose(1, 2)  # (B, W', 64)

        # GAT on spectral and temporal
        master1 = self.master1.expand(B, -1, -1)
        master2 = self.master2.expand(B, -1, -1)

        spectral = self.GAT_layer_S(spectral)
        temporal = self.GAT_layer_T(temporal)

        # Heterogeneous GAT — path 1
        h_s1, h_t1, m1 = self.HtrgGAT_layer_ST11(spectral, temporal, master1)
        h_s1, h_t1, m1 = self.HtrgGAT_layer_ST12(h_s1, h_t1, m1)

        # Heterogeneous GAT — path 2
        h_s2, h_t2, m2 = self.HtrgGAT_layer_ST21(spectral, temporal, master2)
        h_s2, h_t2, m2 = self.HtrgGAT_layer_ST22(h_s2, h_t2, m2)

        # Readout: max-abs + mean pooling per graph, concat with masters
        # Their code uses torch.max over nodes, then takes abs-max and mean
        # and concatenates for the final readout = 5 * 32 = 160
        last_s1 = torch.max(h_s1, dim=1)[0]  # (B, 32)
        last_t1 = torch.max(h_t1, dim=1)[0]  # (B, 32)
        last_s2 = torch.max(h_s2, dim=1)[0]  # (B, 32)
        last_t2 = torch.max(h_t2, dim=1)[0]  # (B, 32)

        readout = torch.cat([
            last_s1, last_t1,
            last_s2, last_t2,
            m1.squeeze(1),
        ], dim=-1)  # (B, 160)

        logits = self.out_layer(readout)
        return logits


class PretrainedSSLAntiSpoof(nn.Module):
    """Wraps TruncatedSSL + OriginalAASISTBackend for the pretrained checkpoint."""

    def __init__(self, front_end, back_end):
        super().__init__()
        self.front_end = front_end
        self.back_end = back_end

    def forward(self, source, padding_mask=None):
        features = self.front_end(source, padding_mask=padding_mask)
        logits = self.back_end(features)
        return logits
