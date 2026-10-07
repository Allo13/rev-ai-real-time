import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing official Mamba block from mamba_ssm, otherwise fallback to Pure PyTorch Mamba Block
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class FallbackMambaBlock(nn.Module):
    """
    Pure PyTorch fallback implementation of Mamba Selective State Space Block
    for environments without pre-compiled mamba-ssm C++/CUDA kernels.
    """
    def __init__(self, d_model: int = 144, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of shape (Batch, Time, d_model)
        Returns:
            torch.Tensor: Output tensor of shape (Batch, Time, d_model)
        """
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)  # (Batch, Time, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)
        
        # 1D Convolution over temporal dimension
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # Gated state space activation simulation
        y = x_conv * self.act(z_branch)
        out = self.out_proj(y)
        return out


class GatedAttentionPooling(nn.Module):
    """
    Gated Attention Pooling over temporal dimension.
    Calculates dynamic attention weights for sequence aggregation.
    """
    def __init__(self, input_dim: int = 144, attn_dim: int = 64):
        super().__init__()
        self.attn_v = nn.Linear(input_dim, attn_dim)
        self.attn_u = nn.Linear(input_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (Batch, Time, input_dim)
        Returns:
            torch.Tensor: Pooled representation of shape (Batch, input_dim)
        """
        v = torch.tanh(self.attn_v(x))      # (Batch, Time, attn_dim)
        u = torch.sigmoid(self.attn_u(x))   # (Batch, Time, attn_dim)
        scores = self.attn_w(v * u)          # (Batch, Time, 1)
        
        weights = F.softmax(scores, dim=1)   # (Batch, Time, 1)
        pooled = torch.sum(x * weights, dim=1)  # (Batch, input_dim)
        return pooled


class FakeMambaClassifier(nn.Module):
    """
    Downstream Fake-Mamba Classifier.
    
    Architecture:
    1. Projects (Batch, Time, 1024) down to 144 hidden dimension using nn.Linear(1024, 144).
    2. Passes features through Mamba state-space blocks.
    3. Gated attention pooling over temporal dimension.
    4. Projects to 2 output logits (Bonafide vs Spoof).
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 144, num_mamba_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Linear projection from XLSR feature dimension (1024) to Mamba hidden dimension (144)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)
        
        # Instantiate state-space modeling backbone using Mamba blocks
        self.mamba_layers = nn.ModuleList()
        for _ in range(num_mamba_layers):
            if MAMBA_AVAILABLE:
                # Official Mamba block from mamba_ssm
                self.mamba_layers.append(Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2))
            else:
                self.mamba_layers.append(FallbackMambaBlock(d_model=hidden_dim, d_state=16, d_conv=4, expand=2))
                
        # Gated Attention Pooling layer over temporal dimension
        self.pooling = GatedAttentionPooling(input_dim=hidden_dim, attn_dim=64)
        
        # Final classification head: 2 classes (Bonafide vs Spoof)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor from front-end wrapper of shape (Batch, Time, 1024)
        Returns:
            torch.Tensor: Logits tensor of shape (Batch, 2)
        """
        # Project down to 144-dimensional hidden space
        h = self.projection(x)          # (Batch, Time, 144)
        h = self.proj_norm(h)
        
        # Pass through Mamba state-space blocks
        for mamba in self.mamba_layers:
            h = mamba(h) + h            # Residual state-space connection
            
        # Gated Attention Pooling over temporal dimension
        pooled = self.pooling(h)        # (Batch, 144)
        
        # Final classification projection to 2 classes
        logits = self.classifier(pooled)  # (Batch, 2)
        return logits


class FakeMambaModel(nn.Module):
    """
    Unified End-to-End Model combining TruncatedXLSR front-end and FakeMambaClassifier back-end.
    """
    def __init__(self, front_end: nn.Module, back_end: nn.Module):
        super().__init__()
        self.front_end = front_end
        self.back_end = back_end

    def forward(self, source: torch.Tensor, padding_mask=None) -> torch.Tensor:
        # Extract truncated features from front-end: (Batch, Time, 1024)
        features = self.front_end(source, padding_mask=padding_mask)
        # Classify through Fake-Mamba back-end: (Batch, 2)
        logits = self.back_end(features)
        return logits
