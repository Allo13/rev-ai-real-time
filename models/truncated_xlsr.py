import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MockXLSREncoderLayer(nn.Module):
    """Mock layer representing one transformer layer in XLSR-300M."""
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.attn = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.attn(x) + residual
        x = self.ffn(x) + x
        return x


class MockXLSREncoder(nn.Module):
    """Mock encoder containing transformer layers."""
    def __init__(self, num_layers: int = 24, embed_dim: int = 1024):
        super().__init__()
        self.layers = nn.ModuleList([MockXLSREncoderLayer(embed_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        states = [x]
        curr = x
        for layer in self.layers:
            curr = layer(curr)
            states.append(curr)
        return curr, states


class MockXLSRModel(nn.Module):
    """
    Mock Fairseq XLSR-300M model for offline usage, unit testing, 
    and environments without pre-trained fairseq checkpoint weights.
    """
    def __init__(self, num_layers: int = 24, embed_dim: int = 1024):
        super().__init__()
        # Conv feature extractor simulating 16kHz audio downsampling (stride 320 -> T_samples / 320)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5),
            nn.GELU(),
            nn.Conv1d(512, embed_dim, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=4, stride=2),
            nn.GELU()
        )
        self.encoder = MockXLSREncoder(num_layers=num_layers, embed_dim=embed_dim)

    def extract_features(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, mask: bool = False) -> Dict[str, Any]:
        """
        Mimics Fairseq XLSR Wav2Vec2 extract_features API.
        Args:
            source (torch.Tensor): Audio waveform of shape (Batch, Samples)
        Returns:
            dict: Containing 'x' and 'inner_states' list
        """
        if source.ndim == 2:
            x = source.unsqueeze(1)  # (Batch, 1, Samples)
        else:
            x = source
            
        feats = self.feature_extractor(x)  # (Batch, 1024, Time)
        feats = feats.transpose(1, 2)       # (Batch, Time, 1024)
        
        _, inner_states = self.encoder(feats)
        return {
            "x": inner_states[-1],
            "inner_states": inner_states  # inner_states[0] = conv output, inner_states[1..num_layers] = layer outputs
        }


class HFXLSRWrapper(nn.Module):
    """
    Wraps HuggingFace Wav2Vec2Model to expose the same extract_features API
    as the Fairseq XLSR-300M model, so the rest of the codebase stays unchanged.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        self.encoder = self.hf_model.encoder

    def extract_features(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, mask: bool = False) -> Dict[str, Any]:
        if source.ndim == 1:
            source = source.unsqueeze(0)
        # CNN feature extraction + projection
        extract_features = self.hf_model.feature_extractor(source)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, _ = self.hf_model.feature_projection(extract_features)
        # Apply positional conv embedding + layer norm (same as Wav2Vec2Encoder.forward)
        position_embeddings = self.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.encoder.layer_norm(hidden_states)
        hidden_states = self.encoder.dropout(hidden_states)
        # Manually iterate encoder layers to collect per-layer hidden states
        inner_states = [hidden_states]
        h = hidden_states
        for layer in self.encoder.layers:
            h = layer(h)
            inner_states.append(h)
        return {
            "x": inner_states[-1],
            "inner_states": inner_states,
        }


def load_xlsr_300m(checkpoint_path: Optional[str] = None):
    """
    Loads XLSR-300M via (in order of preference):
    1. Fairseq checkpoint from disk (requires fairseq)
    2. HuggingFace transformers (facebook/wav2vec2-xls-r-300m, auto-downloaded)
    3. MockXLSRModel fallback for offline testing
    """
    # --- Try Fairseq first ---
    default_locations = [
        checkpoint_path,
        os.path.join(".", "weights", "xlsr_53_56k.pt"),
        os.path.join(".", "models", "xlsr_53_56k.pt"),
    ]
    target_path = None
    for loc in default_locations:
        if loc and os.path.exists(loc):
            target_path = loc
            break

    if target_path:
        try:
            import fairseq
            print(f"[load_xlsr_300m] Loading official Fairseq XLSR-300M weights from: {target_path}")
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([target_path])
            xlsr = model[0]
            xlsr.eval()
            return xlsr
        except ImportError:
            print("[load_xlsr_300m] fairseq not installed, trying HuggingFace transformers...")
        except Exception as e:
            print(f"[load_xlsr_300m] Fairseq load failed ({e}), trying HuggingFace transformers...")

    # --- Try HuggingFace transformers ---
    try:
        from transformers import Wav2Vec2Model
        model_id = "facebook/wav2vec2-xls-r-300m"
        print(f"[load_xlsr_300m] Loading XLSR-300M via HuggingFace: {model_id}")
        print("[load_xlsr_300m] (First run downloads ~1.2 GB of weights, cached afterwards)")
        hf_model = Wav2Vec2Model.from_pretrained(model_id)
        hf_model.eval()
        print(f"[load_xlsr_300m] Loaded {model_id} successfully ({sum(p.numel() for p in hf_model.parameters())/1e6:.0f}M params)")
        return HFXLSRWrapper(hf_model)
    except ImportError:
        print("[load_xlsr_300m] transformers not installed.")
    except Exception as e:
        print(f"[load_xlsr_300m] HuggingFace load failed: {e}")

    # --- Fallback to mock ---
    print("[load_xlsr_300m] Falling back to MockXLSRModel (untrained, for testing only).")
    return MockXLSRModel(num_layers=24, embed_dim=1024)



class TruncatedXLSR(nn.Module):
    """
    Truncated XLSR-300M Front-End Wrapper with Learnable Layer Aggregation.
    
    Constraints & Specifications:
    - Initialized with copy.deepcopy of provided Fairseq XLSR model to prevent in-place mutation of base model.
    - Encoder layers physically sliced to num_layers.
    - Learnable layer weight parameter initialized as zeros of size num_layers.
    - Softmax-weighted sum over transformer layer inner_states.
    - Strictly returns tensor of shape (Batch, Time, 1024) with assertion check.
    """
    def __init__(self, base_xlsr_model: nn.Module, num_layers: int = 24):
        super().__init__()
        self.num_layers = num_layers
        
        # 1. Deepcopy base model to prevent in-place mutation across different ablation runs
        self.xlsr = copy.deepcopy(base_xlsr_model)
        
        # 2. Physically slice the encoder layers
        if hasattr(self.xlsr, 'encoder') and hasattr(self.xlsr.encoder, 'layers'):
            total_available = len(self.xlsr.encoder.layers)
            if num_layers > total_available:
                raise ValueError(f"Requested {num_layers} layers, but base model only has {total_available} layers.")
            self.xlsr.encoder.layers = self.xlsr.encoder.layers[:num_layers]
        else:
            raise AttributeError("Base XLSR model structure does not contain 'encoder.layers'.")
        
        # 3. Define learnable weight parameter for layer aggregation
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Truncated XLSR.
        
        Args:
            source (torch.Tensor): Audio waveform of shape (Batch, Samples)
            padding_mask (Optional[torch.Tensor]): Padding mask
            
        Returns:
            torch.Tensor: Weighted sum representation of shape (Batch, Time, 1024)
        """
        # Execute extract_features with mask=False
        res = self.xlsr.extract_features(source, padding_mask=padding_mask, mask=False)
        
        # Strict diagnostic check: verify result is a dictionary containing "inner_states" key
        if not isinstance(res, dict) or "inner_states" not in res:
            raise ValueError(
                "Diagnostic check failed: XLSR extract_features output must be a dictionary "
                f"containing the 'inner_states' key. Received type: {type(res)}, keys: {res.keys() if isinstance(res, dict) else None}"
            )
            
        inner_states = res["inner_states"]
        # inner_states[0] is feature extractor output.
        # inner_states[1:num_layers + 1] correspond to the transformer encoder layer outputs.
        selected_states = inner_states[1:self.num_layers + 1]
        
        if len(selected_states) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} inner states, but received {len(selected_states)}."
            )
        
        # Stack states along dimension 0: shape (num_layers, Batch, Time, 1024)
        stacked_states = torch.stack(selected_states, dim=0)
        
        # Compute softmax weights over layer_weights
        weights = F.softmax(self.layer_weights, dim=0)  # Shape: (num_layers,)
        
        # Reshape weights for broadcasting: (num_layers, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)
        
        # Compute weighted sum across layers: shape (Batch, Time, 1024)
        weighted_sum = torch.sum(stacked_states * weights, dim=0)
        
        # Strict assertion check on output shape
        assert weighted_sum.shape[-1] == 1024, (
            f"Expected channel dimension 1024, got {weighted_sum.shape[-1]} (Tensor shape: {weighted_sum.shape})"
        )
        
        return weighted_sum
