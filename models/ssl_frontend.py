import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class HFWav2VecWrapper(nn.Module):
    """Wraps any HuggingFace Wav2Vec2Model to expose an extract_features API
    that returns per-layer hidden states."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        self.encoder = self.hf_model.encoder
        self.embed_dim = self.hf_model.config.hidden_size
        self.num_layers = self.hf_model.config.num_hidden_layers

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
    ) -> Dict[str, Any]:
        if source.ndim == 1:
            source = source.unsqueeze(0)
        features = self.hf_model.feature_extractor(source)
        features = features.transpose(1, 2)
        hidden_states, _ = self.hf_model.feature_projection(features)
        position_embeddings = self.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.encoder.layer_norm(hidden_states)
        hidden_states = self.encoder.dropout(hidden_states)
        inner_states = [hidden_states]
        h = hidden_states
        for layer in self.encoder.layers:
            h = layer(h)
            inner_states.append(h)
        return {"x": inner_states[-1], "inner_states": inner_states}


def load_wav2vec2_base() -> HFWav2VecWrapper:
    """Loads facebook/wav2vec2-base (95M params, 768-dim, 12 layers)."""
    from transformers import Wav2Vec2Model

    model_id = "facebook/wav2vec2-base"
    print(f"[ssl_frontend] Loading {model_id}...")
    hf_model = Wav2Vec2Model.from_pretrained(model_id, low_cpu_mem_usage=True)
    gc.collect()
    hf_model.eval()
    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"[ssl_frontend] Loaded {model_id} ({n_params / 1e6:.0f}M params, dim={hf_model.config.hidden_size})")
    return HFWav2VecWrapper(hf_model)


def load_xlsr_large() -> HFWav2VecWrapper:
    """Loads facebook/wav2vec2-xls-r-300m (315M params, 1024-dim, 24 layers)."""
    from transformers import Wav2Vec2Model

    model_id = "facebook/wav2vec2-xls-r-300m"
    print(f"[ssl_frontend] Loading {model_id}...")
    hf_model = Wav2Vec2Model.from_pretrained(model_id, low_cpu_mem_usage=True)
    gc.collect()
    hf_model.eval()
    n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"[ssl_frontend] Loaded {model_id} ({n_params / 1e6:.0f}M params, dim={hf_model.config.hidden_size})")
    return HFWav2VecWrapper(hf_model)


class TruncatedSSL(nn.Module):
    """Truncated SSL front-end with learnable layer aggregation.
    Works with any HFWav2VecWrapper (wav2vec2-base 768-dim or XLSR 1024-dim)."""

    def __init__(self, base_model: HFWav2VecWrapper, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = base_model.embed_dim
        self.ssl = base_model

        total = len(self.ssl.encoder.layers)
        if num_layers > total:
            raise ValueError(f"Requested {num_layers} layers but model has {total}")
        self.ssl.encoder.layers = self.ssl.encoder.layers[:num_layers]

        self.layer_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(
        self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = self.ssl.extract_features(source, padding_mask=padding_mask, mask=False)
        inner_states = res["inner_states"]
        # inner_states[0] = post-CNN features, [1..num_layers] = transformer layer outputs
        selected = inner_states[1 : self.num_layers + 1]
        stacked = torch.stack(selected, dim=0)  # (L, B, T, D)
        w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        out = torch.sum(stacked * w, dim=0)  # (B, T, D)
        return out


# Convenience aliases matching the old naming convention
TruncatedW2V2 = TruncatedSSL
