"""
Convert a pretrained checkpoint from TakHemlata/SSL_Anti-spoofing
into the format used by this project.

Strategy: Instead of loading the full HF model (which needs ~4GB RAM),
we save the converted SSL state dict and backend state dict separately.
At load time, app.py applies them.

Usage:
    python convert_pretrained.py <path_to_LA_model.pth> [--out checkpoints/v2_pretrained/best_model.pt]
"""

import argparse
import re
import gc
from collections import OrderedDict
from pathlib import Path

import torch


def build_fairseq_to_hf_map():
    return [
        (r"^feature_extractor\.conv_layers\.(\d+)\.0\.weight", r"feature_extractor.conv_layers.\1.conv.weight"),
        (r"^feature_extractor\.conv_layers\.(\d+)\.0\.bias", r"feature_extractor.conv_layers.\1.conv.bias"),
        (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.weight", r"feature_extractor.conv_layers.\1.layer_norm.weight"),
        (r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.bias", r"feature_extractor.conv_layers.\1.layer_norm.bias"),
        (r"^layer_norm\.weight", r"feature_projection.layer_norm.weight"),
        (r"^layer_norm\.bias", r"feature_projection.layer_norm.bias"),
        (r"^post_extract_proj\.weight", r"feature_projection.projection.weight"),
        (r"^post_extract_proj\.bias", r"feature_projection.projection.bias"),
        (r"^encoder\.pos_conv\.0\.weight_g", r"encoder.pos_conv_embed.conv.weight_g"),
        (r"^encoder\.pos_conv\.0\.weight_v", r"encoder.pos_conv_embed.conv.weight_v"),
        (r"^encoder\.pos_conv\.0\.bias", r"encoder.pos_conv_embed.conv.bias"),
        (r"^encoder\.layer_norm\.weight", r"encoder.layer_norm.weight"),
        (r"^encoder\.layer_norm\.bias", r"encoder.layer_norm.bias"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.k_proj", r"encoder.layers.\1.attention.k_proj"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.v_proj", r"encoder.layers.\1.attention.v_proj"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.q_proj", r"encoder.layers.\1.attention.q_proj"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.out_proj", r"encoder.layers.\1.attention.out_proj"),
        (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm", r"encoder.layers.\1.layer_norm"),
        (r"^encoder\.layers\.(\d+)\.fc1", r"encoder.layers.\1.feed_forward.intermediate_dense"),
        (r"^encoder\.layers\.(\d+)\.fc2", r"encoder.layers.\1.feed_forward.output_dense"),
        (r"^encoder\.layers\.(\d+)\.final_layer_norm", r"encoder.layers.\1.final_layer_norm"),
    ]


def convert_fairseq_key(key, patterns):
    for pat, repl in patterns:
        new_key, n = re.subn(pat, repl, key)
        if n > 0:
            return new_key
    return None


def convert_checkpoint(src_path, dst_path, num_layers=24):
    print(f"Loading source checkpoint: {src_path}")
    src_state = torch.load(src_path, map_location="cpu", weights_only=False)

    if "model" in src_state:
        src_state = src_state["model"]
    elif "state_dict" in src_state:
        src_state = src_state["state_dict"]

    print(f"Source has {len(src_state)} keys")

    ssl_prefix = "ssl_model.model."
    patterns = build_fairseq_to_hf_map()
    hf_ssl_state = OrderedDict()
    backend_state = OrderedDict()
    ssl_skipped = 0

    for key, val in src_state.items():
        if key.startswith(ssl_prefix):
            inner = key[len(ssl_prefix):]
            if any(inner.startswith(s) for s in ["quantizer", "project_q", "mask_emb", "final_proj"]):
                ssl_skipped += 1
                continue
            hf_key = convert_fairseq_key(inner, patterns)
            if hf_key:
                hf_ssl_state[hf_key] = val
            else:
                ssl_skipped += 1
        else:
            backend_state[key] = val

    del src_state
    gc.collect()

    print(f"SSL mapped: {len(hf_ssl_state)}, skipped: {ssl_skipped}")
    print(f"Backend keys: {len(backend_state)}")

    # Save both parts — app.py will load HF model and apply these on top
    out = {
        "ssl_state_dict": hf_ssl_state,
        "backend_state_dict": backend_state,
        "num_layers": num_layers,
        "arch": "xlsr_aasist_pretrained",
        "converted_from": str(src_path),
        "note": "Converted from TakHemlata/SSL_Anti-spoofing (XLS-R 300M + original AASIST, ASVspoof 2019 LA)",
    }

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst_path)
    ssl_params = sum(v.numel() for v in hf_ssl_state.values()) / 1e6
    be_params = sum(v.numel() for v in backend_state.values()) / 1e6
    print(f"\nSaved: {dst_path}")
    print(f"  SSL params: {ssl_params:.1f}M, Backend params: {be_params:.1f}M")
    print(f"  arch=xlsr_aasist_pretrained, layers={num_layers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SSL_Anti-spoofing checkpoint to Reveal AI format")
    parser.add_argument("src", help="Path to their pretrained .pth checkpoint")
    parser.add_argument("--out", default="./checkpoints/v2_pretrained/best_model.pt")
    parser.add_argument("--layers", type=int, default=24)
    args = parser.parse_args()
    convert_checkpoint(args.src, args.out, args.layers)
