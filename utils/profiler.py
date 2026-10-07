import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


def profile_model(
    model: nn.Module, 
    sample_len: int = 64000, 
    device: Optional[torch.device] = None
) -> Tuple[float, str, str]:
    """
    Calculates model parameters, MACs, and GFLOPs using ptflops.
    
    Args:
        model (nn.Module): PyTorch model to profile
        sample_len (int): 1D Audio sequence sample length (default: 64000 samples = 4s at 16kHz)
        device (torch.device): CUDA or CPU device
        
    Returns:
        Tuple[float, str, str]: (gflops_value, macs_str, params_str)
    """
    if device is None:
        device = next(model.parameters()).device if any(model.parameters()) else torch.device("cpu")

    model.eval()

    # Custom 1D audio input constructor for ptflops as strictly specified
    custom_input_constructor = lambda input_shape: {'source': torch.randn(1, input_shape[0]).to(device)}

    if PTFLOPS_AVAILABLE:
        try:
            macs_str, params_str = get_model_complexity_info(
                model,
                input_res=(sample_len,),
                input_constructor=custom_input_constructor,
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
            
            # Also calculate numeric GFLOPs
            macs_num, _ = get_model_complexity_info(
                model,
                input_res=(sample_len,),
                input_constructor=custom_input_constructor,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            # 1 MAC is approximately 2 FLOPs -> GFLOPs = (MACs * 2) / 1e9
            gflops = (macs_num * 2) / 1e9
            return gflops, macs_str, params_str
        except Exception as e:
            print(f"[profiler] ptflops profiling notice: {e}")

    # Fallback profiling calculation if ptflops isn't available or fails
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_str = f"{total_params / 1e6:.2f} M"
    # Estimate GFLOPs based on parameter count and sequence time steps
    time_steps = sample_len // 320  # XLSR 16kHz stride downsampling factor
    gflops = (total_params * time_steps * 2) / 1e9
    macs_str = f"{gflops / 2:.2f} GMac"
    return gflops, macs_str, params_str


def benchmark_rtf_and_gflops(
    model: nn.Module, 
    sample_len: int = 64000, 
    sample_rate: int = 16000,
    warmup_trials: int = 20,
    eval_trials: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Isolated benchmarking script to measure Real-Time Factor (RTF) and GFLOPs securely without CPU wall-clock skew.
    
    Execution Protocol:
    1. Calculate GFLOPs using ptflops with custom 1D input constructor.
    2. Execute 20 warm-up forward passes inside a torch.no_grad() block to purge caches.
    3. Use torch.cuda.synchronize() immediately before and after the timed evaluation loop (100 trials).
    4. Calculate and return RTF, throughput, and GFLOPs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Audio clip length in seconds (e.g. 64000 samples / 16000 Hz = 4.0 seconds)
    audio_duration_sec = sample_len / float(sample_rate)
    dummy_input = torch.randn(1, sample_len, device=device)

    # Step 1: Calculate GFLOPs
    gflops, macs_str, params_str = profile_model(model, sample_len=sample_len, device=device)

    # Step 2: Warm-up loop (20 iterations inside torch.no_grad() to purge caches & initialize CUDA contexts)
    with torch.no_grad():
        for _ in range(warmup_trials):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Step 3: Timed Evaluation Loop (100 iterations)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(eval_trials):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    end_time = time.perf_counter()

    # Step 4: Calculate Metrics
    total_time_sec = end_time - start_time
    avg_latency_sec = total_time_sec / float(eval_trials)
    avg_latency_ms = avg_latency_sec * 1000.0
    
    # Real-Time Factor (RTF) = Average Processing Time / Total Audio Duration
    rtf = avg_latency_sec / audio_duration_sec

    results = {
        "device": str(device),
        "sample_len": sample_len,
        "audio_duration_sec": audio_duration_sec,
        "eval_trials": eval_trials,
        "total_eval_time_sec": total_time_sec,
        "avg_latency_ms": avg_latency_ms,
        "rtf": rtf,
        "gflops": gflops,
        "macs_str": macs_str,
        "params_str": params_str
    }

    print("\n" + "="*55)
    print(f"   REAL-TIME FACTOR (RTF) & GFLOPS BENCHMARK RESULTS   ")
    print("="*55)
    print(f" Device                 : {results['device']}")
    print(f" Audio Length           : {audio_duration_sec:.2f} seconds ({sample_len} samples @ {sample_rate}Hz)")
    print(f" Model Parameters       : {params_str}")
    print(f" MACs Complexity        : {macs_str}")
    print(f" Computational GFLOPs   : {gflops:.4f} GFLOPs")
    print(f" Average Latency        : {avg_latency_ms:.2f} ms / clip")
    print(f" Real-Time Factor (RTF) : {rtf:.6f} {'[REAL-TIME OK!]' if rtf < 1.0 else '[LATENCY EXCEEDED]'}")
    print("="*55 + "\n")

    return results


if __name__ == "__main__":
    # Test script standalone functionality
    from models.truncated_xlsr import MockXLSRModel, TruncatedXLSR
    from models.fake_mamba import FakeMambaClassifier, FakeMambaModel

    print("Running profiler self-test...")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mock_base = MockXLSRModel(num_layers=24, embed_dim=1024)
    front_end = TruncatedXLSR(mock_base, num_layers=12)
    back_end = FakeMambaClassifier(input_dim=1024, hidden_dim=144)
    full_model = FakeMambaModel(front_end, back_end)

    benchmark_rtf_and_gflops(full_model, sample_len=64000, device=dev)
