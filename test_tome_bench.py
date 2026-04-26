"""
Comprehensive benchmark for Token Merging (ToMe) on HuggingFace ViT models.

Covers:
  1. Speed benchmark — original vs smashed across batch sizes
  2. Quality check  — original top-1 must appear in smashed top-5
  3. Attention backends — sdpa, eager, flash_attention_2 (fp16)

Usage:
    uv run python test_tome_bench.py
"""

import copy
import time

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

from pruna import SmashConfig, smash

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_IDS = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]
BATCH_SIZES = [8, 32, 128]
R_VALUE = 16
N_WARMUP = 10
N_RUNS = 100
N_QUALITY_SAMPLES = 100

# Attention backends to test: (name, attn_implementation kwarg, dtype)
ATTN_BACKENDS = [
    ("sdpa", "sdpa", torch.float32),
    ("eager", "eager", torch.float32),
    ("flash_attention_2", "flash_attention_2", torch.float16),
]

SEPARATOR = "=" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_input(batch_size: int, processor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a normalised random input batch on DEVICE."""
    dummy = torch.randn(batch_size, 3, 224, 224, device=DEVICE, dtype=dtype)
    mean = torch.tensor(processor.image_mean, device=DEVICE, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=DEVICE, dtype=dtype).view(1, 3, 1, 1)
    return (dummy - mean) / std


def _smash_model(hf_model, device=DEVICE):
    """Apply ToMe with the shared SmashConfig."""
    smash_config = SmashConfig(device=device)
    smash_config.add("token_merging")
    smash_config.add(dict(token_merging_r=R_VALUE))
    return smash(model=copy.deepcopy(hf_model), smash_config=smash_config).to(device).eval()


def benchmark_model(model, input_batch, n_warmup=N_WARMUP, n_runs=N_RUNS, label="model"):
    """Time *n_runs* forward passes with proper CUDA synchronisation."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(pixel_values=input_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(pixel_values=input_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_runs) * 1000
    throughput = (n_runs * input_batch.shape[0]) / elapsed
    print(f"  {label:<20s}: {avg_ms:7.2f} ms/batch | {throughput:8.1f} img/s")
    return avg_ms


# ---------------------------------------------------------------------------
# 1. Speed benchmark
# ---------------------------------------------------------------------------
def run_speed_benchmark():
    """Benchmark original vs ToMe across models and batch sizes."""
    print(f"\n{SEPARATOR}")
    print("PART 1 — SPEED BENCHMARK  (default sdpa backend, fp32)")
    print(SEPARATOR)

    for model_id in MODEL_IDS:
        print(f"\nModel: {model_id}  |  r={R_VALUE}")
        print("-" * 60)
        processor = AutoImageProcessor.from_pretrained(model_id)
        hf_model = ViTForImageClassification.from_pretrained(model_id).to(DEVICE).eval()
        smashed = _smash_model(hf_model)

        for bs in BATCH_SIZES:
            inp = _make_input(bs, processor)
            print(f"\n  batch_size={bs}")
            orig_ms = benchmark_model(hf_model, inp, label="Original ViT")
            tome_ms = benchmark_model(smashed, inp, label="ToMe ViT")
            speedup = orig_ms / tome_ms
            tag = "FASTER" if speedup > 1 else "SLOWER"
            print(f"  {'-> speedup':<20s}: {speedup:.3f}x ({tag}, {abs(speedup - 1) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# 2. Quality comparison
# ---------------------------------------------------------------------------
def run_quality_comparison():
    """Compare top-1/top-1 and top-1-in-top-5 agreement between original and smashed."""
    print(f"\n{SEPARATOR}")
    print(f"PART 2 — QUALITY CHECK  ({N_QUALITY_SAMPLES} samples)")
    print(SEPARATOR)

    dataset = load_dataset("timm/mini-imagenet", split="test")

    for model_id in MODEL_IDS:
        print(f"\nModel: {model_id}  |  r={R_VALUE}")
        print("-" * 60)
        processor = AutoImageProcessor.from_pretrained(model_id)
        hf_model = ViTForImageClassification.from_pretrained(model_id).to(DEVICE).eval()
        smashed = _smash_model(hf_model)

        top1_top5_failures = []
        top1_top1_failures = []
        with torch.no_grad():
            for i in range(N_QUALITY_SAMPLES):
                img = dataset[i]["image"].convert("RGB")
                inp = processor(img, return_tensors="pt")["pixel_values"].to(DEVICE)
                orig_top1 = hf_model(inp).logits.argmax(-1).item()
                tome_logits = smashed(inp).logits
                tome_top1 = tome_logits.argmax(-1).item()
                tome_top5 = tome_logits.topk(5).indices[0].tolist()
                if orig_top1 != tome_top1:
                    top1_top1_failures.append(i)
                if orig_top1 not in tome_top5:
                    top1_top5_failures.append(i)

        top1_passed = N_QUALITY_SAMPLES - len(top1_top1_failures)
        top5_passed = N_QUALITY_SAMPLES - len(top1_top5_failures)
        top1_status = "PASS" if not top1_top1_failures else "WARN"
        top5_status = "PASS" if not top1_top5_failures else "WARN"
        print(f"  top-1 == top-1     : {top1_passed}/{N_QUALITY_SAMPLES} samples passed  [{top1_status}]")
        print(f"  top-1 in top-5     : {top5_passed}/{N_QUALITY_SAMPLES} samples passed  [{top5_status}]")
        if top1_top1_failures:
            print(f"  top-1 mismatch indices: {top1_top1_failures}")
        if top1_top5_failures:
            print(f"  top-5 miss indices    : {top1_top5_failures}")


# ---------------------------------------------------------------------------
# 3. Attention-backend comparison
# ---------------------------------------------------------------------------
def run_attention_backend_benchmark():
    """Compare speedup across attention implementations."""
    print(f"\n{SEPARATOR}")
    print("PART 3 — ATTENTION BACKEND COMPARISON  (batch_size=32)")
    print(SEPARATOR)

    for model_id in MODEL_IDS:
        print(f"\nModel: {model_id}  |  r={R_VALUE}")
        print("-" * 60)
        processor = AutoImageProcessor.from_pretrained(model_id)

        for backend_name, attn_impl, dtype in ATTN_BACKENDS:
            print(f"\n  backend={backend_name}  dtype={dtype}")
            try:
                hf_model = ViTForImageClassification.from_pretrained(
                    model_id, attn_implementation=attn_impl, torch_dtype=dtype,
                ).to(DEVICE).eval()
                smashed = _smash_model(hf_model)

                inp = _make_input(32, processor, dtype=dtype)

                orig_ms = benchmark_model(hf_model, inp, label=f"Original ({backend_name})")
                tome_ms = benchmark_model(smashed, inp, label=f"ToMe ({backend_name})")
                speedup = orig_ms / tome_ms
                tag = "FASTER" if speedup > 1 else "SLOWER"
                print(f"  {'-> speedup':<20s}: {speedup:.3f}x ({tag}, {abs(speedup - 1) * 100:.1f}%)")
            except Exception as e:
                print(f"  SKIPPED — {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Config: r={R_VALUE}, warmup={N_WARMUP}, runs={N_RUNS}")

    run_speed_benchmark()
    run_quality_comparison()
    run_attention_backend_benchmark()

    print(f"\n{SEPARATOR}")
    print("ALL BENCHMARKS COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
