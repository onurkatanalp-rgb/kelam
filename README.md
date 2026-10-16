# Kelam

A Turkish-first language model trained **from scratch**: own tokenizer, own architecture, no pretrained weights.
This repository contains the model, optimizer and TPU (JAX) training code. Model weights and data are not included.

*Sıfırdan eğitilen Türkçe dil modeli. Bu depoda model, optimizer ve TPU eğitim kodu var; model ağırlıkları ve veri yoktur.*

## What is here

| File | What it does |
|---|---|
| `model.py` | GPT in PyTorch: RMSNorm, RoPE, SwiGLU, tied embeddings, SDPA. Optional fine-grained **Mixture-of-Experts** layer (`UzmanKatmani`): many small experts + 1 shared expert, sigmoid router, auxiliary-loss-free load balancing (DeepSeek-V3 style), grouped matmul path for speed |
| `optimizer.py` | **Muon** for matrices and MoE experts + AdamW for embeddings, router and norms |
| `tpu/kelam_jax.py` | The same model in JAX (identical weight names). MoE uses `jax.lax.ragged_dot` |
| `tpu/optim_jax.py` | Muon + AdamW in JAX, same rules as `optimizer.py` |
| `tpu/egit_tpu.py` | TPU training: FSDP sharding, gradient accumulation with `lax.scan`, warmup-stable-decay schedule, checkpoint/resume, NaN/spike watchdog (rolls back and skips data), dead-expert warning, export to PyTorch |
| `tpu/esitlik_testi.py` | Numerical-equivalence tests PyTorch ⇔ JAX (logits, loss, gradients, expert loads, Muon) |
| `moe_testleri.py`, `optimizer_testleri.py` | Unit tests for MoE and Muon |
| `moe_karsilastir.py` | Same-budget comparison runs (dense vs MoE, AdamW vs Muon) |
| `sozluk.py` | Tokenizer wrapper and chat format |

Identifiers and comments are in Turkish.

## Results so far

| Experiment | Setup | Result |
|---|---|---|
| Dense pretraining | 336M params, ~3.3B Turkish tokens (FineWeb-2 tr + Wikipedia), context 1024 | val loss **2.802** |
| Decay to zero (WSD) | last 20% of steps, lr → 0 | val loss 3.007 → **2.802** |
| MoE vs dense | same 42M active params, 6000 steps × 8k tokens | 3.613 → **3.572**; gap grew during training (0.016 → 0.041) |
| Muon vs AdamW | MoE 98M total / 42M active, same budget | 3.567 → **3.421** (≈ 25–30% fewer tokens for the same loss) |
| MoE throughput (H100) | 1.55B total / 233M active, grouped matmul, compiled | 157k tok/s = 65% of the dense model with the same active size |
| TPU v5e-8 (Kaggle) | same MoE, JAX | *measuring* |
| PyTorch ⇔ JAX | `python -m tpu.esitlik_testi` | 7/7 pass (Muon identical in fp32) |

The 336M model was instruction-tuned into a working Turkish chat assistant (val loss 1.668).

## Plan

1. MoE + Muon, ~8B total / ~1B active parameters, ~30B Turkish tokens on TPU.
2. Data-quality filtering (small classifier trained on LLM-graded samples), model growth (stacking).
3. Python coding with execution feedback in a sandbox; retrieval-augmented answers with sources.

## Running the tests

```bash
python -m tpu.esitlik_testi      # PyTorch vs JAX
python moe_testleri.py
python optimizer_testleri.py
python -m tpu.egit_tpu --deneme  # CPU, 4 fake devices, synthetic data
```

Requirements: `torch`, `jax`, `optax`, `numpy`, `sentencepiece`.

## Data credits

FineWeb-2 (HuggingFaceFW, ODC-By) · Turkish Wikipedia (CC BY-SA).
