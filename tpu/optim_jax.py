"""Muon + AdamW (JAX). optimizer.py (PyTorch) ile ayni kurallar, ayni sayilar (tpu/esitlik_testi.py kontrol eder).

Bolusum (optimizer.py optimizer_kur ile ayni):
  1 boyutlu (normlar)                     -> AdamW, agirlik azaltma 0
  gomme, yonlendirici                     -> AdamW, agirlik azaltma 0.1
  diger matrisler ve MoE uzmanlari        -> Muon (tur='muon') ya da AdamW 0.1 (tur='adamw')
  sapma (MoE dengesi)                     -> optimizer disi (denge_guncelle ile degisir)
"""
import math

import jax
import jax.numpy as jnp

B1, B2, EPS = 0.9, 0.95, 1e-8
MOMENTUM, AGIRLIK_AZALTMA = 0.95, 0.1
NS_TURU = jnp.bfloat16  # optimizer.py NS_TURU ile ayni


def sinif(ad, sekil, tur):
    if ad.endswith("sapma"):
        return None
    if len(sekil) < 2:
        return "adamw0"
    if "gomme" in ad or "yonlendirici" in ad or tur != "muon":
        return "adamw"
    return "muon"


def dikgenlestir(G, adim=5):
    """Newton-Schulz (optimizer.py ile ayni katsayilar ve bf16 hesap)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(NS_TURU)
    devrik = X.shape[-2] > X.shape[-1]
    if devrik:
        X = jnp.swapaxes(X, -2, -1)
    X = X / (jnp.sqrt(jnp.sum(jnp.square(X.astype(jnp.float32)), axis=(-2, -1), keepdims=True)).astype(X.dtype) + 1e-7)
    for _ in range(adim):
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if devrik:
        X = jnp.swapaxes(X, -2, -1)
    return X


def kur(a, tur="muon"):
    durum = {"adim": jnp.zeros((), jnp.int32)}
    for ad, p in a.items():
        s = sinif(ad, p.shape, tur)
        if s == "muon":
            durum[ad] = {"m": jnp.zeros_like(p, jnp.float32)}
        elif s:
            durum[ad] = {"m": jnp.zeros_like(p, jnp.float32), "v": jnp.zeros_like(p, jnp.float32)}
    return durum


def adim(a, egim, durum, lr, tur="muon"):
    """Bir guncelleme. lr: bu adimin ogrenme hizi (takvim disarida hesaplanir). Donus: (a, durum)."""
    yeni_a, yeni_d = dict(a), {"adim": durum["adim"] + 1}
    t = (durum["adim"] + 1).astype(jnp.float32)
    for ad, p in a.items():
        s = sinif(ad, p.shape, tur)
        if s is None:
            continue
        g = egim[ad].astype(jnp.float32)
        p32 = p.astype(jnp.float32)
        if s == "muon":
            m = MOMENTUM * durum[ad]["m"] + g
            guncelleme = dikgenlestir(g + MOMENTUM * m).astype(jnp.float32)
            olcek = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
            p32 = p32 * (1 - lr * AGIRLIK_AZALTMA) - lr * olcek * guncelleme
            yeni_d[ad] = {"m": m}
        else:
            wd = AGIRLIK_AZALTMA if s == "adamw" else 0.0
            m = B1 * durum[ad]["m"] + (1 - B1) * g
            v = B2 * durum[ad]["v"] + (1 - B2) * g * g
            mhat, vhat = m / (1 - B1 ** t), v / (1 - B2 ** t)
            p32 = p32 * (1 - lr * wd) - lr * mhat / (jnp.sqrt(vhat) + EPS)
            yeni_d[ad] = {"m": m, "v": v}
        yeni_a[ad] = p32.astype(p.dtype)
    return yeni_a, yeni_d
