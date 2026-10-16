"""Kelam modelinin JAX surumu (TPU icin). model.py (PyTorch) ile ayni matematik, ayni agirlik adlari.

Agirliklar duz bir sozlukte tutulur, adlar PyTorch state_dict ile birebir ayni:
    gomme.weight, bloklar.{i}.norm1.agirlik, bloklar.{i}.dikkat.qkv.weight, ... , son_norm.agirlik
Boylece torch <-> jax agirlik tasimak tek satir: {ad: np.asarray(t)}. Dogruluk: tpu/esitlik_testi.py

MoE: yonlendirici 32 bitte (sigmoid), secim sapmali puanla, agirlik sapmasiz puanla (DeepSeek-V3);
uzmanlar jax.lax.ragged_dot ile (bosluksuz gruplu carpim; PyTorch'taki grouped_mm'in karsiligi).
"""
import math

import jax
import jax.numpy as jnp
import numpy as np


def yapi_ara(genislik):
    return int(8 * genislik / 3 / 64 + 0.999) * 64  # model.py IleriBesleme ile ayni


def ilk_agirliklar(anahtar, sozluk, katman, kafa, genislik, baglam, moe=None):
    """model.py GPT ile ayni dagilimlar (degerler farkli rastgele; esitlik testi torch agirliklarini tasir)."""
    del kafa, baglam
    a = {}
    anahtarlar = iter(jax.random.split(anahtar, 64 * katman + 8))

    def normal(sekil, std=0.02):
        return np.asarray(jax.random.normal(next(anahtarlar), sekil, jnp.float32) * std)

    artik_std = 0.02 / math.sqrt(2 * katman)
    a["gomme.weight"] = normal((sozluk, genislik))
    for i in range(katman):
        b = f"bloklar.{i}."
        a[b + "norm1.agirlik"] = np.ones(genislik, np.float32)
        a[b + "norm2.agirlik"] = np.ones(genislik, np.float32)
        a[b + "dikkat.qkv.weight"] = normal((3 * genislik, genislik))
        a[b + "dikkat.cikis.weight"] = normal((genislik, genislik), artik_std)
        if moe:
            u, ara = moe["uzman"], moe["uzman_ara"]
            a[b + "ileri.yonlendirici.weight"] = normal((u, genislik))
            a[b + "ileri.sapma"] = np.zeros(u, np.float32)
            a[b + "ileri.kapi"] = normal((u, genislik, ara))
            a[b + "ileri.yukari"] = normal((u, genislik, ara))
            a[b + "ileri.asagi"] = normal((u, ara, genislik), artik_std)
            for j in range(moe["ortak"]):
                o = b + f"ileri.ortak.{j}."
                a[o + "kapi.weight"] = normal((ara, genislik))
                a[o + "yukari.weight"] = normal((ara, genislik))
                a[o + "asagi.weight"] = normal((genislik, ara), artik_std)
        else:
            ara = yapi_ara(genislik)
            a[b + "ileri.kapi.weight"] = normal((ara, genislik))
            a[b + "ileri.yukari.weight"] = normal((ara, genislik))
            a[b + "ileri.asagi.weight"] = normal((genislik, ara), artik_std)
    a["son_norm.agirlik"] = np.ones(genislik, np.float32)
    return a


def rms_norm(x, w, eps=1e-6):
    # model.py: x * rsqrt(mean(x^2) + eps).type_as(x) * w ; ortalama 32 bitte
    olcek = jax.lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), -1, keepdims=True) + eps)
    return x * olcek.astype(x.dtype) * w.astype(x.dtype)


def rope_hazirla(kafa_boyutu, uzunluk, taban=10000.0):
    frekans = 1.0 / (taban ** (np.arange(0, kafa_boyutu, 2, dtype=np.float32) / kafa_boyutu))
    acilar = np.outer(np.arange(uzunluk, dtype=np.float32), frekans)
    return np.cos(acilar), np.sin(acilar)


def rope_uygula(x, cos, sin):
    # x: (G, kafa, U, kb); ciftler (0::2, 1::2) dondurulur ve yeniden araya serpistirilir (model.py ile ayni)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None].astype(x.dtype), sin[None, None].astype(x.dtype)
    return jnp.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1).reshape(x.shape)


def dogrusal(x, w):
    """PyTorch nn.Linear (bias yok): x @ w.T ; w: (cikis, giris)."""
    return x @ w.T.astype(x.dtype)


def dikkat(a, b, x, kafa, cos, sin):
    G, U, C = x.shape
    qkv = dogrusal(x, a[b + "qkv.weight"])
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = (t.reshape(G, U, kafa, C // kafa).transpose(0, 2, 1, 3) for t in (q, k, v))
    q, k = rope_uygula(q, cos, sin), rope_uygula(k, cos, sin)
    puan = (q @ k.transpose(0, 1, 3, 2)).astype(jnp.float32) / math.sqrt(C // kafa)
    maske = jnp.tril(jnp.ones((U, U), bool))
    puan = jnp.where(maske, puan, -jnp.inf)
    y = jax.nn.softmax(puan, axis=-1).astype(v.dtype) @ v
    return dogrusal(y.transpose(0, 2, 1, 3).reshape(G, U, C), a[b + "cikis.weight"])


def ileri_besleme(a, b, x):
    return dogrusal(jax.nn.silu(dogrusal(x, a[b + "kapi.weight"])) * dogrusal(x, a[b + "yukari.weight"]),
                    a[b + "asagi.weight"])


def yonlendir(a, b, x, secilen):
    """(secilen uzmanlar, agirliklar); 32 bitte. Secimde sapma, agirlikta sapmasiz puan."""
    puan = jax.nn.sigmoid(x.astype(jnp.float32) @ a[b + "yonlendirici.weight"].astype(jnp.float32).T)
    sapma = jax.lax.stop_gradient(a[b + "sapma"]).astype(jnp.float32)
    _, sec = jax.lax.top_k(puan + sapma, secilen)
    agirlik = jnp.take_along_axis(puan, sec, axis=-1)
    return sec, agirlik / agirlik.sum(-1, keepdims=True)


def uzman_katmani(a, b, x, moe):
    """Donus: (cikti, her uzmana giden token sayisi). x: (G, U, C)."""
    G, U, C = x.shape
    duz_x = x.reshape(-1, C)
    k, n_uzman = moe["secilen"], moe["uzman"]
    sec, agirlik = yonlendir(a, b, duz_x, k)
    y = jnp.zeros(duz_x.shape, jnp.float32)
    for j in range(moe["ortak"]):
        y = y + ileri_besleme(a, b + f"ortak.{j}.", duz_x).astype(jnp.float32)
    duz = sec.reshape(-1)
    sira = jnp.argsort(duz, stable=True)
    token_sirali = sira // k
    sayilar = jnp.bincount(duz, length=n_uzman)                  # sabit boy: derleyici dostu
    xs = duz_x[token_sirali]
    tur = x.dtype
    h = (jax.nn.silu(jax.lax.ragged_dot(xs, a[b + "kapi"].astype(tur), sayilar))
         * jax.lax.ragged_dot(xs, a[b + "yukari"].astype(tur), sayilar))
    cikti = jax.lax.ragged_dot(h, a[b + "asagi"].astype(tur), sayilar)       # uzman sirasinda
    katki = jnp.zeros_like(cikti).at[sira].set(cikti)                       # tokenin kendi sirasina
    katki = katki.reshape(-1, k, C).astype(jnp.float32) * agirlik[..., None]
    y = y + katki.sum(1)
    return y.astype(tur).reshape(G, U, C), sayilar


def ileri(a, girdi, yapi, hedef=None, tur=jnp.float32):
    """Donus: (puanlar, kayip ya da None, uzman yukleri listesi). yapi: model.yapi sozlugu (moe dahil)."""
    katman, kafa, genislik = yapi["katman"], yapi["kafa"], yapi["genislik"]
    moe = yapi.get("moe")
    U = girdi.shape[1]
    cos, sin = rope_hazirla(genislik // kafa, yapi["baglam"])
    cos, sin = cos[:U], sin[:U]
    x = a["gomme.weight"].astype(tur)[girdi]
    yukler = []
    for i in range(katman):
        b = f"bloklar.{i}."
        x = x + dikkat(a, b + "dikkat.", rms_norm(x, a[b + "norm1.agirlik"]), kafa, cos, sin)
        h = rms_norm(x, a[b + "norm2.agirlik"])
        if moe:
            d, sayi = uzman_katmani(a, b + "ileri.", h, moe)
            yukler.append(sayi)
        else:
            d = ileri_besleme(a, b + "ileri.", h)
        x = x + d
    puan = dogrusal(rms_norm(x, a["son_norm.agirlik"]), a["gomme.weight"])  # ortak gomme (bas = gomme)
    if hedef is None:
        return puan, None, yukler
    logp = jax.nn.log_softmax(puan.astype(jnp.float32), axis=-1)
    kayip = -jnp.take_along_axis(logp, hedef[..., None], axis=-1).mean()
    return puan, kayip, yukler


def denge_guncelle(a, yukler, yapi, hiz=1e-3):
    """DeepSeek-V3 kayipsiz denge: az kullanilan uzmanin sapmasi artar (model.py denge_guncelle ile ayni)."""
    a = dict(a)
    for i, yuk in enumerate(yukler):
        anahtar = f"bloklar.{i}.ileri.sapma"
        yuk = yuk.astype(jnp.float32)
        a[anahtar] = a[anahtar] + hiz * jnp.sign(yuk.mean() - yuk)
    return a
