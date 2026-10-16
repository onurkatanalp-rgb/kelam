"""PyTorch (model.py) ile JAX (tpu/kelam_jax.py) ayni agirliklarla ayni sonucu veriyor mu?

Calistirma (yapay_zeka klasorunden):  .venv\\Scripts\\python -m tpu.esitlik_testi
Kontrol: cikti puanlari, kayip, butun agirliklarin egimleri, uzman yukleri (32 bit, islemcide).
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

from model import GPT
from tpu import kelam_jax as kj

jax.config.update("jax_enable_x64", False)
SONUCLAR = []


def torch_kur(moe, tohum=0):
    torch.manual_seed(tohum)
    m = GPT(300, 3, 4, 64, 32, moe=moe)
    for b in m.bloklar:
        if moe:
            b.ileri.hizli = True
            with torch.no_grad():
                b.ileri.sapma.uniform_(-0.05, 0.05)
    return m


def jax_agirliklari(m):
    d = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    d.pop("bas.weight", None)  # gomme ile ortak, ayrica tutulmaz
    return d


def karsilastir(ad, moe):
    m = torch_kur(moe)
    m.train()
    torch.manual_seed(1)
    girdi = torch.randint(0, 300, (2, 32))
    hedef = torch.randint(0, 300, (2, 32))
    puan_t, kayip_t = m(girdi, hedef)
    kayip_t.backward()
    egim_t = {k: p.grad.numpy() for k, p in m.named_parameters()}
    yuk_t = [b.ileri.yuk.numpy().copy() for b in m.bloklar] if moe else []

    a = jax_agirliklari(m)

    def kayip_f(a_):
        return kj.ileri(a_, jnp.asarray(girdi.numpy()), m.yapi, jnp.asarray(hedef.numpy()))[1]

    puan_j, kayip_j, yuk_j = kj.ileri(a, jnp.asarray(girdi.numpy()), m.yapi, jnp.asarray(hedef.numpy()))
    egim_j = jax.grad(kayip_f)({k: jnp.asarray(v) for k, v in a.items()})

    hatalar = []
    fp = np.abs(np.asarray(puan_j) - puan_t.detach().numpy()).max()
    if fp > 1e-4:
        hatalar.append(f"puan farki {fp:.2e}")
    fk = abs(float(kayip_j) - kayip_t.item())
    if fk > 1e-5:
        hatalar.append(f"kayip farki {fk:.2e}")
    for k, gt in egim_t.items():
        gj = np.asarray(egim_j[k])
        goreli = np.linalg.norm(gj - gt) / max(np.linalg.norm(gt), 1e-12)
        if goreli > 1e-4:
            hatalar.append(f"{k} egimi goreli {goreli:.2e}")
    for i, (yt, yj) in enumerate(zip(yuk_t, yuk_j)):
        if not np.array_equal(yt.astype(np.int64), np.asarray(yj).astype(np.int64)):
            hatalar.append(f"katman {i} uzman yuku farkli")
    SONUCLAR.append((ad, not hatalar, "; ".join(hatalar[:4]) + (f" (+{len(hatalar) - 4})" if len(hatalar) > 4 else ""),
                     f"kayip torch {kayip_t.item():.6f} jax {float(kayip_j):.6f}"))


def denge_esit():
    moe = dict(uzman=8, secilen=2, ortak=1, uzman_ara=16)
    m = torch_kur(moe)
    a = jax_agirliklari(m)
    yukler = [jnp.asarray(np.random.default_rng(i).integers(0, 50, 8)) for i in range(3)]
    for b, y in zip(m.bloklar, yukler):
        b.ileri.yuk.copy_(torch.tensor(np.asarray(y), dtype=torch.float32))
    m.denge_guncelle(1e-3)
    a2 = kj.denge_guncelle(a, yukler, m.yapi, 1e-3)
    fark = max(np.abs(np.asarray(a2[f"bloklar.{i}.ileri.sapma"]) - m.bloklar[i].ileri.sapma.numpy()).max()
               for i in range(3))
    SONUCLAR.append(("denge guncellemesi ayni", fark < 1e-7, f"fark {fark:.2e}" if fark >= 1e-7 else "", ""))


def optimizer_esit(tur):
    """Ayni egimlerle 2 guncelleme adimi (momentum ve Adam duzeltmesi de sinanir): agirliklar ayni mi?
    Muon dikgenlestirmesi burada 32 bitte yapilir: algoritma birebir karsilastirilir (bf16 yuvarlamasi iki
    kutuphanede farkli; o ayri testte, kabul edilebilir sapma olarak olculur)."""
    import optimizer as ot
    from optimizer import optimizer_kur
    from tpu import optim_jax as oj
    ot.NS_TURU, oj.NS_TURU = torch.float32, jnp.float32
    moe = dict(uzman=8, secilen=2, ortak=1, uzman_ara=16)
    m = torch_kur(moe)
    a = {k: jnp.asarray(v) for k, v in jax_agirliklari(m).items()}
    opt_t = optimizer_kur(m, 1e-2, tur)
    durum = oj.kur(a, tur)
    torch.manual_seed(3)
    for _ in range(2):
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 0.01
        egim = {k: jnp.asarray(p.grad.numpy()) for k, p in m.named_parameters()}
        opt_t.step()
        a, durum = oj.adim(a, egim, durum, 1e-2, tur)
    hatalar = []
    for k, p in m.named_parameters():
        pt, pj = p.detach().numpy(), np.asarray(a[k])
        # Muon bf16 dikgenlestirme yapar: iki kutuphanede bf16 carpimlarin yuvarlamasi azicik farkli olabilir
        sinir = 1e-5
        goreli = np.linalg.norm(pj - pt) / max(np.linalg.norm(pt), 1e-12)
        if goreli > sinir:
            hatalar.append(f"{k} goreli {goreli:.2e}")
    ot.NS_TURU, oj.NS_TURU = torch.bfloat16, jnp.bfloat16
    SONUCLAR.append((f"optimizer ({tur}): 2 adim sonra butun agirliklar ayni", not hatalar,
                     "; ".join(hatalar[:4]), ""))


def ns_bf16_sapmasi():
    """bf16 dikgenlestirme iki kutuphanede de 32 bitlik sonuca yakin kalmali (Muon bu yaklasikliga dayanir)."""
    import optimizer as ot
    from tpu import optim_jax as oj
    G = np.random.default_rng(0).standard_normal((4, 192, 64)).astype(np.float32)
    ot.NS_TURU = torch.float32
    ref = ot.dikgenlestir(torch.tensor(G)).float().numpy()
    ot.NS_TURU = torch.bfloat16
    t = ot.dikgenlestir(torch.tensor(G)).float().numpy()
    j = np.asarray(oj.dikgenlestir(jnp.asarray(G)).astype(jnp.float32))
    st, sj = (np.linalg.norm(x - ref) / np.linalg.norm(ref) for x in (t, j))
    SONUCLAR.append(("Muon bf16 dikgenlestirme 32 bite yakin (iki kutuphane)", st < 0.1 and sj < 0.1,
                     "" if st < 0.1 and sj < 0.1 else f"torch {st:.3f} jax {sj:.3f}", f"sapma torch {st:.3f}, jax {sj:.3f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    karsilastir("normal model: puan, kayip, butun egimler ayni", None)
    karsilastir("MoE (8 uzman, 2 secilen, 1 ortak): puan, kayip, egimler, uzman yukleri ayni",
                dict(uzman=8, secilen=2, ortak=1, uzman_ara=16))
    karsilastir("MoE (16 uzman, 3 secilen, ortaksiz)", dict(uzman=16, secilen=3, ortak=0, uzman_ara=24))
    denge_esit()
    optimizer_esit("adamw")
    optimizer_esit("muon")
    ns_bf16_sapmasi()
    for ad, gecti, neden, bilgi in SONUCLAR:
        print(f"{'✓' if gecti else '✗'} {ad}" + (f"  ({neden})" if neden else "") + (f"  [{bilgi}]" if bilgi else ""))
    hata = sum(not g for _, g, _, _ in SONUCLAR)
    print(f"\n{len(SONUCLAR) - hata}/{len(SONUCLAR)} esitlik testi gecti")
    sys.exit(1 if hata else 0)
