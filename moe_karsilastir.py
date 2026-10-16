"""MoE ile normal modeli ayni aktif boyut ve ayni veriyle yan yana egitir (bu bilgisayarda, bedava).

Soru: Ayni hesap maliyetinde MoE gercekten daha iyi ogreniyor mu, uzmanlar dengeli kullaniliyor mu?
Veri: veri/egitim.bin (Vikipedi, v1 sozluk). Ikisi de ayni tohumla ayni gruplari gorur.
Cikti: moe_karsilastirma_log.txt ve moe_karsilastirma.json

Calistirma:  .venv\\Scripts\\python -u moe_karsilastir.py
"""
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT
from optimizer import optimizer_kur

ADIM = int(os.environ.get("MOE_ADIM", 6000))
GRUP, BAGLAM = 16, 512                     # adim basina 8.192 token
LR, ISINMA = 1e-3, 300
DEGERLENDIRME = 500
ORTAK = dict(sozluk=32000, katman=8, kafa=8, genislik=512, baglam=BAGLAM)
# 16 uzman, her tokende 1 ortak + 3 secilen; uzman ara boyutu 352 -> aktif ileri besleme normalinkiyle ayni (4 x 352 = 1408)
MOE = dict(uzman=16, secilen=3, ortak=1, uzman_ara=352)
DENEYLER = {  # DENEY ortam degiskeniyle secilir: ad -> [(model adi, moe, optimizer), ...]
    "moe": [("normal", None, "adamw"), ("moe", MOE, "adamw")],
    "muon": [("moe adamw", MOE, "adamw"), ("moe muon", MOE, "muon")],
}
MODELLER = DENEYLER[os.environ.get("DENEY", "moe")]


def yukle(ad):
    return np.memmap(os.path.join("veri", ad + ".bin"), dtype=np.uint16, mode="r")


def grup(veri, rastgele):
    bas = rastgele.integers(0, len(veri) - BAGLAM - 1, GRUP)
    x = np.stack([veri[i:i + BAGLAM + 1] for i in bas]).astype(np.int64)
    x = torch.from_numpy(x).cuda()
    return x[:, :-1], x[:, 1:]


def lr(adim):
    if adim < ISINMA:
        return LR * (adim + 1) / ISINMA
    return LR * max(0.0, 1 - (adim - ISINMA) / (ADIM - ISINMA))  # duz cizgiyle sifira (soğutma dersi)


@torch.no_grad()
def sinav(model, veri):
    model.eval()
    r = np.random.default_rng(123)  # her olcumde ayni sinav gruplari
    toplam = 0.0
    for _ in range(40):
        x, y = grup(veri, r)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            toplam += model(x, y)[1].item()
    model.train()
    return toplam / 40


def egit(ad, moe, opt_tur, log):
    torch.manual_seed(0)
    model = GPT(**ORTAK, moe=moe).cuda()
    opt = optimizer_kur(model, LR, opt_tur)
    egitim, dogrulama = yukle("egitim"), yukle("dogrulama")
    r = np.random.default_rng(0)  # iki model de ayni gruplari ayni sirayla gorur
    log(f"== {ad}: toplam {model.parametre_sayisi() / 1e6:.1f}M, aktif {model.aktif_parametre_sayisi() / 1e6:.1f}M")
    egri, t0 = [], time.time()
    for adim in range(ADIM):
        for g in opt.param_groups:
            g["lr"] = lr(adim)
        x, y = grup(egitim, r)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, kayip = model(x, y)
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(kayip.item()):
            log(f"!! {ad}: kayip bozuldu (adim {adim})")
            break
        if (adim + 1) % DEGERLENDIRME == 0 or adim + 1 == ADIM:
            yukler = [y.cpu() for y in model.uzman_yukleri()] if moe else []
            denge = max((y.max() / y.mean()).item() for y in yukler) if yukler else None
            s = sinav(model, dogrulama)
            egri.append({"adim": adim + 1, "sinav": s, "denge": denge})
            log(f"{ad} adim {adim + 1:5d} | sinav {s:.3f}" + (f" | en yuklu uzman/ortalama {denge:.2f}" if denge else "")
                + f" | {(adim + 1) * GRUP * BAGLAM / (time.time() - t0) / 1000:.0f}k token/sn")
        if moe:
            model.denge_guncelle(1e-3)  # yuk sayacini da sifirlar
    return egri


def main():
    with open("moe_karsilastirma_log.txt", "a", encoding="utf-8") as f:
        def log(m):
            print(m, flush=True)
            f.write(m + "\n")
            f.flush()
        log(f"--- {time.strftime('%d.%m %H:%M')} | {ADIM} adim x {GRUP * BAGLAM} token")
        sonuc = {ad: egit(ad, moe, opt, log) for ad, moe, opt in MODELLER}
        (a1, e1), (a2, e2) = list(sonuc.items())[:2]
        n, m = e1[-1]["sinav"], e2[-1]["sinav"]
        log(f"SONUC: {a1} {n:.3f} | {a2} {m:.3f} | fark {n - m:+.3f} ({a2 + ' daha iyi' if m < n else a2 + ' daha iyi DEGIL'})")
    with open(f"karsilastirma_{os.environ.get('DENEY', 'moe')}.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=1)


if __name__ == "__main__":
    main()
