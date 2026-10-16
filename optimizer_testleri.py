"""Muon testleri:  .venv\\Scripts\\python optimizer_testleri.py"""
import sys

import torch

from model import GPT
from optimizer import Birlesik, Muon, dikgenlestir, optimizer_kur

CIHAZ = "cuda" if torch.cuda.is_available() else "cpu"
SONUCLAR = []


def test(ad):
    def sar(f):
        try:
            f()
            SONUCLAR.append((ad, True, ""))
        except Exception as h:  # noqa: BLE001
            SONUCLAR.append((ad, False, f"{type(h).__name__}: {h}"))
        return f
    return sar


@test("dikgenlestirme tekil degerleri ~1'e ceker (kare, uzun, genis, 3 boyutlu)")
def _():
    torch.manual_seed(0)
    for sekil in ((64, 64), (128, 32), (32, 128), (4, 48, 96)):
        G = torch.randn(*sekil, device=CIHAZ)
        s = torch.linalg.svdvals(dikgenlestir(G).float()).flatten()
        # Keller'in polinomu bilerek tam 1 degil ~[0.5, 1.2] verir (hiz icin); rastgele kare matriste en kucuk
        # tekil deger 0'a cok yakin baslar, onu da yukari ceker ama tam yetistiremez. Buyuk cogunluk bantta olmali.
        bantta = ((s > 0.5) & (s < 1.3)).float().mean().item()
        assert s.max() < 1.3 and bantta > 0.95, f"{sekil}: bantta %{100 * bantta:.0f}, en buyuk {s.max():.2f}"


@test("parametreler dogru bolunur: matrisler Muon, gomme/yonlendirici/normlar AdamW")
def _():
    m = GPT(500, 2, 4, 64, 32, moe=dict(uzman=4, secilen=2, ortak=1, uzman_ara=16))
    o = optimizer_kur(m, 1e-3, "muon")
    assert isinstance(o, Birlesik)
    muon = {id(p) for p in o.parcalar[0].param_groups[0]["params"]}
    for ad, p in m.named_parameters():
        beklenen = p.dim() >= 2 and "gomme" not in ad and "yonlendirici" not in ad
        assert (id(p) in muon) == beklenen, ad
    hepsi = sum(len(g["params"]) for g in o.param_groups)
    assert hepsi == len(list(m.parameters()))


@test("Muon guncelleme buyuklugu AdamW'ninkine yakin (RMS 0.2-0.4 x lr)")
def _():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(256, 512, device=CIHAZ) * 0.02)
    once = p.detach().clone()
    o = Muon([p], lr=1e-2, weight_decay=0.0)
    p.grad = torch.randn_like(p)
    o.step()
    rms = ((p - once).pow(2).mean().sqrt() / 1e-2).item()
    assert 0.15 < rms < 0.45, f"rms/lr {rms:.3f}"


@test("model Muon ile ogreniyor (kucuk veriyi ezberler), kaydet-yukle calisir")
def _():
    torch.manual_seed(2)
    m = GPT(100, 2, 4, 64, 32, moe=dict(uzman=4, secilen=2, ortak=1, uzman_ara=32)).to(CIHAZ)
    o = optimizer_kur(m, 3e-3, "muon")
    g = torch.randint(0, 100, (8, 33), device=CIHAZ)
    for i in range(300):
        _, kayip = m(g[:, :-1], g[:, 1:])
        kayip.backward()
        o.step()
        o.zero_grad()
        m.denge_guncelle()
        if i == 0:
            ilk = kayip.item()
    assert kayip.item() < 0.3 * ilk, f"{ilk:.2f} -> {kayip.item():.2f}"
    o2 = optimizer_kur(m, 3e-3, "muon")
    o2.load_state_dict(o.state_dict())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for ad, gecti, neden in SONUCLAR:
        print(f"{'✓' if gecti else '✗'} {ad}" + (f"  ({neden})" if neden else ""))
    hata = sum(not g for _, g, _ in SONUCLAR)
    print(f"\n{len(SONUCLAR) - hata}/{len(SONUCLAR)} optimizer testi geçti")
    sys.exit(1 if hata else 0)
