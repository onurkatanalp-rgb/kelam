"""MoE (uzmanlar karisimi) dogruluk testleri:  .venv\\Scripts\\python moe_testleri.py

Aylarca surecek egitimden once katmanin matematiksel olarak dogru calistigini kanitlar."""
import copy
import sys

import torch
import torch.nn.functional as F

from model import GPT, IleriBesleme, UzmanKatmani

CIHAZ = "cuda" if torch.cuda.is_available() else "cpu"
SONUCLAR = []


def test(ad):
    def sar(f):
        try:
            f()
            SONUCLAR.append((ad, True, ""))
        except Exception as h:  # noqa: BLE001 - her hatayi rapora yaz
            SONUCLAR.append((ad, False, f"{type(h).__name__}: {h}"))
        return f
    return sar


def katman(uzman=8, secilen=2, ortak=1, ara=32, genislik=64):
    torch.manual_seed(0)
    k = UzmanKatmani(genislik, uzman, secilen, ortak, ara).to(CIHAZ)
    for p in (k.kapi, k.yukari, k.asagi):
        torch.nn.init.normal_(p, std=0.1)
    return k


def elle_hesapla(k, x):
    """Her tokeni tek tek, tanimdan hesaplar (hizli yol bununla karsilastirilir)."""
    C = x.shape[-1]
    duz = x.reshape(-1, C).float()
    cikti = torch.zeros_like(duz)
    for i, t in enumerate(duz):
        puan = torch.sigmoid(k.yonlendirici.weight.float() @ t)
        sec = torch.topk(puan + k.sapma, k.secilen).indices
        w = puan[sec] / puan[sec].sum()
        for o in k.ortak:
            cikti[i] += o.asagi.weight.float() @ (F.silu(o.kapi.weight.float() @ t) * (o.yukari.weight.float() @ t))
        for e, a in zip(sec.tolist(), w):
            h = F.silu(t @ k.kapi[e].float()) * (t @ k.yukari[e].float())
            cikti[i] += a * (h @ k.asagi[e].float())
    return cikti.view(x.shape)


@test("hizli yol = tanimdan elle hesap")
def _():
    k = katman()
    x = torch.randn(2, 5, 64, device=CIHAZ)
    fark = (k(x) - elle_hesapla(k, x)).abs().max().item()
    assert fark < 1e-4, f"fark {fark}"


@test("hizli (toplu) yol = tek tek yol: cikti ve butun egimler ayni")
def _():
    sonuc = {}
    for hizli in (False, True):
        k = katman(uzman=16, secilen=3, ortak=1, ara=32)
        k.hizli = hizli
        torch.manual_seed(5)
        x = torch.randn(3, 50, 64, device=CIHAZ, requires_grad=True)
        y = k(x)
        y.pow(2).sum().backward()
        sonuc[hizli] = (y.detach(), x.grad, {a: p.grad for a, p in k.named_parameters()})
    (y0, gx0, g0), (y1, gx1, g1) = sonuc[False], sonuc[True]
    assert (y0 - y1).abs().max() < 1e-5, "cikti farkli"
    assert (gx0 - gx1).abs().max() < 1e-5, "girdi egimi farkli"
    for a in g0:
        assert (g0[a] - g1[a]).abs().max() < 1e-5, f"{a} egimi farkli"


@test("hizli yol bf16'da da tek tek yola yakin")
def _():
    if CIHAZ != "cuda":
        return
    k = katman(uzman=16, secilen=3, ortak=1, ara=32)
    x = torch.randn(4, 64, 64, device=CIHAZ)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        k.hizli = False
        a = k(x).float()
        k.hizli = True
        b = k(x).float()
    goreli = ((a - b).norm() / a.norm()).item()
    assert goreli < 0.01, f"goreli fark {goreli}"


@test("gruplu carpim yolu (egitimde kullanilan) = tek tek yol, bf16")
def _():
    if CIHAZ != "cuda":
        return
    sonuc = {}
    for hizli in (False, True):
        k = katman(uzman=16, secilen=3, ortak=1, ara=32)
        k.hizli = hizli
        torch.manual_seed(6)
        x = torch.randn(4, 64, 64, device=CIHAZ, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = k(x)
        y.float().pow(2).sum().backward()
        sonuc[hizli] = (y.float().detach(), x.grad, {a: p.grad for a, p in k.named_parameters()})
    (y0, gx0, g0), (y1, gx1, g1) = sonuc[False], sonuc[True]

    def goreli(a, b):
        return ((a - b).norm() / a.norm().clamp(min=1e-12)).item()
    assert goreli(y0, y1) < 0.01, f"cikti {goreli(y0, y1)}"
    assert goreli(gx0, gx1) < 0.02, f"girdi egimi {goreli(gx0, gx1)}"
    for a in g0:
        assert goreli(g0[a], g1[a]) < 0.02, f"{a} egimi {goreli(g0[a], g1[a])}"


@test("1 uzman + secilen 1 = normal ileri besleme")
def _():
    k = katman(uzman=1, secilen=1, ortak=0, ara=48)
    d = IleriBesleme.__new__(IleriBesleme)
    torch.nn.Module.__init__(d)
    d.kapi = torch.nn.Linear(64, 48, bias=False).to(CIHAZ)
    d.yukari = torch.nn.Linear(64, 48, bias=False).to(CIHAZ)
    d.asagi = torch.nn.Linear(48, 64, bias=False).to(CIHAZ)
    with torch.no_grad():
        d.kapi.weight.copy_(k.kapi[0].T)
        d.yukari.weight.copy_(k.yukari[0].T)
        d.asagi.weight.copy_(k.asagi[0].T)
    x = torch.randn(3, 7, 64, device=CIHAZ)
    fark = (k(x) - d(x)).abs().max().item()
    assert fark < 1e-5, f"fark {fark}"


@test("egim (gradient) her parcaya ulasiyor")
def _():
    k = katman()
    x = torch.randn(4, 16, 64, device=CIHAZ, requires_grad=True)
    k(x).pow(2).sum().backward()
    for ad, p in k.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{ad} egim almadi"
    assert x.grad.abs().sum() > 0


@test("egimler sayisal turevle tutarli (float64)")
def _():
    k = katman(uzman=4, secilen=2, ortak=1, ara=8, genislik=16).double()
    x = torch.randn(1, 3, 16, device=CIHAZ, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: k(t), (x,), eps=1e-6, atol=1e-4)


@test("bf16: uzman secimi 32 bitle ayni, hata normal katmandan buyuk degil")
def _():
    if CIHAZ != "cuda":
        return
    k = katman()
    d = IleriBesleme(64).to(CIHAZ)
    x = torch.randn(4, 32, 64, device=CIHAZ)

    def goreli(f):
        tam = f(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yarim = f(x).float()
        return ((yarim - tam).norm() / tam.norm()).item()

    duz = x.reshape(-1, 64)
    sec_t, w_t = k.yonlendir(duz)
    with torch.autocast("cuda", dtype=torch.bfloat16):  # katmanin kendi secimi bf16 altinda da ayni olmali
        sec_b, w_b = k.yonlendir(duz)
    assert torch.equal(sec_t, sec_b) and torch.equal(w_t, w_b), "bf16 altinda uzman secimi degisti"
    moe_h, normal_h = goreli(k), goreli(d)
    assert moe_h < 2 * normal_h + 1e-3, f"moe {moe_h:.4f}, normal {normal_h:.4f}"


@test("denge: dengesiz baslayan yonlendirici zamanla dengelenir")
def _():
    k = katman(uzman=8, secilen=2)
    with torch.no_grad():
        k.yonlendirici.weight[0] += 0.5  # uzman 0'i bilerek cok cekici yap
    x = torch.randn(64, 32, 64, device=CIHAZ)
    k.train()

    def oran():
        k.yuk.zero_()
        k(x)
        y = k.yuk / k.yuk.sum()
        return (y.max() / y.mean()).item()

    once = oran()
    for _ in range(400):
        k(x)
        k.denge_guncelle(1e-2)
    sonra = oran()
    assert sonra < once and sonra < 1.5, f"en yuklu/ortalama {once:.2f} -> {sonra:.2f}"


@test("degerlendirmede (eval) yuk sayaci degismez")
def _():
    k = katman()
    k.eval()
    k(torch.randn(2, 8, 64, device=CIHAZ))
    assert k.yuk.sum() == 0


@test("kaydet-yukle: sapma dahil birebir ayni cikti")
def _():
    torch.manual_seed(1)
    moe = dict(uzman=8, secilen=2, ortak=1, uzman_ara=32)
    a = GPT(500, 2, 4, 64, 32, moe=moe).to(CIHAZ)
    for b in a.bloklar:
        b.ileri.sapma.uniform_(-0.1, 0.1)
    durum = copy.deepcopy(a.state_dict())
    yapi = a.yapi
    b = GPT(yapi["sozluk"], yapi["katman"], yapi["kafa"], yapi["genislik"], yapi["baglam"], moe=yapi["moe"]).to(CIHAZ)
    b.load_state_dict(durum)
    a.eval(); b.eval()
    g = torch.randint(0, 500, (2, 32), device=CIHAZ)
    assert torch.equal(a(g)[0], b(g)[0])
    assert all(torch.equal(x.ileri.sapma, y.ileri.sapma) for x, y in zip(a.bloklar, b.bloklar))


@test("tam model ogreniyor (kucuk veriyi ezberler)")
def _():
    torch.manual_seed(2)
    m = GPT(100, 2, 4, 64, 32, moe=dict(uzman=8, secilen=2, ortak=1, uzman_ara=32)).to(CIHAZ)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    g = torch.randint(0, 100, (8, 33), device=CIHAZ)
    for i in range(300):
        _, kayip = m(g[:, :-1], g[:, 1:])
        kayip.backward()
        opt.step()
        opt.zero_grad()
        m.denge_guncelle()
        if i == 0:
            ilk = kayip.item()
    assert kayip.item() < 0.3 * ilk, f"kayip {ilk:.2f} -> {kayip.item():.2f}"


@test("aktif parametre sayisi dogru")
def _():
    m = GPT(1000, 2, 4, 64, 32, moe=dict(uzman=8, secilen=2, ortak=1, uzman_ara=32))
    uzman_basi = 3 * 64 * 32
    assert m.parametre_sayisi() - m.aktif_parametre_sayisi() == 2 * 6 * uzman_basi


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for ad, gecti, neden in SONUCLAR:
        print(f"{'✓' if gecti else '✗'} {ad}" + (f"  ({neden})" if neden else ""))
    hata = sum(not g for _, g, _ in SONUCLAR)
    print(f"\n{len(SONUCLAR) - hata}/{len(SONUCLAR)} MoE testi geçti")
    sys.exit(1 if hata else 0)
