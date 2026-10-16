"""Muon: matris agirliklari icin "dik" guncelleme yapan optimizer (Keller Jordan 2024; Moonshot, arXiv 2502.16982).

Fikir: AdamW her dugmeyi ayri ayri olcekler. Muon ise bir agirlik matrisinin guncellemesini butun olarak ele alir
ve "dikgenlestirir" (Newton-Schulz yinelemesi): guncelleme her yone esit guc verir, birkac yone yigilmaz.
Olculen kazanc: ayni kayba %10-30 daha az veriyle (1B alti modellerde; bkz. CLAUDE.md arastirma notlari).

Moonshot ayarlari (buyuk olcekte kararli calisan surum):
  - agirlik azaltma (weight decay) Muon parametrelerine de uygulanir (yoksa agirliklar buyur, bf16 tasar)
  - guncelleme buyuklugu AdamW'ninkine esitlenir: 0.2 * sqrt(max(satir, sutun)) -> AdamW'nin ogrenme hizi aynen kullanilir
  - gomme (embedding), cikis basi, normlar, yonlendirici ve tek boyutlu her sey AdamW ile
MoE uzmanlari (uzman, giris, cikis) tensorunde her uzmanin matrisi ayri dikgenlestirilir.
"""
import math

import torch

NS_TURU = torch.bfloat16  # dikgenlestirme hassasiyeti (testler 32 bitle algoritmayi birebir karsilastirir)


def dikgenlestir(G, adim=5):
    """Newton-Schulz: G'yi en yakin 'dik' matrise yaklastirir (tekil degerleri ~1'e ceker). G: (..., a, b).
    Katsayilar Keller Jordan'in 5. derece polinomu; bf16'da calisir (hizli), sonuc yaklasik dik."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(NS_TURU)
    devrik = X.size(-2) > X.size(-1)
    if devrik:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(adim):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if devrik:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    """Yalniz 2 ya da 3 boyutlu (MoE uzmanlari) matrisler icin. Digerleri AdamW ile ayri gruptadir (bkz. optimizer_kur)."""

    def __init__(self, parametreler, lr=3e-4, momentum=0.95, weight_decay=0.1, ns_adim=5):
        super().__init__(parametreler, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_adim=ns_adim))

    @torch.no_grad()
    def step(self):
        for grup in self.param_groups:
            for p in grup["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                durum = self.state[p]
                if "momentum" not in durum:
                    durum["momentum"] = torch.zeros_like(g)
                m = durum["momentum"]
                m.mul_(grup["momentum"]).add_(g)
                g = g.add(m, alpha=grup["momentum"])                    # Nesterov
                guncelleme = dikgenlestir(g, grup["ns_adim"]).float()
                satir, sutun = p.shape[-2], p.shape[-1]
                olcek = 0.2 * math.sqrt(max(satir, sutun))               # AdamW ile ayni guncelleme buyuklugu
                p.mul_(1 - grup["lr"] * grup["weight_decay"])
                p.add_(guncelleme.to(p.dtype), alpha=-grup["lr"] * olcek)


class Birlesik:
    """Muon + AdamW tek optimizer gibi: step, zero_grad, param_groups (ogrenme hizi takvimi icin), state_dict."""

    def __init__(self, muon, adamw):
        self.parcalar = [o for o in (muon, adamw) if o is not None]

    @property
    def param_groups(self):
        return [g for o in self.parcalar for g in o.param_groups]

    def step(self):
        for o in self.parcalar:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.parcalar:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"parcalar": [o.state_dict() for o in self.parcalar]}

    def load_state_dict(self, durum):
        for o, d in zip(self.parcalar, durum["parcalar"]):
            o.load_state_dict(d)


def optimizer_kur(model, lr, tur="adamw", weight_decay=0.1):
    """tur='adamw': bugunku duzen. tur='muon': matrisler Muon, gerisi AdamW (Moonshot duzeni)."""
    muon_p, karar_p, diger_p = [], [], []
    for ad, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            diger_p.append(p)
        elif tur == "muon" and "gomme" not in ad and "bas" not in ad and "yonlendirici" not in ad:
            muon_p.append(p)
        else:
            karar_p.append(p)
    adamw = torch.optim.AdamW([{"params": karar_p, "weight_decay": weight_decay},
                               {"params": diger_p, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    if tur != "muon":
        return adamw
    return Birlesik(Muon(muon_p, lr=lr, weight_decay=weight_decay), adamw)
