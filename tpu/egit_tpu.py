"""Kelam'i TPU'da (Kaggle v5e-8, Google TRC) egitir. JAX; model tpu/kelam_jax.py, optimizer tpu/optim_jax.py.
Kesilirse ayni komutla kaldigi yerden devam eder (Kaggle oturumu 9 saatte kapanir).

Calistirma:  python -m tpu.egit_tpu --ayar tpu/ayarlar/<ad>.json
Yerel deneme (islemci, 4 sahte cihaz):  python -m tpu.egit_tpu --deneme

Ne yapar:
  - Agirliklar 32 bit tutulur, hesap bf16 (egit.py ile ayni duzen).
  - Cihazlara bolme (FSDP): her agirlik en buyuk bolunebilir ekseninden cihazlara dagitilir, grup da veriden bolunur.
    8B MoE tek cihaza sigmaz; bolununce her cihaz 1/8'ini tutar.
  - Biriktirme: bir adimdaki kucuk gruplar jax.lax.scan ile tek derlenmis dongude toplanir.
  - Bekci: kayip NaN olursa ya da ust uste sicrarsa son kayda geri doner, veriyi baska yerden surdurur.
    Uzman yukleri izlenir: hic token almayan (olu) uzman uyarisi.
  - Kayit: agirliklar + optimizer + adim tek .npz; son 2 tutulur. PyTorch kaydi (.pt) da cikarilabilir: sunucu.py okur.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

AYAR_VARSAYILAN = dict(
    ad="deneme",
    sozluk=32000, katman=4, kafa=4, genislik=128, baglam=128,
    moe=dict(uzman=8, secilen=2, ortak=1, uzman_ara=64),   # null = normal model
    optimizer="muon",            # muon | adamw
    ogrenme_hizi=1e-3, isinma=100,
    toplam_adim=400,             # takvim bu adima gore
    sogutma_orani=0.2,           # son %20'de ogrenme hizi duz cizgiyle sifira iner (WSD)
    durma_adimi=None,            # bu oturumda burada dur (None = toplam_adim)
    grup=8, biriktirme=2,        # adim basina grup x biriktirme x baglam token
    denge_hizi=1e-3,
    veri="veri/v2/orta", kayit="kayitlar_tpu",
    kayit_araligi=100, degerlendirme_araligi=100, gunluk_araligi=10,
    sure_siniri_saat=None,       # Kaggle: 8.5 -> oturum kapanmadan kaydeder ve cikar
    tohum=0,
)


# ---------------------------------------------------------------- yardimcilar

def ogrenme_hizi(adim, ay):
    """Isinma, sabit, son sogutma_orani kadarinda sifira duz inis (WSD). Sabit kisim uzatilabilir: egitim surdurulebilir."""
    if adim < ay["isinma"]:
        return ay["ogrenme_hizi"] * (adim + 1) / ay["isinma"]
    sogutma_bas = int(ay["toplam_adim"] * (1 - ay["sogutma_orani"]))
    if adim < sogutma_bas:
        return ay["ogrenme_hizi"]
    return ay["ogrenme_hizi"] * max(0.0, (ay["toplam_adim"] - adim) / max(1, ay["toplam_adim"] - sogutma_bas))


def veri_ac(klasor, ad):
    return np.memmap(os.path.join(klasor, ad + ".bin"), dtype=np.uint16, mode="r")


def grup_al(veri, adet, uzunluk, rng):
    """(adet, uzunluk+1) token; girdi = [:, :-1], hedef = [:, 1:]."""
    bas = rng.integers(0, len(veri) - uzunluk - 1, adet)
    return np.stack([veri[i:i + uzunluk + 1] for i in bas]).astype(np.int32)


def veri_rng(ay, adim, kaydirma):
    """Her adimin verisi (tohum, adim, kaydirma)'dan belirlenir: devam edince ayni veri, geri sarinca farkli veri."""
    return np.random.default_rng([ay["tohum"], adim, kaydirma])


# ---------------------------------------------------------------- bolme (sharding)

def bolme_kur(a, cihaz_sayisi):
    """Her agirlik icin PartitionSpec: cihaz sayisina bolunen en buyuk eksen 'veri' eksenine dagitilir."""
    from jax.sharding import PartitionSpec as P
    sonuc = {}
    for ad, p in a.items():
        sekil = p.shape
        eksen = None
        for e in sorted(range(len(sekil)), key=lambda e: -sekil[e]):
            if sekil[e] % cihaz_sayisi == 0 and sekil[e] >= cihaz_sayisi:
                eksen = e
                break
        spec = [None] * len(sekil)
        if eksen is not None and cihaz_sayisi > 1:
            spec[eksen] = "veri"
        sonuc[ad] = P(*spec)
    return sonuc


def durum_bolmesi(durum, agirlik_bolmesi):
    from jax.sharding import PartitionSpec as P
    return {k: (P() if k == "adim" else {kk: agirlik_bolmesi[k] for kk in v}) for k, v in durum.items()}


# ---------------------------------------------------------------- kayit

def _duzle(a, durum):
    d = {f"a/{k}": np.asarray(v) for k, v in a.items()}
    for k, v in durum.items():
        if k == "adim":
            d["o/adim"] = np.asarray(v)
        else:
            for kk, vv in v.items():
                d[f"o/{k}/{kk}"] = np.asarray(vv)
    return d


def _ac(d):
    a, durum = {}, {}
    for anahtar in d.files:
        if anahtar.startswith("a/"):
            a[anahtar[2:]] = d[anahtar]
        elif anahtar == "bilgi":
            continue
        elif anahtar == "o/adim":
            durum["adim"] = d[anahtar]
        else:
            ad, kk = anahtar[2:].rsplit("/", 1)
            durum.setdefault(ad, {})[kk] = d[anahtar]
    return a, durum


def kaydet(klasor, a, durum, bilgi, sakla=2):
    import jax
    os.makedirs(klasor, exist_ok=True)
    yol = os.path.join(klasor, f"adim_{bilgi['adim']:07d}.npz")
    d = _duzle(jax.device_get(a), jax.device_get(durum))
    d["bilgi"] = np.frombuffer(json.dumps(bilgi).encode(), dtype=np.uint8)
    with open(yol + ".gecici", "wb") as f:
        np.savez(f, **d)
    os.replace(yol + ".gecici", yol)  # yarim kalan kayit eskisini bozmaz
    for eski in sorted(glob.glob(os.path.join(klasor, "adim_*.npz")))[:-sakla]:
        os.remove(eski)
    return yol


def son_kayit(klasor):
    kayitlar = sorted(glob.glob(os.path.join(klasor, "adim_*.npz")))
    if not kayitlar:
        return None
    with np.load(kayitlar[-1]) as d:
        a, durum = _ac(d)
        bilgi = json.loads(bytes(d["bilgi"]).decode())
    return a, durum, bilgi


def torch_kaydi(a, yapi, yol, adim):
    """sunucu.py / konus.py'nin okudugu PyTorch kaydi (bas.weight = gomme, ortak agirlik)."""
    import torch
    sd = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in a.items()}
    sd["bas.weight"] = sd["gomme.weight"]
    torch.save({"model": sd, "yapi": yapi, "adim": adim}, yol)


# ---------------------------------------------------------------- egitim adimi

def adim_fonksiyonu(yapi, ay):
    import jax
    import jax.numpy as jnp
    from tpu import kelam_jax as kj
    from tpu import optim_jax as oj

    def kayip_f(a, parca):
        _, kayip, yukler = kj.ileri(a, parca[:, :-1], yapi, parca[:, 1:], tur=jnp.bfloat16)
        return kayip, yukler

    def adim(a, durum, veri, lr):
        """veri: (biriktirme, grup, baglam+1). Donus: a, durum, kayip, egim normu, uzman yukleri (katman, uzman)."""
        def tek(toplam, parca):
            (kayip, yukler), egim = jax.value_and_grad(kayip_f, has_aux=True)(a, parca)
            egim_t, kayip_t, yuk_t = toplam
            egim_t = jax.tree.map(lambda x, y: x + y.astype(jnp.float32), egim_t, egim)
            yuk_t = yuk_t + (jnp.stack(yukler).astype(jnp.float32) if yukler else 0.0)
            return (egim_t, kayip_t + kayip, yuk_t), None

        n = veri.shape[0]
        sifir = jax.tree.map(lambda x: jnp.zeros(x.shape, jnp.float32), a)
        yuk0 = jnp.zeros((yapi["katman"], yapi["moe"]["uzman"]), jnp.float32) if yapi.get("moe") else jnp.zeros(())
        (egim, kayip, yukler), _ = jax.lax.scan(tek, (sifir, jnp.zeros(()), yuk0), veri)
        egim = jax.tree.map(lambda x: x / n, egim)
        # egit.py: clip_grad_norm_(1.0)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for k, g in egim.items() if not k.endswith("sapma")))
        olcek = jnp.minimum(1.0, 1.0 / (norm + 1e-6))
        egim = jax.tree.map(lambda g: g * olcek, egim)
        a, durum = oj.adim(a, egim, durum, lr, ay["optimizer"])
        if yapi.get("moe"):
            a = kj.denge_guncelle(a, list(yukler), yapi, ay["denge_hizi"])
        return a, durum, kayip / n, norm, yukler

    return adim


def degerlendirme_fonksiyonu(yapi):
    import jax.numpy as jnp
    from tpu import kelam_jax as kj

    def degerlendir(a, parca):
        return kj.ileri(a, parca[:, :-1], yapi, parca[:, 1:], tur=jnp.bfloat16)[1]

    return degerlendir


# ---------------------------------------------------------------- ana dongu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ayar", help="JSON ayar dosyasi (AYAR_VARSAYILAN uzerine yazilir)")
    ap.add_argument("--deneme", action="store_true", help="islemcide 4 sahte cihaz + sahte veriyle sina")
    ap.add_argument("--torch", action="store_true", help="son kayittan PyTorch .pt cikar ve cik")
    arg = ap.parse_args()

    if arg.deneme:
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from tpu import kelam_jax as kj
    from tpu import optim_jax as oj

    ay = dict(AYAR_VARSAYILAN)
    if arg.ayar:
        with open(arg.ayar, encoding="utf-8") as f:
            ay.update(json.load(f))
    if arg.deneme:
        ay.update(sozluk=512, kayit=os.environ.get("KELAM_DENEME_KAYIT", "kayitlar_tpu_deneme"),
                  toplam_adim=int(os.environ.get("KELAM_DENEME_ADIM", 60)), isinma=10,
                  kayit_araligi=20, degerlendirme_araligi=20, gunluk_araligi=5,
                  durma_adimi=int(os.environ["KELAM_DENEME_DUR"]) if os.environ.get("KELAM_DENEME_DUR") else None)
    yapi = dict(sozluk=ay["sozluk"], katman=ay["katman"], kafa=ay["kafa"], genislik=ay["genislik"],
                baglam=ay["baglam"], tokenizer="v2")
    if ay.get("moe"):
        yapi["moe"] = dict(ay["moe"])
    klasor = os.path.join(ay["kayit"], ay["ad"])

    if arg.torch:
        a, _, bilgi = son_kayit(klasor)
        yol = os.path.join(klasor, "en_son.pt")
        torch_kaydi(a, yapi, yol, bilgi["adim"])
        print(f"PyTorch kaydi: {yol} (adim {bilgi['adim']})")
        return

    cihazlar = jax.devices()
    n = len(cihazlar)
    agi = Mesh(np.array(cihazlar), ("veri",))
    assert ay["grup"] % n == 0, f"grup ({ay['grup']}) cihaz sayisina ({n}) bolunmeli"
    print(f"{n} cihaz: {cihazlar[0].platform} | yapi {yapi}", flush=True)

    # veri
    if arg.deneme:
        rng0 = np.random.default_rng(0)
        # ogrenilebilir sahte veri: tekrar eden kisa kaliplar
        kalip = rng0.integers(0, ay["sozluk"], 50)
        egitim_v = np.tile(kalip, 20000).astype(np.uint16)
        dogrulama_v = np.tile(kalip, 200).astype(np.uint16)
    else:
        egitim_v, dogrulama_v = veri_ac(ay["veri"], "egitim"), veri_ac(ay["veri"], "dogrulama")
    print(f"veri: egitim {len(egitim_v) / 1e6:.0f}M token, dogrulama {len(dogrulama_v) / 1e6:.1f}M", flush=True)

    # agirliklar + optimizer (kayit varsa oradan)
    kayit = son_kayit(klasor)
    if kayit:
        a_np, durum_np, bilgi = kayit
        print(f"Kaldigi yerden devam: adim {bilgi['adim']}", flush=True)
    else:
        a_np = kj.ilk_agirliklar(jax.random.PRNGKey(ay["tohum"]), ay["sozluk"], ay["katman"], ay["kafa"],
                                 ay["genislik"], ay["baglam"], yapi.get("moe"))
        durum_np = None
        bilgi = dict(adim=0, en_iyi=float("inf"), kaydirma=0, geri_sarma=0)
    bolme = bolme_kur(a_np, n)
    a = {k: jax.device_put(jnp.asarray(v, jnp.float32), NamedSharding(agi, bolme[k])) for k, v in a_np.items()}
    durum = oj.kur(a, ay["optimizer"]) if durum_np is None else durum_np
    d_bolme = durum_bolmesi(durum, bolme)
    durum = jax.tree.map(lambda v, s: jax.device_put(jnp.asarray(v), NamedSharding(agi, s)), durum, d_bolme,
                         is_leaf=lambda x: isinstance(x, P))
    toplam_p = sum(int(np.prod(v.shape)) for k, v in a_np.items() if not k.endswith("sapma"))
    aktif_p = toplam_p
    if yapi.get("moe"):
        m = yapi["moe"]
        aktif_p -= ay["katman"] * (m["uzman"] - m["secilen"]) * 3 * ay["genislik"] * m["uzman_ara"]
    print(f"parametre: toplam {toplam_p / 1e6:.1f}M, aktif {aktif_p / 1e6:.1f}M | optimizer {ay['optimizer']}",
          flush=True)
    del a_np, durum_np

    veri_bolme = NamedSharding(agi, P(None, "veri", None))
    a_bolme = {k: NamedSharding(agi, s) for k, s in bolme.items()}
    d_named = jax.tree.map(lambda s: NamedSharding(agi, s), d_bolme, is_leaf=lambda x: isinstance(x, P))
    adim_j = jax.jit(adim_fonksiyonu(yapi, ay), donate_argnums=(0, 1),
                     out_shardings=(a_bolme, d_named, None, None, None))
    deger_j = jax.jit(degerlendirme_fonksiyonu(yapi))

    def parti(adim, kaydirma):
        v = grup_al(egitim_v, ay["biriktirme"] * ay["grup"], ay["baglam"], veri_rng(ay, adim, kaydirma))
        return jax.device_put(v.reshape(ay["biriktirme"], ay["grup"], -1), veri_bolme)

    def degerlendir():
        rng = np.random.default_rng(12345)  # hep ayni sinav parcalari: sonuclar karsilastirilabilir
        kayiplar = [float(deger_j(a, jax.device_put(grup_al(dogrulama_v, ay["grup"], ay["baglam"], rng),
                                                     NamedSharding(agi, P("veri", None))))) for _ in range(20)]
        return sum(kayiplar) / len(kayiplar)

    son = min(ay["toplam_adim"], ay["durma_adimi"] or ay["toplam_adim"])
    adim = bilgi["adim"]
    baslangic = time.time()
    t0, ort, sicrama = time.time(), None, 0
    yuk_pencere = None
    token_adim = ay["grup"] * ay["biriktirme"] * ay["baglam"]
    sonraki = parti(adim, bilgi["kaydirma"])
    while adim < son:
        lr = ogrenme_hizi(adim, ay)
        a, durum, kayip, norm, yukler = adim_j(a, durum, sonraki, jnp.float32(lr))
        sonraki = parti(adim + 1, bilgi["kaydirma"])  # TPU calisirken sonraki parti hazirlanir
        kayip = float(kayip)
        adim += 1
        if arg.deneme and adim == int(os.environ.get("KELAM_DENEME_NAN", -1)) and bilgi["geri_sarma"] == 0:
            kayip = float("nan")  # bekci sinamasi

        # bekci: NaN ya da ust uste 5 sicrama (ortalamanin 1.5 kati) -> son kayda don, veriyi kaydir
        if ort is not None and (not math.isfinite(kayip) or kayip > 1.5 * ort):
            sicrama = sicrama + 1 if math.isfinite(kayip) else 99
        else:
            sicrama = 0
        if sicrama >= 5:
            bilgi["geri_sarma"] += 1
            print(f"!!! adim {adim}: kayip {kayip:.3f} (ortalama {ort:.3f}) -> son kayda donuluyor "
                  f"({bilgi['geri_sarma']}. kez)", flush=True)
            if bilgi["geri_sarma"] > 3:
                raise SystemExit("3'ten fazla geri sarma: ayarlar incelenmeli (ogrenme hizi?). Durdu.")
            kayit = son_kayit(klasor)
            if kayit is None:
                raise SystemExit("Kayit yok, geri donulemedi. Durdu.")
            a_np, durum_np, eski = kayit
            a = {k: jax.device_put(jnp.asarray(v), a_bolme[k]) for k, v in a_np.items()}
            durum = jax.tree.map(lambda v, s: jax.device_put(jnp.asarray(v), s), durum_np, d_named)
            adim, ort, sicrama = eski["adim"], None, 0
            bilgi["kaydirma"] += 1  # ayni veriye bir daha carpmasin
            sonraki = parti(adim, bilgi["kaydirma"])
            continue
        if math.isfinite(kayip):
            ort = kayip if ort is None else 0.98 * ort + 0.02 * kayip

        if yapi.get("moe"):
            yuk_pencere = yukler if yuk_pencere is None else yuk_pencere + yukler
        if adim % ay["gunluk_araligi"] == 0:
            gecen = time.time() - t0
            satir = (f"adim {adim:7d} | kayip {kayip:.3f} | norm {float(norm):.2f} | hiz {lr:.1e} "
                     f"| {ay['gunluk_araligi'] * token_adim / gecen / 1000:.0f}k token/sn")
            try:
                ist = cihazlar[0].memory_stats() or {}
                if "peak_bytes_in_use" in ist:
                    satir += f" | bellek {ist['peak_bytes_in_use'] / 2**30:.1f}/{ist.get('bytes_limit', 0) / 2**30:.0f} GB"
            except Exception:
                pass
            if yuk_pencere is not None:
                y = np.asarray(yuk_pencere)
                oran = y / np.maximum(y.mean(-1, keepdims=True), 1)
                satir += f" | uzman yuku en az {oran.min():.2f} en cok {oran.max():.2f}"
                olu = int((y == 0).sum())
                if olu:
                    satir += f" | UYARI {olu} olu uzman"
                yuk_pencere = None
            print(satir, flush=True)
            t0 = time.time()

        if adim % ay["degerlendirme_araligi"] == 0 or adim == son:
            s = degerlendir()
            print(f"--- adim {adim}: dogrulama {s:.3f}", flush=True)
            bilgi["en_iyi"] = min(bilgi["en_iyi"], s)

        sure_doldu = ay["sure_siniri_saat"] and time.time() - baslangic > ay["sure_siniri_saat"] * 3600
        if adim % ay["kayit_araligi"] == 0 or adim == son or sure_doldu:
            bilgi["adim"] = adim
            print(f"kayit: {kaydet(klasor, a, durum, bilgi)}", flush=True)
        if sure_doldu:
            print(f"Sure siniri: adim {adim}'de kaydedildi; ayni komutla devam eder.", flush=True)
            return
    print("Egitim bitti." if adim >= ay["toplam_adim"] else f"Durma adimina gelindi ({adim}).", flush=True)


if __name__ == "__main__":
    main()
