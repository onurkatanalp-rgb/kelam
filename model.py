"""Sifirdan yazilmis GPT tipi dil modeli.

Yapi (modern modellerdeki gibi):
  token -> gomme -> [Blok x katman] -> son normalizasyon -> her parca icin puan
  Blok = Dikkat (RoPE konum bilgisiyle) + Ileri besleme (SwiGLU), ikisinde de RMSNorm ve artik baglanti.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """Sayilarin buyuklugunu dengeler; egitimin patlamamasini saglar."""

    def __init__(self, boyut, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.agirlik = nn.Parameter(torch.ones(boyut))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.agirlik


def rope_hazirla(kafa_boyutu, uzunluk, taban=10000.0):
    """Her konum icin donme acilari: model kelimelerin sirasini bunlarla bilir."""
    frekans = 1.0 / (taban ** (torch.arange(0, kafa_boyutu, 2).float() / kafa_boyutu))
    acilar = torch.outer(torch.arange(uzunluk).float(), frekans)
    return torch.cos(acilar), torch.sin(acilar)


def rope_uygula(x, cos, sin):
    # x: (grup, kafa, uzunluk, kafa_boyutu) - ciftler halinde dondurulur
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Dikkat(nn.Module):
    """Her token, kendinden onceki tokenlara bakip hangisinin onemli oldugunu secer."""

    def __init__(self, genislik, kafa):
        super().__init__()
        self.kafa = kafa
        self.qkv = nn.Linear(genislik, 3 * genislik, bias=False)
        self.cikis = nn.Linear(genislik, genislik, bias=False)

    def forward(self, x, cos, sin):
        G, U, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(G, U, self.kafa, C // self.kafa).transpose(1, 2) for t in (q, k, v))
        q, k = rope_uygula(q, cos, sin), rope_uygula(k, cos, sin)
        # is_causal: model gelecegi goremez, yalniz gecmise bakar
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.cikis(y.transpose(1, 2).contiguous().view(G, U, C))


class IleriBesleme(nn.Module):
    """Dikkatin topladigi bilgiyi isleyip "dusunen" kisim (SwiGLU)."""

    def __init__(self, genislik):
        super().__init__()
        ara = int(8 * genislik / 3 / 64 + 0.999) * 64
        self.kapi = nn.Linear(genislik, ara, bias=False)
        self.yukari = nn.Linear(genislik, ara, bias=False)
        self.asagi = nn.Linear(ara, genislik, bias=False)

    def forward(self, x):
        return self.asagi(F.silu(self.kapi(x)) * self.yukari(x))


class UzmanKatmani(nn.Module):
    """MoE (uzmanlar karisimi): ileri besleme yerine cok sayida kucuk uzman. Her token icin yonlendirici
    en uygun `secilen` uzmani secer; `ortak` uzman(lar) her tokende calisir (genel bilgi tekrar ogrenilmez).
    Denge: kayip terimi yok (DeepSeek-V3 yontemi). Her uzmanin secim puanina bir sapma eklenir; az kullanilan
    uzmanin sapmasi artar, cok kullanilaninki azalir (denge_guncelle). Sapma yalniz secimi etkiler, agirligi degil."""

    def __init__(self, genislik, uzman, secilen, ortak, uzman_ara):
        super().__init__()
        self.uzman, self.secilen = uzman, secilen
        self.yonlendirici = nn.Linear(genislik, uzman, bias=False)
        self.register_buffer("sapma", torch.zeros(uzman))
        self.register_buffer("yuk", torch.zeros(uzman), persistent=False)  # son adimlarda her uzmana giden token
        # Butun uzmanlar tek tensorde: (uzman, giris, cikis)
        self.kapi = nn.Parameter(torch.empty(uzman, genislik, uzman_ara))
        self.yukari = nn.Parameter(torch.empty(uzman, genislik, uzman_ara))
        self.asagi = nn.Parameter(torch.empty(uzman, uzman_ara, genislik))
        self.ortak = nn.ModuleList(OrtakUzman(genislik, uzman_ara) for _ in range(ortak))
        self.hizli = True  # False: uzmanlari tek tek hesaplayan basit yol (testlerde karsilastirma icin)

    def forward(self, x):
        G, U, C = x.shape
        x = x.reshape(-1, C)
        ust = torch.promote_types(x.dtype, torch.float32)  # en az 32 bit (float64 testinde 64)
        sec, agirlik = self.yonlendir(x)
        if self.training:
            self.yuk += self._say(sec.flatten()).float()
        # Toplama 32 bitte: bf16'da cok sayida kucuk katki toplanirken hassasiyet kaybolmasin
        y = torch.zeros(x.shape, dtype=ust, device=x.device)
        for o in self.ortak:
            y = y + o(x).to(ust)
        if not self.hizli:
            y = self._tek_tek(x, sec, agirlik, y)
        elif x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16:
            y = self._gruplu(x, sec, agirlik, y)  # egitim: bosluksuz gruplu carpim (en hizli)
        else:
            y = self._toplu(x, sec, agirlik, y)
        return y.type_as(x).reshape(G, U, C)

    def _tek_tek(self, x, sec, agirlik, y):
        """Basit yol (dogrulugun olcusu): uzmanlar sirayla, her biri kendi tokenlariyla."""
        ust = y.dtype
        duz = sec.flatten()
        sira = duz.argsort()
        token_no = sira // self.secilen
        sayilar = torch.bincount(duz, minlength=self.uzman).tolist()
        bas = 0
        for e, n in enumerate(sayilar):
            if n == 0:
                continue
            t = token_no[bas:bas + n]
            xe = x[t]
            h = F.silu(xe @ self.kapi[e]) * (xe @ self.yukari[e])
            w = agirlik.reshape(-1)[sira[bas:bas + n]].unsqueeze(1)
            y = y.index_add(0, t, (h @ self.asagi[e]).to(ust) * w)
            bas += n
        return y

    def _say(self, duz):
        """Her uzmana giden token sayisi. bincount yerine sabit boyutlu toplama: derleyici (torch.compile)
        katmani bolmeden tek parca derleyebilsin (bincount'un cikti boyu veriye bagli sayiliyor)."""
        return torch.zeros(self.uzman, dtype=torch.int64, device=duz.device).scatter_add_(0, duz, torch.ones_like(duz))

    def _gruplu(self, x, sec, agirlik, y):
        """En hizli yol (bf16, kart): tokenlar uzmana gore uc uca dizilir, her uzman yalniz kendi tokenlariyla
        tek bir gruplu carpimda (grouped_mm) hesaplanir. Tablo/bosluk yok: zaman ve bellek bosa gitmez."""
        ust = y.dtype
        duz = sec.flatten()
        sira = duz.argsort(stable=True)
        token_sirali = sira // self.secilen
        sinirlar = torch.cumsum(self._say(duz), 0).to(torch.int32)  # her uzmanin bitisi
        xs = x[token_sirali].to(torch.bfloat16)
        kapi, yukari, asagi = (t.to(torch.bfloat16) for t in (self.kapi, self.yukari, self.asagi))
        h = F.silu(F.grouped_mm(xs, kapi, offs=sinirlar)) * F.grouped_mm(xs, yukari, offs=sinirlar)
        cikti = F.grouped_mm(h, asagi, offs=sinirlar)              # (token*secilen, genislik), uzman sirasinda
        katki = torch.empty_like(cikti)
        katki[sira] = cikti                                         # tokenin kendi sirasina (bf16, az bellek)
        # Agirlikli toplam tek carpimla: (token, 1, secilen) @ (token, secilen, genislik); kart icte 32 bitte toplar
        birlesik = torch.bmm(agirlik.to(torch.bfloat16).unsqueeze(1), katki.view(-1, self.secilen, x.shape[1]))
        return y + birlesik.squeeze(1).to(ust)

    def _toplu(self, x, sec, agirlik, y):
        """Hizli yol: her uzmanin tokenlari (uzman, yer, genislik) tablosuna dizilir, butun uzmanlar tek
        toplu carpimla (bmm) hesaplanir. Python dongusu yok; tablo en kalabalik uzmana gore doldurulur."""
        ust = y.dtype
        duz = sec.flatten()                                         # (token*secilen,) hangi uzman
        sira = duz.argsort(stable=True)
        uzman_sirali = duz[sira]
        token_sirali = sira // self.secilen
        sayilar = torch.bincount(duz, minlength=self.uzman)
        baslangic = torch.cumsum(sayilar, 0) - sayilar
        yer = torch.arange(duz.numel(), device=x.device) - baslangic[uzman_sirali]  # uzman icindeki sira
        kapasite = int(sayilar.max())                               # tek senkron: tablo boyu
        kapasite = (kapasite + 63) // 64 * 64                       # 64'un kati: kart carpimi verimli
        tablo = x.new_zeros(self.uzman, kapasite, x.shape[1])
        tablo[uzman_sirali, yer] = x[token_sirali]
        kapi, yukari, asagi = self.kapi, self.yukari, self.asagi
        if torch.is_autocast_enabled(x.device.type):
            tur = torch.get_autocast_dtype(x.device.type)
            tablo, kapi, yukari, asagi = (t.to(tur) for t in (tablo, kapi, yukari, asagi))
        else:
            kapi, yukari, asagi = (t.to(tablo.dtype) for t in (kapi, yukari, asagi))
        h = F.silu(torch.bmm(tablo, kapi)) * torch.bmm(tablo, yukari)
        cikti = torch.bmm(h, asagi)                                 # (uzman, kapasite, genislik)
        # Sonuclari tokenin kendi sirasina geri koy ve her tokenin `secilen` katkisini topla. index_add kullanilmaz:
        # kartta ayni satira es zamanli eklemenin sirasi rastgeledir, sonuc son basamakta her calismada degisirdi.
        katki = torch.empty(duz.numel(), x.shape[1], dtype=ust, device=x.device)
        katki[sira] = cikti[uzman_sirali, yer].to(ust)
        katki = katki.view(-1, self.secilen, x.shape[1]) * agirlik.unsqueeze(-1)
        return y + katki.sum(1)

    def yonlendir(self, x):
        """Her token icin secilen uzmanlar ve agirliklari. Tam hassasiyette: bf16 yuvarlamasi tokeni yanlis
        uzmana gonderebilir. Secimde sapmali puan, agirlikta sapmasiz puan (DeepSeek-V3)."""
        ust = torch.promote_types(x.dtype, torch.float32)
        with torch.autocast(x.device.type, enabled=False):
            puan = torch.sigmoid(F.linear(x.to(ust), self.yonlendirici.weight.to(ust)))  # (token, uzman)
            _, sec = torch.topk(puan + self.sapma.to(ust), self.secilen, dim=-1)
            agirlik = puan.gather(1, sec)
            return sec, agirlik / agirlik.sum(-1, keepdim=True)

    @torch.no_grad()
    def denge_guncelle(self, hiz):
        if self.yuk.sum() > 0:
            self.sapma += hiz * torch.sign(self.yuk.mean() - self.yuk)
            self.yuk.zero_()


class OrtakUzman(IleriBesleme):
    """Her tokende calisan uzman: uzmanlarla ayni boyutta kucuk bir SwiGLU."""

    def __init__(self, genislik, ara):
        nn.Module.__init__(self)
        self.kapi = nn.Linear(genislik, ara, bias=False)
        self.yukari = nn.Linear(genislik, ara, bias=False)
        self.asagi = nn.Linear(ara, genislik, bias=False)


class Blok(nn.Module):
    def __init__(self, genislik, kafa, moe=None):
        super().__init__()
        self.norm1 = RMSNorm(genislik)
        self.dikkat = Dikkat(genislik, kafa)
        self.norm2 = RMSNorm(genislik)
        self.ileri = UzmanKatmani(genislik, **moe) if moe else IleriBesleme(genislik)

    def forward(self, x, cos, sin):
        x = x + self.dikkat(self.norm1(x), cos, sin)
        return x + self.ileri(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, sozluk, katman, kafa, genislik, baglam, moe=None):
        """moe: None = normal model; ya da dict(uzman=, secilen=, ortak=, uzman_ara=) = uzmanlar karisimi."""
        super().__init__()
        self.baglam = baglam
        self.bellek_tasarrufu = False
        self.yapi = dict(sozluk=sozluk, katman=katman, kafa=kafa, genislik=genislik, baglam=baglam)
        if moe:
            self.yapi["moe"] = dict(moe)
        self.gomme = nn.Embedding(sozluk, genislik)
        self.bloklar = nn.ModuleList(Blok(genislik, kafa, moe) for _ in range(katman))
        self.son_norm = RMSNorm(genislik)
        self.bas = nn.Linear(genislik, sozluk, bias=False)
        self.bas.weight = self.gomme.weight  # giris ve cikis ayni sozlugu paylasir
        cos, sin = rope_hazirla(genislik // kafa, baglam)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._ilk_deger)
        # Artik baglantilara giden katmanlari kucuk baslat: derin modelde egitim dengeli kalir
        for ad, p in self.named_parameters():
            if ad.endswith("cikis.weight") or ad.endswith("asagi.weight") or ad.endswith("ileri.asagi"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * katman))
            elif ad.endswith("ileri.kapi") or ad.endswith("ileri.yukari"):  # uzman tensorleri
                nn.init.normal_(p, mean=0.0, std=0.02)

    @staticmethod
    def _ilk_deger(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def parametre_sayisi(self):
        return sum(p.numel() for p in self.parameters())

    def aktif_parametre_sayisi(self):
        """Her token icin calisan parametre (MoE'de secilmeyen uzmanlar sayilmaz)."""
        toplam = self.parametre_sayisi()
        for b in self.bloklar:
            if isinstance(b.ileri, UzmanKatmani):
                u = b.ileri
                toplam -= (u.uzman - u.secilen) * (u.kapi[0].numel() + u.yukari[0].numel() + u.asagi[0].numel())
        return toplam

    def denge_guncelle(self, hiz=1e-3):
        """MoE: her optimizer adimindan sonra cagrilir; uzmanlarin yukunu dengeler."""
        for b in self.bloklar:
            if isinstance(b.ileri, UzmanKatmani):
                b.ileri.denge_guncelle(hiz)

    def uzman_yukleri(self):
        """Her katmanda uzmanlara giden token oranlari (denge kontrolu icin)."""
        return [b.ileri.yuk / b.ileri.yuk.sum().clamp(min=1) for b in self.bloklar if isinstance(b.ileri, UzmanKatmani)]

    def forward(self, girdi, hedef=None):
        U = girdi.shape[1]
        x = self.gomme(girdi)
        cos, sin = self.cos[:U], self.sin[:U]
        for blok in self.bloklar:
            if self.bellek_tasarrufu and self.training:
                # Ara sonuclari saklama, geri donuste yeniden hesapla: daha az bellek, biraz daha yavas
                x = checkpoint(blok, x, cos, sin, use_reentrant=False)
            else:
                x = blok(x, cos, sin)
        puan = self.bas(self.son_norm(x))
        if hedef is None:
            return puan, None
        # Kayip: modelin dogru sonraki tokena ne kadar dusuk ihtimal verdigi
        kayip = F.cross_entropy(puan.float().view(-1, puan.size(-1)), hedef.reshape(-1))
        return puan, kayip

    @torch.no_grad()
    def uret(self, girdi, yeni_token, sicaklik=0.8, en_iyi_k=50, bitis_id=None):
        """Bir sonraki tokeni tahmin et, ekle, tekrarla (Ders 1)."""
        for _ in range(yeni_token):
            parca = girdi[:, -self.baglam:]
            puan, _ = self(parca)
            puan = puan[:, -1, :] / sicaklik
            if en_iyi_k:
                esik = torch.topk(puan, en_iyi_k).values[:, -1, None]
                puan[puan < esik] = -float("inf")
            sonraki = torch.multinomial(F.softmax(puan, dim=-1), 1)
            girdi = torch.cat((girdi, sonraki), dim=1)
            if bitis_id is not None and sonraki.item() == bitis_id:
                break
        return girdi
