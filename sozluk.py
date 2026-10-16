"""Tokenizer surumleri tek yerde. Her model hangi sozlukle egitildiyse onunla konusur.

v1: 110M (kucuk). Satir sonlarini ve fazla bosluklari siler.
v2: 336M (orta) ve sonrasi. Satir sonu, bosluk ve sohbet isaretleri korunur (liste, paragraf, kod).

v2'de satir basindaki kelime "▁kelime" olarak parcalansin diye her satir sonunun arkasina
bir bosluk eklenir, cozerken geri alinir. Bu islem kayipsizdir: kodla -> coz ayni metni verir.
"""
import os

import sentencepiece as spm

KOK = os.path.dirname(os.path.abspath(__file__))
YOLLAR = {"v1": os.path.join("veri", "tokenizer.model"), "v2": os.path.join("veri", "v2", "tokenizer.model")}

# v2 sozlugundeki ozel parcalar (sozluk egitilirken bu sirayla eklenir)
OZEL = ["\n", "<|kullanici|>", "<|kelam|>", "<|sistem|>"]
KULLANICI, KELAM, SISTEM = "<|kullanici|>", "<|kelam|>", "<|sistem|>"


class Sozluk:
    def __init__(self, surum="v2", yol=None):
        self.surum = surum
        self.sp = spm.SentencePieceProcessor(model_file=yol or os.path.join(KOK, YOLLAR[surum]))
        self.eos = self.sp.eos_id()
        self.bos = self.sp.bos_id()

    def __len__(self):
        return self.sp.get_piece_size()

    def kodla(self, metin):
        if self.surum == "v2":
            metin = metin.replace("\n", "\n ")
        return self.sp.encode(metin)

    def kodla_hepsi(self, metinler, num_threads=-1):
        if self.surum == "v2":
            metinler = [m.replace("\n", "\n ") for m in metinler]
        return self.sp.encode(metinler, num_threads=num_threads)

    def coz(self, ids):
        metin = self.sp.decode(ids)
        return metin.replace("\n ", "\n") if self.surum == "v2" else metin

    def id(self, parca):
        return self.sp.piece_to_id(parca)

    # --- Sohbet bicimi ---
    # <bos><|kullanici|>soru\n<|kelam|>cevap<eos><|kullanici|>soru\n<|kelam|>cevap<eos> ...
    # v1 sozlukte ozel isaret yok; "Kullanıcı:" / "Kelam:" duz yazi olarak kullanilir.

    # Arac kullanimi: Kelam cevabinda ```kelam ... ``` kodu yazarsa kod guvenli alanda calisir, sonucu
    # "sistem" rolüyle konusmaya eklenir, Kelam okuyup devam eder (hata varsa duzeltir):
    # ...<|kelam|>kod<eos><|sistem|>Çıktı: ...\n<|kelam|>aciklama<eos>

    def _soru(self, metin, rol="kullanici"):
        """Kullanicinin ya da sistemin sozu + ardindan Kelam'in sirasi (bu tokenlardan ogrenilmez)."""
        if self.surum == "v2":
            isaret = KULLANICI if rol == "kullanici" else SISTEM
            return self.kodla(f"{isaret}{metin.strip()}\n{KELAM}")
        ad = "Kullanıcı" if rol == "kullanici" else "Sistem"
        return self.kodla(f"{ad}: {metin.strip()} Kelam:")

    def sohbet_kodla(self, mesajlar):
        """mesajlar: [{"rol": "kullanici"|"kelam"|"sistem", "metin": ...}, ...]
        Dondurur: (ids, ogren) - ogren[i] True ise o token Kelam'in cevabidir, kayip yalniz onlardan hesaplanir."""
        ids, ogren = [self.bos], [False]
        for m in mesajlar:
            if m["rol"] in ("kullanici", "sistem"):
                parca = self._soru(m["metin"], m["rol"])
                ids += parca
                ogren += [False] * len(parca)
            else:
                parca = self.kodla(m["metin"].strip()) + [self.eos]
                ids += parca
                # "ogren": False -> bu Kelam mesaji yalniz baglamdir (ör. hata-duzeltme orneklerindeki bozuk kod)
                ogren += [m.get("ogren", True)] * len(parca)
        return ids, ogren

    def sohbet_istemi(self, mesajlar):
        """Konusma gecmisi + son soru -> modelin cevaba baslayacagi token dizisi."""
        ids, _ = self.sohbet_kodla(mesajlar)
        return ids
