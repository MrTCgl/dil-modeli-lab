/**
 * İLERİ GEÇİŞİN AŞAMALARI
 * =======================
 *
 * Modelin bir tahmin üretirken yaptığı işi sıralı adımlara böler. Katman
 * sayısı değişince liste de değişir: her MLP bloğu dört aşama ekler
 * (genişlet → ReLU → daralt → artık bağlantı).
 *
 * Çarpım aşamalarının bir de "alt adımı" vardır: matristeki her tek tek
 * çarpma. Toplam alt adım = satır × sütun. Bu sayı Wgiriş için 2048'dir;
 * yani modelin tek bir karakteri tahmin etmesi, sadece o aşamada iki binden
 * fazla çarpma demek. Sayının kendisi de öğretici olduğu için ekranda yazıyor.
 */

import type { IleriIz, Model } from "./model.ts";

export type AsamaTuru =
  | "tokenler"
  | "gomme"
  | "giris"
  | "genisle"
  | "relu"
  | "daralt"
  | "artik"
  | "cikis"
  | "softmax";

export interface Asama {
  tur: AsamaTuru;
  /** MLP bloğu aşamalarında bloğun sırası. */
  blok: number;
  baslik: string;
  aciklama: string;
  /** Çarpım aşamalarında tek tek çarpma sayısı; diğerlerinde 0. */
  altAdim: number;
}

export function asamalariCikar(model: Model, iz: IleriIz): Asama[] {
  const D = model.ayar.D;
  const C = model.ayar.baglam;
  const V = iz.logits.length;
  const liste: Asama[] = [
    {
      tur: "tokenler",
      blok: -1,
      baslik: "Karakterler token id'ye dönüyor",
      aciklama: `Model harf görmez, sayı görür. Son ${C} karakterin her biri sözlükteki sırasıyla değiştirilir.`,
      altAdim: 0,
    },
    {
      tur: "gomme",
      blok: -1,
      baslik: "Gömme araması",
      aciklama: `Her token id, gömme tablosunda bir satırı işaret eder: ${D} sayı. ${C} karakter için ${C} vektör alınır ve uç uca eklenir — ${C * D} sayılık tek bir dizi.`,
      altAdim: 0,
    },
    {
      tur: "giris",
      blok: -1,
      baslik: "Giriş projeksiyonu",
      aciklama: `${C * D} sayı tek bir ${D}'lik vektöre iniyor. Bağlamdaki sıra bilgisi burada korunur: her pozisyonun kendi ağırlıkları var.`,
      altAdim: model.Wgiris.satir * model.Wgiris.sutun,
    },
  ];

  model.bloklar.forEach((blok, l) => {
    liste.push({
      tur: "genisle",
      blok: l,
      baslik: `Blok ${l + 1}: genişleme`,
      aciklama: `${D} sayı ${4 * D}'e açılıyor. Geniş ara katman, modelin ayırt edebileceği örüntü sayısını artırır.`,
      altAdim: blok.W1.satir * blok.W1.sutun,
    });
    liste.push({
      tur: "relu",
      blok: l,
      baslik: `Blok ${l + 1}: ReLU`,
      aciklama:
        "Negatif olan her şey sıfırlanır. Modelin doğrusal olmayan tek yeri burası — bu olmasaydı üst üste konan bütün katmanlar tek bir çarpıma indirgenirdi.",
      altAdim: 0,
    });
    liste.push({
      tur: "daralt",
      blok: l,
      baslik: `Blok ${l + 1}: daralma`,
      aciklama: `${4 * D} sayı tekrar ${D}'ya iniyor. Çıkan vektör ana yola eklenmek üzere hazır.`,
      altAdim: blok.W2.satir * blok.W2.sutun,
    });
    liste.push({
      tur: "artik",
      blok: l,
      baslik: `Blok ${l + 1}: artık bağlantı`,
      aciklama:
        "Bloğun ürettiği vektör, bloğa giren vektörün üstüne eklenir — yerine geçmez. Blok böylece ana yolu değiştirmek zorunda kalmadan ona küçük bir düzeltme ekler.",
      altAdim: 0,
    });
  });

  liste.push({
    tur: "cikis",
    blok: -1,
    baslik: "Çıkış katmanı",
    aciklama: `Son ${D}'lik vektör, sözlükteki ${V} karakterin her biri için bir skora çevriliyor. Bu ham skorlara logits denir.`,
    altAdim: model.Wcikis.satir * model.Wcikis.sutun,
  });

  liste.push({
    tur: "softmax",
    blok: -1,
    baslik: "Softmax",
    aciklama:
      "Skorlar olasılığa dönüşüyor: her biri üstel alınır, sonra toplamlarına bölünür. Çıkan sayılar artık toplamı 1 olan bir dağılım.",
    altAdim: 0,
  });

  return liste;
}

/** Bir aşamada ekranda "o anki ham vektör" olarak ne gösterilecek. */
export function asamaVektoru(asama: Asama, iz: IleriIz): { ad: string; veri: Float32Array } {
  switch (asama.tur) {
    case "tokenler":
    case "gomme":
      return { ad: `birleşik bağlam (${iz.birlesik.length})`, veri: iz.birlesik };
    case "giris":
      return { ad: `h0 (${iz.h0.length})`, veri: iz.h0 };
    case "genisle":
      return { ad: `ReLU öncesi (${iz.bloklar[asama.blok].oncesi.length})`, veri: iz.bloklar[asama.blok].oncesi };
    case "relu":
      return { ad: `ReLU sonrası (${iz.bloklar[asama.blok].relu.length})`, veri: iz.bloklar[asama.blok].relu };
    case "daralt":
      return { ad: `dal çıktısı (${iz.bloklar[asama.blok].dal.length})`, veri: iz.bloklar[asama.blok].dal };
    case "artik":
      return { ad: `blok çıktısı (${iz.bloklar[asama.blok].cikti.length})`, veri: iz.bloklar[asama.blok].cikti };
    case "cikis":
    case "softmax":
      return { ad: `logits (${iz.logits.length})`, veri: iz.logits };
  }
}
