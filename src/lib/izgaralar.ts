/**
 * IZGARA TANIMLARI
 * ================
 * Hangi matrisin hangi açıklamayla ve hangi eksen etiketleriyle
 * gösterileceği. Hem ağırlık ekranı hem eğitim paneli aynı listeyi
 * kullanıyor — iki yerde ayrı ayrı tanımlanıp zamanla ayrışmasınlar.
 */

import type { Matris, Model } from "./model.ts";
import { gorunurAd } from "./tokenizer.ts";

export interface IzgaraTanimi {
  matris: Matris;
  aciklama: string;
  satirEtiketi?: (i: number) => string | null;
  sutunEtiketi?: (j: number) => string | null;
}

/** Sayısal eksen etiketi: her hücreye yazmak kalabalık eder, seyreltiyoruz. */
function seyrek(aralik: number) {
  return (i: number) => (i % aralik === 0 ? String(i) : null);
}

export function izgaralariTopla(model: Model): IzgaraTanimi[] {
  const D = model.ayar.D;
  const C = model.ayar.baglam;

  const liste: IzgaraTanimi[] = [
    {
      matris: model.E,
      aciklama:
        "Gömme tablosu. Her satır bir karakterin vektörü. Bir satırı bozmak o harfin modeldeki anlamını bozar.",
      satirEtiketi: (i) => gorunurAd(i),
      sutunEtiketi: seyrek(4),
    },
  ];

  if (model.P) {
    liste.push({
      matris: model.P,
      aciklama:
        "Pozisyon gömmeleri. Dikkat kendi başına sıraya kör — \"ab\" ile \"ba\" ona aynı görünür. Her konuma öğrenilebilir bir vektör eklemek sırayı geri kazandırır.",
      satirEtiketi: (i) => `t−${C - i}`,
      sutunEtiketi: seyrek(4),
    });
  }

  if (model.dikkat) {
    const d = model.dikkat;
    liste.push(
      {
        matris: d.Wq,
        aciklama:
          "Sorgu projeksiyonu. Her konumun \"ben ne arıyorum\" vektörünü üretir; bu vektör diğer konumların anahtarlarıyla çarpılarak dikkat skorlarını verir.",
        satirEtiketi: seyrek(4),
        sutunEtiketi: seyrek(4),
      },
      {
        matris: d.Wk,
        aciklama: "Anahtar projeksiyonu. Her konumun \"bende ne var\" vektörü — sorgularla eşleşen taraf.",
        satirEtiketi: seyrek(4),
        sutunEtiketi: seyrek(4),
      },
      {
        matris: d.Wv,
        aciklama:
          "Değer projeksiyonu. Bir konuma bakılırsa ne aktarılacağı. Dikkat ağırlıkları bu vektörleri harmanlar.",
        satirEtiketi: seyrek(4),
        sutunEtiketi: seyrek(4),
      },
      {
        matris: d.Wo,
        aciklama: "Dikkatin çıkış projeksiyonu. Harmanlanan değeri ana yola eklenecek biçime sokar.",
        satirEtiketi: seyrek(4),
        sutunEtiketi: seyrek(4),
      },
    );
  }

  if (model.Wgiris) {
    liste.push({
      matris: model.Wgiris,
      aciklama: `Giriş projeksiyonu. Bağlamdaki ${C} gömme uç uca eklenir (${C}×${D}=${C * D} sayı) ve buradan tek bir ${D}'lık vektöre iner. Satır etiketleri hangi pozisyona ait olduğunu gösterir: t−1 en son karakter.`,
      satirEtiketi: (i) => (i % D === 0 ? `t−${C - Math.floor(i / D)}` : null),
      sutunEtiketi: seyrek(4),
    });
  }

  model.bloklar.forEach((blok) => {
    liste.push({
      matris: blok.W1,
      aciklama: `Bloğun genişleyen katmanı: ${D} sayı ${4 * D}'e açılır, ardından ReLU negatifleri siler.`,
      satirEtiketi: seyrek(4),
      sutunEtiketi: seyrek(8),
    });
    liste.push({
      matris: blok.W2,
      aciklama: `Bloğun daralan katmanı: ${4 * D} sayı ${D}'ya iner ve ana yola eklenir (artık bağlantı).`,
      satirEtiketi: seyrek(8),
      sutunEtiketi: seyrek(4),
    });
  });

  liste.push({
    matris: model.Wcikis,
    aciklama:
      "Çıkış katmanı. Her sütun bir karaktere ait: o karakterin skoru bu sütunla çarpımdan çıkar. Bir sütunu yukarı çekmek o harfi modele sevdirir.",
    satirEtiketi: seyrek(4),
    sutunEtiketi: (j) => gorunurAd(j),
  });

  return liste;
}
