/**
 * ÇARPIM ANİMASYONUNUN ARİTMETİĞİ
 * ===============================
 *
 * Animasyonun her karesinde "şu an hangi terim ekleniyor, ara toplam ne"
 * sorusunu cevaplar. Bileşenden ayrı bir dosyada durmasının sebebi, konsol
 * betiğinin bunu doğrudan sınayabilmesi: ekranda biriken toplamın modelin
 * ileri geçişte ürettiği sayıyla birebir aynı çıktığını kanıtlamak istiyoruz.
 * Animasyon "yaklaşık olarak şöyle oluyor" demiyor; aynı işlemi yapıyor.
 *
 * FLOAT32 AYRINTISI
 * Model ağırlıkları ve ara sonuçları Float32Array'de tutar; yani her toplama
 * işleminden sonra sonuç 32 bit kesinliğe yuvarlanır. JavaScript'te normal
 * sayı aritmetiği 64 bittir. Ara toplamı düz toplasaydık ekrandaki sayı
 * modelinkinden son basamaklarda ayrılırdı. Math.fround her adımda aynı
 * yuvarlamayı uygulayarak ikisini birebir eşitler — konsol betiği bunu
 * her çıkış için tek tek sınıyor.
 */

import type { Matris } from "./model.ts";

export interface CarpimKonum {
  /** Şu an işlenen giriş (matrisin satırı). */
  i: number;
  /** Şu an doldurulan çıkış (matrisin sütunu). */
  j: number;
  bitti: boolean;
  /** i'inci terim eklenmeden önceki ara toplam (yanlılık dahil). */
  oncekiToplam: number;
  /** x[i] * W[i][j] */
  terim: number;
  sonrakiToplam: number;
}

export function carpimKonumu(
  x: Float32Array,
  W: Matris,
  b: Float32Array | undefined,
  altAdim: number,
): CarpimKonum {
  const toplamAdim = W.satir * W.sutun;
  const bitti = altAdim >= toplamAdim;
  const guvenli = Math.max(0, Math.min(altAdim, toplamAdim - 1));
  const j = Math.floor(guvenli / W.satir);
  const i = guvenli - j * W.satir;

  // Ara toplam yanlılıkla başlar: y[j] = b[j] + toplam_i x[i]*W[i][j].
  // Ekranda ilk terim eklenmeden önce görünen sayı b[j]'dir, sıfır değil.
  let oncekiToplam = b ? b[j] : 0;
  for (let k = 0; k < i; k++) {
    oncekiToplam = Math.fround(oncekiToplam + x[k] * W.veri[k * W.sutun + j]);
  }
  const terim = x[i] * W.veri[i * W.sutun + j];

  return {
    i,
    j,
    bitti,
    oncekiToplam,
    terim,
    sonrakiToplam: Math.fround(oncekiToplam + terim),
  };
}

/** Bir çıkışın bütün terimleri toplandığındaki değer. */
export function carpimSonucu(
  x: Float32Array,
  W: Matris,
  b: Float32Array | undefined,
  j: number,
): number {
  let toplam = b ? b[j] : 0;
  for (let k = 0; k < W.satir; k++) {
    toplam = Math.fround(toplam + x[k] * W.veri[k * W.sutun + j]);
  }
  return toplam;
}
