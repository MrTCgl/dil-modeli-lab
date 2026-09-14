/**
 * AĞIRLIK RENK SKALASI
 * ====================
 *
 * Izgaradaki renk iki şeyi birden taşır: işaret ve büyüklük.
 *   negatif -> soğuk (turkuaz)
 *   sıfır   -> zeminle aynı koyu ton
 *   pozitif -> sıcak (kehribar)
 *
 * Kırmızı-yeşil bilerek kullanılmadı: en yaygın renk körlüğü türünde bu iki
 * uç ayırt edilemez. Turkuaz-kehribar çifti hem renk körlüğünde hem gri
 * basımda ayrışır, çünkü aradaki fark sadece renkte değil parlaklıkta da var.
 *
 * Geçişler Lab renk uzayında hesaplanır; sRGB'de doğrusal karıştırma ara
 * tonları çamurlaştırır.
 */

import { interpolateLab } from "d3-interpolate";

export const HUCRE_SIFIR = "#1b212b";
export const HUCRE_NEGATIF = "#45c8bb";
export const HUCRE_POZITIF = "#eda14a";

const negatifSkala = interpolateLab(HUCRE_SIFIR, HUCRE_NEGATIF);
const pozitifSkala = interpolateLab(HUCRE_SIFIR, HUCRE_POZITIF);

/**
 * Bir ağırlığı renge çevirir. `olcek` skalanın ucu: bu değerdeki (ve
 * üstündeki) ağırlıklar tam doygun görünür.
 *
 * Karekök neden var? Ağırlıkların çoğu sıfıra yakındır; doğrusal eşleme
 * ızgaranın tamamını zemin rengine boğar ve yapı görünmez olur. Karekök
 * küçük değerlerin kontrastını açar, büyükleri sıkıştırır. Bu bir yorumdur
 * ve gizlenmemeli: skala göstergesi hangi rengin hangi sayıya karşılık
 * geldiğini sayıyla birlikte yazar.
 */
export function agirlikRengi(deger: number, olcek: number): string {
  if (!(olcek > 0) || !Number.isFinite(deger)) return HUCRE_SIFIR;
  const oran = Math.min(1, Math.abs(deger) / olcek);
  const t = Math.sqrt(oran);
  return deger >= 0 ? pozitifSkala(t) : negatifSkala(t);
}

/**
 * Skalanın ucunu verinin kendisinden bulur.
 * En büyük mutlak değeri almak yerine %99,5'lik dilimi alıyoruz: tek bir
 * aykırı ağırlık bütün ızgarayı soluklaştırmasın. Aykırı değer yine de
 * doygun görünür, sadece skalayı ele geçirmez.
 */
export function olcekBul(veri: Float32Array, dilim = 0.995): number {
  const n = veri.length;
  if (n === 0) return 1;
  const mutlak = new Float32Array(n);
  for (let i = 0; i < n; i++) mutlak[i] = Math.abs(veri[i]);
  mutlak.sort();
  const indeks = Math.min(n - 1, Math.floor(n * dilim));
  return Math.max(mutlak[indeks], 1e-6);
}

/** Izgara başlığında gösterilen özet istatistikler. */
export interface Ozet {
  enKucuk: number;
  enBuyuk: number;
  ortalama: number;
  olcek: number;
}

export function ozetCikar(veri: Float32Array): Ozet {
  let enKucuk = Infinity;
  let enBuyuk = -Infinity;
  let toplam = 0;
  for (let i = 0; i < veri.length; i++) {
    const v = veri[i];
    if (v < enKucuk) enKucuk = v;
    if (v > enBuyuk) enBuyuk = v;
    toplam += v;
  }
  return {
    enKucuk,
    enBuyuk,
    ortalama: toplam / veri.length,
    olcek: olcekBul(veri),
  };
}
