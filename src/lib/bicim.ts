/** Sayı biçimlendirme. Ekranda her sayı aynı genişlikte durmalı. */

/** Ağırlık ve aktivasyon değerleri için: işaretli, sabit basamaklı. */
export function sayi(x: number, basamak = 3): string {
  if (!Number.isFinite(x)) return x > 0 ? "∞" : Number.isNaN(x) ? "—" : "-∞";
  const s = x.toFixed(basamak);
  return s === "-" + (0).toFixed(basamak) ? (0).toFixed(basamak) : s;
}

/** Yüzde gösterimi: olasılık çubuklarının yanında. */
export function yuzde(x: number, basamak = 1): string {
  return (x * 100).toFixed(basamak) + "%";
}

/** Büyük sayılar: parametre adedi gibi. */
export function adet(x: number): string {
  return x.toLocaleString("tr-TR");
}
