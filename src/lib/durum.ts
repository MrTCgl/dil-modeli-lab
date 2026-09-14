/**
 * URL DURUMU
 * ==========
 *
 * Ekrandaki her ayar adres çubuğunda yaşar. Sebebi basit: birine "şuna bak"
 * demek istediğinizde link yeterli olsun. Aynı tohum aynı modeli ürettiği
 * için link aynı sayıları açar.
 *
 * Durum tarayıcı geçmişini kirletmeden güncellenir (replaceState): kaydıracı
 * oynatan biri geri tuşuna bastığında elli adım geri gitmesin, sayfadan
 * çıksın.
 */

export interface Durum {
  seed: number;
  D: number;
  katman: number;
  sicaklik: number;
  metin: string;
}

export const VARSAYILAN_DURUM: Durum = {
  seed: 1,
  D: 16,
  katman: 2,
  sicaklik: 1,
  metin: "kasabada deniz",
};

function sayiOku(deger: string | null, varsayilan: number, enAz: number, enCok: number): number {
  if (deger === null) return varsayilan;
  const n = Number(deger);
  if (!Number.isFinite(n)) return varsayilan;
  return Math.min(enCok, Math.max(enAz, n));
}

export function durumOku(): Durum {
  if (typeof window === "undefined") return VARSAYILAN_DURUM;
  const p = new URLSearchParams(window.location.search);
  return {
    seed: Math.round(sayiOku(p.get("seed"), VARSAYILAN_DURUM.seed, 0, 999999)),
    D: Math.round(sayiOku(p.get("d"), VARSAYILAN_DURUM.D, 4, 64)),
    katman: Math.round(sayiOku(p.get("katman"), VARSAYILAN_DURUM.katman, 1, 4)),
    sicaklik: sayiOku(p.get("sicaklik"), VARSAYILAN_DURUM.sicaklik, 0.1, 2),
    metin: p.get("metin") ?? VARSAYILAN_DURUM.metin,
  };
}

export function durumYaz(durum: Durum): void {
  if (typeof window === "undefined") return;
  const p = new URLSearchParams();
  if (durum.seed !== VARSAYILAN_DURUM.seed) p.set("seed", String(durum.seed));
  if (durum.D !== VARSAYILAN_DURUM.D) p.set("d", String(durum.D));
  if (durum.katman !== VARSAYILAN_DURUM.katman) p.set("katman", String(durum.katman));
  if (durum.sicaklik !== VARSAYILAN_DURUM.sicaklik) p.set("sicaklik", String(durum.sicaklik));
  if (durum.metin !== VARSAYILAN_DURUM.metin) p.set("metin", durum.metin);
  const sorgu = p.toString();
  const yeni = window.location.pathname + (sorgu ? `?${sorgu}` : "");
  window.history.replaceState(null, "", yeni);
}
