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
  /** Gradyan inişinin adım büyüklüğü. */
  ogrenmeOrani: number;
  /** "Eğit" düğmesine basınca kaç adım atılacak. */
  adimSayisi: number;
  /** İleri geçiş ekranındaki çarpım canlandırmasının hızı (kare başına çarpma). */
  hiz: number;
  /** Dikkat katmanı açık mı. */
  dikkat: boolean;
  /** Her konum kendi sonraki karakterini tahmin etsin mi (yalnızca dikkat açıkken). */
  tumKonumlar: boolean;
}

export const VARSAYILAN_DURUM: Durum = {
  seed: 1,
  D: 16,
  katman: 2,
  sicaklik: 1,
  metin: "kasabada deniz",
  ogrenmeOrani: 0.2,
  adimSayisi: 600,
  hiz: 24,
  dikkat: true,
  tumKonumlar: true,
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
    ogrenmeOrani: sayiOku(p.get("lr"), VARSAYILAN_DURUM.ogrenmeOrani, 0.001, 1),
    adimSayisi: Math.round(sayiOku(p.get("adim"), VARSAYILAN_DURUM.adimSayisi, 1, 2000)),
    hiz: sayiOku(p.get("hiz"), VARSAYILAN_DURUM.hiz, 0.25, 160),
    dikkat: p.get("dikkat") === null ? VARSAYILAN_DURUM.dikkat : p.get("dikkat") !== "0",
    tumKonumlar:
      p.get("konumlar") === null ? VARSAYILAN_DURUM.tumKonumlar : p.get("konumlar") !== "0",
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
  if (durum.ogrenmeOrani !== VARSAYILAN_DURUM.ogrenmeOrani) p.set("lr", String(durum.ogrenmeOrani));
  if (durum.adimSayisi !== VARSAYILAN_DURUM.adimSayisi) p.set("adim", String(durum.adimSayisi));
  if (durum.hiz !== VARSAYILAN_DURUM.hiz) p.set("hiz", String(durum.hiz));
  if (durum.dikkat !== VARSAYILAN_DURUM.dikkat) p.set("dikkat", durum.dikkat ? "1" : "0");
  if (durum.tumKonumlar !== VARSAYILAN_DURUM.tumKonumlar)
    p.set("konumlar", durum.tumKonumlar ? "1" : "0");
  const sorgu = p.toString();
  const yeni = window.location.pathname + (sorgu ? `?${sorgu}` : "");
  window.history.replaceState(null, "", yeni);
}
