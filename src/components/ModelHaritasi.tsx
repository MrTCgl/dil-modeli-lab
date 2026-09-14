/**
 * MODEL HARİTASI
 * ==============
 *
 * Modelin bütün matrislerini küçük ızgaralar hâlinde yan yana koyar ve
 * hangilerinin çıkarım sırasında güncellendiğini gösterir.
 *
 * Ayrım için yeni bir renk eklemedik, mevcut rengi kaldırdık: donmuş
 * matrisler gri, güncellenen matris normal turkuaz-kehribar skalasında.
 * Renk yine tek bir şey anlatıyor (değerin işareti), üstüne "canlı olan
 * renkli" sezgisi bedavaya biniyor.
 */

import { useEffect, useMemo, useRef } from "react";
import type { Matris } from "../lib/model.ts";
import { agirlikRengi, donukRenk, olcekBul } from "../lib/renk.ts";
import { adet, yuzde } from "../lib/bicim.ts";

export interface HaritaOgesi {
  matris: Matris;
  hizli: boolean;
}

function MiniIzgara({ matris, hizli, enBuyukKenar }: HaritaOgesi & { enBuyukKenar: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const olcek = useMemo(() => olcekBul(matris.veri), [matris]);

  // Uzun-dar matrisler burada da çevrilir, yoksa harita bir iğneler dizisi olur.
  const cevrik = matris.satir > matris.sutun;
  const gSatir = cevrik ? matris.sutun : matris.satir;
  const gSutun = cevrik ? matris.satir : matris.sutun;
  const hucre = Math.max(1, Math.floor(enBuyukKenar / Math.max(gSatir, gSutun)));

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const g = gSutun * hucre;
    const y = gSatir * hucre;
    cv.width = Math.round(g * dpr);
    cv.height = Math.round(y * dpr);
    cv.style.width = `${g}px`;
    cv.style.height = `${y}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, g, y);
    for (let i = 0; i < gSatir; i++) {
      for (let j = 0; j < gSutun; j++) {
        const giris = cevrik ? j : i;
        const cikis = cevrik ? i : j;
        const deger = matris.veri[giris * matris.sutun + cikis];
        ctx.fillStyle = hizli ? agirlikRengi(deger, olcek) : donukRenk(deger, olcek);
        ctx.fillRect(j * hucre, i * hucre, hucre, hucre);
      }
    }
  }, [matris, hizli, olcek, hucre, gSatir, gSutun, cevrik]);

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={ref}
        className={hizli ? "outline outline-1 outline-cizgi-parlak" : "opacity-80"}
        aria-label={`${matris.ad}, ${hizli ? "çıkarım sırasında güncelleniyor" : "donmuş"}`}
      />
      <span className={`text-[10px] ${hizli ? "text-metin" : "text-cok-soluk"}`}>{matris.ad}</span>
      <span className="sayi text-[9px] text-cok-soluk">{adet(matris.veri.length)}</span>
    </div>
  );
}

export function ModelHaritasi({
  ogeler,
  hizliSayi,
  toplamSayi,
}: {
  ogeler: HaritaOgesi[];
  hizliSayi: number;
  toplamSayi: number;
}) {
  return (
    <section className="border border-cizgi bg-yuzey">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-cizgi px-3 py-2">
        <h2 className="text-[13px] font-medium text-metin">Hangi parametreler oynuyor</h2>
        <span className="sayi text-[11px] text-soluk">
          {adet(hizliSayi)} / {adet(toplamSayi)} ={" "}
          <span className="text-metin">{yuzde(hizliSayi / toplamSayi, 1)}</span>
        </span>
        <span className="ml-auto text-[10px] text-cok-soluk">
          renkli = çıkarım sırasında güncelleniyor · gri = donmuş
        </span>
      </header>
      <div className="flex flex-wrap items-end gap-4 px-3 py-3">
        {ogeler.map((o) => (
          <MiniIzgara key={o.matris.ad} matris={o.matris} hizli={o.hizli} enBuyukKenar={64} />
        ))}
      </div>
    </section>
  );
}
