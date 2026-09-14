/**
 * KAYIP EĞRİSİ
 * ============
 *
 * Eğitim sırasında modelin ne kadar yanıldığını zamana karşı çizer.
 *
 * ÜÇ ÇİZGİ, ÜÇ AYRI ŞEY
 *   - soluk ince çizgi: her adımın ham kaybı. Gürültülü, çünkü her adım
 *     metinden rastgele seçilmiş 32 örneğe bakıyor.
 *   - kalın çizgi: ham kaybın yürüyen ortalaması. Eğilim burada okunur.
 *   - kesikli çizgi: doğrulama kaybı — modelin hiç eğitilmediği metin
 *     parçasındaki başarısı. Eğitim kaybı düşerken bu düşmeyi bırakırsa
 *     model ezberlemeye başlamış demektir.
 *   - en üstteki noktalı yatay çizgi: ln(32) = 3.47, yani hiçbir şey
 *     bilmeyen bir modelin kaybı. Eğrinin altına inmesi "öğrendi" demek.
 *
 * RENK YOK, BİLEREK
 * Bu uygulamada renk zaten iki şey anlatıyor: ağırlığın işareti
 * (turkuaz/kehribar) ve karakterin insan etiketi (küredeki mor/gri/pembe).
 * Eğriye üçüncü bir renk dili eklemek yerine çizgileri kalınlık ve
 * kesiklilikle ayırdık.
 */

import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { RASTGELE_KAYIP } from "../lib/model.ts";
import { sayi } from "../lib/bicim.ts";

export interface DogrulamaNoktasi {
  adim: number;
  deger: number;
}

export interface KayipEgrisiProps {
  /** Her eğitim adımının ham kaybı, sırayla. */
  kayiplar: number[];
  dogrulama: DogrulamaNoktasi[];
  genislik: number;
  yukseklik?: number;
}

/** Yürüyen ortalama: eğilimi gürültüden ayırır. */
function yumusat(veri: number[], pencere: number): number[] {
  if (veri.length === 0) return [];
  const sonuc = new Array<number>(veri.length);
  let toplam = 0;
  for (let i = 0; i < veri.length; i++) {
    toplam += veri[i];
    if (i >= pencere) toplam -= veri[i - pencere];
    sonuc[i] = toplam / Math.min(i + 1, pencere);
  }
  return sonuc;
}

/** Çok nokta varsa seyrelt: 4000 noktalı bir SVG yolu boşuna ağırdır. */
function seyrelt<T>(veri: T[], enFazla: number): Array<[number, T]> {
  if (veri.length <= enFazla) return veri.map((v, i) => [i, v]);
  const adim = veri.length / enFazla;
  const sonuc: Array<[number, T]> = [];
  for (let i = 0; i < enFazla; i++) {
    const indeks = Math.min(veri.length - 1, Math.floor(i * adim));
    sonuc.push([indeks, veri[indeks]]);
  }
  return sonuc;
}

const KENAR = { ust: 12, sag: 12, alt: 26, sol: 42 };

export function KayipEgrisi({ kayiplar, dogrulama, genislik, yukseklik = 230 }: KayipEgrisiProps) {
  const cizimG = Math.max(120, genislik - KENAR.sol - KENAR.sag);
  const cizimY = yukseklik - KENAR.ust - KENAR.alt;

  const { hamYol, yumusakYol, dogrulamaYol, xOlcek, yOlcek, sonKayip, sonDogrulama } = useMemo(() => {
    const n = kayiplar.length;
    const enBuyukAdim = Math.max(1, n - 1);

    let enBuyukKayip = RASTGELE_KAYIP;
    for (const k of kayiplar) if (Number.isFinite(k) && k > enBuyukKayip) enBuyukKayip = k;
    for (const d of dogrulama) if (Number.isFinite(d.deger) && d.deger > enBuyukKayip) enBuyukKayip = d.deger;

    const x = scaleLinear().domain([0, enBuyukAdim]).range([0, cizimG]);
    const y = scaleLinear().domain([0, enBuyukKayip * 1.06]).range([cizimY, 0]).nice();

    const ciz = line<[number, number]>()
      .x((d) => x(d[0]))
      .y((d) => y(d[1]));

    const ham = seyrelt(kayiplar, 900).map(([i, v]) => [i, v] as [number, number]);
    const pencere = Math.max(5, Math.round(n / 60));
    const yumusak = seyrelt(yumusat(kayiplar, pencere), 900).map(([i, v]) => [i, v] as [number, number]);
    const dogr = dogrulama
      .filter((d) => Number.isFinite(d.deger))
      .map((d) => [d.adim, d.deger] as [number, number]);

    return {
      hamYol: n > 1 ? ciz(ham) : null,
      yumusakYol: n > 1 ? ciz(yumusak) : null,
      dogrulamaYol: dogr.length > 1 ? ciz(dogr) : null,
      xOlcek: x,
      yOlcek: y,
      sonKayip: n > 0 ? yumusat(kayiplar.slice(-60), 60).at(-1) ?? null : null,
      sonDogrulama: dogrulama.length > 0 ? dogrulama[dogrulama.length - 1].deger : null,
    };
  }, [kayiplar, dogrulama, cizimG, cizimY]);

  const yTikler = yOlcek.ticks(5);
  const xTikler = xOlcek.ticks(6);
  const rastgeleY = yOlcek(RASTGELE_KAYIP);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="sayi text-[13px] text-metin">
          eğitim kaybı {sonKayip == null ? "—" : sayi(sonKayip, 4)}
        </span>
        <span className="sayi text-[13px] text-soluk">
          doğrulama {sonDogrulama == null ? "—" : sayi(sonDogrulama, 4)}
        </span>
        <span className="sayi ml-auto text-[11px] text-cok-soluk">
          rastgele tahminin kaybı ln(32) = {sayi(RASTGELE_KAYIP, 4)}
        </span>
      </div>

      <svg width={genislik} height={yukseklik} role="img" aria-label="eğitim kaybı eğrisi">
        <g transform={`translate(${KENAR.sol},${KENAR.ust})`}>
          {yTikler.map((t) => (
            <g key={t} transform={`translate(0,${yOlcek(t)})`}>
              <line x1={0} x2={cizimG} stroke="#1d222c" strokeWidth={1} />
              <text x={-8} dy="0.32em" textAnchor="end" className="sayi" fill="#5e6878" fontSize={10}>
                {t}
              </text>
            </g>
          ))}
          {xTikler.map((t) => (
            <text
              key={t}
              x={xOlcek(t)}
              y={cizimY + 16}
              textAnchor="middle"
              className="sayi"
              fill="#5e6878"
              fontSize={10}
            >
              {t}
            </text>
          ))}

          {/* rastgele modelin kaybı: eğrinin buranın altına inmesi öğrenmenin tanımı */}
          <line
            x1={0}
            x2={cizimG}
            y1={rastgeleY}
            y2={rastgeleY}
            stroke="#4a5462"
            strokeWidth={1}
            strokeDasharray="2 4"
          />

          {hamYol && <path d={hamYol} fill="none" stroke="#6d7787" strokeWidth={1} opacity={0.5} />}
          {dogrulamaYol && (
            <path d={dogrulamaYol} fill="none" stroke="#aab4c4" strokeWidth={1.5} strokeDasharray="5 4" />
          )}
          {yumusakYol && <path d={yumusakYol} fill="none" stroke="#e6ebf2" strokeWidth={2} />}
        </g>
      </svg>

      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-cok-soluk">
        <Gosterge kalinlik={2} renk="#e6ebf2" ad="eğitim kaybı (yürüyen ortalama)" />
        <Gosterge kalinlik={1} renk="#6d7787" ad="her adımın ham kaybı" />
        <Gosterge kalinlik={1.5} renk="#aab4c4" kesik="5 4" ad="doğrulama kaybı (görülmemiş metin)" />
        <Gosterge kalinlik={1} renk="#4a5462" kesik="2 4" ad="rastgele tahmin" />
        <span className="ml-auto">yatay eksen: eğitim adımı</span>
      </div>
    </div>
  );
}

function Gosterge({
  kalinlik,
  renk,
  kesik,
  ad,
}: {
  kalinlik: number;
  renk: string;
  kesik?: string;
  ad: string;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width={18} height={6} aria-hidden>
        <line x1={0} x2={18} y1={3} y2={3} stroke={renk} strokeWidth={kalinlik} strokeDasharray={kesik} />
      </svg>
      {ad}
    </span>
  );
}
