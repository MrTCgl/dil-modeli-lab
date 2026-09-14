/**
 * MATRİS ÇARPIMI, HÜCRE HÜCRE
 * ===========================
 *
 * Tek bir formülü canlandırır:
 *
 *     y[j] = b[j] + toplam_i x[i] * W[i][j]
 *
 * Yani: çıkıştaki her sayı, girişteki bütün sayıların bir ağırlıklı
 * toplamıdır. Ekranda o toplamın tek tek birikişini izliyorsunuz — hangi
 * giriş hangi ağırlıkla çarpılıyor, ara toplam nereye gidiyor.
 *
 * DÜZEN
 * Matrisin satırları girişe, sütunları çıkışa karşılık gelir. Uzun ve dar
 * matrisleri (Wgiriş 128×16 gibi) olduğu gibi çizince ekranda okunmaz bir
 * iğne oluyor; bu yüzden satır sayısı sütun sayısını aşınca çizimi 90°
 * çeviriyoruz. İki düzende de şu değişmez: x girişin şeridi, y çıkışın
 * şeridi, aradaki dikdörtgen de ağırlıklar.
 *
 * Ekrandaki ara toplam gerçekten x ve W'den yeniden hesaplanır — animasyonu
 * süslemek için önceden hesaplanmış bir sayı gösterilmiyor. Son terim
 * eklendiğinde çıkan değer, modelin ileri geçişte ürettiği değerin aynısıdır.
 */

import { useCallback, useEffect, useMemo, useRef } from "react";
import type { Matris } from "../lib/model.ts";
import { carpimKonumu } from "../lib/carpim.ts";
import { agirlikRengi, olcekBul } from "../lib/renk.ts";
import { sayi } from "../lib/bicim.ts";

const ETIKET = 28;
const SERIT = 16;
const ARA = 10;

export interface CarpimCanlandirmaProps {
  /** Giriş vektörü; uzunluğu W.satir kadar. */
  x: Float32Array;
  W: Matris;
  b?: Float32Array;
  /** Modelin hesapladığı çıkış; tamamlanan sütunlarda bu gösterilir. */
  y: Float32Array;
  /** Kaç terim işlendi. 0 = hiç, W.satir*W.sutun = bitti. */
  altAdim: number;
  maxGenislik?: number;
  /** Giriş konumlarının adı (bağlam gömmelerinde "t−3 · boyut 5" gibi). */
  xEtiketi?: (i: number) => string | null;
  yEtiketi?: (j: number) => string | null;
}

export function CarpimCanlandirma({
  x,
  W,
  b,
  y,
  altAdim,
  maxGenislik = 900,
  xEtiketi,
  yEtiketi,
}: CarpimCanlandirmaProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Satır sayısı sütun sayısını aşarsa çizimi çeviriyoruz: geniş bir
  // dikdörtgen her zaman uzun bir iğneden okunaklı.
  const cevrik = W.satir > W.sutun;
  const gSatir = cevrik ? W.sutun : W.satir;
  const gSutun = cevrik ? W.satir : W.sutun;

  const hucre = useMemo(() => {
    const genislikPayi = maxGenislik - ETIKET - SERIT - ARA - 30;
    const yukseklikPayi = 300;
    return Math.max(3, Math.min(26, Math.min(
      Math.floor(genislikPayi / gSutun),
      Math.floor(yukseklikPayi / gSatir),
    )));
  }, [maxGenislik, gSatir, gSutun]);

  const konum = useMemo(() => carpimKonumu(x, W, b, altAdim), [x, W, b, altAdim]);
  const wOlcek = useMemo(() => olcekBul(W.veri), [W]);
  const xOlcek = useMemo(() => olcekBul(x), [x]);
  const yOlcek = useMemo(() => olcekBul(y), [y]);

  const wGenislik = gSutun * hucre;
  const wYukseklik = gSatir * hucre;

  // Çizim alanının toplam boyutu düzene göre değişir.
  const tuvalGenislik = cevrik ? ETIKET + wGenislik + ARA + SERIT + 34 : ETIKET + SERIT + ARA + wGenislik;
  const tuvalYukseklik = cevrik ? SERIT + ARA + wYukseklik + 16 : wYukseklik + ARA + SERIT + 16;

  const ciz = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(tuvalGenislik * dpr);
    cv.height = Math.round(tuvalYukseklik * dpr);
    cv.style.width = `${tuvalGenislik}px`;
    cv.style.height = `${tuvalYukseklik}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, tuvalGenislik, tuvalYukseklik);

    // Düzene göre yerleşim noktaları
    const wX = cevrik ? ETIKET : ETIKET + SERIT + ARA;
    const wY = cevrik ? SERIT + ARA : 0;
    const bosluk = hucre >= 8 ? 1 : 0;

    // --- ağırlık matrisi ---------------------------------------------------
    for (let gi = 0; gi < gSatir; gi++) {
      for (let gj = 0; gj < gSutun; gj++) {
        // ekran konumundan gerçek (giriş, çıkış) çiftine
        const giris = cevrik ? gj : gi;
        const cikis = cevrik ? gi : gj;
        const deger = W.veri[giris * W.sutun + cikis];

        // Şu an işlenen çıkış dışındaki her şey söner: gözün bir yere baksın.
        const etkin = cikis === konum.j;
        ctx.globalAlpha = etkin ? 1 : 0.22;
        ctx.fillStyle = agirlikRengi(deger, wOlcek);
        ctx.fillRect(wX + gj * hucre, wY + gi * hucre, hucre - bosluk, hucre - bosluk);

        // Bu çıkışta zaten toplanmış girişler: ince bir üst şerit
        if (etkin && giris < konum.i) {
          ctx.globalAlpha = 0.5;
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(wX + gj * hucre, wY + gi * hucre, Math.max(1, hucre - bosluk), 1);
        }
      }
    }
    ctx.globalAlpha = 1;

    // şu an çarpılan hücre
    if (!konum.bitti) {
      const gi = cevrik ? konum.j : konum.i;
      const gj = cevrik ? konum.i : konum.j;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(wX + gj * hucre - 1, wY + gi * hucre - 1, hucre - bosluk + 2, hucre - bosluk + 2);
    }

    // --- giriş şeridi (x) --------------------------------------------------
    for (let i = 0; i < W.satir; i++) {
      ctx.globalAlpha = i === konum.i && !konum.bitti ? 1 : 0.55;
      ctx.fillStyle = agirlikRengi(x[i], xOlcek);
      if (cevrik) ctx.fillRect(wX + i * hucre, 0, hucre - bosluk, SERIT);
      else ctx.fillRect(ETIKET, wY + i * hucre, SERIT, hucre - bosluk);
    }

    // --- çıkış şeridi (y) --------------------------------------------------
    // Henüz hesaplanmamış hücreler neredeyse görünmez olduğu için şeridin
    // sınırını önce çiziyoruz: kullanıcı daha en baştan "burası kaç sayıyla
    // dolacak" bilsin.
    ctx.strokeStyle = "#272e3a";
    ctx.lineWidth = 1;
    if (cevrik) ctx.strokeRect(wX + wGenislik + ARA - 0.5, wY - 0.5, SERIT + 1, W.sutun * hucre + 1);
    else ctx.strokeRect(wX - 0.5, wY + wYukseklik + ARA - 0.5, W.sutun * hucre + 1, SERIT + 1);

    for (let j = 0; j < W.sutun; j++) {
      const tamam = j < konum.j || konum.bitti;
      const suAn = j === konum.j && !konum.bitti;
      const deger = tamam ? y[j] : suAn ? konum.sonrakiToplam : 0;
      ctx.globalAlpha = tamam || suAn ? 1 : 0.12;
      ctx.fillStyle = agirlikRengi(deger, yOlcek);
      if (cevrik) ctx.fillRect(wX + wGenislik + ARA, wY + j * hucre, SERIT, hucre - bosluk);
      else ctx.fillRect(wX + j * hucre, wY + wYukseklik + ARA, hucre - bosluk, SERIT);
    }
    ctx.globalAlpha = 1;

    // şu an dolan çıkış hücresi
    if (!konum.bitti) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      if (cevrik) ctx.strokeRect(wX + wGenislik + ARA - 1, wY + konum.j * hucre - 1, SERIT + 2, hucre - bosluk + 2);
      else ctx.strokeRect(wX + konum.j * hucre - 1, wY + wYukseklik + ARA - 1, hucre - bosluk + 2, SERIT + 2);
    }

    // --- etiketler ---------------------------------------------------------
    ctx.fillStyle = "#5e6878";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    if (cevrik) {
      ctx.fillText("x", ETIKET - 6, SERIT / 2);
      ctx.fillText("W", ETIKET - 6, wY + wYukseklik / 2);
      ctx.textAlign = "left";
      ctx.fillText("y", wX + wGenislik + ARA + SERIT + 5, wY + (W.sutun * hucre) / 2);
    } else {
      ctx.fillText("x", ETIKET - 6, wYukseklik / 2);
      ctx.fillText("W", ETIKET - 6, 8);
      ctx.textAlign = "left";
      ctx.fillText("y", wX + wGenislik + 5, wY + wYukseklik + ARA + SERIT / 2);
    }
  }, [x, W, b, y, konum, hucre, cevrik, gSatir, gSutun, wGenislik, wYukseklik, tuvalGenislik, tuvalYukseklik, wOlcek, xOlcek, yOlcek]);

  useEffect(() => {
    ciz();
  }, [ciz]);

  const xAd = xEtiketi?.(konum.i);
  const yAd = yEtiketi?.(konum.j);

  return (
    <div>
      <div className="overflow-x-auto">
        <canvas ref={canvasRef} className="block" />
      </div>

      {konum.bitti ? (
        <p className="sayi mt-3 text-[12px] text-soluk">
          Çarpım bitti: {W.sutun} çıkışın her biri {W.satir} terimin toplamı — toplam{" "}
          {(W.satir * W.sutun).toLocaleString("tr-TR")} çarpma ve toplama.
        </p>
      ) : (
        <div className="mt-3 space-y-1">
          <div className="sayi flex flex-wrap items-baseline gap-x-2 text-[13px]">
            <span className="text-cok-soluk">x[{konum.i}]</span>
            <span className="text-metin">{sayi(x[konum.i], 4)}</span>
            <span className="text-cok-soluk">×</span>
            <span className="text-cok-soluk">
              W[{konum.i},{konum.j}]
            </span>
            <span className="text-metin">{sayi(W.veri[konum.i * W.sutun + konum.j], 4)}</span>
            <span className="text-cok-soluk">=</span>
            <span className={konum.terim >= 0 ? "text-[#eda14a]" : "text-[#45c8bb]"}>
              {sayi(konum.terim, 4)}
            </span>
          </div>
          <div className="sayi text-[12px] text-soluk">
            ara toplam {sayi(konum.oncekiToplam, 4)} → <span className="text-metin">{sayi(konum.sonrakiToplam, 4)}</span>
            <span className="text-cok-soluk">
              {" "}
              · y[{konum.j}] için {konum.i + 1}/{W.satir} terim
            </span>
          </div>
          {(xAd || yAd) && (
            <div className="text-[11px] text-cok-soluk">
              {xAd ? `giriş: ${xAd}` : ""}
              {xAd && yAd ? " · " : ""}
              {yAd ? `çıkış: ${yAd}` : ""}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
