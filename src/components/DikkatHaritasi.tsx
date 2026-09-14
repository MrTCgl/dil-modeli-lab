/**
 * DİKKAT HARİTASI
 * ===============
 *
 * Bağlamdaki her konumun, her konuma verdiği ağırlık. Satır "kim bakıyor",
 * sütun "nereye bakıyor". Her satırın toplamı 1'dir — dikkat bir dağılımdır,
 * konumlar arasında paylaştırılan sabit bir bütçedir.
 *
 * Sağ üst üçgen kapalıdır: nedensel maske. Bir konum kendinden sonrasına
 * bakamaz, yoksa tahmin edeceği karakteri görmüş olurdu.
 *
 * SON SATIR NEDEN VURGULU
 * Tahmini son konum üretiyor, dolayısıyla çıktıyı belirleyen satır o.
 * Diğer satırlar da gerçekten hesaplanıyor (gerçek eğitimde her konum kendi
 * sonraki karakterini tahmin eder) ama bu uygulamada tek hedef olduğu için
 * gradyan yalnızca son satırdan akıyor. Gösteriyoruz çünkü mekanizmanın
 * tamamı bu; soluk çiziyoruz çünkü tahmini onlar yapmıyor.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DikkatIzi } from "../lib/model.ts";
import { gorunurAd } from "../lib/tokenizer.ts";
import { sayi, yuzde } from "../lib/bicim.ts";
import { HUCRE_POZITIF, HUCRE_SIFIR } from "../lib/renk.ts";
import { interpolateLab } from "d3-interpolate";

const skala = interpolateLab(HUCRE_SIFIR, HUCRE_POZITIF);
const KENAR = 30;

export function DikkatHaritasi({
  dikkat,
  baglamIds,
  genislik,
  skorGoster = false,
}: {
  dikkat: DikkatIzi;
  baglamIds: number[];
  genislik: number;
  /** true ise softmax öncesi ham skorlar, false ise ağırlıklar gösterilir. */
  skorGoster?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [uzerinde, setUzerinde] = useState<[number, number] | null>(null);
  const C = baglamIds.length;
  const hucre = Math.max(18, Math.min(48, Math.floor((Math.min(genislik, 560) - KENAR) / C)));
  const boy = KENAR + C * hucre;

  // Ham skorlar için ayrı ölçek: negatif de olabiliyorlar.
  const skorOlcek = useMemo(() => {
    let enBuyuk = 1e-6;
    for (const satir of dikkat.skorlar) {
      for (const v of satir) if (Number.isFinite(v)) enBuyuk = Math.max(enBuyuk, Math.abs(v));
    }
    return enBuyuk;
  }, [dikkat]);

  const ciz = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(boy * dpr);
    cv.height = Math.round(boy * dpr);
    cv.style.width = `${boy}px`;
    cv.style.height = `${boy}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, boy, boy);
    ctx.font = "11px ui-monospace, monospace";
    ctx.textBaseline = "middle";

    for (let t = 0; t < C; t++) {
      for (let k = 0; k < C; k++) {
        const x = KENAR + k * hucre;
        const y = KENAR + t * hucre;
        const maskeli = k > t;
        if (maskeli) {
          // Kapalı hücreler: çapraz tarama. Boş bırakmak "sıfır ağırlık" ile
          // karışırdı; bunlar hiç hesaplanmıyor, sıfır değil.
          ctx.fillStyle = "#14181f";
          ctx.fillRect(x, y, hucre - 1, hucre - 1);
          ctx.strokeStyle = "#232a34";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x + 3, y + hucre - 4);
          ctx.lineTo(x + hucre - 4, y + 3);
          ctx.stroke();
          continue;
        }
        const deger = skorGoster ? dikkat.skorlar[t][k] : dikkat.agirliklar[t][k];
        const oran = skorGoster
          ? Math.min(1, Math.abs(deger) / skorOlcek)
          : Math.min(1, Math.sqrt(deger));
        ctx.globalAlpha = t === C - 1 ? 1 : 0.45;
        ctx.fillStyle = skala(oran);
        ctx.fillRect(x, y, hucre - 1, hucre - 1);
        ctx.globalAlpha = 1;

        if (hucre >= 30 && !skorGoster) {
          ctx.fillStyle = oran > 0.55 ? "#12151a" : "#8d97a8";
          ctx.textAlign = "center";
          ctx.fillText(Math.round(deger * 100) + "", x + hucre / 2, y + hucre / 2);
        }
      }
    }

    // son satırı çerçevele: çıktıyı belirleyen satır bu
    ctx.strokeStyle = "#e6ebf2";
    ctx.lineWidth = 2;
    ctx.strokeRect(KENAR - 1, KENAR + (C - 1) * hucre - 1, C * hucre + 1, hucre + 1);

    // etiketler
    ctx.fillStyle = "#5e6878";
    ctx.textAlign = "center";
    for (let k = 0; k < C; k++) {
      ctx.fillText(gorunurAd(baglamIds[k]), KENAR + k * hucre + hucre / 2, KENAR / 2 + 2);
    }
    ctx.textAlign = "right";
    for (let t = 0; t < C; t++) {
      ctx.fillText(gorunurAd(baglamIds[t]), KENAR - 7, KENAR + t * hucre + hucre / 2);
    }
  }, [dikkat, baglamIds, C, hucre, boy, skorGoster, skorOlcek]);

  useEffect(() => {
    ciz();
  }, [ciz]);

  const secili = uzerinde ?? [C - 1, C - 1];
  const [st, sk] = secili;
  const maskeli = sk > st;

  return (
    <div>
      <div className="overflow-x-auto">
        <canvas
          ref={canvasRef}
          className="block"
          onMouseMove={(e) => {
            const kutu = e.currentTarget.getBoundingClientRect();
            const k = Math.floor((e.clientX - kutu.left - KENAR) / hucre);
            const t = Math.floor((e.clientY - kutu.top - KENAR) / hucre);
            setUzerinde(t >= 0 && t < C && k >= 0 && k < C ? [t, k] : null);
          }}
          onMouseLeave={() => setUzerinde(null)}
          aria-label="dikkat haritası"
        />
      </div>
      <p className="sayi mt-2 text-[11px] text-soluk">
        <span className="text-metin">{gorunurAd(baglamIds[st])}</span>
        <span className="text-cok-soluk"> (t−{C - st}) → </span>
        <span className="text-metin">{gorunurAd(baglamIds[sk])}</span>
        <span className="text-cok-soluk"> (t−{C - sk}): </span>
        {maskeli ? (
          <span className="text-cok-soluk">maske kapalı — geleceğe bakılamaz</span>
        ) : (
          <>
            ağırlık <span className="text-metin">{yuzde(dikkat.agirliklar[st][sk])}</span>
            <span className="text-cok-soluk"> · ham skor {sayi(dikkat.skorlar[st][sk], 3)}</span>
          </>
        )}
      </p>
      <p className="mt-1 text-[10px] leading-relaxed text-cok-soluk">
        Satır: kim bakıyor. Sütun: nereye bakıyor. Her satırın toplamı 100. Çerçeveli son satır
        çıktıyı belirleyen satırdır; diğerleri hesaplanıyor ama bu tahminde kullanılmıyor.
      </p>
    </div>
  );
}
