/**
 * VEKTÖR ŞERİDİ
 * =============
 *
 * Tek bir vektörü yan yana dizilmiş kareler olarak gösterir. Kareler yeterince
 * genişse sayı doğrudan karenin içine yazılır — 16 boyutlu bir vektörün on
 * altı sayısı da böylece aynı anda görünür. Dar olduğunda (64 ya da 128
 * boyutlu ara vektörlerde) sayılar sığmaz; o zaman renk kalır, sayı ise
 * üstüne gelince görünür.
 *
 * Renk skalası ağırlık ızgarasıyla aynı: negatif turkuaz, pozitif kehribar.
 * Aynı rengin her yerde aynı anlama gelmesi, ekranlar arasında geçerken
 * kullanıcının yeniden öğrenmesi gereken bir şey bırakmıyor.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { agirlikRengi, olcekBul } from "../lib/renk.ts";
import { sayi } from "../lib/bicim.ts";

export interface VektorSeridiProps {
  veri: Float32Array;
  /** Kaç piksel genişlik ayrılabilir. */
  genislik?: number;
  /** Beyaz çerçeveyle işaretlenecek konum. */
  vurgu?: number | null;
  /** Söndürülmüş (ReLU'nun sıfırladığı) konumlar. */
  sonuk?: ReadonlySet<number> | null;
  /** Henüz hesaplanmamış konumlar soluk çizilir. */
  hesaplanan?: number;
  etiket?: (i: number) => string | null;
  yukseklik?: number;
  /** Ortak ölçek: iki vektörü karşılaştırırken aynı skalayı kullanmak için. */
  olcek?: number;
}

export function VektorSeridi({
  veri,
  genislik = 640,
  vurgu = null,
  sonuk = null,
  hesaplanan,
  etiket,
  yukseklik = 34,
  olcek,
}: VektorSeridiProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [uzerinde, setUzerinde] = useState<number | null>(null);

  const n = veri.length;
  const hucre = Math.max(4, Math.min(52, Math.floor(genislik / n)));
  const toplamGenislik = hucre * n;
  const sayiSigar = hucre >= 40;
  const kullanilanOlcek = useMemo(() => olcek ?? olcekBul(veri), [veri, olcek]);

  const ciz = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(toplamGenislik * dpr);
    cv.height = Math.round(yukseklik * dpr);
    cv.style.width = `${toplamGenislik}px`;
    cv.style.height = `${yukseklik}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, toplamGenislik, yukseklik);

    const bosluk = hucre >= 8 ? 1 : 0;
    for (let i = 0; i < n; i++) {
      const x = i * hucre;
      const beklemede = hesaplanan != null && i >= hesaplanan;
      ctx.globalAlpha = beklemede ? 0.16 : 1;
      ctx.fillStyle = agirlikRengi(veri[i], kullanilanOlcek);
      ctx.fillRect(x, 0, hucre - bosluk, yukseklik);

      if (sonuk?.has(i)) {
        // Sönmüş nöron: üstüne çapraz tarama. Renk tek başına "sıfır"ı
        // anlatamaz, çünkü sıfıra yakın bir değer de aynı tona düşer.
        ctx.globalAlpha = beklemede ? 0.1 : 0.55;
        ctx.strokeStyle = "#8d97a8";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + 1, yukseklik - 1);
        ctx.lineTo(x + hucre - bosluk - 1, 1);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      if (sayiSigar && !beklemede) {
        ctx.fillStyle = "#0d1016";
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(sayi(veri[i], 2), x + (hucre - bosluk) / 2, yukseklik / 2);
      }
    }

    if (vurgu != null && vurgu >= 0 && vurgu < n) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(vurgu * hucre - 1, 1, hucre - (hucre >= 8 ? 1 : 0) + 2, yukseklik - 2);
    }
  }, [veri, n, hucre, toplamGenislik, yukseklik, vurgu, sonuk, hesaplanan, kullanilanOlcek, sayiSigar]);

  useEffect(() => {
    ciz();
  }, [ciz]);

  const gosterilen = uzerinde ?? vurgu;

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => {
          const kutu = e.currentTarget.getBoundingClientRect();
          const i = Math.floor((e.clientX - kutu.left) / hucre);
          setUzerinde(i >= 0 && i < n ? i : null);
        }}
        onMouseLeave={() => setUzerinde(null)}
        className="block"
      />
      {gosterilen != null && (
        <div className="sayi mt-1 text-[11px] text-soluk">
          [{gosterilen}]
          {etiket?.(gosterilen) ? ` ${etiket(gosterilen)}` : ""} = {sayi(veri[gosterilen], 4)}
        </div>
      )}
    </div>
  );
}
