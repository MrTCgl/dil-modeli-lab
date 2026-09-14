/**
 * KÜRE EKRANI
 * ===========
 *
 * Küreyi, modeli ve küreyi hareket ettiren küçük bir eğitim düğmesini bir
 * araya getirir.
 *
 * Buradaki eğitim GEÇİCİ: ana iş parçacığında, küçük parçalar hâlinde
 * çalışıyor. Asıl eğitim paneli (Web Worker'da, kayıp eğrisiyle, ayarlarıyla)
 * sıralamanın bir sonraki adımı. Bunu şimdiden koymamın sebebi, kürenin en
 * önemli özelliğinin — noktaların eğitimle yer değiştirmesi — iddia olarak
 * değil gösterilerek doğrulanabilmesi.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { type Model, modelOlustur, parametreSayisi } from "../lib/model.ts";
import { type Egitici, egiticiOlustur, egitimAdimi, veriHazirla } from "../lib/train.ts";
import { EGITIM_METNI } from "../data/metin.ts";
import { durumOku, durumYaz, VARSAYILAN_DURUM } from "../lib/durum.ts";
import { adet, sayi } from "../lib/bicim.ts";
import { GommeKuresi } from "./GommeKuresi.tsx";

/** Bir karede kaç eğitim adımı atılacak. Arayüz akıcı kalsın diye küçük. */
const KARE_BASINA_ADIM = 4;

export default function KureEkrani() {
  const [durum, setDurum] = useState(VARSAYILAN_DURUM);
  const [hazir, setHazir] = useState(false);
  useEffect(() => {
    setDurum(durumOku());
    setHazir(true);
  }, []);
  useEffect(() => {
    if (hazir) durumYaz(durum);
  }, [durum, hazir]);

  const [yenileme, setYenileme] = useState(0);
  const model = useMemo<Model>(
    () => modelOlustur({ D: durum.D, katmanSayisi: durum.katman, baglam: 8, seed: durum.seed }),
    [durum.D, durum.katman, durum.seed, yenileme],
  );

  const veri = useMemo(() => veriHazirla(EGITIM_METNI), []);
  const [surum, setSurum] = useState(0);
  const [adim, setAdim] = useState(0);
  const [kayip, setKayip] = useState<number | null>(null);
  const [egitiliyor, setEgitiliyor] = useState(false);
  const egiticiRef = useRef<Egitici | null>(null);

  // Model yenilenince eğitici de sıfırlanır.
  useEffect(() => {
    egiticiRef.current = egiticiOlustur(model, veri, {
      ogrenmeOrani: 0.2,
      yigin: 32,
      sicaklik: 1,
      kirpma: 5,
    });
    setAdim(0);
    setKayip(null);
    setSurum((s) => s + 1);
    setEgitiliyor(false);
  }, [model, veri]);

  useEffect(() => {
    if (!egitiliyor) return;
    let raf = 0;
    const kare = () => {
      const egitici = egiticiRef.current;
      if (!egitici) return;
      let son = 0;
      for (let i = 0; i < KARE_BASINA_ADIM; i++) son = egitimAdimi(egitici);
      setAdim(egitici.adim);
      setKayip(son);
      setSurum((s) => s + 1);
      raf = requestAnimationFrame(kare);
    };
    raf = requestAnimationFrame(kare);
    return () => cancelAnimationFrame(raf);
  }, [egitiliyor]);

  return (
    <div className="space-y-4">
      <GommeKuresi model={model} surum={surum} yukseklik={480} />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border border-cizgi bg-yuzey px-3 py-2">
        <button
          onClick={() => setEgitiliyor((e) => !e)}
          className="border border-cizgi-parlak px-3 py-1 text-[11px] text-metin hover:bg-yuzey-2"
        >
          {egitiliyor ? "■ eğitimi durdur" : "▶ eğit ve izle"}
        </button>
        <span className="sayi text-[11px] text-cok-soluk">
          {adet(adim)} adım
          {kayip != null && <> · kayıp {sayi(kayip, 3)}</>}
        </span>

        <div className="ml-auto flex items-center gap-2">
          <label htmlFor="tohum" className="text-[11px] text-cok-soluk">
            tohum
          </label>
          <input
            id="tohum"
            type="number"
            value={durum.seed}
            onChange={(e) => setDurum((d) => ({ ...d, seed: Number(e.target.value) || 0 }))}
            className="sayi w-16 border border-cizgi bg-zemin px-2 py-1 text-[12px] text-metin"
          />
          <button
            onClick={() => setYenileme((y) => y + 1)}
            className="border border-cizgi px-2 py-1 text-[11px] text-soluk hover:border-cizgi-parlak hover:text-metin"
          >
            sıfırla
          </button>
        </div>

        <p className="w-full text-[10px] leading-relaxed text-cok-soluk">
          {adet(parametreSayisi(model))} parametre · Buradaki eğitim geçici ve ana iş parçacığında
          çalışıyor; kayıp eğrisi, ayarlar ve Web Worker'lı sürüm eğitim panelinde gelecek. Amacı
          noktaların gerçekten yer değiştirdiğini göstermek.
        </p>
      </div>
    </div>
  );
}
