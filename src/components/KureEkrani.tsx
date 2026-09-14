/**
 * KÜRE EKRANI
 * ===========
 *
 * Küreyi, modeli ve küreyi hareket ettiren küçük bir eğitim düğmesini bir
 * araya getirir.
 *
 * Bu ekran küreyi tek başına, sakin bir ortamda inceleme içindir: model
 * durur, siz döndürür, seçer, sayılara bakarsınız. Eğitim sürerken canlı
 * hareket eden sürümü eğitim panelinde — küre bileşeni ikisinde de aynı.
 */

import { useEffect, useMemo, useState } from "react";
import { type Model, modelOlustur, parametreSayisi } from "../lib/model.ts";
import { durumOku, durumYaz, VARSAYILAN_DURUM } from "../lib/durum.ts";
import { adet } from "../lib/bicim.ts";
import { GommeKuresi } from "./GommeKuresi.tsx";

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

  return (
    <div className="space-y-4">
      <GommeKuresi model={model} surum={yenileme} yukseklik={480} />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border border-cizgi bg-yuzey px-3 py-2">
        <a
          href="/egitim"
          className="border border-cizgi-parlak px-3 py-1 text-[11px] text-metin hover:bg-yuzey-2"
        >
          eğitim paneline git →
        </a>
        <span className="text-[11px] text-cok-soluk">
          noktaların eğitimle yer değiştirmesini orada canlı izleyebilirsiniz
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
          {adet(parametreSayisi(model))} parametre · D={durum.D} · {durum.katman} katman. Bu
          ekrandaki model eğitilmemiş; tohumu değiştirerek farklı rastgele başlangıçların nasıl
          göründüğüne bakabilirsiniz.
        </p>
      </div>
    </div>
  );
}
