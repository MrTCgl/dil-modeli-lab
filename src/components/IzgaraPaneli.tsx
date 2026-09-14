/**
 * IZGARA PANELİ
 * =============
 *
 * Modelin bütün ağırlık matrislerini ızgara olarak gösterir ve elle
 * değiştirmeyi yönetir. Bu adada tek bir model örneği yaşar; ızgaralar ve
 * tahmin şeridi aynı sayıları okur, yani ekranda gördüğünüz şey modelin
 * kendisidir — kopyası ya da temsili değil.
 *
 * Model nesnesi yerinde (mutable) değiştiği için React'in değişimi kendi
 * başına fark etmesi mümkün değil; bu yüzden her düzenlemede artan bir
 * `sürüm` sayacı tutuyoruz. Alternatif her adımda modelin kopyasını
 * çıkarmaktı: 7 binden fazla sayıyı saniyede onlarca kez kopyalamak,
 * eğitim canlı aktığında boşa giden iş olurdu.
 */

import { useMemo, useRef, useState } from "react";
import { type Matris, type Model, modelOlustur, parametreSayisi } from "../lib/model.ts";
import { izgaralariTopla } from "../lib/izgaralar.ts";
import { adet } from "../lib/bicim.ts";
import { AgirlikIzgarasi } from "./AgirlikIzgarasi.tsx";
import { TahminSeridi } from "./TahminSeridi.tsx";

/** Elle bozulan bir hücrenin kaydı: geri alabilmek için ilk değeri saklarız. */
interface Bozma {
  matris: Matris;
  indeks: number;
  ilkDeger: number;
}

export default function IzgaraPaneli() {
  const [seed, setSeed] = useState(1);
  const [surum, setSurum] = useState(0);
  const [metin, setMetin] = useState("deniz kı");

  const model = useMemo<Model>(
    () => modelOlustur({ D: 16, katmanSayisi: 2, baglam: 8, seed }),
    [seed],
  );

  // Bozmalar model değişince sıfırlanır (yeni modelin eski kaydı anlamsız).
  const bozmalarRef = useRef<Map<string, Bozma>>(new Map());
  const suAnkiSeed = useRef(seed);
  if (suAnkiSeed.current !== seed) {
    suAnkiSeed.current = seed;
    bozmalarRef.current = new Map();
  }

  const matrisler = useMemo(() => izgaralariTopla(model), [model]);

  const degistir = (matris: Matris, indeks: number, deger: number) => {
    const anahtar = `${matris.ad}#${indeks}`;
    if (!bozmalarRef.current.has(anahtar)) {
      bozmalarRef.current.set(anahtar, { matris, indeks, ilkDeger: matris.veri[indeks] });
    }
    matris.veri[indeks] = deger;
    setSurum((s) => s + 1);
  };

  const hepsiniGeriAl = () => {
    for (const { matris, indeks, ilkDeger } of bozmalarRef.current.values()) {
      matris.veri[indeks] = ilkDeger;
    }
    bozmalarRef.current.clear();
    setSurum((s) => s + 1);
  };

  // Her matris için o matriste bozulmuş hücrelerin indeksleri.
  const bozulanlar = useMemo(() => {
    const harita = new Map<string, Set<number>>();
    for (const { matris, indeks } of bozmalarRef.current.values()) {
      let kume = harita.get(matris.ad);
      if (!kume) {
        kume = new Set();
        harita.set(matris.ad, kume);
      }
      kume.add(indeks);
    }
    return harita;
  }, [surum]);

  const bozmaSayisi = bozmalarRef.current.size;

  return (
    <div className="space-y-4">
      <TahminSeridi model={model} metin={metin} onMetin={setMetin} surum={surum} />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border border-cizgi bg-yuzey px-3 py-2">
        <span className="sayi text-[11px] text-cok-soluk">
          {adet(parametreSayisi(model))} parametre · D=16 · 2 katman · bağlam 8
        </span>
        <div className="ml-auto flex items-center gap-2">
          <label className="text-[11px] text-cok-soluk" htmlFor="seed">
            tohum
          </label>
          <input
            id="seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            className="sayi w-16 border border-cizgi bg-zemin px-2 py-1 text-[12px] text-metin"
          />
          <button
            onClick={hepsiniGeriAl}
            disabled={bozmaSayisi === 0}
            className="border border-cizgi px-2 py-1 text-[11px] text-soluk enabled:hover:border-cizgi-parlak enabled:hover:text-metin disabled:text-cok-soluk disabled:opacity-40"
          >
            bozduklarımı geri al{bozmaSayisi > 0 ? ` (${bozmaSayisi})` : ""}
          </button>
        </div>
      </div>

      {matrisler.map(({ matris, satirEtiketi, sutunEtiketi, aciklama }) => (
        <div key={matris.ad}>
          <p className="mb-1 px-0.5 text-[11px] text-cok-soluk">{aciklama}</p>
          <AgirlikIzgarasi
            matris={matris}
            surum={surum}
            satirEtiketi={satirEtiketi}
            sutunEtiketi={sutunEtiketi}
            bozulan={bozulanlar.get(matris.ad)}
            onDegistir={(indeks, deger) => degistir(matris, indeks, deger)}
            parlamaAktif
          />
        </div>
      ))}
    </div>
  );
}
