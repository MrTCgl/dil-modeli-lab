/**
 * ÜRETİM
 * ======
 *
 * Tahmin edilen karakteri metnin sonuna ekleyip döngüyü sürdürmek — modelin
 * "yazması" dediğimiz şey tam olarak bu ve başka bir şey değil.
 *
 * Üretilen her metni adım sayısıyla birlikte listede tutuyoruz. Eğitimin
 * başında çıkan çöp ile birkaç yüz adım sonra çıkan Türkçe benzeri metin
 * yan yana durduğunda, eğitimin ne yaptığı tek bakışta görülüyor. Bu yüzden
 * eski üretimleri silmiyoruz.
 */

import { useState } from "react";
import { type Model, rastgeleUretec, uret } from "../lib/model.ts";
import { decode, encode } from "../lib/tokenizer.ts";
import { adet, sayi } from "../lib/bicim.ts";

interface Kayit {
  adim: number;
  sicaklik: number;
  metin: string;
  anahtar: number;
}

export function Uretim({
  model,
  adim,
  sicaklik,
  baslangic,
}: {
  model: Model;
  adim: number;
  sicaklik: number;
  baslangic: string;
}) {
  const [kayitlar, setKayitlar] = useState<Kayit[]>([]);
  const [uzunluk, setUzunluk] = useState(200);
  const [tohum, setTohum] = useState(1);

  const uretVeEkle = () => {
    const ids = encode(baslangic);
    const rnd = rastgeleUretec(tohum);
    const yeni = uret(model, ids, uzunluk, sicaklik, rnd);
    setKayitlar((eski) =>
      [{ adim, sicaklik, metin: decode(ids) + decode(yeni), anahtar: Date.now() }, ...eski].slice(0, 6),
    );
    setTohum((t) => t + 1);
  };

  return (
    <section className="border border-cizgi bg-yuzey">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-cizgi px-3 py-2">
        <h2 className="text-[13px] font-medium text-metin">Üretim</h2>
        <span className="text-[11px] text-cok-soluk">
          "{baslangic}" ile başlayıp devamını model yazıyor
        </span>
        <div className="ml-auto flex items-center gap-2">
          <label htmlFor="uzunluk" className="text-[11px] text-cok-soluk">
            uzunluk
          </label>
          <input
            id="uzunluk"
            type="number"
            min={20}
            max={600}
            step={20}
            value={uzunluk}
            onChange={(e) => setUzunluk(Math.max(20, Math.min(600, Number(e.target.value) || 200)))}
            className="sayi w-16 border border-cizgi bg-zemin px-2 py-1 text-[12px] text-metin"
          />
          <button
            onClick={uretVeEkle}
            className="border border-cizgi-parlak px-3 py-1 text-[11px] text-metin hover:bg-yuzey-2"
          >
            üret
          </button>
        </div>
      </header>

      {kayitlar.length === 0 ? (
        <p className="px-3 py-3 text-[11px] leading-relaxed text-cok-soluk">
          Şimdi üretirseniz çöp çıkar — model henüz hiçbir şey bilmiyor. Birkaç yüz adım eğitip
          tekrar üretin; iki metin burada alt alta durur ve farkı kendiniz görürsünüz.
        </p>
      ) : (
        <ul className="divide-y divide-cizgi">
          {kayitlar.map((k) => (
            <li key={k.anahtar} className="px-3 py-2.5">
              <div className="sayi mb-1 text-[10px] text-cok-soluk">
                {adet(k.adim)} adım · sıcaklık {sayi(k.sicaklik, 2)}
              </div>
              <p className="sayi text-[12px] leading-relaxed break-words text-soluk">{k.metin}</p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
