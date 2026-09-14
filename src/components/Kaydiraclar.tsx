/**
 * KAYDIRAÇ PANELİ
 * ===============
 *
 * Bütün ayarlar tek yerde. Her kaydıracın yanında ne işe yaradığını anlatan
 * tek cümle var — jargonsuz, çünkü bu uygulamanın varlık sebebi jargonu
 * açmak.
 *
 * Bazı ayarlar modeli yeniden kurar (gömme boyutu, katman sayısı, tohum):
 * yeni bir mimari, yeni rastgele ağırlıklar, sıfırlanan eğitim. Bunlar
 * ayrıca işaretli — kullanıcı yarım saatlik eğitimini bilmeden silmesin.
 */

import type { ReactNode } from "react";

export interface KaydiracProps {
  ad: string;
  aciklama: string;
  min: number;
  max: number;
  adim: number;
  deger: number;
  onDegisti: (v: number) => void;
  bicim?: (v: number) => string;
  /** true ise değişiklik modeli baştan kurar ve eğitimi sıfırlar. */
  yenidenKurar?: boolean;
  devreDisi?: boolean;
}

export function Kaydirac({
  ad,
  aciklama,
  min,
  max,
  adim,
  deger,
  onDegisti,
  bicim,
  yenidenKurar,
  devreDisi,
}: KaydiracProps) {
  const kimlik = `kaydirac-${ad.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <div className={devreDisi ? "opacity-45" : undefined}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={kimlik} className="text-[12px] text-metin">
          {ad}
        </label>
        <span className="sayi text-[12px] text-soluk">{bicim ? bicim(deger) : deger}</span>
      </div>
      <input
        id={kimlik}
        type="range"
        min={min}
        max={max}
        step={adim}
        value={deger}
        disabled={devreDisi}
        onChange={(e) => onDegisti(Number(e.target.value))}
        className="mt-1 h-1 w-full accent-white"
      />
      <p className="mt-1 text-[10px] leading-snug text-cok-soluk">
        {aciklama}
        {yenidenKurar && (
          <span className="text-soluk"> · değiştirmek modeli baştan kurar, eğitim sıfırlanır.</span>
        )}
      </p>
    </div>
  );
}

export function KaydiracPaneli({ children }: { children: ReactNode }) {
  return (
    <section className="border border-cizgi bg-yuzey">
      <header className="border-b border-cizgi px-3 py-2">
        <h2 className="text-[13px] font-medium text-metin">Ayarlar</h2>
      </header>
      <div className="grid gap-4 p-3 sm:grid-cols-2 lg:grid-cols-3">{children}</div>
    </section>
  );
}
