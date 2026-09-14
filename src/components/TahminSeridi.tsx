/**
 * TAHMİN ŞERİDİ
 * =============
 *
 * Ağırlık ızgarasının hemen üstünde durur ve modelin o anki tahminini
 * gösterir. Tek işi var: bir hücreyi elle bozduğunuzda çıktının anında
 * değiştiğini görünür kılmak. Bu olmadan ızgara sadece renkli bir tablo;
 * bununla birlikte "model = sayı tablosu" fikri yerine oturuyor.
 *
 * (Tam ileri geçiş gösterimi ayrı bir ekran olacak. Burada sadece sonuç var.)
 */

import { useMemo } from "react";
import { type Model, ileriGecis } from "../lib/model.ts";
import { encode, gorunurAd } from "../lib/tokenizer.ts";
import { sayi, yuzde } from "../lib/bicim.ts";

export function TahminSeridi({
  model,
  metin,
  onMetin,
  surum,
}: {
  model: Model;
  metin: string;
  onMetin: (m: string) => void;
  surum: number;
}) {
  const { enIyiler, baglam } = useMemo(() => {
    const ids = encode(metin);
    const iz = ileriGecis(model, ids, 1);
    const sirali = Array.from(iz.olasilik)
      .map((olasilik, id) => ({ olasilik, id }))
      .sort((a, b) => b.olasilik - a.olasilik)
      .slice(0, 6);
    return { enIyiler: sirali, baglam: iz.baglamIds };
  }, [model, metin, surum]);

  return (
    <div className="border border-cizgi bg-yuzey">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-cizgi px-3 py-2">
        <label htmlFor="baglam-girdi" className="text-[12px] text-soluk">
          Metin
        </label>
        <input
          id="baglam-girdi"
          value={metin}
          onChange={(e) => onMetin(e.target.value)}
          className="sayi min-w-[180px] flex-1 border border-cizgi bg-zemin px-2 py-1 text-[13px] text-metin"
          spellCheck={false}
        />
        <span className="sayi text-[11px] text-cok-soluk">
          modele giren son 8:{" "}
          <span className="text-soluk">
            {baglam.map((id) => gorunurAd(id)).join("")}
          </span>
        </span>
      </div>

      <div className="px-3 py-2">
        <p className="mb-2 text-[11px] text-cok-soluk">
          Bir sonraki karakter için modelin olasılıkları. Aşağıdaki ızgaralarda bir hücreyi
          değiştirin — bu çubuklar aynı anda değişir.
        </p>
        <ul className="space-y-1">
          {enIyiler.map(({ id, olasilik }) => (
            <li key={id} className="flex items-center gap-2">
              <span className="sayi w-5 text-center text-[13px] text-metin">{gorunurAd(id)}</span>
              <span className="sayi w-14 text-right text-[11px] text-soluk">{yuzde(olasilik)}</span>
              <span className="h-3 flex-1 bg-yuzey-2">
                <span
                  className="block h-full bg-cizgi-parlak transition-[width] duration-150"
                  style={{ width: `${Math.max(0.4, olasilik * 100)}%` }}
                />
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Ham logits'i de gösteren küçük yardımcı — hata ayıklarken işe yarıyor. */
export function LogitOzeti({ model, metin }: { model: Model; metin: string }) {
  const iz = useMemo(() => ileriGecis(model, encode(metin), 1), [model, metin]);
  const enBuyuk = Math.max(...Array.from(iz.logits));
  const enKucuk = Math.min(...Array.from(iz.logits));
  return (
    <span className="sayi text-[11px] text-cok-soluk">
      logits {sayi(enKucuk, 2)} … {sayi(enBuyuk, 2)}
    </span>
  );
}
