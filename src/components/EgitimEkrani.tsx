/**
 * EĞİTİM PANELİ
 * =============
 *
 * Eğitim ayrı bir iş parçacığında (Web Worker) döner; bu ekran onun
 * aynasıdır. Worker belirli aralıklarla ağırlıkların kopyasını gönderir,
 * ekran kendi model nesnesinin üzerine yazar ve küre, ızgaralar, üretim —
 * hepsi aynı anda tazelenir.
 *
 * NEDEN ARAYÜZ DONMUYOR
 * Tek bir eğitim adımı 32 örnek üzerinde ileri geçiş ve geri yayılım demek.
 * Saniyede yüzlercesi atılıyor. Ana iş parçacığında yapılsaydı kaydıraç
 * oynatılamaz, küre döndürülemezdi. Worker hesabı alıp götürüyor, buraya
 * sadece sonuçlar geliyor.
 *
 * YENİLEME HIZI
 * Worker saniyede kırk mesaj gönderebiliyor ama ekranı o hızda yeniden
 * çizmenin anlamı yok: göz ayırt edemez, işlemci yanar. Gelen veriler
 * ref'lerde birikiyor, ekran saniyede sekiz kez tazeleniyor. Kayıp eğrisi
 * yine de her adımı çiziyor — hiçbir ölçüm atılmıyor, sadece boyama
 * seyrekleşiyor.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  type Model,
  agirliklariYaz,
  ileriGecis,
  modelOlustur,
  parametreSayisi,
} from "../lib/model.ts";
import { encode } from "../lib/tokenizer.ts";
import type { EgitimCevabi, EgitimIstegi } from "../workers/egitim.ts";
import { izgaralariTopla } from "../lib/izgaralar.ts";
import { durumOku, durumYaz, VARSAYILAN_DURUM } from "../lib/durum.ts";
import { adet, sayi } from "../lib/bicim.ts";
import { AgirlikIzgarasi } from "./AgirlikIzgarasi.tsx";
import { GommeKuresi } from "./GommeKuresi.tsx";
import { DikkatHaritasi } from "./DikkatHaritasi.tsx";
import { type DogrulamaNoktasi, KayipEgrisi } from "./KayipEgrisi.tsx";
import { Kaydirac, KaydiracPaneli } from "./Kaydiraclar.tsx";
import { Uretim } from "./Uretim.tsx";

/** Ekran saniyede kaç kez tazelensin. */
const TAZELEME_ARALIGI = 125;

export default function EgitimEkrani() {
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

  // Ekranın aynası: worker'dan gelen ağırlıklar bunun üzerine yazılır.
  const ayna = useMemo<Model>(
    () => modelOlustur({ D: durum.D, katmanSayisi: durum.katman, baglam: 8, seed: durum.seed, dikkat: durum.dikkat }),
    [durum.D, durum.katman, durum.seed, durum.dikkat, yenileme],
  );

  const workerRef = useRef<Worker | null>(null);
  const kayipRef = useRef<number[]>([]);
  const dogrulamaRef = useRef<DogrulamaNoktasi[]>([]);
  const sonTazelemeRef = useRef(0);
  const zamanlayiciRef = useRef<number | null>(null);

  const [adim, setAdim] = useState(0);
  const [calisiyor, setCalisiyor] = useState(false);
  const [surum, setSurum] = useState(0);
  const [cizim, setCizim] = useState<{ kayiplar: number[]; dogrulama: DogrulamaNoktasi[] }>({
    kayiplar: [],
    dogrulama: [],
  });
  const [tokenSayisi, setTokenSayisi] = useState(0);

  const tazele = useCallback(() => {
    sonTazelemeRef.current = performance.now();
    setCizim({ kayiplar: kayipRef.current.slice(), dogrulama: dogrulamaRef.current.slice() });
    setSurum((s) => s + 1);
  }, []);

  /** Gelen her mesajda değil, en fazla TAZELEME_ARALIGI'de bir çiz. */
  const tazelemeIste = useCallback(() => {
    const gecen = performance.now() - sonTazelemeRef.current;
    if (gecen >= TAZELEME_ARALIGI) {
      if (zamanlayiciRef.current != null) {
        clearTimeout(zamanlayiciRef.current);
        zamanlayiciRef.current = null;
      }
      tazele();
    } else if (zamanlayiciRef.current == null) {
      zamanlayiciRef.current = window.setTimeout(() => {
        zamanlayiciRef.current = null;
        tazele();
      }, TAZELEME_ARALIGI - gecen);
    }
  }, [tazele]);

  // --- worker'ı kur --------------------------------------------------------
  useEffect(() => {
    const worker = new Worker(new URL("../workers/egitim.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    worker.onmessage = (olay: MessageEvent<EgitimCevabi>) => {
      const cevap = olay.data;
      if (cevap.tur === "hazir") {
        agirliklariYaz(ayna, cevap.agirliklar);
        kayipRef.current = [];
        dogrulamaRef.current = [];
        setAdim(0);
        setCalisiyor(false);
        setTokenSayisi(cevap.tokenSayisi);
        tazele();
        return;
      }
      for (let i = 0; i < cevap.kayiplar.length; i++) kayipRef.current.push(cevap.kayiplar[i]);
      if (cevap.dogrulama != null) {
        const son = dogrulamaRef.current[dogrulamaRef.current.length - 1];
        if (!son || son.adim !== cevap.adim) {
          dogrulamaRef.current.push({ adim: cevap.adim, deger: cevap.dogrulama });
        }
      }
      if (cevap.agirliklar) agirliklariYaz(ayna, cevap.agirliklar);
      setAdim(cevap.adim);
      setCalisiyor(cevap.calisiyor);
      if (!cevap.calisiyor) tazele();
      else tazelemeIste();
    };

    const istek: EgitimIstegi = {
      tur: "kur",
      ayar: ayna.ayar,
      egitim: { ogrenmeOrani: durum.ogrenmeOrani, yigin: 32, sicaklik: 1, kirpma: 5 },
    };
    worker.postMessage(istek);

    return () => {
      worker.terminate();
      workerRef.current = null;
      if (zamanlayiciRef.current != null) clearTimeout(zamanlayiciRef.current);
      zamanlayiciRef.current = null;
    };
    // Model biçimi değiştiğinde worker baştan kurulur.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ayna]);

  // Öğrenme oranı eğitim sürerken bile canlı uygulanır.
  useEffect(() => {
    workerRef.current?.postMessage({
      tur: "egitimAyari",
      egitim: { ogrenmeOrani: durum.ogrenmeOrani },
    } satisfies EgitimIstegi);
  }, [durum.ogrenmeOrani]);

  const gonder = (istek: EgitimIstegi) => workerRef.current?.postMessage(istek);

  const [olcumRef, genislik] = genislikIzle();
  const izgaralar = useMemo(() => izgaralariTopla(ayna), [ayna]);

  return (
    <div className="space-y-4" ref={olcumRef}>
      <section className="flex flex-wrap items-center gap-x-3 gap-y-2 border border-cizgi bg-yuzey px-3 py-2">
        <button
          onClick={() =>
            calisiyor ? gonder({ tur: "dur" }) : gonder({ tur: "basla", adimSayisi: durum.adimSayisi })
          }
          className="border border-cizgi-parlak px-3 py-1 text-[12px] text-metin hover:bg-yuzey-2"
        >
          {calisiyor ? "■ durdur" : `▶ ${adet(durum.adimSayisi)} adım eğit`}
        </button>
        <button
          onClick={() => setYenileme((y) => y + 1)}
          className="border border-cizgi px-2.5 py-1 text-[11px] text-soluk hover:border-cizgi-parlak hover:text-metin"
        >
          sıfırla
        </button>
        <span className="sayi text-[11px] text-cok-soluk">
          {adet(adim)} adım · {adet(parametreSayisi(ayna))} parametre
          {tokenSayisi > 0 && <> · {adet(tokenSayisi)} token metin</>}
        </span>
        <span className="sayi ml-auto text-[11px] text-cok-soluk">
          {calisiyor ? "eğitim sürüyor — sayfa serbest, küreyi döndürebilirsiniz" : "durdu"}
        </span>
      </section>

      <section className="border border-cizgi bg-yuzey p-3">
        <KayipEgrisi
          kayiplar={cizim.kayiplar}
          dogrulama={cizim.dogrulama}
          genislik={Math.max(300, genislik - 26)}
        />
      </section>

      <KaydiracPaneli>
        <div className={calisiyor ? "opacity-45" : undefined}>
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-[12px] text-metin">Dikkat katmanı</span>
            <button
              onClick={() => setDurum((d) => ({ ...d, dikkat: !d.dikkat }))}
              disabled={calisiyor}
              className={`border px-2 py-0.5 text-[11px] transition-colors ${
                durum.dikkat
                  ? "border-cizgi-parlak text-metin"
                  : "border-cizgi text-cok-soluk hover:text-soluk"
              }`}
            >
              {durum.dikkat ? "açık" : "kapalı"}
            </button>
          </div>
          <p className="mt-2 text-[10px] leading-snug text-cok-soluk">
            Açıkken bağlamdaki karakterler birbirine bakabilir: her konum, öncesindeki hangi
            konumlara ne kadar ağırlık vereceğine kendi karar verir. Kapalıyken bağlam uç uca
            eklenip sabit bir projeksiyondan geçer — pozisyon başına ağırlıklar öğrenilir ama
            karakterler birbirine bakamaz.
            <span className="text-soluk"> · değiştirmek modeli baştan kurar, eğitim sıfırlanır.</span>
          </p>
        </div>
        <Kaydirac
          ad="Sıcaklık"
          aciklama="Softmax'tan önce skorları böler. Düşük değer modeli en olası harfe sabitler, yüksek değer dağıtır. Yalnızca üretimi etkiler; eğitim her zaman 1 ile yapılır."
          min={0.1}
          max={2}
          adim={0.05}
          deger={durum.sicaklik}
          onDegisti={(v) => setDurum((d) => ({ ...d, sicaklik: v }))}
          bicim={(v) => sayi(v, 2)}
        />
        <Kaydirac
          ad="Öğrenme oranı"
          aciklama="Her adımda ağırlıkların gradyan yönünde ne kadar oynatılacağı. Küçükse öğrenme yavaş, büyükse kayıp zıplar ve dağılabilir."
          min={0.001}
          max={1}
          adim={0.001}
          deger={durum.ogrenmeOrani}
          onDegisti={(v) => setDurum((d) => ({ ...d, ogrenmeOrani: v }))}
          bicim={(v) => sayi(v, 3)}
        />
        <Kaydirac
          ad="Eğitim adımı"
          aciklama="Eğit düğmesine basınca kaç adım atılacağı. Her adım metinden rastgele 32 örnek görür."
          min={1}
          max={2000}
          adim={1}
          deger={durum.adimSayisi}
          onDegisti={(v) => setDurum((d) => ({ ...d, adimSayisi: Math.round(v) }))}
          bicim={(v) => adet(Math.round(v))}
        />
        <Kaydirac
          ad="Gömme boyutu (D)"
          aciklama="Her karakteri kaç sayıyla temsil edeceğimiz. Büyüdükçe model daha ince ayrımlar yapabilir ama daha yavaş öğrenir."
          min={4}
          max={64}
          adim={1}
          deger={durum.D}
          onDegisti={(v) => setDurum((d) => ({ ...d, D: Math.round(v) }))}
          yenidenKurar
          devreDisi={calisiyor}
        />
        <Kaydirac
          ad="Katman sayısı"
          aciklama="Üst üste kaç MLP bloğu olacağı. Her blok ana yola bir düzeltme ekler."
          min={1}
          max={4}
          adim={1}
          deger={durum.katman}
          onDegisti={(v) => setDurum((d) => ({ ...d, katman: Math.round(v) }))}
          yenidenKurar
          devreDisi={calisiyor}
        />
        <Kaydirac
          ad="Tohum"
          aciklama="Rastgele başlangıcı belirler. Aynı tohum her zaman aynı modeli üretir; adres çubuğundaki link de bu yüzden aynı sayıları açar."
          min={0}
          max={200}
          adim={1}
          deger={durum.seed}
          onDegisti={(v) => setDurum((d) => ({ ...d, seed: Math.round(v) }))}
          yenidenKurar
          devreDisi={calisiyor}
        />
        <Kaydirac
          ad="Animasyon hızı"
          aciklama="İleri geçiş ekranındaki matris çarpımı canlandırmasının hızı: bir karede kaç çarpma işleniyor."
          min={0.25}
          max={160}
          adim={0.25}
          deger={durum.hiz}
          onDegisti={(v) => setDurum((d) => ({ ...d, hiz: v }))}
          bicim={(v) => (v < 1 ? sayi(v, 2) : String(Math.round(v)))}
        />
      </KaydiracPaneli>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="border border-cizgi bg-yuzey p-3">
          <h2 className="mb-2 text-[13px] font-medium text-metin">Gömme küresi</h2>
          <GommeKuresi model={ayna} surum={surum} yukseklik={400} />
        </section>
        {durum.dikkat ? (
          <CanliDikkat model={ayna} metin={durum.metin} surum={surum} />
        ) : (
          <section className="border border-cizgi bg-yuzey p-3">
            <h2 className="mb-2 text-[13px] font-medium text-metin">Dikkat haritası</h2>
            <p className="text-[11px] leading-relaxed text-cok-soluk">
              Dikkat katmanı kapalı. Açarsanız bağlamdaki karakterlerin birbirine ne kadar
              baktığını buradan canlı izleyebilirsiniz — eğitimin başında dağılım neredeyse düz
              olur, eğitim ilerledikçe keskinleşir.
            </p>
          </section>
        )}
      </div>

      <Uretim model={ayna} adim={adim} sicaklik={durum.sicaklik} baslangic={durum.metin} />

      <section className="space-y-3">
        <h2 className="text-[13px] font-medium text-metin">
          Ağırlıklar
          <span className="ml-2 text-[11px] font-normal text-cok-soluk">
            eğitim sürerken değişen hücreler kısa süre parlıyor · burada salt okunur, elle
            değiştirmek için ağırlıklar ekranı
          </span>
        </h2>
        {izgaralar.map(({ matris, aciklama, satirEtiketi, sutunEtiketi }) => (
          <div key={matris.ad}>
            <p className="mb-1 px-0.5 text-[11px] text-cok-soluk">{aciklama}</p>
            <AgirlikIzgarasi
              matris={matris}
              surum={surum}
              satirEtiketi={satirEtiketi}
              sutunEtiketi={sutunEtiketi}
              parlamaAktif
            />
          </div>
        ))}
      </section>
    </div>
  );
}

/**
 * Eğitim sürerken dikkat haritası. Eğitimin başında satırlar neredeyse düz
 * dağılır (model henüz nereye bakacağını bilmiyor); eğitim ilerledikçe
 * belirli konumlara yığılmaya başlar. Ölçüldü: dağılımın ortalama entropisi
 * 2,08'den (düz dağılım ln(8) = 2,079) 1,63'e iniyor.
 */
function CanliDikkat({ model, metin, surum }: { model: Model; metin: string; surum: number }) {
  const iz = useMemo(() => ileriGecis(model, encode(metin), 1), [model, metin, surum]);
  const [olcumRef, genislik] = genislikIzle();
  return (
    <section className="border border-cizgi bg-yuzey p-3" ref={olcumRef}>
      <h2 className="mb-1 text-[13px] font-medium text-metin">Dikkat haritası</h2>
      <p className="mb-3 text-[11px] text-cok-soluk">
        "{metin}" bağlamında · eğitim ilerledikçe dağılımın keskinleşmesini izleyin
      </p>
      {iz.dikkat && (
        <DikkatHaritasi
          dikkat={iz.dikkat}
          baglamIds={iz.baglamIds}
          genislik={Math.max(260, genislik - 26)}
        />
      )}
    </section>
  );
}

function genislikIzle(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [genislik, setGenislik] = useState(880);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const g = new ResizeObserver((girdiler) => {
      for (const girdi of girdiler) setGenislik(girdi.contentRect.width);
    });
    g.observe(el);
    setGenislik(el.clientWidth);
    return () => g.disconnect();
  }, []);
  return [ref, genislik];
}
