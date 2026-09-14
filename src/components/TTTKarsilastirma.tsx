/**
 * TTT MODU — DONMUŞ MODEL İLE ÇIKARIM ANINDA ÖĞRENEN MODELİN KARŞILAŞTIRMASI
 * ==========================================================================
 *
 * Aynı metin iki modele aynı anda okutuluyor:
 *   SOL  — bütün ağırlıklar donmuş. Bugün yaygın olan çıkarım budur:
 *          model okuduğu şeyden öğrenmez.
 *   SAĞ  — sadece son projeksiyon matrisi her karakterden sonra bir adım
 *          güncelleniyor. Model okudukça kendini metne uyduruyor.
 *
 * Her konumda ikisinin de bir sonraki karaktere verdiği olasılık ölçülüyor.
 * Aradaki fark tek bir karakterde gürültüdür; anlamlı olan biriken fark, o
 * yüzden eğri kümülatif ortalama kaybı çiziyor.
 *
 * DÜRÜSTLÜK NOTU
 * TTT'nin kazancı metne bağlı. Modelin zaten iyi bildiği bir metinde
 * neredeyse hiçbir şey kazandırmaz; tekrar eden ya da eğitim metninden
 * farklı bir metinde belirgin biçimde öne geçer. Ekran hangi sonuç çıkarsa
 * onu gösteriyor — kazandığını varsayan bir yerleşim kurmadık.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import {
  type IleriIz,
  type Model,
  agirliklariYaz,
  ileriGecis,
  kayip,
  modelOlustur,
  parametreSayisi,
} from "../lib/model.ts";
import { cikisFarki, hizliAgirlikAdimi, hizliParametreSayisi, modelKopyala } from "../lib/ttt.ts";
import { encode, gorunurAd } from "../lib/tokenizer.ts";
import type { EgitimCevabi, EgitimIstegi } from "../workers/egitim.ts";
import { durumOku, durumYaz, VARSAYILAN_DURUM } from "../lib/durum.ts";
import { adet, sayi, yuzde } from "../lib/bicim.ts";
import { ModelHaritasi, type HaritaOgesi } from "./ModelHaritasi.tsx";

const SINAMA_PARCASI =
  "okyanusun ortasında bir fener durur. fenerin bekçisi her gece lambayı yakar, sabaha kadar bekler. deniz karardıkça ışık uzaklara gider. ";
const VARSAYILAN_SINAMA = SINAMA_PARCASI.repeat(4);

interface Olcum {
  donuk: number;
  hizli: number;
}

export default function TTTKarsilastirma() {
  const [durum, setDurum] = useState(VARSAYILAN_DURUM);
  const [hazir, setHazir] = useState(false);
  useEffect(() => {
    setDurum(durumOku());
    setHazir(true);
  }, []);
  useEffect(() => {
    if (hazir) durumYaz(durum);
  }, [durum, hazir]);

  const [sinamaMetni, setSinamaMetni] = useState(VARSAYILAN_SINAMA);
  /**
   * Hızlı ağırlık öğrenme oranının varsayılanı ölçümle seçildi (bkz.
   * scripts/cekirdek.ts, 7. bölüm). Tarama sonucu: 0,01 civarı her üç
   * sınamada da ya kazanıyor ya başabaş; 0,02–0,03 tekrar eden metinde en
   * yüksek kazancı veriyor ama tek turluk metinde zarar ediyor; 0,06'dan
   * sonra her durumda bozuyor. Kullanıcı bozulduğu yeri de görebilsin diye
   * kaydıracın üst sınırını 0,15'te bıraktık.
   */
  const [hizliOran, setHizliOran] = useState(0.01);

  // --- temel model: worker'da eğitilir, burada aynalanır -------------------
  const temel = useMemo<Model>(
    () => modelOlustur({ D: durum.D, katmanSayisi: durum.katman, baglam: 8, seed: durum.seed }),
    [durum.D, durum.katman, durum.seed],
  );
  const workerRef = useRef<Worker | null>(null);
  const [temelAdim, setTemelAdim] = useState(0);
  const [temelKayip, setTemelKayip] = useState<number | null>(null);
  const [egitiliyor, setEgitiliyor] = useState(false);
  const [temelSurum, setTemelSurum] = useState(0);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/egitim.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (olay: MessageEvent<EgitimCevabi>) => {
      const cevap = olay.data;
      if (cevap.tur === "hazir") {
        agirliklariYaz(temel, cevap.agirliklar);
        setTemelAdim(0);
        setEgitiliyor(false);
        setTemelSurum((s) => s + 1);
        return;
      }
      if (cevap.agirliklar) agirliklariYaz(temel, cevap.agirliklar);
      setTemelAdim(cevap.adim);
      setEgitiliyor(cevap.calisiyor);
      if (cevap.kayiplar.length > 0) setTemelKayip(cevap.kayiplar[cevap.kayiplar.length - 1]);
      if (cevap.agirliklar) setTemelSurum((s) => s + 1);
    };
    worker.postMessage({
      tur: "kur",
      ayar: temel.ayar,
      egitim: { ogrenmeOrani: durum.ogrenmeOrani, yigin: 32, sicaklik: 1, kirpma: 5 },
    } satisfies EgitimIstegi);
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [temel]);

  // --- iki kopya ve akış ---------------------------------------------------
  const ids = useMemo(() => encode(sinamaMetni), [sinamaMetni]);
  const donukRef = useRef<Model | null>(null);
  const hizliRef = useRef<Model | null>(null);
  const olcumlerRef = useRef<Olcum[]>([]);
  const [konum, setKonum] = useState(0);
  const [olcumSurumu, setOlcumSurumu] = useState(0);
  const [sonDurum, setSonDurum] = useState<{
    hedef: number;
    donuk: IleriIz;
    hizli: IleriIz;
  } | null>(null);
  const [oynatiliyor, setOynatiliyor] = useState(false);
  const [surukleme, setSurukleme] = useState(0);

  const sifirla = useCallback(() => {
    donukRef.current = modelKopyala(temel);
    hizliRef.current = modelKopyala(temel);
    olcumlerRef.current = [];
    setKonum(0);
    setSonDurum(null);
    setSurukleme(0);
    setOlcumSurumu((s) => s + 1);
    setOynatiliyor(false);
  }, [temel]);

  // Temel model her değiştiğinde (eğitim, tohum, sıfırlama) akış baştan başlar:
  // yarısı eski ağırlıklarla ölçülmüş bir karşılaştırma yanıltıcı olurdu.
  useEffect(() => {
    sifirla();
  }, [temelSurum, ids, sifirla]);

  const adimAt = useCallback(
    (adet: number) => {
      const donuk = donukRef.current;
      const hizli = hizliRef.current;
      if (!donuk || !hizli) return;
      const C = temel.ayar.baglam;
      let t = konum;
      let sonuncu: typeof sonDurum = null;

      for (let n = 0; n < adet && t + 1 < ids.length; n++) {
        const baglam = ids.slice(Math.max(0, t + 1 - C), t + 1);
        const hedef = ids[t + 1];

        const izDonuk = ileriGecis(donuk, baglam, 1);
        const izHizli = ileriGecis(hizli, baglam, 1);
        olcumlerRef.current.push({
          donuk: kayip(izDonuk.olasilik, hedef),
          hizli: kayip(izHizli.olasilik, hedef),
        });
        sonuncu = { hedef, donuk: izDonuk, hizli: izHizli };

        // Çıkarım anı güncellemesi — yalnızca sağdaki modelde, yalnızca
        // son projeksiyon matrisinde.
        hizliAgirlikAdimi(hizli, izHizli, hedef, hizliOran);
        t++;
      }

      setKonum(t);
      if (sonuncu) setSonDurum(sonuncu);
      setSurukleme(cikisFarki(hizli, donuk));
      setOlcumSurumu((s) => s + 1);
      if (t + 1 >= ids.length) setOynatiliyor(false);
    },
    [konum, ids, temel.ayar.baglam, hizliOran],
  );

  useEffect(() => {
    if (!oynatiliyor) return;
    let raf = 0;
    const kare = () => {
      adimAt(3);
      raf = requestAnimationFrame(kare);
    };
    raf = requestAnimationFrame(kare);
    return () => cancelAnimationFrame(raf);
  }, [oynatiliyor, adimAt]);

  const [olcumRef, genislik] = genislikIzle();

  const ozet = useMemo(() => {
    const o = olcumlerRef.current;
    if (o.length === 0) return null;
    let dToplam = 0;
    let hToplam = 0;
    for (const x of o) {
      dToplam += x.donuk;
      hToplam += x.hizli;
    }
    return {
      adet: o.length,
      donuk: dToplam / o.length,
      hizli: hToplam / o.length,
    };
  }, [olcumSurumu]);

  const harita = useMemo<HaritaOgesi[]>(() => {
    const liste: HaritaOgesi[] = [
      { matris: temel.E, hizli: false },
      { matris: temel.Wgiris, hizli: false },
    ];
    temel.bloklar.forEach((b) => {
      liste.push({ matris: b.W1, hizli: false });
      liste.push({ matris: b.W2, hizli: false });
    });
    liste.push({ matris: temel.Wcikis, hizli: true });
    return liste;
  }, [temel, temelSurum]);

  return (
    <div className="space-y-4" ref={olcumRef}>
      {/* temel model */}
      <section className="flex flex-wrap items-center gap-x-3 gap-y-2 border border-cizgi bg-yuzey px-3 py-2">
        <span className="text-[12px] text-soluk">Temel model</span>
        <button
          onClick={() =>
            egitiliyor
              ? workerRef.current?.postMessage({ tur: "dur" } satisfies EgitimIstegi)
              : workerRef.current?.postMessage({
                  tur: "basla",
                  adimSayisi: durum.adimSayisi,
                } satisfies EgitimIstegi)
          }
          className="border border-cizgi-parlak px-3 py-1 text-[11px] text-metin hover:bg-yuzey-2"
        >
          {egitiliyor ? "■ durdur" : `▶ ${adet(durum.adimSayisi)} adım eğit`}
        </button>
        <span className="sayi text-[11px] text-cok-soluk">
          {adet(temelAdim)} adım
          {temelKayip != null && <> · kayıp {sayi(temelKayip, 3)}</>}
        </span>
        <p className="w-full text-[10px] leading-relaxed text-cok-soluk">
          TTT'yi eğitilmemiş bir modelde denemek anlamsız: ikisi de çöp üretir. Önce birkaç yüz
          adım eğitin, sonra aşağıdaki akışı başlatın. Temel model değiştiğinde karşılaştırma
          kendiliğinden sıfırlanır.
        </p>
      </section>

      <ModelHaritasi
        ogeler={harita}
        hizliSayi={hizliParametreSayisi(temel)}
        toplamSayi={parametreSayisi(temel)}
      />

      {/* akış kontrolü */}
      <section className="space-y-2 border border-cizgi bg-yuzey px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setOynatiliyor((o) => !o)}
            disabled={konum + 1 >= ids.length}
            className="border border-cizgi-parlak px-3 py-1 text-[11px] text-metin hover:bg-yuzey-2 disabled:opacity-40"
          >
            {oynatiliyor ? "■ duraklat" : "▶ oku"}
          </button>
          <button
            onClick={() => {
              setOynatiliyor(false);
              adimAt(1);
            }}
            disabled={konum + 1 >= ids.length}
            className="border border-cizgi px-2.5 py-1 text-[11px] text-soluk hover:border-cizgi-parlak hover:text-metin disabled:opacity-40"
          >
            tek karakter ▸
          </button>
          <button
            onClick={sifirla}
            className="border border-cizgi px-2.5 py-1 text-[11px] text-soluk hover:border-cizgi-parlak hover:text-metin"
          >
            baştan
          </button>
          <span className="sayi text-[11px] text-cok-soluk">
            {adet(konum)} / {adet(Math.max(0, ids.length - 1))} karakter
          </span>

          <div className="ml-auto flex items-center gap-2">
            <label htmlFor="hizli-oran" className="text-[11px] text-cok-soluk">
              hızlı ağırlık öğrenme oranı
            </label>
            <input
              id="hizli-oran"
              type="range"
              min={0}
              max={0.15}
              step={0.0025}
              value={hizliOran}
              onChange={(e) => setHizliOran(Number(e.target.value))}
              className="h-1 w-28 accent-white"
            />
            <span className="sayi w-14 text-right text-[11px] text-soluk">{sayi(hizliOran, 4)}</span>
          </div>
        </div>
        <p className="text-[10px] leading-relaxed text-cok-soluk">
          Çıkarım anında ne kadar hızlı uyum sağlanacağı. Küçük değerler güvenli ama yavaş;
          büyüttükçe tekrar eden metinde kazanç artar, ardından model kendi kendini bozmaya başlar.
          Sıfırda iki panel aynı şeyi yapar — karşılaştırmanın kontrol grubu budur.
        </p>

        <label htmlFor="sinama" className="block text-[11px] text-cok-soluk">
          okunacak metin — tekrar eden ya da eğitim metninden farklı bir metinde fark daha belirgin
        </label>
        <textarea
          id="sinama"
          value={sinamaMetni}
          onChange={(e) => setSinamaMetni(e.target.value)}
          rows={3}
          spellCheck={false}
          className="sayi w-full resize-y border border-cizgi bg-zemin px-2 py-1 text-[12px] leading-relaxed text-metin"
        />
        <OkunanMetin ids={ids} konum={konum} />
      </section>

      {/* iki panel */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          baslik="Donmuş"
          altBaslik="bütün ağırlıklar sabit — bugünkü çıkarım"
          vurgulu={false}
          iz={sonDurum?.donuk ?? null}
          hedef={sonDurum?.hedef ?? null}
          ortalama={ozet?.donuk ?? null}
        />
        <Panel
          baslik="Hızlı ağırlık"
          altBaslik="son projeksiyon matrisi her karakterde güncelleniyor"
          vurgulu
          iz={sonDurum?.hizli ?? null}
          hedef={sonDurum?.hedef ?? null}
          ortalama={ozet?.hizli ?? null}
          ek={
            <span className="sayi text-[11px] text-cok-soluk">
              donmuş kopyadan uzaklık {sayi(surukleme, 4)}
            </span>
          }
        />
      </div>

      <section className="border border-cizgi bg-yuzey p-3">
        <FarkEgrisi
          olcumler={olcumlerRef.current}
          surum={olcumSurumu}
          genislik={Math.max(300, genislik - 26)}
        />
      </section>
    </div>
  );
}

/** Okunan metin: işlenmiş kısım parlak, sıradaki karakter işaretli. */
function OkunanMetin({ ids, konum }: { ids: number[]; konum: number }) {
  const pencere = 90;
  const bas = Math.max(0, konum - pencere + 20);
  const parca = ids.slice(bas, bas + pencere);
  return (
    <p className="sayi overflow-hidden border border-cizgi bg-zemin px-2 py-1.5 text-[12px] whitespace-pre">
      {parca.map((id, i) => {
        const mutlak = bas + i;
        const okundu = mutlak <= konum;
        const sirada = mutlak === konum + 1;
        return (
          <span
            key={mutlak}
            className={
              sirada
                ? "bg-cizgi-parlak text-zemin"
                : okundu
                  ? "text-metin"
                  : "text-cok-soluk"
            }
          >
            {gorunurAd(id) === "␣" ? " " : gorunurAd(id)}
          </span>
        );
      })}
    </p>
  );
}

function Panel({
  baslik,
  altBaslik,
  vurgulu,
  iz,
  hedef,
  ortalama,
  ek,
}: {
  baslik: string;
  altBaslik: string;
  vurgulu: boolean;
  iz: IleriIz | null;
  hedef: number | null;
  ortalama: number | null;
  ek?: React.ReactNode;
}) {
  const ilkBes = useMemo(() => {
    if (!iz) return [];
    return Array.from(iz.olasilik)
      .map((olasilik, id) => ({ olasilik, id }))
      .sort((a, b) => b.olasilik - a.olasilik)
      .slice(0, 5);
  }, [iz]);

  const dogruOlasilik = iz && hedef != null ? iz.olasilik[hedef] : null;

  return (
    <section className={`border bg-yuzey ${vurgulu ? "border-cizgi-parlak" : "border-cizgi"}`}>
      <header className="border-b border-cizgi px-3 py-2">
        <h2 className="text-[13px] font-medium text-metin">{baslik}</h2>
        <p className="text-[11px] text-cok-soluk">{altBaslik}</p>
      </header>
      <div className="space-y-3 px-3 py-3">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <span className="sayi text-[18px] text-metin">
            {ortalama == null ? "—" : sayi(ortalama, 4)}
          </span>
          <span className="text-[11px] text-cok-soluk">ortalama kayıp (baştan beri)</span>
          {ek}
        </div>

        {hedef != null && (
          <div className="sayi text-[11px] text-soluk">
            doğru karakter <span className="text-metin">{gorunurAd(hedef)}</span> · bu modelin ona
            verdiği olasılık{" "}
            <span className="text-metin">{dogruOlasilik == null ? "—" : yuzde(dogruOlasilik)}</span>
          </div>
        )}

        <ul className="space-y-1">
          {ilkBes.length === 0 && (
            <li className="text-[11px] text-cok-soluk">akış başlamadı</li>
          )}
          {ilkBes.map(({ id, olasilik }) => (
            <li key={id} className="flex items-center gap-2">
              <span
                className={`sayi w-5 text-center text-[13px] ${
                  id === hedef ? "text-metin" : "text-soluk"
                }`}
              >
                {gorunurAd(id)}
              </span>
              <span className="sayi w-14 text-right text-[11px] text-soluk">{yuzde(olasilik)}</span>
              <span className="h-3 flex-1 bg-yuzey-2">
                <span
                  className={`block h-full ${id === hedef ? "bg-metin" : "bg-cizgi-parlak"}`}
                  style={{ width: `${Math.max(0.5, olasilik * 100)}%` }}
                />
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

const KENAR = { ust: 12, sag: 12, alt: 26, sol: 42 };
/** Fark grafiğinin yüksekliği. */
const FARK_Y = 70;
/** Eğrinin başında çizilmeyen konum sayısı (ölçüme dahil, resme değil). */
const ATLANAN = 10;

/**
 * Biriken fark. Tek bir karakterdeki fark gürültü; anlamlı olan kümülatif
 * ortalama, çünkü TTT'nin iddiası "okudukça iyileşir".
 */
function FarkEgrisi({
  olcumler,
  surum,
  genislik,
  yukseklik = 210,
}: {
  olcumler: Olcum[];
  surum: number;
  genislik: number;
  yukseklik?: number;
}) {
  const cizimG = Math.max(120, genislik - KENAR.sol - KENAR.sag);
  const cizimY = yukseklik - KENAR.ust - KENAR.alt;

  const { donukYol, hizliYol, farkYol, x, y, yFark, son } = useMemo(() => {
    const n = olcumler.length;
    const donukBirikim: Array<[number, number]> = [];
    const hizliBirikim: Array<[number, number]> = [];
    const farkBirikim: Array<[number, number]> = [];
    let dT = 0;
    let hT = 0;
    let enBuyuk = 0.5;
    let farkEnBuyuk = 0.05;
    for (let i = 0; i < n; i++) {
      dT += olcumler[i].donuk;
      hT += olcumler[i].hizli;
      const d = dT / (i + 1);
      const h = hT / (i + 1);
      // İlk birkaç konumun "ortalaması" tek iki örnekten ibaret ve beş kat
      // büyük çıkıyor; çizilirse bütün ekseni ezip asıl farkı görünmez
      // kılıyor. O yüzden ilk ATLANAN konumu çizmiyoruz — ölçüme dahiller,
      // sadece resme değiller.
      if (i >= ATLANAN) {
        donukBirikim.push([i, d]);
        hizliBirikim.push([i, h]);
        farkBirikim.push([i, d - h]);
        if (d > enBuyuk) enBuyuk = d;
        if (h > enBuyuk) enBuyuk = h;
        farkEnBuyuk = Math.max(farkEnBuyuk, Math.abs(d - h));
      }
    }
    const xO = scaleLinear().domain([0, Math.max(1, n - 1)]).range([0, cizimG]);
    const yO = scaleLinear().domain([0, enBuyuk * 1.08]).range([cizimY, 0]).nice();
    const yF = scaleLinear().domain([-farkEnBuyuk * 1.15, farkEnBuyuk * 1.15]).range([FARK_Y, 0]).nice();
    const ciz = line<[number, number]>()
      .x((p) => xO(p[0]))
      .y((p) => yO(p[1]));
    const cizFark = line<[number, number]>()
      .x((p) => xO(p[0]))
      .y((p) => yF(p[1]));
    return {
      donukYol: donukBirikim.length > 1 ? ciz(donukBirikim) : null,
      hizliYol: hizliBirikim.length > 1 ? ciz(hizliBirikim) : null,
      farkYol: farkBirikim.length > 1 ? cizFark(farkBirikim) : null,
      x: xO,
      y: yO,
      yFark: yF,
      son:
        n > 0
          ? { donuk: dT / n, hizli: hT / n }
          : null,
    };
    // olcumler yerinde büyüyen bir dizi; sürüm sayacı yeniden hesabı tetikler.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surum, cizimG, cizimY]);

  const fark = son ? son.donuk - son.hizli : null;

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="text-[13px] text-metin">Biriken fark</span>
        {fark != null && (
          <span className="sayi text-[12px] text-soluk">
            hızlı ağırlık {fark >= 0 ? "önde" : "geride"}: {sayi(Math.abs(fark), 4)} kayıp{" "}
            {fark >= 0 ? "daha az" : "daha çok"}
          </span>
        )}
      </div>

      <svg width={genislik} height={yukseklik} role="img" aria-label="biriken kayıp farkı">
        <g transform={`translate(${KENAR.sol},${KENAR.ust})`}>
          {y.ticks(5).map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line x1={0} x2={cizimG} stroke="#1d222c" />
              <text x={-8} dy="0.32em" textAnchor="end" className="sayi" fill="#5e6878" fontSize={10}>
                {t}
              </text>
            </g>
          ))}
          {x.ticks(6).map((t) => (
            <text
              key={t}
              x={x(t)}
              y={cizimY + 16}
              textAnchor="middle"
              className="sayi"
              fill="#5e6878"
              fontSize={10}
            >
              {t}
            </text>
          ))}
          {donukYol && <path d={donukYol} fill="none" stroke="#aab4c4" strokeWidth={1.5} strokeDasharray="5 4" />}
          {hizliYol && <path d={hizliYol} fill="none" stroke="#e6ebf2" strokeWidth={2} />}
        </g>
      </svg>

      {/*
        İki eğri birbirine yakın seyrettiği için aradaki farkı ayrı bir
        eksende çiziyoruz. Üstteki grafiğin sıfır tabanını bozup farkı
        büyütmek daha "etkileyici" olurdu ama yanıltıcı olurdu; fark kendi
        ekseninde, sıfır çizgisiyle birlikte duruyor.
      */}
      <div className="mt-2 border-t border-cizgi pt-2">
        <p className="mb-1 text-[10px] text-cok-soluk">
          fark (donmuş − hızlı) · sıfırın üstü hızlı ağırlığın önde olduğu yer
        </p>
        <svg width={genislik} height={FARK_Y + KENAR.alt} role="img" aria-label="biriken fark eğrisi">
          <g transform={`translate(${KENAR.sol},4)`}>
            {yFark.ticks(3).map((t) => (
              <g key={t} transform={`translate(0,${yFark(t)})`}>
                <line x1={0} x2={cizimG} stroke={t === 0 ? "#38414f" : "#1d222c"} />
                <text x={-8} dy="0.32em" textAnchor="end" className="sayi" fill="#5e6878" fontSize={9}>
                  {t}
                </text>
              </g>
            ))}
            {farkYol && <path d={farkYol} fill="none" stroke="#e6ebf2" strokeWidth={1.5} />}
          </g>
        </svg>
      </div>

      <div className="mt-1 flex flex-wrap gap-x-4 text-[10px] text-cok-soluk">
        <span className="flex items-center gap-1.5">
          <svg width={18} height={6} aria-hidden>
            <line x1={0} x2={18} y1={3} y2={3} stroke="#aab4c4" strokeWidth={1.5} strokeDasharray="5 4" />
          </svg>
          donmuş model
        </span>
        <span className="flex items-center gap-1.5">
          <svg width={18} height={6} aria-hidden>
            <line x1={0} x2={18} y1={3} y2={3} stroke="#e6ebf2" strokeWidth={2} />
          </svg>
          hızlı ağırlık
        </span>
        <span className="ml-auto">
          dikey eksen: baştan beri ortalama kayıp · yatay: okunan karakter · ilk {ATLANAN} konum
          çizilmiyor (birkaç örneklik ortalama gürültüden ibaret)
        </span>
      </div>
    </div>
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
