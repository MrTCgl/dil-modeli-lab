/**
 * İLERİ GEÇİŞ — ADIM ADIM
 * =======================
 *
 * Kullanıcı bir cümle yazar; model son 8 karaktere bakıp bir sonrakini
 * tahmin eder. Bu ekran o tahminin nasıl çıktığını aşama aşama gösterir.
 *
 * Aşağıdaki her sayı, ileri geçişin kaydından (IleriIz) okunur. Yani
 * ekrandaki değerler modelin gerçekten hesapladığı değerlerdir; gösterim
 * için ayrıca hesaplanmış, yuvarlanmış ya da "temsilî" hiçbir sayı yok.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { type IleriIz, type Model, ileriGecis, modelOlustur } from "../lib/model.ts";
import { ALFABE, encode, gorunurAd } from "../lib/tokenizer.ts";
import { type Asama, asamaVektoru, asamalariCikar } from "../lib/asamalar.ts";
import { durumOku, durumYaz, VARSAYILAN_DURUM } from "../lib/durum.ts";
import { sayi, yuzde } from "../lib/bicim.ts";
import { VektorSeridi } from "./VektorSeridi.tsx";
import { CarpimCanlandirma } from "./CarpimCanlandirma.tsx";

/** Sabit süreli (çarpım olmayan) aşamaların oynatma süresi, kare cinsinden. */
const DURAKLAMA_KARESI = 42;

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

export default function IleriGecisEkrani() {
  const [durum, setDurum] = useState(VARSAYILAN_DURUM);
  const [hazir, setHazir] = useState(false);

  // URL'deki durum ilk boyamadan sonra okunur: sunucuda üretilen HTML ile
  // istemcideki ilk render'ın ayrışmaması için.
  useEffect(() => {
    setDurum(durumOku());
    setHazir(true);
  }, []);
  useEffect(() => {
    if (hazir) durumYaz(durum);
  }, [durum, hazir]);

  const model = useMemo<Model>(
    () => modelOlustur({ D: durum.D, katmanSayisi: durum.katman, baglam: 8, seed: durum.seed }),
    [durum.D, durum.katman, durum.seed],
  );
  const iz = useMemo<IleriIz>(
    () => ileriGecis(model, encode(durum.metin), durum.sicaklik),
    [model, durum.metin, durum.sicaklik],
  );
  const asamalar = useMemo(() => asamalariCikar(model, iz), [model, iz]);

  const [asamaIndeks, setAsamaIndeks] = useState(0);
  const [altAdim, setAltAdim] = useState(0);
  const [oynatiliyor, setOynatiliyor] = useState(false);
  // Animasyon hızı ortak durumda: eğitim panelindeki kaydıraçla aynı değer,
  // ve adres çubuğuyla paylaşılabiliyor.
  const hiz = durum.hiz;
  const setHiz = (v: number) => setDurum((d) => ({ ...d, hiz: v }));

  const asamaRef = useRef(0);
  const altRef = useRef(0);
  const hizRef = useRef(hiz);
  hizRef.current = hiz;

  // Model ya da metin değişince baştan başla.
  useEffect(() => {
    asamaRef.current = 0;
    altRef.current = 0;
    setAsamaIndeks(0);
    setAltAdim(0);
    setOynatiliyor(false);
  }, [model, durum.metin]);

  const git = (indeks: number, alt = 0) => {
    const sinirli = Math.max(0, Math.min(asamalar.length - 1, indeks));
    asamaRef.current = sinirli;
    altRef.current = alt;
    setAsamaIndeks(sinirli);
    setAltAdim(alt);
  };

  // --- oynatma döngüsü -----------------------------------------------------
  useEffect(() => {
    if (!oynatiliyor) return;
    let id = 0;
    let onceki = performance.now();
    let birikim = 0;
    let sonAsama = asamaRef.current;

    const kare = (t: number) => {
      const dt = Math.min(64, t - onceki);
      onceki = t;
      const kareSayisi = dt / 16.67;

      let a = asamaRef.current;
      let alt = altRef.current;
      if (a !== sonAsama) {
        birikim = 0;
        sonAsama = a;
      }
      const asama = asamalar[a];

      if (asama.altAdim > 0) {
        birikim += hizRef.current * kareSayisi;
        const artis = Math.floor(birikim);
        if (artis > 0) {
          birikim -= artis;
          alt += artis;
        }
        if (alt >= asama.altAdim) {
          if (a + 1 < asamalar.length) {
            a++;
            alt = 0;
            birikim = 0;
            sonAsama = a;
          } else {
            alt = asama.altAdim;
            setOynatiliyor(false);
          }
        }
      } else {
        birikim += kareSayisi;
        if (birikim >= DURAKLAMA_KARESI) {
          birikim = 0;
          if (a + 1 < asamalar.length) {
            a++;
            alt = 0;
            sonAsama = a;
          } else {
            setOynatiliyor(false);
          }
        }
      }

      asamaRef.current = a;
      altRef.current = alt;
      setAsamaIndeks(a);
      setAltAdim(alt);
      id = requestAnimationFrame(kare);
    };

    id = requestAnimationFrame(kare);
    return () => cancelAnimationFrame(id);
  }, [oynatiliyor, asamalar]);

  const asama = asamalar[asamaIndeks];
  const [olcumRef, genislik] = genislikIzle();

  return (
    <div className="space-y-4" ref={olcumRef}>
      <MetinGirdisi
        metin={durum.metin}
        onMetin={(m) => setDurum((d) => ({ ...d, metin: m }))}
        iz={iz}
      />

      <AsamaSeridi asamalar={asamalar} secili={asamaIndeks} onSec={(i) => git(i)} />

      <Kontroller
        asama={asama}
        asamaIndeks={asamaIndeks}
        toplamAsama={asamalar.length}
        altAdim={altAdim}
        oynatiliyor={oynatiliyor}
        hiz={hiz}
        setHiz={setHiz}
        onOynat={() => setOynatiliyor((o) => !o)}
        onSonraki={() => {
          setOynatiliyor(false);
          if (asama.altAdim > 0 && altAdim + 1 < asama.altAdim) git(asamaIndeks, altAdim + 1);
          else git(asamaIndeks + 1);
        }}
        onOnceki={() => {
          setOynatiliyor(false);
          if (asama.altAdim > 0 && altAdim > 0) git(asamaIndeks, altAdim - 1);
          else git(asamaIndeks - 1, 0);
        }}
        onSonrakiCikis={() => {
          setOynatiliyor(false);
          const satir = asama.altAdim > 0 ? altAdimSatiri(asama, model) : 0;
          if (satir > 0) git(asamaIndeks, Math.min(asama.altAdim, (Math.floor(altAdim / satir) + 1) * satir));
        }}
        onBastan={() => {
          setOynatiliyor(false);
          git(0);
        }}
        onBitir={() => {
          setOynatiliyor(false);
          git(asamaIndeks, asama.altAdim);
        }}
      />

      <section className="border border-cizgi bg-yuzey">
        <header className="border-b border-cizgi px-4 py-3">
          <h2 className="text-[15px] font-medium text-metin">{asama.baslik}</h2>
          <p className="mt-1 max-w-[70ch] text-[12px] leading-relaxed text-soluk">{asama.aciklama}</p>
        </header>
        <div className="p-4">
          <AsamaGovdesi
            asama={asama}
            model={model}
            iz={iz}
            altAdim={altAdim}
            genislik={Math.max(320, genislik - 34)}
            sicaklik={durum.sicaklik}
          />
        </div>
      </section>

      <HamVektor
        asama={asama}
        iz={iz}
        genislik={Math.max(320, genislik - 34)}
        hesaplanan={
          asama.altAdim > 0
            ? Math.floor(altAdim / Math.max(1, altAdimSatiri(asama, model)))
            : undefined
        }
      />
    </div>
  );
}

/** Bir çarpım aşamasında tek bir çıkış kaç terimden oluşuyor. */
function altAdimSatiri(asama: Asama, model: Model): number {
  switch (asama.tur) {
    case "giris":
      return model.Wgiris.satir;
    case "genisle":
      return model.bloklar[asama.blok].W1.satir;
    case "daralt":
      return model.bloklar[asama.blok].W2.satir;
    case "cikis":
      return model.Wcikis.satir;
    default:
      return 0;
  }
}

function MetinGirdisi({
  metin,
  onMetin,
  iz,
}: {
  metin: string;
  onMetin: (m: string) => void;
  iz: IleriIz;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border border-cizgi bg-yuzey px-3 py-2">
      <label htmlFor="metin" className="text-[12px] text-soluk">
        Metin
      </label>
      <input
        id="metin"
        value={metin}
        onChange={(e) => onMetin(e.target.value)}
        spellCheck={false}
        className="sayi min-w-[200px] flex-1 border border-cizgi bg-zemin px-2 py-1 text-[13px] text-metin"
      />
      <span className="sayi text-[11px] text-cok-soluk">
        modele giren son 8:{" "}
        <span className="text-soluk">{iz.baglamIds.map((id) => gorunurAd(id)).join("")}</span>
      </span>
    </div>
  );
}

function AsamaSeridi({
  asamalar,
  secili,
  onSec,
}: {
  asamalar: Asama[];
  secili: number;
  onSec: (i: number) => void;
}) {
  return (
    <nav className="flex flex-wrap gap-1" aria-label="ileri geçiş aşamaları">
      {asamalar.map((a, i) => (
        <button
          key={i}
          onClick={() => onSec(i)}
          title={a.baslik}
          className={`border px-2 py-1 text-[11px] transition-colors ${
            i === secili
              ? "border-cizgi-parlak bg-yuzey-2 text-metin"
              : i < secili
                ? "border-cizgi text-soluk hover:text-metin"
                : "border-cizgi text-cok-soluk hover:text-soluk"
          }`}
        >
          {kisaAd(a)}
        </button>
      ))}
    </nav>
  );
}

function kisaAd(a: Asama): string {
  switch (a.tur) {
    case "tokenler":
      return "token";
    case "gomme":
      return "gömme";
    case "giris":
      return "giriş";
    case "genisle":
      return `${a.blok + 1}· genişle`;
    case "relu":
      return `${a.blok + 1}· ReLU`;
    case "daralt":
      return `${a.blok + 1}· daralt`;
    case "artik":
      return `${a.blok + 1}· artık`;
    case "cikis":
      return "çıkış";
    case "softmax":
      return "softmax";
  }
}

function Kontroller({
  asama,
  asamaIndeks,
  toplamAsama,
  altAdim,
  oynatiliyor,
  hiz,
  setHiz,
  onOynat,
  onSonraki,
  onOnceki,
  onSonrakiCikis,
  onBastan,
  onBitir,
}: {
  asama: Asama;
  asamaIndeks: number;
  toplamAsama: number;
  altAdim: number;
  oynatiliyor: boolean;
  hiz: number;
  setHiz: (h: number) => void;
  onOynat: () => void;
  onSonraki: () => void;
  onOnceki: () => void;
  onSonrakiCikis: () => void;
  onBastan: () => void;
  onBitir: () => void;
}) {
  const dugme =
    "border border-cizgi px-2.5 py-1 text-[11px] text-soluk hover:border-cizgi-parlak hover:text-metin";
  return (
    <div className="flex flex-wrap items-center gap-2 border border-cizgi bg-yuzey px-3 py-2">
      <button onClick={onBastan} className={dugme}>
        başa dön
      </button>
      <button onClick={onOnceki} className={dugme}>
        ◂ önceki
      </button>
      <button
        onClick={onOynat}
        className="border border-cizgi-parlak px-3 py-1 text-[11px] text-metin hover:bg-yuzey-2"
      >
        {oynatiliyor ? "■ duraklat" : "▶ oynat"}
      </button>
      <button onClick={onSonraki} className={dugme}>
        sonraki ▸
      </button>
      {asama.altAdim > 0 && (
        <>
          <button onClick={onSonrakiCikis} className={dugme}>
            sonraki çıkış ▸▸
          </button>
          <button onClick={onBitir} className={dugme}>
            çarpımı bitir
          </button>
        </>
      )}

      <div className="ml-auto flex items-center gap-2">
        <label htmlFor="hiz" className="text-[11px] text-cok-soluk">
          hız
        </label>
        <input
          id="hiz"
          type="range"
          min={0.25}
          max={160}
          step={0.25}
          value={hiz}
          onChange={(e) => setHiz(Number(e.target.value))}
          className="h-1 w-28 accent-white"
        />
        <span className="sayi w-20 text-right text-[11px] text-cok-soluk">
          {hiz < 1 ? `${sayi(hiz, 2)} çarpma` : `${Math.round(hiz)} çarpma`}/kare
        </span>
      </div>

      <div className="sayi w-full text-[11px] text-cok-soluk">
        aşama {asamaIndeks + 1}/{toplamAsama}
        {asama.altAdim > 0 && (
          <>
            {" · "}
            çarpma {Math.min(altAdim, asama.altAdim).toLocaleString("tr-TR")}/
            {asama.altAdim.toLocaleString("tr-TR")}
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Aşama gövdeleri
// ---------------------------------------------------------------------------

function AsamaGovdesi({
  asama,
  model,
  iz,
  altAdim,
  genislik,
  sicaklik,
}: {
  asama: Asama;
  model: Model;
  iz: IleriIz;
  altAdim: number;
  genislik: number;
  sicaklik: number;
}) {
  const D = model.ayar.D;
  const C = model.ayar.baglam;

  switch (asama.tur) {
    case "tokenler":
      return <Tokenler iz={iz} C={C} />;

    case "gomme":
      return <Gommeler iz={iz} D={D} C={C} genislik={genislik} />;

    case "giris":
      return (
        <CarpimCanlandirma
          x={iz.birlesik}
          W={model.Wgiris}
          b={model.bgiris}
          y={iz.h0}
          altAdim={altAdim}
          maxGenislik={genislik}
          xEtiketi={(i) => {
            const t = Math.floor(i / D);
            return `${gorunurAd(iz.baglamIds[t])} (t−${C - t}) · boyut ${i % D}`;
          }}
          yEtiketi={(j) => `h0 boyut ${j}`}
        />
      );

    case "genisle":
      return (
        <CarpimCanlandirma
          x={iz.bloklar[asama.blok].girdi}
          W={model.bloklar[asama.blok].W1}
          b={model.bloklar[asama.blok].b1}
          y={iz.bloklar[asama.blok].oncesi}
          altAdim={altAdim}
          maxGenislik={genislik}
        />
      );

    case "relu":
      return <Relu iz={iz} blok={asama.blok} genislik={genislik} />;

    case "daralt":
      return (
        <CarpimCanlandirma
          x={iz.bloklar[asama.blok].relu}
          W={model.bloklar[asama.blok].W2}
          b={model.bloklar[asama.blok].b2}
          y={iz.bloklar[asama.blok].dal}
          altAdim={altAdim}
          maxGenislik={genislik}
        />
      );

    case "artik":
      return <Artik iz={iz} blok={asama.blok} genislik={genislik} />;

    case "cikis":
      return (
        <CarpimCanlandirma
          x={iz.bloklar.length > 0 ? iz.bloklar[iz.bloklar.length - 1].cikti : iz.h0}
          W={model.Wcikis}
          b={model.bcikis}
          y={iz.logits}
          altAdim={altAdim}
          maxGenislik={genislik}
          yEtiketi={(j) => `"${gorunurAd(j)}" karakterinin skoru`}
        />
      );

    case "softmax":
      return <Softmax iz={iz} sicaklik={sicaklik} genislik={genislik} />;
  }
}

function Tokenler({ iz, C }: { iz: IleriIz; C: number }) {
  return (
    <div className="flex flex-wrap gap-2">
      {iz.baglamIds.map((id, t) => (
        <div
          key={t}
          className={`border px-3 py-2 text-center ${
            t === C - 1 ? "border-cizgi-parlak bg-yuzey-2" : "border-cizgi"
          }`}
        >
          <div className="sayi text-[18px] text-metin">{gorunurAd(id)}</div>
          <div className="sayi text-[11px] text-soluk">id {id}</div>
          <div className="text-[10px] text-cok-soluk">t−{C - t}</div>
        </div>
      ))}
    </div>
  );
}

function Gommeler({ iz, D, C, genislik }: { iz: IleriIz; D: number; C: number; genislik: number }) {
  const serit = Math.min(genislik - 90, D * 46);
  return (
    <div className="space-y-3">
      {iz.gommeler.map((vektor, t) => (
        <div key={t} className="flex items-center gap-3">
          <div className="w-16 shrink-0">
            <div className="sayi text-[14px] text-metin">{gorunurAd(iz.baglamIds[t])}</div>
            <div className="text-[10px] text-cok-soluk">t−{C - t}</div>
          </div>
          <VektorSeridi veri={vektor} genislik={serit} yukseklik={26} />
        </div>
      ))}
      <div className="border-t border-cizgi pt-3">
        <p className="mb-2 text-[11px] text-cok-soluk">
          Uç uca eklenmiş hali — {C} × {D} = {C * D} sayı. Sıradaki aşama bunu tek bir {D}'lik
          vektöre indirecek.
        </p>
        <VektorSeridi veri={iz.birlesik} genislik={genislik - 20} yukseklik={22} />
      </div>
    </div>
  );
}

function Relu({ iz, blok, genislik }: { iz: IleriIz; blok: number; genislik: number }) {
  const b = iz.bloklar[blok];
  const sonuk = useMemo(() => {
    const kume = new Set<number>();
    for (let i = 0; i < b.oncesi.length; i++) if (b.oncesi[i] <= 0) kume.add(i);
    return kume;
  }, [b.oncesi]);

  const olcek = useMemo(() => {
    let enBuyuk = 0;
    for (let i = 0; i < b.oncesi.length; i++) enBuyuk = Math.max(enBuyuk, Math.abs(b.oncesi[i]));
    return Math.max(enBuyuk, 1e-6);
  }, [b.oncesi]);

  const oran = sonuk.size / b.oncesi.length;

  return (
    <div className="space-y-4">
      <div>
        <p className="mb-1 text-[11px] text-cok-soluk">önce — negatifler turkuaz</p>
        <VektorSeridi veri={b.oncesi} genislik={genislik - 20} yukseklik={30} olcek={olcek} />
      </div>
      <div>
        <p className="mb-1 text-[11px] text-cok-soluk">
          sonra — sıfırlananlar çapraz çizgili
        </p>
        <VektorSeridi
          veri={b.relu}
          genislik={genislik - 20}
          yukseklik={30}
          sonuk={sonuk}
          olcek={olcek}
        />
      </div>
      <p className="sayi text-[13px] text-metin">
        {sonuk.size} / {b.oncesi.length} nöron sıfırlandı{" "}
        <span className="text-cok-soluk">({yuzde(oran, 0)})</span>
      </p>
      <p className="max-w-[70ch] text-[11px] leading-relaxed text-cok-soluk">
        Sönen nöron o tahmine hiç katkı vermez. Eğitim sırasında da gradyanı geçirmez, yani o
        adımda öğrenmez. Bir modelin nöronlarının büyük kısmının her girdide sönük olması
        olağandır: farklı girdilerde farklı alt kümeler uyanır.
      </p>
    </div>
  );
}

function Artik({ iz, blok, genislik }: { iz: IleriIz; blok: number; genislik: number }) {
  const b = iz.bloklar[blok];
  const olcek = useMemo(() => {
    let enBuyuk = 0;
    for (const dizi of [b.girdi, b.dal, b.cikti]) {
      for (let i = 0; i < dizi.length; i++) enBuyuk = Math.max(enBuyuk, Math.abs(dizi[i]));
    }
    return Math.max(enBuyuk, 1e-6);
  }, [b]);

  const serit = Math.min(genislik - 40, b.girdi.length * 48);
  const satirlar: Array<[string, Float32Array]> = [
    ["bloğa giren", b.girdi],
    ["bloğun ürettiği", b.dal],
    ["toplam", b.cikti],
  ];

  return (
    <div className="space-y-3">
      {satirlar.map(([ad, veri], i) => (
        <div key={ad}>
          <p className="mb-1 flex items-baseline gap-2 text-[11px] text-cok-soluk">
            <span className="sayi w-4 text-metin">{i === 1 ? "+" : i === 2 ? "=" : " "}</span>
            {ad}
          </p>
          <div className="pl-6">
            <VektorSeridi veri={veri} genislik={serit} yukseklik={30} olcek={olcek} />
          </div>
        </div>
      ))}
    </div>
  );
}

function Softmax({ iz, sicaklik, genislik }: { iz: IleriIz; sicaklik: number; genislik: number }) {
  const sirali = useMemo(
    () =>
      Array.from(iz.olasilik)
        .map((olasilik, id) => ({ olasilik, id, logit: iz.logits[id] }))
        .sort((a, b) => b.olasilik - a.olasilik),
    [iz],
  );
  const ilkBes = sirali.slice(0, 5);
  const kalan = sirali.slice(5).reduce((a, s) => a + s.olasilik, 0);

  return (
    <div className="space-y-4">
      <ul className="space-y-1.5">
        {ilkBes.map(({ id, olasilik, logit }) => (
          <li key={id} className="flex items-center gap-3">
            <span className="sayi w-6 text-center text-[16px] text-metin">{gorunurAd(id)}</span>
            <span className="sayi w-16 text-right text-[12px] text-metin">{yuzde(olasilik)}</span>
            <span className="h-4 flex-1 bg-yuzey-2">
              <span
                className="block h-full bg-cizgi-parlak"
                style={{ width: `${Math.max(0.5, olasilik * 100)}%` }}
              />
            </span>
            <span className="sayi w-24 text-right text-[11px] text-cok-soluk">
              logit {sayi(logit, 2)}
            </span>
          </li>
        ))}
      </ul>
      <p className="sayi text-[11px] text-cok-soluk">
        kalan {ALFABE.length - 5} karakterin toplamı {yuzde(kalan)} · sıcaklık {sayi(sicaklik, 2)}
      </p>
      <div>
        <p className="mb-1 text-[11px] text-cok-soluk">bütün dağılım, sözlük sırasında</p>
        <VektorSeridi veri={iz.olasilik} genislik={genislik - 20} yukseklik={26} etiket={(i) => `"${gorunurAd(i)}"`} />
      </div>
    </div>
  );
}

/**
 * Aşama ne olursa olsun altta duran şerit: o anda üzerinde çalışılan ham
 * vektör. Çarpım sürerken henüz hesaplanmamış konumlar soluk çizilir —
 * yoksa animasyon daha üçüncü çıkışta iken altta bitmiş vektörü göstermek
 * gibi bir tutarsızlık olurdu.
 */
function HamVektor({
  asama,
  iz,
  genislik,
  hesaplanan,
}: {
  asama: Asama;
  iz: IleriIz;
  genislik: number;
  hesaplanan?: number;
}) {
  const { ad, veri } = asamaVektoru(asama, iz);
  const eksik = hesaplanan != null && hesaplanan < veri.length;
  return (
    <section className="border border-cizgi bg-yuzey px-4 py-3">
      <p className="mb-2 text-[11px] text-cok-soluk">
        o anki ham vektör · <span className="sayi text-soluk">{ad}</span>
        {eksik && (
          <span className="sayi"> · {hesaplanan}/{veri.length} hesaplandı</span>
        )}
      </p>
      <VektorSeridi veri={veri} genislik={genislik - 20} yukseklik={22} hesaplanan={hesaplanan} />
    </section>
  );
}
