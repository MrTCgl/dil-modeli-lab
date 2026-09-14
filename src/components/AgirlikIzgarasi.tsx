/**
 * AĞIRLIK IZGARASI
 * ================
 *
 * Bir matrisi ısı haritası olarak gösterir: satır × sütun, her hücre bir
 * kare, değer renge eşlenmiş. Hücrenin üstüne gelince tam değeri görünür,
 * tıklayınca elle değiştirilebilir.
 *
 * Elle değiştirme bu uygulamanın en önemli düğmesi olabilir: "model" denen
 * şeyin bir sayı tablosundan ibaret olduğunu, tek bir hücreyi bozup çıktının
 * anında değiştiğini görmek kadar iyi anlatan bir açıklama yok.
 *
 * NEDEN CANVAS, NEDEN SVG DEĞİL?
 * Wgiriş matrisi tek başına 128×16 = 2048 hücre. Bütün ızgaralar toplamda
 * dört binden fazla kare eder ve eğitim sırasında bunlar saniyede onlarca
 * kez güncellenecek. Bu kadar DOM düğümünü canlı tutmak tarayıcıyı dizginler.
 * Canvas tek düğümde çizer, akıcı kalır. Renk skalası yine D3'ün Lab
 * ara değerlemesiyle hesaplanır; sadece boyama işi canvas'a devredilmiştir.
 * Etiketler de canvas'a çizilir — böylece hizalama her hücre boyutunda tam.
 *
 * DEVRİK GÖSTERİM
 * Bazı matrisler çok uzun ve dar: Wgiriş 128 satır × 16 sütun. Olduğu gibi
 * çizilince ekranda okunmaz bir şerit oluyor ve yanındaki yer boş kalıyor.
 * Bu tür matrisleri varsayılan olarak devrik (satır ↔ sütun yer değiştirmiş)
 * gösteriyoruz — Wgiriş böylece 16 satır × 128 sütunluk geniş bir bant olur
 * ve bağlamdaki 8 pozisyonun blokları tek bakışta ayırt edilir.
 *
 * Devrik gösterim SADECE bir görüntüleme tercihidir: matrisin kendisine
 * dokunulmaz, hücre düzenlemeleri doğru yere yazılır ve başlık hangi kipte
 * olduğunu açıkça yazar. Gizli bir dönüşüm bırakmıyoruz.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Matris } from "../lib/model.ts";
import { HUCRE_NEGATIF, HUCRE_POZITIF, agirlikRengi, ozetCikar } from "../lib/renk.ts";
import { sayi } from "../lib/bicim.ts";

const SOL_MARJ = 40;
const UST_MARJ = 20;
const ETIKET_ESIGI = 9; // bu boyutun altında etiket çizilmez, okunmuyor
const PARLAMA_SURESI = 700; // ms
/**
 * Parlama döngüsü saniyede en fazla bu kadar kez yeniden çizer.
 *
 * Neden sınır var: eğitim sürerken matristeki her hücre her güncellemede
 * değişiyor, yani parlama hiç bitmiyor ve döngü 60 fps'te dönüyordu. Yedi
 * ızgaranın tuvali saniyede altmış kez yeniden boyanınca tarayıcının
 * birleştiricisi tıkanıyor ve sayfa — hesap worker'da olmasına rağmen —
 * takılıyordu. Ölçüm: kareler arası süre 16,7 ms'ten 70 ms'e çıkıyordu.
 * Gözün ayırt edemeyeceği bir tazeleme hızı için ödenecek bedel değil.
 * 700 ms süren bir sönme için saniyede on iki kare fazlasıyla yeter; bu
 * sınırı 20 fps'ten 12 fps'e çekmek eğitim sırasındaki ortalama kare
 * süresini 41 ms'ten 35 ms'e indirdi.
 */
const PARLAMA_ARALIGI = 80; // ms (~12 fps)
/**
 * Parlama eşiği: o güncellemedeki en büyük değişimin bu oranını aşan
 * hücreler parlar.
 *
 * Neden sabit bir sayı değil: eğitimde her hücre her adımda bir miktar
 * oynuyor, ama ne kadar oynadığı öğrenme oranına, adım sayısına ve eğitimin
 * hangi aşamasında olduğuna göre kat kat değişiyor. Sabit eşik ya hiçbir
 * şeyi ya her şeyi parlatıyordu. Orana bağlayınca soru "çok mu değişti"den
 * "bu turda en çok değişenler hangileri"ye dönüyor — zaten sorulması
 * gereken de bu.
 */
const PARLAMA_ORANI = 0.35;
const YUKSEKLIK_BUTCESI = 420; // "sığdır" kipinde hedeflenen çizim yüksekliği

export type BoyutKipi = "sigdir" | "orta" | "buyuk";

export interface AgirlikIzgarasiProps {
  matris: Matris;
  /** Dışarıdaki her değişimde artan sayaç; yeniden çizimi tetikler. */
  surum: number;
  /** Satır etiketi (null = etiket yok). Gömme tablosunda harfler buraya gelir. */
  satirEtiketi?: (i: number) => string | null;
  /** Sütun etiketi. Çıkış katmanında harfler buraya gelir. */
  sutunEtiketi?: (j: number) => string | null;
  /** Elle bozulmuş hücrelerin indeksleri — ızgarada işaretlenir. */
  bozulan?: ReadonlySet<number>;
  /** Hücre değiştiğinde çağrılır. Verilmezse ızgara salt okunur olur. */
  onDegistir?: (indeks: number, deger: number) => void;
  /** Değişen hücreleri kısa süre parlat (eğitim sırasında). */
  parlamaAktif?: boolean;
}

export function AgirlikIzgarasi({
  matris,
  surum,
  satirEtiketi,
  sutunEtiketi,
  bozulan,
  onDegistir,
  parlamaAktif = false,
}: AgirlikIzgarasiProps) {
  const kapsayiciRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const oncekiRef = useRef<Float32Array | null>(null);
  const parlamaRef = useRef<Float64Array | null>(null);
  const cerceveRef = useRef<number | null>(null);
  const zamanRef = useRef<number | null>(null);

  // Ekranda görünmeyen ızgara çizilmez. Eğitim sayfasında yedi ızgara var ve
  // çoğu katlamanın altında kalıyor; görünmeyen tuvalleri saniyede yirmi kez
  // yeniden boyamak, kimsenin bakmadığı bir yere işlemci harcamak demek.
  const [gorunur, setGorunur] = useState(true);
  const [kip, setKip] = useState<BoyutKipi>("sigdir");
  // Çok uzun ve dar matrisler varsayılan olarak devrik gösterilir.
  const [devrik, setDevrik] = useState(matris.satir > matris.sutun * 3);
  const [genislik, setGenislik] = useState(640);
  const [uzerinde, setUzerinde] = useState<number | null>(null);
  const [secili, setSecili] = useState<number | null>(null);
  const [fare, setFare] = useState<{ x: number; y: number } | null>(null);

  // Not: burada özeti seyrek hesaplamayı denedim (matrisin tamamını sıralamak
  // gerekiyor), ama profil çıkarınca renk hesabının toplam CPU'nun binde
  // beşi olduğu görüldü — darboğaz oradan değil, tuvallerin yeniden
  // boyanmasındandı. Erken optimizasyonu geri aldım: başlıktaki en
  // küçük/ortalama/en büyük değerleri her güncellemede doğru olsun.
  const ozet = useMemo(() => ozetCikar(matris.veri), [matris, surum]);

  // Ekranda görünen satır/sütun sayısı (devrik kipte yer değiştirir).
  const gSatir = devrik ? matris.sutun : matris.satir;
  const gSutun = devrik ? matris.satir : matris.sutun;

  /** Ekrandaki (i, j) konumundan matrisin düz dizisindeki indekse. */
  const veriIndeksi = useCallback(
    (i: number, j: number) => (devrik ? j * matris.sutun + i : i * matris.sutun + j),
    [devrik, matris.sutun],
  );

  /** Ters yön: düz indeksten ekrandaki konuma. */
  const gorunenKonum = useCallback(
    (indeks: number): [number, number] => {
      const di = Math.floor(indeks / matris.sutun);
      const dj = indeks - di * matris.sutun;
      return devrik ? [dj, di] : [di, dj];
    },
    [devrik, matris.sutun],
  );

  // Hücre boyutu: "sığdır" kipinde matris hem genişliğe hem yükseklik
  // bütçesine sığacak şekilde küçülür; kaydırmaya gerek kalmaz.
  const boyut = useMemo(() => {
    if (kip === "orta") return 11;
    if (kip === "buyuk") return 19;
    const kullanilabilirG = Math.max(120, genislik - SOL_MARJ - 8);
    const kullanilabilirY = YUKSEKLIK_BUTCESI - UST_MARJ;
    const sigan = Math.min(
      Math.floor(kullanilabilirG / gSutun),
      Math.floor(kullanilabilirY / gSatir),
    );
    // Alt sınır 7: dar ekranda hücreler okunmaz hale gelmesin. Sığmadığında
    // ızgara küçülmek yerine kaydırılabilir olur — telefonda istediğimiz de bu.
    return Math.max(7, Math.min(28, sigan));
  }, [kip, genislik, gSatir, gSutun]);

  const bosluk = boyut >= 8 ? 1 : 0;
  const cizimGenislik = SOL_MARJ + gSutun * boyut;
  const cizimYukseklik = UST_MARJ + gSatir * boyut;

  useEffect(() => {
    const el = kapsayiciRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const gozlemci = new IntersectionObserver(
      (girdiler) => setGorunur(girdiler.some((g) => g.isIntersecting)),
      { rootMargin: "150px" },
    );
    gozlemci.observe(el);
    return () => gozlemci.disconnect();
  }, []);

  // Kapsayıcı genişliğini izle: "sığdır" kipi buna göre hesaplanır.
  useEffect(() => {
    const el = kapsayiciRef.current;
    if (!el) return;
    const gozlemci = new ResizeObserver((girdiler) => {
      for (const g of girdiler) setGenislik(g.contentRect.width);
    });
    gozlemci.observe(el);
    setGenislik(el.clientWidth);
    return () => gozlemci.disconnect();
  }, []);

  // --- Çizim ---------------------------------------------------------------
  const ciz = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(cizimGenislik * dpr)) {
      cv.width = Math.round(cizimGenislik * dpr);
      cv.height = Math.round(cizimYukseklik * dpr);
      cv.style.width = `${cizimGenislik}px`;
      cv.style.height = `${cizimYukseklik}px`;
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cizimGenislik, cizimYukseklik);

    const veri = matris.veri;
    const olcek = ozet.olcek;
    const simdi = performance.now();
    let parlayanVar = false;

    // hücreler
    for (let i = 0; i < gSatir; i++) {
      const y = UST_MARJ + i * boyut;
      for (let j = 0; j < gSutun; j++) {
        ctx.fillStyle = agirlikRengi(veri[veriIndeksi(i, j)], olcek);
        ctx.fillRect(SOL_MARJ + j * boyut, y, boyut - bosluk, boyut - bosluk);
      }
    }

    // eğitimde değişen hücreler: kısa süreli parlama
    const parlama = parlamaRef.current;
    if (parlamaAktif && parlama) {
      for (let indeks = 0; indeks < parlama.length; indeks++) {
        const t = parlama[indeks];
        if (t === 0) continue;
        const gecen = simdi - t;
        if (gecen > PARLAMA_SURESI) {
          parlama[indeks] = 0;
          continue;
        }
        parlayanVar = true;
        const [i, j] = gorunenKonum(indeks);
        ctx.fillStyle = `rgba(255,255,255,${0.42 * (1 - gecen / PARLAMA_SURESI)})`;
        ctx.fillRect(SOL_MARJ + j * boyut, UST_MARJ + i * boyut, boyut - bosluk, boyut - bosluk);
      }
    }

    // elle bozulan hücreler: kalıcı çerçeve
    if (bozulan && bozulan.size > 0) {
      ctx.strokeStyle = "#e8ecf2";
      ctx.lineWidth = 1;
      for (const indeks of bozulan) {
        const [i, j] = gorunenKonum(indeks);
        ctx.strokeRect(SOL_MARJ + j * boyut + 0.5, UST_MARJ + i * boyut + 0.5, boyut - bosluk - 1, boyut - bosluk - 1);
      }
    }

    // seçili hücre
    if (secili != null) {
      const [i, j] = gorunenKonum(secili);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(SOL_MARJ + j * boyut - 1, UST_MARJ + i * boyut - 1, boyut - bosluk + 2, boyut - bosluk + 2);
    }

    // etiketler
    if (boyut >= ETIKET_ESIGI) {
      ctx.fillStyle = "#5e6878";
      ctx.font = `${Math.min(11, boyut - 2)}px ui-monospace, monospace`;
      ctx.textBaseline = "middle";
      const yanEtiket = devrik ? sutunEtiketi : satirEtiketi;
      const ustEtiket = devrik ? satirEtiketi : sutunEtiketi;
      if (yanEtiket) {
        ctx.textAlign = "right";
        for (let i = 0; i < gSatir; i++) {
          const etiket = yanEtiket(i);
          if (etiket) ctx.fillText(etiket, SOL_MARJ - 6, UST_MARJ + i * boyut + boyut / 2);
        }
      }
      if (ustEtiket) {
        ctx.textAlign = "center";
        for (let j = 0; j < gSutun; j++) {
          const etiket = ustEtiket(j);
          if (etiket) ctx.fillText(etiket, SOL_MARJ + j * boyut + boyut / 2, UST_MARJ - 9);
        }
      }
    }

    if (parlayanVar) {
      // Sönme sürerken yeniden çiz — ama saniyede yirmi kereden fazla değil.
      zamanRef.current = window.setTimeout(() => {
        zamanRef.current = null;
        cerceveRef.current = requestAnimationFrame(ciz);
      }, PARLAMA_ARALIGI);
    } else {
      cerceveRef.current = null;
    }
  }, [matris, ozet, boyut, bosluk, cizimGenislik, cizimYukseklik, secili, bozulan, satirEtiketi, sutunEtiketi, parlamaAktif, gSatir, gSutun, devrik, veriIndeksi, gorunenKonum]);

  // Değişen hücreleri tespit et, sonra çiz.
  useEffect(() => {
    if (!gorunur) {
      // Görünür alana geri dönünce her hücre "değişmiş" sayılıp topluca
      // parlamasın diye karşılaştırma anlık görüntüsünü de bırakıyoruz.
      oncekiRef.current = null;
      return;
    }
    if (parlamaAktif) {
      const onceki = oncekiRef.current;
      if (!parlamaRef.current || parlamaRef.current.length !== matris.veri.length) {
        parlamaRef.current = new Float64Array(matris.veri.length);
      }
      if (onceki && onceki.length === matris.veri.length) {
        const simdi = performance.now();
        const parlama = parlamaRef.current;
        let enBuyukFark = 0;
        for (let i = 0; i < matris.veri.length; i++) {
          const fark = Math.abs(onceki[i] - matris.veri[i]);
          if (fark > enBuyukFark) enBuyukFark = fark;
        }
        if (enBuyukFark > 0) {
          const esik = enBuyukFark * PARLAMA_ORANI;
          for (let i = 0; i < matris.veri.length; i++) {
            if (Math.abs(onceki[i] - matris.veri[i]) >= esik) parlama[i] = simdi;
          }
        }
      }
      oncekiRef.current = new Float32Array(matris.veri);
    }
    if (cerceveRef.current != null) cancelAnimationFrame(cerceveRef.current);
    if (zamanRef.current != null) clearTimeout(zamanRef.current);
    cerceveRef.current = requestAnimationFrame(ciz);
    return () => {
      if (cerceveRef.current != null) cancelAnimationFrame(cerceveRef.current);
      if (zamanRef.current != null) clearTimeout(zamanRef.current);
      cerceveRef.current = null;
      zamanRef.current = null;
    };
  }, [ciz, surum, parlamaAktif, matris, gorunur]);

  // --- Etkileşim -----------------------------------------------------------
  const indeksBul = useCallback(
    (olay: { clientX: number; clientY: number }): number | null => {
      const cv = canvasRef.current;
      if (!cv) return null;
      const kutu = cv.getBoundingClientRect();
      const x = olay.clientX - kutu.left - SOL_MARJ;
      const y = olay.clientY - kutu.top - UST_MARJ;
      if (x < 0 || y < 0) return null;
      const j = Math.floor(x / boyut);
      const i = Math.floor(y / boyut);
      if (i < 0 || j < 0 || i >= gSatir || j >= gSutun) return null;
      return veriIndeksi(i, j);
    },
    [boyut, gSatir, gSutun, veriIndeksi],
  );

  const klavye = (olay: React.KeyboardEvent) => {
    if (secili == null) return;
    const [i, j] = gorunenKonum(secili);
    let yeniI = i;
    let yeniJ = j;
    if (olay.key === "ArrowUp") yeniI--;
    else if (olay.key === "ArrowDown") yeniI++;
    else if (olay.key === "ArrowLeft") yeniJ--;
    else if (olay.key === "ArrowRight") yeniJ++;
    else if (olay.key === "Escape") {
      setSecili(null);
      return;
    } else return;
    olay.preventDefault();
    yeniI = Math.max(0, Math.min(gSatir - 1, yeniI));
    yeniJ = Math.max(0, Math.min(gSutun - 1, yeniJ));
    setSecili(veriIndeksi(yeniI, yeniJ));
  };

  const gosterilen = uzerinde ?? secili;

  return (
    <section className="border border-cizgi bg-yuzey">
      <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-cizgi px-3 py-2">
        <h3 className="text-[13px] font-medium text-metin">{matris.ad}</h3>
        <span className="sayi text-[11px] text-cok-soluk">
          {matris.satir} × {matris.sutun} = {matris.veri.length} sayı
        </span>
        <span className="sayi ml-auto text-[11px] text-soluk">
          en küçük {sayi(ozet.enKucuk)} · ortalama {sayi(ozet.ortalama)} · en büyük {sayi(ozet.enBuyuk)}
        </span>
        <button
          onClick={() => setDevrik((d) => !d)}
          className={`border px-2 py-0.5 text-[11px] transition-colors ${
            devrik ? "border-cizgi-parlak text-metin" : "border-cizgi text-cok-soluk hover:text-soluk"
          }`}
          title="Satır ve sütunları yer değiştirerek göster. Matris değişmez, sadece görüntü döner."
        >
          devrik
        </button>
        <BoyutSecici kip={kip} setKip={setKip} />
      </header>
      {devrik && (
        <p className="border-b border-cizgi bg-yuzey-2 px-3 py-1 text-[10px] text-cok-soluk">
          Devrik gösterim: ekranda satırlar ile sütunlar yer değiştirdi ({matris.satir}×{matris.sutun} matris,{" "}
          {gSatir}×{gSutun} olarak çiziliyor). Matrisin kendisi değişmedi; ipucundaki [satır, sütun] gerçek konumdur.
        </p>
      )}

      <div ref={kapsayiciRef} className="relative overflow-auto px-3 py-3" style={{ maxHeight: 450 }}>
        <div className="mx-auto w-fit">
        <canvas
          ref={canvasRef}
          tabIndex={0}
          onKeyDown={klavye}
          onMouseMove={(e) => {
            setUzerinde(indeksBul(e));
            const kap = kapsayiciRef.current;
            if (kap) {
              const kutu = kap.getBoundingClientRect();
              setFare({
                x: e.clientX - kutu.left + kap.scrollLeft,
                y: e.clientY - kutu.top + kap.scrollTop,
              });
            }
          }}
          onMouseLeave={() => {
            setUzerinde(null);
            setFare(null);
          }}
          onClick={(e) => {
            const indeks = indeksBul(e);
            if (indeks != null) setSecili(indeks);
          }}
          className="cursor-crosshair touch-pan-x touch-pan-y"
          aria-label={`${matris.ad} ağırlık ızgarası`}
        />
        </div>
        {uzerinde != null && fare && (
          <Ipucu
            x={fare.x}
            y={fare.y}
            matris={matris}
            indeks={uzerinde}
            satirEtiketi={satirEtiketi}
            sutunEtiketi={sutunEtiketi}
          />
        )}
      </div>

      <Skala olcek={ozet.olcek} />

      {secili != null && onDegistir && (
        <Duzenleyici
          matris={matris}
          indeks={secili}
          olcek={ozet.olcek}
          onDegistir={onDegistir}
          onKapat={() => setSecili(null)}
        />
      )}
      {gosterilen == null && (
        <p className="border-t border-cizgi px-3 py-2 text-[11px] text-cok-soluk">
          Hücrenin üstüne gelin: tam değeri görünür. Tıklayın: elle değiştirilebilir.
        </p>
      )}
    </section>
  );
}

function BoyutSecici({ kip, setKip }: { kip: BoyutKipi; setKip: (k: BoyutKipi) => void }) {
  const secenekler: Array<[BoyutKipi, string]> = [
    ["sigdir", "sığdır"],
    ["orta", "orta"],
    ["buyuk", "büyük"],
  ];
  return (
    <div className="flex border border-cizgi">
      {secenekler.map(([deger, etiket]) => (
        <button
          key={deger}
          onClick={() => setKip(deger)}
          className={`px-2 py-0.5 text-[11px] transition-colors ${
            kip === deger ? "bg-cizgi text-metin" : "text-cok-soluk hover:text-soluk"
          }`}
        >
          {etiket}
        </button>
      ))}
    </div>
  );
}

function Ipucu({
  x,
  y,
  matris,
  indeks,
  satirEtiketi,
  sutunEtiketi,
}: {
  x: number;
  y: number;
  matris: Matris;
  indeks: number;
  satirEtiketi?: (i: number) => string | null;
  sutunEtiketi?: (j: number) => string | null;
}) {
  const i = Math.floor(indeks / matris.sutun);
  const j = indeks - i * matris.sutun;
  const sEt = satirEtiketi?.(i);
  const suEt = sutunEtiketi?.(j);
  return (
    <div
      className="pointer-events-none absolute z-10 border border-cizgi-parlak bg-zemin px-2 py-1 shadow-lg"
      style={{ left: x + 14, top: y + 14 }}
    >
      <div className="sayi text-[13px] text-metin">{sayi(matris.veri[indeks], 4)}</div>
      <div className="sayi text-[10px] text-cok-soluk">
        satır {i}
        {sEt ? ` (${sEt})` : ""} · sütun {j}
        {suEt ? ` (${suEt})` : ""}
      </div>
    </div>
  );
}

/** Renk skalasının göstergesi: hangi renk hangi sayı. */
function Skala({ olcek }: { olcek: number }) {
  return (
    <div className="flex items-center gap-2 border-t border-cizgi px-3 py-1.5">
      <span className="sayi text-[10px] text-cok-soluk">{sayi(-olcek, 2)}</span>
      <div
        className="h-2 flex-1"
        style={{
          background: `linear-gradient(to right, ${HUCRE_NEGATIF}, #1b212b 50%, ${HUCRE_POZITIF})`,
        }}
        aria-hidden
      />
      <span className="sayi text-[10px] text-cok-soluk">+{sayi(olcek, 2)}</span>
      <span className="ml-1 text-[10px] text-cok-soluk" title="Renk, değerin karekökü ile ölçeklenir; küçük değerler böylece kaybolmaz.">
        negatif · sıfır · pozitif
      </span>
    </div>
  );
}

/** Seçili hücreyi elle değiştirme paneli. */
function Duzenleyici({
  matris,
  indeks,
  olcek,
  onDegistir,
  onKapat,
}: {
  matris: Matris;
  indeks: number;
  olcek: number;
  onDegistir: (indeks: number, deger: number) => void;
  onKapat: () => void;
}) {
  const i = Math.floor(indeks / matris.sutun);
  const j = indeks - i * matris.sutun;
  const deger = matris.veri[indeks];
  const sinir = Math.max(olcek * 3, 0.5);

  // Sayı kutusu kendi metnini tutar: aksi halde yazarken "0." gibi ara
  // durumlar anında sayıya çevrilip imleç yerinden oynar.
  const [kutuMetni, setKutuMetni] = useState(deger.toFixed(4));
  useEffect(() => {
    if (Number(kutuMetni) !== deger) setKutuMetni(deger.toFixed(4));
    // kutuMetni bilerek bağımlılık değil: kullanıcı yazarken üzerine yazmayalım.
  }, [deger, indeks]);

  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-cizgi bg-yuzey-2 px-3 py-2">
      <span className="sayi text-[11px] text-soluk">
        [{i}, {j}]
      </span>
      <input
        type="range"
        min={-sinir}
        max={sinir}
        step={sinir / 500}
        value={deger}
        onChange={(e) => onDegistir(indeks, Number(e.target.value))}
        className="h-1 flex-1 min-w-[140px] accent-white"
        aria-label="ağırlık değeri"
      />
      <input
        type="number"
        step={0.01}
        value={kutuMetni}
        onChange={(e) => {
          setKutuMetni(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== "" && Number.isFinite(v)) onDegistir(indeks, v);
        }}
        className="sayi w-24 border border-cizgi bg-zemin px-2 py-1 text-[13px] text-metin"
      />
      <button
        onClick={() => onDegistir(indeks, 0)}
        className="border border-cizgi px-2 py-1 text-[11px] text-soluk hover:border-cizgi-parlak hover:text-metin"
      >
        sıfırla
      </button>
      <button onClick={onKapat} className="px-2 py-1 text-[11px] text-cok-soluk hover:text-metin">
        kapat
      </button>
    </div>
  );
}
