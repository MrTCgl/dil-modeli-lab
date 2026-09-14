/**
 * GÖMME KÜRESİ
 * ============
 *
 * Her karakterin gömme vektörü birim uzunluğa getirilir ve küre üzerinde bir
 * nokta olur. Yön anlam taşır, uzunluk taşımaz: iki karakterin benzerliği
 * aralarındaki açıdır. Küre bu yüzden doğru ev — düzlemde açı diye bir şey
 * yok, kürede var.
 *
 * NE GÖRÜYORSUNUZ, NE GÖRMÜYORSUNUZ
 * Gömmeler 16 boyutlu, ekran 3 boyutlu. Aradaki farkı PCA kapatıyor ve bu
 * kayıplı. O yüzden:
 *   - başlıkta ilk üç bileşenin taşıdığı değişkenlik oranı yazıyor,
 *   - her noktanın izdüşümde ne kadarının kaldığı boyutuna yansıyor,
 *   - seçilen çiftin GERÇEK 16 boyutlu açısı ile küredeki görünen açısı
 *     yan yana yazılıyor.
 * İkisi ayrıldığında resim yanılıyor demektir, sayı değil.
 *
 * RENKLER
 * Sesli / sessiz / noktalama ayrımı bir insan etiketlemesi. Model bunu hiç
 * görmedi — sadece metni okudu. Renkler kümelenirse, o kümeyi model buldu.
 * Ağırlık ızgarasının turkuaz-kehribar skalasından bilerek uzak duruldu:
 * orada renk sayının işaretini anlatıyor, burada insanın etiketini.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type Model } from "../lib/model.ts";
import { ALFABE, gorunurAd } from "../lib/tokenizer.ts";
import { type Izdusum, aciDerece, izdusumCikar, kosinus } from "../lib/pca.ts";
import { sayi, yuzde } from "../lib/bicim.ts";

const SESLILER = "aeıioöuü";
const RENK_SESLI = 0xb9a0f0;
const RENK_SESSIZ = 0x93a2b8;
const RENK_NOKTALAMA = 0xd98a8a;

function karakterRengi(id: number): number {
  const ch = ALFABE[id];
  if (SESLILER.includes(ch)) return RENK_SESLI;
  if (/\p{L}/u.test(ch)) return RENK_SESSIZ;
  return RENK_NOKTALAMA;
}

/** Harfi küçük bir tuvale çizip doku olarak kullanıyoruz: etiket her zaman kameraya bakar. */
function etiketDokusu(metin: string, renk: number): THREE.CanvasTexture {
  const boyut = 128;
  const cv = document.createElement("canvas");
  cv.width = boyut;
  cv.height = boyut;
  const ctx = cv.getContext("2d")!;
  ctx.clearRect(0, 0, boyut, boyut);
  ctx.fillStyle = `#${renk.toString(16).padStart(6, "0")}`;
  ctx.font = "600 78px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(metin, boyut / 2, boyut / 2 + 4);
  const doku = new THREE.CanvasTexture(cv);
  doku.colorSpace = THREE.SRGBColorSpace;
  return doku;
}

export interface GommeKuresiProps {
  model: Model;
  /** Ağırlıklar yerinde değiştiğinde artan sayaç (eğitim sırasında). */
  surum: number;
  yukseklik?: number;
}

export function GommeKuresi({ model, surum, yukseklik = 460 }: GommeKuresiProps) {
  const kapRef = useRef<HTMLDivElement>(null);
  const [secili, setSecili] = useState<number[]>([]);
  const [izdusum, setIzdusum] = useState<Izdusum | null>(null);

  // Three.js nesneleri React durumunda tutulmaz: her karede değişirler,
  // yeniden render tetiklemeleri anlamsız olur.
  const sahneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    sahne: THREE.Scene;
    kamera: THREE.PerspectiveCamera;
    kontrol: OrbitControls;
    noktalar: THREE.Mesh[];
    etiketler: THREE.Sprite[];
    yay: THREE.Line | null;
    /** Yayın örnek noktaları: her karede derinliğe göre renklendirmek için. */
    yayNoktalari: THREE.Vector3[];
    kirli: boolean;
  } | null>(null);

  const oncekiBilesenler = useRef<Float64Array[] | undefined>(undefined);
  const gorunurRef = useRef<() => boolean>(() => true);
  const seciliRef = useRef<number[]>([]);
  seciliRef.current = secili;

  // --- izdüşüm: ağırlıklar her değiştiğinde yeniden hesaplanır -------------
  useEffect(() => {
    const yeni = izdusumCikar(model.E, oncekiBilesenler.current);
    oncekiBilesenler.current = yeni.bilesenler;
    setIzdusum(yeni);
  }, [model, surum]);

  // Model değişince seçim anlamını yitirir.
  useEffect(() => {
    setSecili([]);
    oncekiBilesenler.current = undefined;
  }, [model]);

  // --- sahneyi bir kez kur -------------------------------------------------
  useEffect(() => {
    const kap = kapRef.current;
    if (!kap) return;

    const azalt = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const genislik = kap.clientWidth;
    const yuk = yukseklik;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(genislik, yuk);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.touchAction = "none";
    kap.appendChild(renderer.domElement);

    const sahne = new THREE.Scene();
    const kamera = new THREE.PerspectiveCamera(42, genislik / yuk, 0.1, 100);
    kamera.position.set(1.1, 0.8, 2.9);

    const kontrol = new OrbitControls(kamera, renderer.domElement);
    kontrol.enableDamping = !azalt;
    kontrol.dampingFactor = 0.075;
    kontrol.enablePan = false;
    kontrol.minDistance = 1.5;
    kontrol.maxDistance = 7;
    kontrol.rotateSpeed = 0.75;

    // referans küresi: ince bir ızgara, öne çıkmadan derinlik hissi versin
    const telGeo = new THREE.WireframeGeometry(new THREE.SphereGeometry(1, 24, 12));
    const tel = new THREE.LineSegments(
      telGeo,
      new THREE.LineBasicMaterial({ color: 0x232b37, transparent: true, opacity: 0.85 }),
    );
    sahne.add(tel);

    const noktaGeo = new THREE.SphereGeometry(0.026, 14, 10);
    const noktalar: THREE.Mesh[] = [];
    const etiketler: THREE.Sprite[] = [];

    for (let id = 0; id < ALFABE.length; id++) {
      const renk = karakterRengi(id);
      const nokta = new THREE.Mesh(
        noktaGeo,
        new THREE.MeshBasicMaterial({ color: renk, transparent: true, opacity: 1 }),
      );
      nokta.userData.id = id;
      sahne.add(nokta);
      noktalar.push(nokta);

      const etiket = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: etiketDokusu(gorunurAd(id), renk),
          transparent: true,
          depthWrite: false,
        }),
      );
      etiket.scale.setScalar(0.17);
      sahne.add(etiket);
      etiketler.push(etiket);
    }

    sahneRef.current = { renderer, sahne, kamera, kontrol, noktalar, etiketler, yay: null, yayNoktalari: [], kirli: true };

    // --- çizim döngüsü: sadece bir şey değiştiğinde çizer ------------------
    let raf = 0;
    const kameraYonu = new THREE.Vector3();
    const dongu = () => {
      raf = requestAnimationFrame(dongu);
      const durum = sahneRef.current;
      if (!durum) return;
      const hareket = durum.kontrol.update();
      if (!hareket && !durum.kirli) return;
      if (!gorunurRef.current()) return;

      // Arkadaki noktalar soluklaşsın: kürenin ön-arka ayrımı böyle okunuyor.
      durum.kamera.getWorldPosition(kameraYonu).normalize();
      for (let i = 0; i < durum.noktalar.length; i++) {
        const nokta = durum.noktalar[i];
        const yon = nokta.position.clone().normalize();
        const on = (yon.dot(kameraYonu) + 1) / 2;
        const seciliMi = seciliRef.current.includes(i);
        const gorunurluk = seciliMi ? 1 : 0.14 + 0.86 * Math.pow(on, 1.6);
        (nokta.material as THREE.MeshBasicMaterial).opacity = gorunurluk;
        (durum.etiketler[i].material as THREE.SpriteMaterial).opacity = seciliMi
          ? 1
          : 0.1 + 0.9 * Math.pow(on, 2);
        const temel = (nokta.userData.temelOlcek as number) ?? 1;
        nokta.scale.setScalar(seciliMi ? temel * 1.9 : temel);
      }

      // Yayın kürenin arkasından geçen yarısı zemin rengine doğru sönüyor.
      // Tel kafes hiçbir şeyi örtmediği için, sönümleme olmadan yay kürenin
      // önünden geçiyormuş gibi görünüp bir ilmek izlenimi veriyordu.
      if (durum.yay && durum.yayNoktalari.length > 0) {
        const renkler = durum.yay.geometry.getAttribute("color") as THREE.BufferAttribute;
        for (let i = 0; i < durum.yayNoktalari.length; i++) {
          const on = (durum.yayNoktalari[i].dot(kameraYonu) + 1) / 2;
          const k = 0.06 + 0.94 * Math.pow(on, 1.9);
          renkler.setXYZ(
            i,
            0.047 + (0.902 - 0.047) * k,
            0.059 + (0.922 - 0.059) * k,
            0.082 + (0.949 - 0.082) * k,
          );
        }
        renkler.needsUpdate = true;
      }

      durum.renderer.render(durum.sahne, durum.kamera);
      durum.kirli = false;
    };
    raf = requestAnimationFrame(dongu);

    // Küre ekranda değilken çizilmez: eğitim panelinde sayfa uzun ve küre
    // rahatlıkla görüş alanının dışında kalabiliyor.
    let gorunur = true;
    const gorunurlukGozlemcisi =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(
            (girdiler) => {
              gorunur = girdiler.some((g) => g.isIntersecting);
              const d = sahneRef.current;
              if (gorunur && d) d.kirli = true;
            },
            { rootMargin: "120px" },
          );
    gorunurlukGozlemcisi?.observe(kap);
    gorunurRef.current = () => gorunur;

    const gozlemci = new ResizeObserver(() => {
      const durum = sahneRef.current;
      if (!durum || !kapRef.current) return;
      const g = kapRef.current.clientWidth;
      durum.renderer.setSize(g, yuk);
      durum.kamera.aspect = g / yuk;
      durum.kamera.updateProjectionMatrix();
      durum.kirli = true;
    });
    gozlemci.observe(kap);

    return () => {
      cancelAnimationFrame(raf);
      gorunurlukGozlemcisi?.disconnect();
      gozlemci.disconnect();
      kontrol.dispose();
      noktalar.forEach((n) => (n.material as THREE.Material).dispose());
      etiketler.forEach((e) => {
        const m = e.material as THREE.SpriteMaterial;
        m.map?.dispose();
        m.dispose();
      });
      noktaGeo.dispose();
      telGeo.dispose();
      (tel.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
      sahneRef.current = null;
    };
  }, [yukseklik]);

  // --- izdüşüm değişince noktaları taşı ------------------------------------
  useEffect(() => {
    const durum = sahneRef.current;
    if (!durum || !izdusum) return;
    for (let id = 0; id < izdusum.noktalar.length && id < durum.noktalar.length; id++) {
      const n = izdusum.noktalar[id];
      durum.noktalar[id].position.set(n.x, n.y, n.z);
      // İzdüşümde az şeyi hayatta kalan noktalar küçük çizilir: küredeki
      // yerlerine daha az güvenilmeli.
      const olcek = 0.55 + 0.75 * n.kalan;
      durum.noktalar[id].userData.temelOlcek = olcek;
      durum.noktalar[id].scale.setScalar(olcek);
      durum.etiketler[id].position.set(n.x * 1.13, n.y * 1.13, n.z * 1.13);
      durum.etiketler[id].scale.setScalar(0.13 + 0.07 * n.kalan);
    }
    durum.kirli = true;
  }, [izdusum]);

  // --- seçili çift arasındaki büyük çember yayı ----------------------------
  useEffect(() => {
    const durum = sahneRef.current;
    if (!durum || !izdusum) return;
    if (durum.yay) {
      durum.sahne.remove(durum.yay);
      durum.yay.geometry.dispose();
      (durum.yay.material as THREE.Material).dispose();
      durum.yay = null;
      durum.yayNoktalari = [];
    }
    if (secili.length === 2) {
      const a = izdusum.noktalar[secili[0]];
      const b = izdusum.noktalar[secili[1]];
      const va = new THREE.Vector3(a.x, a.y, a.z);
      const vb = new THREE.Vector3(b.x, b.y, b.z);
      const noktalar: THREE.Vector3[] = [];
      for (let i = 0; i <= 64; i++) {
        // küre üzerinde iki nokta arasındaki en kısa yol
        const v = va.clone().lerp(vb, i / 64).normalize().multiplyScalar(1.004);
        noktalar.push(v);
      }
      const geo = new THREE.BufferGeometry().setFromPoints(noktalar);
      geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(noktalar.length * 3), 3));
      const yay = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }),
      );
      durum.sahne.add(yay);
      durum.yay = yay;
      durum.yayNoktalari = noktalar.map((v) => v.clone().normalize());
    }
    durum.kirli = true;
  }, [secili, izdusum]);

  // --- tıklama ile seçim ---------------------------------------------------
  // Işın izleme yerine ekran uzayında en yakın noktayı arıyoruz: noktalar
  // küçük, ışınla isabet ettirmek dokunmatik ekranda neredeyse imkânsız.
  const tiklama = (olay: React.MouseEvent) => {
    const durum = sahneRef.current;
    if (!durum) return;
    const kutu = durum.renderer.domElement.getBoundingClientRect();
    const fx = olay.clientX - kutu.left;
    const fy = olay.clientY - kutu.top;

    let enYakin = -1;
    let enYakinUzaklik = Infinity;
    const gecici = new THREE.Vector3();
    for (let id = 0; id < durum.noktalar.length; id++) {
      gecici.copy(durum.noktalar[id].position).project(durum.kamera);
      const ex = ((gecici.x + 1) / 2) * kutu.width;
      const ey = ((1 - gecici.y) / 2) * kutu.height;
      const uzaklik = Math.hypot(ex - fx, ey - fy);
      if (uzaklik < enYakinUzaklik) {
        enYakinUzaklik = uzaklik;
        enYakin = id;
      }
    }
    if (enYakin < 0 || enYakinUzaklik > 26) {
      setSecili([]);
      return;
    }
    setSecili((onceki) => {
      if (onceki.includes(enYakin)) return onceki.filter((x) => x !== enYakin);
      if (onceki.length >= 2) return [onceki[1], enYakin];
      return [...onceki, enYakin];
    });
  };

  return (
    <div className="space-y-3">
      <div
        ref={kapRef}
        onClick={tiklama}
        className="relative cursor-pointer border border-cizgi bg-[#0c0f15]"
        style={{ height: yukseklik }}
      />
      <Aciklama izdusum={izdusum} />
      <SecimPaneli izdusum={izdusum} secili={secili} onTemizle={() => setSecili([])} />
    </div>
  );
}

function Aciklama({ izdusum }: { izdusum: Izdusum | null }) {
  const ortKalan = useMemo(() => {
    if (!izdusum) return 0;
    return izdusum.noktalar.reduce((a, n) => a + n.kalan, 0) / izdusum.noktalar.length;
  }, [izdusum]);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border border-cizgi bg-yuzey px-3 py-2 text-[11px]">
      <span className="text-soluk">
        16 boyutlu uzayın 3 boyuta izdüşümü ·{" "}
        <span className="sayi text-metin">{izdusum ? yuzde(izdusum.aciklananOran, 1) : "—"}</span>{" "}
        <span className="text-cok-soluk">değişkenlik taşınıyor</span>
      </span>
      <span className="sayi text-cok-soluk">
        noktalarda ortalama {yuzde(ortKalan, 0)} hayatta kalıyor
      </span>
      <span className="ml-auto flex items-center gap-3 text-cok-soluk">
        <Yuvarlak renk="#b9a0f0" ad="sesli" />
        <Yuvarlak renk="#93a2b8" ad="sessiz" />
        <Yuvarlak renk="#d98a8a" ad="noktalama" />
        <span className="text-[10px]">— model bu ayrımı görmedi</span>
      </span>
    </div>
  );
}

function Yuvarlak({ renk, ad }: { renk: string; ad: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-2 w-2 rounded-full" style={{ background: renk }} aria-hidden />
      {ad}
    </span>
  );
}

function SecimPaneli({
  izdusum,
  secili,
  onTemizle,
}: {
  izdusum: Izdusum | null;
  secili: number[];
  onTemizle: () => void;
}) {
  if (!izdusum) return null;

  if (secili.length === 0) {
    return (
      <p className="border border-cizgi bg-yuzey px-3 py-2 text-[11px] text-cok-soluk">
        Küreyi sürükleyerek döndürün, tekerlekle yakınlaştırın. Bir noktaya tıklayın; ikinci bir
        nokta seçince aradaki benzerlik sayıyla yazılır.
      </p>
    );
  }

  const ilk = secili[0];
  const komsular = izdusum.birimler
    .map((v, id) => ({ id, benzerlik: kosinus(izdusum.birimler[ilk], v) }))
    .filter((k) => k.id !== ilk)
    .sort((a, b) => b.benzerlik - a.benzerlik)
    .slice(0, 5);

  let cift: null | { gercek: number; gorunen: number } = null;
  if (secili.length === 2) {
    const [a, b] = secili;
    const gercek = kosinus(izdusum.birimler[a], izdusum.birimler[b]);
    const pa = izdusum.noktalar[a];
    const pb = izdusum.noktalar[b];
    const gorunen = pa.x * pb.x + pa.y * pb.y + pa.z * pb.z;
    cift = { gercek, gorunen };
  }

  return (
    <div className="border border-cizgi bg-yuzey">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-cizgi px-3 py-2">
        <span className="text-[11px] text-cok-soluk">seçili</span>
        {secili.map((id) => (
          <span key={id} className="sayi border border-cizgi-parlak px-2 py-0.5 text-[13px] text-metin">
            {gorunurAd(id)}
          </span>
        ))}
        <button onClick={onTemizle} className="ml-auto text-[11px] text-cok-soluk hover:text-metin">
          seçimi temizle
        </button>
      </div>

      {cift && (
        <div className="space-y-1 border-b border-cizgi px-3 py-2">
          <div className="sayi text-[13px] text-metin">
            kosinüs benzerliği {sayi(cift.gercek, 4)} · açı {aciDerece(cift.gercek).toFixed(1)}°
            <span className="ml-2 text-[11px] text-cok-soluk">16 boyutta, gerçek değer</span>
          </div>
          <div className="sayi text-[11px] text-soluk">
            küredeki görünen açı {aciDerece(cift.gorunen).toFixed(1)}°
            <span className="ml-2 text-cok-soluk">
              {Math.abs(aciDerece(cift.gercek) - aciDerece(cift.gorunen)) > 12
                ? "— izdüşüm bu çifti yanıltıcı gösteriyor, üstteki sayıya güvenin"
                : "— izdüşüm bu çiftte gerçeğe yakın"}
            </span>
          </div>
        </div>
      )}

      <div className="px-3 py-2">
        <p className="mb-1.5 text-[11px] text-cok-soluk">
          <span className="sayi text-soluk">{gorunurAd(ilk)}</span> karakterine 16 boyutta en yakın
          beş karakter
        </p>
        <ul className="flex flex-wrap gap-2">
          {komsular.map(({ id, benzerlik }) => (
            <li key={id} className="sayi border border-cizgi px-2 py-1 text-[12px]">
              <span className="text-metin">{gorunurAd(id)}</span>{" "}
              <span className="text-cok-soluk">{sayi(benzerlik, 3)}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
