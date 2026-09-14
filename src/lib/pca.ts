/**
 * PCA: 16 BOYUTU 3 BOYUTA İNDİRMEK
 * ================================
 *
 * Gömme vektörleri D boyutlu (varsayılan 16). Ekran üç boyutlu. Aradaki
 * farkı kapatmanın kayıpsız bir yolu yok; bu yüzden en az kaybedeni
 * seçiyoruz: verinin en çok yayıldığı üç yönü bulup herkesi o üç yöne
 * izdüşürmek. Buna temel bileşen analizi (PCA) deniyor.
 *
 * KAYBI GİZLEMİYORUZ
 * Ekranda üç şey birden yazıyor:
 *   - ilk üç bileşenin toplam değişkenliğin yüzde kaçını taşıdığı,
 *   - her noktanın izdüşümde ne kadarının hayatta kaldığı ("kalan"),
 *   - seçilen iki karakterin GERÇEK D boyutlu kosinüs benzerliği.
 * Küredeki açı ile gerçek açı birbirini tutmayabilir — tutmadığında da
 * ekranda ikisi yan yana yazılı. Resim bir özet; sayı ise ölçüm.
 *
 * YÖNTEM
 * Kütüphane yok: güç yinelemesi (power iteration) ve sönümleme (deflation).
 * Kovaryans matrisini bir vektörle tekrar tekrar çarparsanız vektör en büyük
 * özdeğerin yönüne döner. O yönü bulup matristen çıkarır, kalanda aynısını
 * tekrarlarsınız. D en fazla 64 olduğu için bu birkaç mikrosaniye sürer ve
 * yirmi satırda okunur.
 */

import type { Matris } from "./model.ts";
import { rastgeleUretec } from "./model.ts";

export interface Nokta {
  /** Birim küre üzerindeki yön — ekranda noktanın durduğu yer. */
  x: number;
  y: number;
  z: number;
  /**
   * İzdüşümün uzunluğu (0–1). 1'e yakınsa bu karakterin yönü üç bileşenle
   * neredeyse tamamen anlatılıyor; 0'a yakınsa küredeki yeri yanıltıcı.
   * Ekranda nokta boyutu ve parlaklığı buna göre ayarlanıyor.
   */
  kalan: number;
}

export interface Izdusum {
  noktalar: Nokta[];
  /** İlk üç bileşenin taşıdığı değişkenlik oranı (0–1). */
  aciklananOran: number;
  /** Üç temel bileşen; sonraki karede işaret hizalamak için saklanır. */
  bilesenler: Float64Array[];
  /** Normalleştirilmiş (birim uzunluklu) gömmeler — gerçek benzerlik bunlardan. */
  birimler: Float64Array[];
}

/** İki vektör arasındaki kosinüs benzerliği. Vektörler birim ise sadece iç çarpım. */
export function kosinus(a: Float64Array, b: Float64Array): number {
  let ic = 0;
  let aKare = 0;
  let bKare = 0;
  for (let i = 0; i < a.length; i++) {
    ic += a[i] * b[i];
    aKare += a[i] * a[i];
    bKare += b[i] * b[i];
  }
  const payda = Math.sqrt(aKare) * Math.sqrt(bKare);
  return payda < 1e-12 ? 0 : ic / payda;
}

/** Kosinüs benzerliğini dereceye çevirir. */
export function aciDerece(kosinusDegeri: number): number {
  return (Math.acos(Math.min(1, Math.max(-1, kosinusDegeri))) * 180) / Math.PI;
}

/** Gömme tablosunun her satırını birim uzunluğa getirir. */
export function birimleHale(E: Matris): Float64Array[] {
  const sonuc: Float64Array[] = [];
  for (let i = 0; i < E.satir; i++) {
    const v = new Float64Array(E.sutun);
    let kare = 0;
    for (let j = 0; j < E.sutun; j++) {
      v[j] = E.veri[i * E.sutun + j];
      kare += v[j] * v[j];
    }
    const uzunluk = Math.sqrt(kare);
    if (uzunluk > 1e-12) for (let j = 0; j < E.sutun; j++) v[j] /= uzunluk;
    sonuc.push(v);
  }
  return sonuc;
}

function uzunluk(v: Float64Array): number {
  let kare = 0;
  for (let i = 0; i < v.length; i++) kare += v[i] * v[i];
  return Math.sqrt(kare);
}

/** Simetrik matris × vektör. C satır öncelikli D×D. */
function carp(C: Float64Array, v: Float64Array, D: number): Float64Array {
  const y = new Float64Array(D);
  for (let i = 0; i < D; i++) {
    let toplam = 0;
    const satir = i * D;
    for (let j = 0; j < D; j++) toplam += C[satir + j] * v[j];
    y[i] = toplam;
  }
  return y;
}

/**
 * Güç yinelemesi: en büyük özdeğere ait yönü bulur.
 * Rastgele bir vektörle başlayıp matrisle çarpmayı tekrarlamak yeter —
 * her çarpımda vektör baskın yöne biraz daha döner.
 */
function enBaskinYon(C: Float64Array, D: number, tohum: number): { yon: Float64Array; ozdeger: number } {
  const rnd = rastgeleUretec(tohum);
  // Tip notu: carp() yeni bir dizi döndürdüğü için v'nin tipini açıkça
  // yazıyoruz; yoksa TypeScript v'yi dar bir tampon tipine bağlıyor.
  let v: Float64Array = new Float64Array(D);
  for (let i = 0; i < D; i++) v[i] = rnd() - 0.5;
  let u = uzunluk(v);
  if (u < 1e-12) {
    v[0] = 1;
    u = 1;
  }
  for (let i = 0; i < D; i++) v[i] /= u;

  let ozdeger = 0;
  for (let yineleme = 0; yineleme < 500; yineleme++) {
    const w = carp(C, v, D);
    const boy = uzunluk(w);
    if (boy < 1e-14) break;
    for (let i = 0; i < D; i++) w[i] /= boy;
    // yakınsama: yön artık kıpırdamıyorsa dur
    let degisim = 0;
    for (let i = 0; i < D; i++) degisim += Math.abs(w[i] - v[i]);
    v = w;
    ozdeger = boy;
    if (degisim < 1e-12) break;
  }
  return { yon: v, ozdeger };
}

/** C := C − λ·v·vᵀ — bulunan yönü matristen çıkarır, sıradaki ortaya çıksın. */
function sonumle(C: Float64Array, D: number, yon: Float64Array, ozdeger: number): void {
  for (let i = 0; i < D; i++) {
    for (let j = 0; j < D; j++) {
      C[i * D + j] -= ozdeger * yon[i] * yon[j];
    }
  }
}

/**
 * Birim gömmeleri üç boyuta indirir.
 *
 * `oncekiBilesenler` verilirse yeni bileşenlerin işareti ona göre hizalanır.
 * Bu eğitim sırasında gerekli: özvektörün işareti matematiksel olarak
 * keyfîdir, dolayısıyla iki kare arasında bileşen ters dönebilir ve küre
 * sebepsiz yere aynalanmış gibi zıplar. Hizalama bunu engelliyor.
 */
export function izdusumCikar(E: Matris, oncekiBilesenler?: Float64Array[]): Izdusum {
  const D = E.sutun;
  const V = E.satir;
  const birimler = birimleHale(E);

  // ortalamayı çıkar: PCA yayılımı ölçer, ortak yönü değil
  const ortalama = new Float64Array(D);
  for (const v of birimler) for (let j = 0; j < D; j++) ortalama[j] += v[j] / V;
  const merkezli = birimler.map((v) => {
    const m = new Float64Array(D);
    for (let j = 0; j < D; j++) m[j] = v[j] - ortalama[j];
    return m;
  });

  // kovaryans
  const C = new Float64Array(D * D);
  for (const m of merkezli) {
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) C[i * D + j] += (m[i] * m[j]) / V;
    }
  }
  let toplamDegisken = 0;
  for (let i = 0; i < D; i++) toplamDegisken += C[i * D + i]; // iz = özdeğerler toplamı

  // ilk üç bileşen
  const bilesenler: Float64Array[] = [];
  let ilkUc = 0;
  for (let k = 0; k < 3; k++) {
    const { yon, ozdeger } = enBaskinYon(C, D, 1000 + k);
    // işaret hizalama
    const onceki = oncekiBilesenler?.[k];
    let isaret = 1;
    if (onceki && onceki.length === D) {
      let ic = 0;
      for (let i = 0; i < D; i++) ic += yon[i] * onceki[i];
      if (ic < 0) isaret = -1;
    } else {
      // İlk seferde de kararlı olsun: en büyük bileşen pozitif olacak şekilde sabitle.
      let enBuyukIndeks = 0;
      for (let i = 1; i < D; i++) if (Math.abs(yon[i]) > Math.abs(yon[enBuyukIndeks])) enBuyukIndeks = i;
      if (yon[enBuyukIndeks] < 0) isaret = -1;
    }
    if (isaret === -1) for (let i = 0; i < D; i++) yon[i] = -yon[i];

    bilesenler.push(yon);
    ilkUc += Math.max(0, ozdeger);
    sonumle(C, D, yon, ozdeger);
  }

  // izdüşüm
  const noktalar: Nokta[] = merkezli.map((m) => {
    const p = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      let toplam = 0;
      for (let i = 0; i < D; i++) toplam += m[i] * bilesenler[k][i];
      p[k] = toplam;
    }
    const boy = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    const merkezliBoy = uzunluk(m);
    // "kalan": merkezli vektörün ne kadarı üç bileşende hayatta kaldı
    const kalan = merkezliBoy < 1e-12 ? 0 : Math.min(1, boy / merkezliBoy);
    if (boy < 1e-12) return { x: 0, y: 1, z: 0, kalan: 0 };
    return { x: p[0] / boy, y: p[1] / boy, z: p[2] / boy, kalan };
  });

  return {
    noktalar,
    aciklananOran: toplamDegisken < 1e-12 ? 0 : Math.min(1, ilkUc / toplamDegisken),
    bilesenler,
    birimler,
  };
}
