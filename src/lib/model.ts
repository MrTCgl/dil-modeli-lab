/**
 * MODEL: MİMARİ VE İLERİ GEÇİŞ
 * ============================
 *
 * Burada hiçbir kütüphane yok. Her matris çarpımı elle yazılmış üç satırlık
 * bir döngüdür ve ekranda gördüğünüz her sayı bu dosyadaki işlemlerden çıkar.
 * Amaç hız değil, okunabilirlik: bu dosyayı baştan sona okuyan biri modelin
 * ne yaptığını tam olarak bilir.
 *
 * MİMARİ
 * ------
 * Son C karakteri (varsayılan 8) alıp bir sonraki karakteri tahmin ederiz:
 *
 *   1) Her karakter bir id'ye, her id bir gömme vektörüne dönüşür:  E[id] -> D sayı
 *   2) C tane gömme uç uca eklenir:                                 C*D sayı
 *   3) Bir giriş projeksiyonu bunu tek bir D'lik vektöre indirir:   h = birlesik @ Wgiris
 *   4) L tane MLP bloğu sırayla uygulanır:  h <- h + W2 · ReLU(h · W1)
 *   5) Çıkış katmanı sözlük kadar skor üretir:                      logits = h @ Wcikis
 *   6) Softmax skorları olasılığa çevirir.
 *
 * NEDEN BİRLEŞTİRME (concat)?
 * Bağlamdaki 8 gömmeyi tek vektöre indirmenin birkaç yolu var. Ortalama almak
 * en kısası ama sıra bilgisini yok eder: "ar" ile "ra" aynı görünür. Biz uç uca
 * ekleyip öğrenilebilir bir projeksiyondan geçiriyoruz; böylece model her
 * pozisyona ayrı ağırlık verebilir. Bu, sinir ağı dil modellerinin klasik
 * kurulumudur (Bengio ve arkadaşları, 2003) ve adım adım göstermeye elverişli:
 * 3. adım gerçek bir matris çarpımıdır, hücre hücre canlandırılabilir.
 *
 * ÖNEMLİ: Attention yok. Bu sürümde MLP yeterli ve anlatması çok daha net.
 *
 * MATRİS DÜZENİ
 * -------------
 * Matrisler satır öncelikli (row-major) tek bir Float32Array'de tutulur:
 *   W[i][j] = W.veri[i * W.sutun + j]
 * Çarpım her zaman "satır vektörü × matris" biçimindedir:
 *   y[j] = toplam_i x[i] * W[i][j]
 * Yani satır = giriş boyutu, sütun = çıkış boyutu. Ağırlık ızgarasında
 * gördüğünüz düzen budur: soldan sağa çıkışlar, yukarıdan aşağıya girişler.
 */

import { SOZLUK_BOYUTU, karakterId } from "./tokenizer.ts";

/** Ekranda ızgara olarak gösterilebilen tek bir ağırlık matrisi. */
export interface Matris {
  /** Izgarada gösterilecek ad, örn. "W1 (katman 0)". */
  ad: string;
  satir: number;
  sutun: number;
  veri: Float32Array;
}

export interface ModelAyarlari {
  /** Gömme boyutu D. */
  D: number;
  /** MLP bloğu sayısı L. */
  katmanSayisi: number;
  /** Bağlam penceresi C: kaç karaktere bakılıyor. */
  baglam: number;
  /** Rastgele başlangıç tohumu. Aynı tohum = aynı model. */
  seed: number;
}

export const VARSAYILAN_AYAR: ModelAyarlari = {
  D: 16,
  katmanSayisi: 2,
  baglam: 8,
  seed: 1,
};

/** Tek bir MLP bloğu: x -> Linear(D, 4D) -> ReLU -> Linear(4D, D), artı artık bağlantı. */
export interface Blok {
  W1: Matris;
  b1: Float32Array;
  W2: Matris;
  b2: Float32Array;
}

export interface Model {
  ayar: ModelAyarlari;
  /** Gömme tablosu: V satır (her karakter için bir satır), D sütun. */
  E: Matris;
  /** Giriş projeksiyonu: (C*D) satır, D sütun. */
  Wgiris: Matris;
  bgiris: Float32Array;
  bloklar: Blok[];
  /** Çıkış katmanı: D satır, V sütun. */
  Wcikis: Matris;
  bcikis: Float32Array;
}

// ---------------------------------------------------------------------------
// Rastgelelik: tohumlanabilir olmalı ki aynı link aynı modeli açsın.
// ---------------------------------------------------------------------------

/** mulberry32: kısa, hızlı, tohumlanabilir sözde rastgele sayı üreteci. */
export function rastgeleUretec(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller ile standart normal dağılım. */
function normalUret(rnd: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function matrisOlustur(ad: string, satir: number, sutun: number): Matris {
  return { ad, satir, sutun, veri: new Float32Array(satir * sutun) };
}

/**
 * Ağırlıkları rastgele başlatır. Ölçek önemlidir: çok büyük olursa ReLU
 * doyar ve eğitim patlar, çok küçük olursa sinyal katmanlar boyunca söner.
 * ReLU'lu katmanlarda yaygın seçim He başlatmasıdır: std = sqrt(2 / girişSayısı).
 */
function rastgeleDoldur(m: Matris, rnd: () => number, std: number): void {
  for (let i = 0; i < m.veri.length; i++) m.veri[i] = normalUret(rnd) * std;
}

export function modelOlustur(ayar: ModelAyarlari = VARSAYILAN_AYAR): Model {
  const { D, katmanSayisi, baglam, seed } = ayar;
  const V = SOZLUK_BOYUTU;
  const rnd = rastgeleUretec(seed);

  const E = matrisOlustur("E (gömmeler)", V, D);
  // Gömmeler biraz daha geniş başlar; küre üzerinde başlangıçta dağınık
  // görünmeleri ve eğitimle toplanmaları böylece izlenebilir olur.
  rastgeleDoldur(E, rnd, 0.5);

  const Wgiris = matrisOlustur("Wgiriş (bağlam → h)", baglam * D, D);
  rastgeleDoldur(Wgiris, rnd, Math.sqrt(2 / (baglam * D)));

  const bloklar: Blok[] = [];
  for (let l = 0; l < katmanSayisi; l++) {
    const W1 = matrisOlustur(`W1 · katman ${l + 1}`, D, 4 * D);
    const W2 = matrisOlustur(`W2 · katman ${l + 1}`, 4 * D, D);
    rastgeleDoldur(W1, rnd, Math.sqrt(2 / D));
    // W2 daha küçük başlar: artık bağlantı yüzünden her blok ana yola ekleme
    // yapar; katman sayısı arttıkça bu eklemeler birikip sinyali şişirmesin.
    rastgeleDoldur(W2, rnd, Math.sqrt(2 / (4 * D)) / Math.sqrt(katmanSayisi));
    bloklar.push({ W1, b1: new Float32Array(4 * D), W2, b2: new Float32Array(D) });
  }

  const Wcikis = matrisOlustur("Wçıkış (h → skorlar)", D, V);
  rastgeleDoldur(Wcikis, rnd, Math.sqrt(2 / D));

  return { ayar, E, Wgiris, bgiris: new Float32Array(D), bloklar, Wcikis, bcikis: new Float32Array(V) };
}

/** Modeldeki toplam öğrenilebilir sayı adedi. Arayüzde gösterilir. */
export function parametreSayisi(model: Model): number {
  let n = model.E.veri.length + model.Wgiris.veri.length + model.bgiris.length;
  for (const b of model.bloklar) {
    n += b.W1.veri.length + b.b1.length + b.W2.veri.length + b.b2.length;
  }
  return n + model.Wcikis.veri.length + model.bcikis.length;
}

// ---------------------------------------------------------------------------
// Temel işlemler
// ---------------------------------------------------------------------------

/**
 * Satır vektörü × matris:  y[j] = toplam_i x[i] * W[i][j]  (+ b[j])
 * Bütün ağın kalbi bu üç satır. Arayüzdeki "hücre hücre" canlandırma da
 * tam olarak bu döngüyü adım adım gösterir.
 */
export function vektorMatris(x: Float32Array, W: Matris, b?: Float32Array): Float32Array {
  const y = new Float32Array(W.sutun);
  if (b) y.set(b);
  for (let i = 0; i < W.satir; i++) {
    const xi = x[i];
    if (xi === 0) continue; // sıfır girişler çıktıyı değiştirmez
    const satirBasi = i * W.sutun;
    for (let j = 0; j < W.sutun; j++) {
      y[j] += xi * W.veri[satirBasi + j];
    }
  }
  return y;
}

/**
 * Softmax: skorları toplamı 1 olan olasılıklara çevirir.
 * En büyük skoru çıkarmak matematiği değiştirmez ama exp() taşmasını önler.
 * Sıcaklık skorları bölerek dağılımın sivriliğini ayarlar: düşük sıcaklık
 * en yüksek skoru öne çıkarır, yüksek sıcaklık dağılımı düzleştirir.
 */
export function softmax(logits: Float32Array, sicaklik = 1): Float32Array {
  const n = logits.length;
  const cikti = new Float32Array(n);
  let enBuyuk = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = logits[i] / sicaklik;
    if (v > enBuyuk) enBuyuk = v;
  }
  let toplam = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.exp(logits[i] / sicaklik - enBuyuk);
    cikti[i] = v;
    toplam += v;
  }
  for (let i = 0; i < n; i++) cikti[i] /= toplam;
  return cikti;
}

// ---------------------------------------------------------------------------
// İleri geçiş
// ---------------------------------------------------------------------------

/** Tek bir bloğun ara değerleri. Hem geri yayılım hem arayüz bunları kullanır. */
export interface BlokIzi {
  /** Bloğa giren vektör (artık bağlantının başlangıcı). */
  girdi: Float32Array;
  /** ReLU'dan önceki ham değerler (4D). */
  oncesi: Float32Array;
  /** ReLU'dan sonra (4D). Kaç tanesinin sıfırlandığı arayüzde yazılır. */
  relu: Float32Array;
  /** İkinci çarpımın çıktısı (D): ana yola eklenecek olan. */
  dal: Float32Array;
  /** girdi + dal. */
  cikti: Float32Array;
}

/**
 * İleri geçişin tam kaydı. Arayüz bu nesneyi alıp adım adım gösterir;
 * geri yayılım da aynı nesneyi kullanır. Yani ekranda görünen sayılarla
 * eğitimde kullanılan sayılar birebir aynıdır.
 */
export interface IleriIz {
  baglamIds: number[];
  /** Her bağlam pozisyonunun gömme vektörü (C tane, D uzunluğunda). */
  gommeler: Float32Array[];
  /** Uç uca eklenmiş hali (C*D). */
  birlesik: Float32Array;
  /** Giriş projeksiyonundan sonra (D). */
  h0: Float32Array;
  bloklar: BlokIzi[];
  /** Softmax öncesi ham skorlar. */
  logits: Float32Array;
  /** Olasılıklar (sıcaklık uygulanmış). */
  olasilik: Float32Array;
  sicaklik: number;
}

export function ileriGecis(model: Model, baglamIds: number[], sicaklik = 1): IleriIz {
  const { D, baglam } = model.ayar;

  // 1-2) Gömme araması. Bağlam kısaysa baştan boşlukla doldururuz.
  const ids = hizala(baglamIds, baglam);
  const gommeler: Float32Array[] = [];
  const birlesik = new Float32Array(baglam * D);
  for (let t = 0; t < baglam; t++) {
    const satirBasi = ids[t] * D;
    const g = model.E.veri.subarray(satirBasi, satirBasi + D);
    const kopya = new Float32Array(g); // izde saklamak için kopya
    gommeler.push(kopya);
    birlesik.set(kopya, t * D);
  }

  // 3) Giriş projeksiyonu: C*D -> D
  const h0 = vektorMatris(birlesik, model.Wgiris, model.bgiris);

  // 4) MLP blokları
  let h = h0;
  const izler: BlokIzi[] = [];
  for (const blok of model.bloklar) {
    const girdi = h;
    const oncesi = vektorMatris(girdi, blok.W1, blok.b1);
    const relu = new Float32Array(oncesi.length);
    for (let i = 0; i < oncesi.length; i++) relu[i] = oncesi[i] > 0 ? oncesi[i] : 0;
    const dal = vektorMatris(relu, blok.W2, blok.b2);
    const cikti = new Float32Array(dal.length);
    for (let i = 0; i < dal.length; i++) cikti[i] = girdi[i] + dal[i]; // artık bağlantı
    izler.push({ girdi, oncesi, relu, dal, cikti });
    h = cikti;
  }

  // 5-6) Çıkış katmanı ve softmax
  const logits = vektorMatris(h, model.Wcikis, model.bcikis);
  const olasilik = softmax(logits, sicaklik);

  return { baglamIds: ids, gommeler, birlesik, h0, bloklar: izler, logits, olasilik, sicaklik };
}

/**
 * Bağlamı tam C uzunluğuna getirir: uzunsa sondan C tanesini alır,
 * kısaysa başına boşluk (id) ekler. Metnin başındaki karakterleri de
 * tahmin edebilmek için gerekli.
 */
export function hizala(ids: readonly number[], baglam: number): number[] {
  const bosluk = karakterId(" ");
  if (ids.length >= baglam) return ids.slice(ids.length - baglam);
  const dolgu = new Array(baglam - ids.length).fill(bosluk);
  return dolgu.concat([...ids]);
}

/**
 * Çapraz entropi kaybı: doğru karaktere verilen olasılığın negatif logaritması.
 * Model doğru karaktere 1 olasılık verirse kayıp 0; 1/V verirse ln(V) olur.
 * V = 32 için rastgele bir modelin beklenen kaybı ln(32) = 3.4657'dir —
 * eğitim grafiğindeki başlangıç çizgisi tam olarak budur.
 */
export function kayip(olasilik: Float32Array, hedef: number): number {
  return -Math.log(Math.max(olasilik[hedef], 1e-12));
}

/** Rastgele bir modelin beklenen kaybı: ln(V). Grafikte referans çizgisi. */
export const RASTGELE_KAYIP = Math.log(SOZLUK_BOYUTU);

// ---------------------------------------------------------------------------
// Üretim: tahmin edilen karakteri geri besleyip metni uzatmak
// ---------------------------------------------------------------------------

/**
 * Olasılık dağılımından tek bir karakter seçer.
 * Dikkat: burada en yüksek olasılıklıyı almıyoruz, dağılımdan örnekliyoruz.
 * En yükseği almak (greedy) modeli kısa sürede aynı heceyi tekrarlamaya
 * sokar; örnekleme ise sıcaklığın ne işe yaradığını görünür kılar.
 */
export function ornekle(olasilik: Float32Array, rnd: () => number): number {
  const esik = rnd();
  let birikim = 0;
  for (let i = 0; i < olasilik.length; i++) {
    birikim += olasilik[i];
    if (birikim >= esik) return i;
  }
  return olasilik.length - 1;
}

/**
 * Verilen başlangıçtan itibaren `adet` karakter üretir.
 * Her adımda: ileri geçiş -> örnekleme -> üretilen karakteri bağlama ekle.
 * Eğitilmemiş modelde çıktı çöptür; eğitilmiş modelde Türkçeye benzemeye
 * başlar. İkisini yan yana görmek bu uygulamanın en öğretici anlarından biri.
 */
export function uret(
  model: Model,
  baslangicIds: readonly number[],
  adet: number,
  sicaklik: number,
  rnd: () => number,
): number[] {
  const ids = [...baslangicIds];
  const uretilen: number[] = [];
  for (let i = 0; i < adet; i++) {
    const iz = ileriGecis(model, ids, sicaklik);
    const yeni = ornekle(iz.olasilik, rnd);
    uretilen.push(yeni);
    ids.push(yeni);
  }
  return uretilen;
}
