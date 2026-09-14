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
  /**
   * Dikkat katmanı açık mı?
   *
   * Açıkken bağlamdaki karakterler birbirine bakabilir: her konum, kendinden
   * önceki konumlardan hangilerine ne kadar ağırlık vereceğine kendi karar
   * verir. Kapalıyken bağlam uç uca eklenip sabit bir projeksiyondan geçer —
   * pozisyon başına ağırlıklar öğrenilir ama karakterler birbirine bakamaz.
   *
   * İkisi de çalışır durumda tutuldu, çünkü asıl öğretici olan fark:
   * "dikkat ne katıyor" sorusu ancak ikisini yan yana koyunca cevaplanıyor.
   */
  dikkat: boolean;
}

export const VARSAYILAN_AYAR: ModelAyarlari = {
  D: 16,
  katmanSayisi: 2,
  baglam: 8,
  seed: 1,
  dikkat: true,
};

/** Tek bir MLP bloğu: x -> Linear(D, 4D) -> ReLU -> Linear(4D, D), artı artık bağlantı. */
export interface Blok {
  W1: Matris;
  b1: Float32Array;
  W2: Matris;
  b2: Float32Array;
}

/**
 * TEK BAŞLI NEDENSEL ÖZ-DİKKAT
 * ----------------------------
 * Her konum üç vektör üretir:
 *   sorgu (q) — "ben ne arıyorum"
 *   anahtar (k) — "bende ne var"
 *   değer (v) — "bana bakılırsa ne veririm"
 *
 * Bir konumun başka bir konuma verdiği ağırlık, sorgusuyla o konumun
 * anahtarının iç çarpımıdır. Skorlar softmax'tan geçirilip toplamı 1 olan
 * ağırlıklara dönüşür, sonra değerlerin ağırlıklı toplamı alınır.
 *
 * "Nedensel" maske: bir konum yalnızca kendine ve kendinden öncekilere
 * bakabilir. Sonraki karaktere bakabilseydi tahmin işi anlamını yitirirdi —
 * cevabı kopya çekmiş olurdu.
 *
 * Tek başlı: gerçek modeller aynı işlemi birkaç "baş" ile paralel yapar ve
 * sonuçları birleştirir. Mekanizma birebir aynı, sadece kaç kez tekrarlandığı
 * değişiyor; tek başlı olanı anlayan çok başlıyı da anlamış olur.
 */
export interface DikkatKatmani {
  /** Sorgu projeksiyonu: D × D. */
  Wq: Matris;
  /** Anahtar projeksiyonu: D × D. */
  Wk: Matris;
  /** Değer projeksiyonu: D × D. */
  Wv: Matris;
  /** Çıkış projeksiyonu: D × D. Dikkatin ürettiğini ana yola hazırlar. */
  Wo: Matris;
}

export interface Model {
  ayar: ModelAyarlari;
  /** Gömme tablosu: V satır (her karakter için bir satır), D sütun. */
  E: Matris;
  /**
   * Pozisyon gömmeleri: C satır, D sütun. Sadece dikkat açıkken var.
   * Dikkat kendi başına sıraya kör: "ab" ile "ba" ona aynı görünür. Her
   * konuma öğrenilebilir bir vektör eklemek sırayı geri kazandırır.
   */
  P: Matris | null;
  /** Dikkat katmanı — kapalıysa null. */
  dikkat: DikkatKatmani | null;
  /** Giriş projeksiyonu: (C*D) satır, D sütun. Dikkat açıkken null. */
  Wgiris: Matris | null;
  bgiris: Float32Array | null;
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
  const { D, katmanSayisi, baglam, seed, dikkat } = ayar;
  const V = SOZLUK_BOYUTU;
  const rnd = rastgeleUretec(seed);

  const E = matrisOlustur("E (gömmeler)", V, D);
  // Gömmeler biraz daha geniş başlar; küre üzerinde başlangıçta dağınık
  // görünmeleri ve eğitimle toplanmaları böylece izlenebilir olur.
  rastgeleDoldur(E, rnd, 0.5);

  let P: Matris | null = null;
  let dikkatKatmani: DikkatKatmani | null = null;
  let Wgiris: Matris | null = null;
  let bgiris: Float32Array | null = null;

  if (dikkat) {
    P = matrisOlustur("P (pozisyonlar)", baglam, D);
    rastgeleDoldur(P, rnd, 0.1);

    const Wq = matrisOlustur("Wq (sorgu)", D, D);
    const Wk = matrisOlustur("Wk (anahtar)", D, D);
    const Wv = matrisOlustur("Wv (değer)", D, D);
    const Wo = matrisOlustur("Wo (dikkat çıkışı)", D, D);
    // Sorgu ve anahtar küçük başlar: skorlar büyük başlarsa softmax daha ilk
    // adımda tek bir konuma kilitlenir ve gradyan neredeyse hiç akmaz.
    rastgeleDoldur(Wq, rnd, Math.sqrt(1 / D) * 0.5);
    rastgeleDoldur(Wk, rnd, Math.sqrt(1 / D) * 0.5);
    rastgeleDoldur(Wv, rnd, Math.sqrt(1 / D));
    // Wo da küçük: dikkat bloğu da artık bağlantıyla ana yola EKLENİYOR.
    rastgeleDoldur(Wo, rnd, Math.sqrt(1 / D) * 0.5);
    dikkatKatmani = { Wq, Wk, Wv, Wo };
  } else {
    Wgiris = matrisOlustur("Wgiriş (bağlam → h)", baglam * D, D);
    rastgeleDoldur(Wgiris, rnd, Math.sqrt(2 / (baglam * D)));
    bgiris = new Float32Array(D);
  }

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

  return {
    ayar,
    E,
    P,
    dikkat: dikkatKatmani,
    Wgiris,
    bgiris,
    bloklar,
    Wcikis,
    bcikis: new Float32Array(V),
  };
}

/** Modeldeki toplam öğrenilebilir sayı adedi. Arayüzde gösterilir. */
export function parametreSayisi(model: Model): number {
  let n = model.E.veri.length;
  if (model.P) n += model.P.veri.length;
  if (model.dikkat) {
    const d = model.dikkat;
    n += d.Wq.veri.length + d.Wk.veri.length + d.Wv.veri.length + d.Wo.veri.length;
  }
  if (model.Wgiris) n += model.Wgiris.veri.length;
  if (model.bgiris) n += model.bgiris.length;
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
/**
 * Dikkat katmanının bütün ara değerleri. Arayüz dikkat haritasını buradan
 * çiziyor, geri yayılım da aynı kaydı kullanıyor.
 */
export interface DikkatIzi {
  /** Gömme + pozisyon: dikkate giren vektörler (C × D). */
  x: Float32Array[];
  sorgu: Float32Array[];
  anahtar: Float32Array[];
  deger: Float32Array[];
  /** Ham skorlar, √D'ye bölünmüş. Maskeli hücreler NaN. (C × C) */
  skorlar: Float32Array[];
  /** Softmax sonrası ağırlıklar; maskeli hücreler 0. Her satırın toplamı 1. */
  agirliklar: Float32Array[];
  /** Değerlerin ağırlıklı toplamı (C × D). */
  karisim: Float32Array[];
  /** x + karisim @ Wo — artık bağlantıdan sonra (C × D). */
  cikti: Float32Array[];
}

/**
 * Tek bir konumun dikkat sonrası zinciri: MLP blokları, çıkış katmanı,
 * softmax. Her konum kendi zincirinden geçer — bloklar ve çıkış katmanı
 * konumlar arasında PAYLAŞILIR, yani aynı ağırlıklar her konuma uygulanır.
 */
export interface KonumIzi {
  /** Zincire giren vektör (dikkat çıkışı). */
  h0: Float32Array;
  bloklar: BlokIzi[];
  logits: Float32Array;
  olasilik: Float32Array;
}

export interface IleriIz {
  baglamIds: number[];
  /** Her bağlam pozisyonunun gömme vektörü (C tane, D uzunluğunda). */
  gommeler: Float32Array[];
  /** Uç uca eklenmiş hali (C*D). Dikkat açıkken null. */
  birlesik: Float32Array | null;
  /** Dikkat katmanının kaydı. Dikkat kapalıyken null. */
  dikkat: DikkatIzi | null;
  /** Giriş projeksiyonundan sonra (D). */
  h0: Float32Array;
  bloklar: BlokIzi[];
  /** Softmax öncesi ham skorlar. */
  logits: Float32Array;
  /** Olasılıklar (sıcaklık uygulanmış). */
  olasilik: Float32Array;
  sicaklik: number;
  /**
   * Her konumun kendi tahmini. Yalnızca istendiğinde ve yalnızca dikkat
   * açıkken doldurulur.
   *
   * Dikkatsiz kurulumda bağlam tek bir vektöre indiği için "konumlar" diye
   * bir şey kalmıyor — her tahmin ayrı bir ileri geçiş gerektiriyor. Aynı
   * geçişte sekiz tahmin birden üretebilmek dikkatin yapısal getirisi.
   */
  konumlar: KonumIzi[] | null;
}

export function ileriGecis(
  model: Model,
  baglamIds: number[],
  sicaklik = 1,
  tumKonumlar = false,
): IleriIz {
  const { D, baglam } = model.ayar;

  // 1-2) Gömme araması. Bağlam kısaysa baştan boşlukla doldururuz.
  const ids = hizala(baglamIds, baglam);
  const gommeler: Float32Array[] = [];
  for (let t = 0; t < baglam; t++) {
    const satirBasi = ids[t] * D;
    const g = model.E.veri.subarray(satirBasi, satirBasi + D);
    gommeler.push(new Float32Array(g)); // izde saklamak için kopya
  }

  let birlesik: Float32Array | null = null;
  let dikkatIzi: DikkatIzi | null = null;
  let h0: Float32Array;

  if (model.dikkat && model.P) {
    dikkatIzi = dikkatGecisi(model, gommeler, baglam, D);
    // Tahmini yapan, dizinin SON konumudur: bir sonraki karakteri o bekliyor.
    h0 = dikkatIzi.cikti[baglam - 1];
  } else if (model.Wgiris && model.bgiris) {
    // 3) Giriş projeksiyonu: C*D -> D
    birlesik = new Float32Array(baglam * D);
    for (let t = 0; t < baglam; t++) birlesik.set(gommeler[t], t * D);
    h0 = vektorMatris(birlesik, model.Wgiris, model.bgiris);
  } else {
    throw new Error("model ne dikkat ne giriş projeksiyonu içeriyor");
  }

  // 4-6) MLP blokları, çıkış katmanı, softmax — son konum için
  const son = konumZinciri(model, h0, sicaklik);

  // İstenirse her konum kendi tahminini de üretir. Bloklar ve çıkış katmanı
  // paylaşıldığı için ek parametre yok, sadece ek hesap var.
  let konumlar: KonumIzi[] | null = null;
  if (tumKonumlar && dikkatIzi) {
    konumlar = dikkatIzi.cikti.map((y, t) =>
      t === baglam - 1 ? son : konumZinciri(model, y, sicaklik),
    );
  }

  return {
    baglamIds: ids,
    gommeler,
    birlesik,
    dikkat: dikkatIzi,
    h0,
    bloklar: son.bloklar,
    logits: son.logits,
    olasilik: son.olasilik,
    sicaklik,
    konumlar,
  };
}

/** Bir vektörü MLP bloklarından ve çıkış katmanından geçirir. */
function konumZinciri(model: Model, h0: Float32Array, sicaklik: number): KonumIzi {
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
  const logits = vektorMatris(h, model.Wcikis, model.bcikis);
  return { h0, bloklar: izler, logits, olasilik: softmax(logits, sicaklik) };
}

/**
 * Dikkat geçişi.
 *
 * Bütün satırlar hesaplanıyor ama tahmini yalnızca son satır üretiyor.
 * Diğer satırları da hesaplamamızın sebebi dikkat haritasının tamamını
 * gösterebilmek — gerçek eğitimde de bütün konumlar aynı anda kendi
 * sonraki karakterini tahmin eder, burada tek hedef olduğu için gradyan
 * yalnızca son satırdan akar. (Sorgular hariç: anahtarlar ve değerler son
 * satır üzerinden bütün konumlara gradyan taşır.)
 */
function dikkatGecisi(
  model: Model,
  gommeler: Float32Array[],
  C: number,
  D: number,
): DikkatIzi {
  const d = model.dikkat!;
  const P = model.P!;
  const olcek = 1 / Math.sqrt(D);

  // Gömme + pozisyon. Dikkat kendi başına sıraya kör; sırayı buradan alıyor.
  const x: Float32Array[] = [];
  for (let t = 0; t < C; t++) {
    const v = new Float32Array(D);
    for (let i = 0; i < D; i++) v[i] = gommeler[t][i] + P.veri[t * D + i];
    x.push(v);
  }

  const sorgu = x.map((v) => vektorMatris(v, d.Wq));
  const anahtar = x.map((v) => vektorMatris(v, d.Wk));
  const deger = x.map((v) => vektorMatris(v, d.Wv));

  const skorlar: Float32Array[] = [];
  const agirliklar: Float32Array[] = [];
  const karisim: Float32Array[] = [];
  const cikti: Float32Array[] = [];

  for (let t = 0; t < C; t++) {
    const satirSkor = new Float32Array(C).fill(NaN);
    // Nedensel maske: t konumu yalnızca 0..t arasına bakabilir.
    let enBuyuk = -Infinity;
    for (let sVal = 0; sVal <= t; sVal++) {
      let ic = 0;
      for (let i = 0; i < D; i++) ic += sorgu[t][i] * anahtar[sVal][i];
      const skor = ic * olcek; // √D'ye bölmek: D büyüdükçe skorlar şişip
      satirSkor[sVal] = skor;  // softmax'ı tek noktaya kilitlemesin diye
      if (skor > enBuyuk) enBuyuk = skor;
    }

    const satirAgirlik = new Float32Array(C);
    let toplam = 0;
    for (let sVal = 0; sVal <= t; sVal++) {
      const e = Math.exp(satirSkor[sVal] - enBuyuk);
      satirAgirlik[sVal] = e;
      toplam += e;
    }
    for (let sVal = 0; sVal <= t; sVal++) satirAgirlik[sVal] /= toplam;

    const c = new Float32Array(D);
    for (let sVal = 0; sVal <= t; sVal++) {
      const a = satirAgirlik[sVal];
      if (a === 0) continue;
      for (let i = 0; i < D; i++) c[i] += a * deger[sVal][i];
    }

    const o = vektorMatris(c, d.Wo);
    const y = new Float32Array(D);
    for (let i = 0; i < D; i++) y[i] = x[t][i] + o[i]; // artık bağlantı

    skorlar.push(satirSkor);
    agirliklar.push(satirAgirlik);
    karisim.push(c);
    cikti.push(y);
  }

  return { x, sorgu, anahtar, deger, skorlar, agirliklar, karisim, cikti };
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

// ---------------------------------------------------------------------------
// Ağırlıkların taşınması
// ---------------------------------------------------------------------------

/**
 * Modelin bütün ağırlıklarını sabit bir sırayla düz diziler hâlinde verir.
 *
 * Buna eğitim Web Worker'ı için ihtiyaç var: eğitim ayrı bir iş parçacığında
 * dönüyor, ekran ise ana iş parçacığında. İkisi aynı nesneyi paylaşamaz
 * (SharedArrayBuffer güvenlik başlıkları istiyor, statik barındırmada yok),
 * bu yüzden worker belirli aralıklarla ağırlıkların bir kopyasını gönderiyor.
 * 7.376 sayı ~30 KB eder; saniyede on beş kez göndermek bile hiçbir şeydir.
 *
 * Sıra iki taraf için de aynı olmak zorunda — bu yüzden tek bir yerde
 * tanımlı ve hem worker hem ekran bu fonksiyonu kullanıyor.
 */
export function agirlikDizileri(model: Model): Float32Array[] {
  const liste: Float32Array[] = [model.E.veri];
  if (model.P) liste.push(model.P.veri);
  if (model.dikkat) {
    const d = model.dikkat;
    liste.push(d.Wq.veri, d.Wk.veri, d.Wv.veri, d.Wo.veri);
  }
  if (model.Wgiris) liste.push(model.Wgiris.veri);
  if (model.bgiris) liste.push(model.bgiris);
  for (const b of model.bloklar) liste.push(b.W1.veri, b.b1, b.W2.veri, b.b2);
  liste.push(model.Wcikis.veri, model.bcikis);
  return liste;
}

/** Gelen ağırlık kopyalarını modelin üzerine yazar. */
export function agirliklariYaz(model: Model, gelen: Float32Array[]): void {
  const hedefler = agirlikDizileri(model);
  if (hedefler.length !== gelen.length) return;
  for (let i = 0; i < hedefler.length; i++) {
    if (hedefler[i].length === gelen[i].length) hedefler[i].set(gelen[i]);
  }
}

/** Worker'a göndermek için ağırlıkların kopyası. */
export function agirlikKopyasi(model: Model): Float32Array[] {
  return agirlikDizileri(model).map((d) => new Float32Array(d));
}
