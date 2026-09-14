/**
 * EĞİTİM: GRADYANLAR VE GRADYAN İNİŞİ
 * ===================================
 *
 * Burada autograd yok. Her türevi elle yazdık. Bunun iki sebebi var:
 * birincisi, geri yayılımın sihir olmadığını göstermek; ikincisi, ekranda
 * gördüğünüz her güncellemenin hangi satırdan geldiğini işaret edebilmek.
 *
 * ZİNCİR KURALI, TEK CÜMLEDE
 * --------------------------
 * İleri geçişte her adım bir sonrakinin girdisini üretir. Geri geçişte aynı
 * yolu tersten yürürüz: "çıktıdaki küçük bir değişiklik kaybı ne kadar
 * değiştirir" sorusunun cevabını alır, bir önceki adıma taşırız.
 *
 * Tek bir kural her şeyi yürütür: y = x @ W ise
 *   dL/dW[i][j] = x[i] * dL/dy[j]        (ağırlığın gradyanı: giriş × çıkış gradyanı)
 *   dL/dx[i]    = toplam_j W[i][j] * dL/dy[j]   (gradyan geriye böyle akar)
 * Aşağıdaki bütün kod bu iki satırın tekrarıdır.
 *
 * BAŞLANGIÇ NOKTASI
 * -----------------
 * Softmax + çapraz entropi birlikte türetildiğinde sonuç şaşırtıcı derecede
 * sade çıkar:
 *   dL/dlogits = olasılık - birSıcak(hedef)
 * Yani "modelin verdiği olasılık eksi olması gereken". Doğru karakterde
 * negatif (skorunu artır), yanlışlarda pozitif (skorunu azalt).
 */

import {
  type Blok,
  type IleriIz,
  type Matris,
  type Model,
  agirlikDizileri,
  ileriGecis,
  kayip,
  matrisOlustur,
  rastgeleUretec,
} from "./model.ts";
import { encode } from "./tokenizer.ts";

// ---------------------------------------------------------------------------
// Gradyan kapları: modelin aynadaki yansıması
// ---------------------------------------------------------------------------

export interface BlokGradyani {
  W1: Matris;
  b1: Float32Array;
  W2: Matris;
  b2: Float32Array;
}

export interface DikkatGradyani {
  Wq: Matris;
  Wk: Matris;
  Wv: Matris;
  Wo: Matris;
}

export interface Gradyan {
  E: Matris;
  P: Matris | null;
  dikkat: DikkatGradyani | null;
  Wgiris: Matris | null;
  bgiris: Float32Array | null;
  bloklar: BlokGradyani[];
  Wcikis: Matris;
  bcikis: Float32Array;
}

function bosMatris(m: Matris): Matris {
  return matrisOlustur(m.ad, m.satir, m.sutun);
}

/** Model ile birebir aynı biçimde, sıfırlarla dolu bir gradyan kabı. */
export function gradyanOlustur(model: Model): Gradyan {
  return {
    E: bosMatris(model.E),
    P: model.P ? bosMatris(model.P) : null,
    dikkat: model.dikkat
      ? {
          Wq: bosMatris(model.dikkat.Wq),
          Wk: bosMatris(model.dikkat.Wk),
          Wv: bosMatris(model.dikkat.Wv),
          Wo: bosMatris(model.dikkat.Wo),
        }
      : null,
    Wgiris: model.Wgiris ? bosMatris(model.Wgiris) : null,
    bgiris: model.bgiris ? new Float32Array(model.bgiris.length) : null,
    bloklar: model.bloklar.map((b: Blok) => ({
      W1: bosMatris(b.W1),
      b1: new Float32Array(b.b1.length),
      W2: bosMatris(b.W2),
      b2: new Float32Array(b.b2.length),
    })),
    Wcikis: bosMatris(model.Wcikis),
    bcikis: new Float32Array(model.bcikis.length),
  };
}

export function gradyanSifirla(g: Gradyan): void {
  g.E.veri.fill(0);
  g.P?.veri.fill(0);
  if (g.dikkat) {
    g.dikkat.Wq.veri.fill(0);
    g.dikkat.Wk.veri.fill(0);
    g.dikkat.Wv.veri.fill(0);
    g.dikkat.Wo.veri.fill(0);
  }
  g.Wgiris?.veri.fill(0);
  g.bgiris?.fill(0);
  for (const b of g.bloklar) {
    b.W1.veri.fill(0);
    b.b1.fill(0);
    b.W2.veri.fill(0);
    b.b2.fill(0);
  }
  g.Wcikis.veri.fill(0);
  g.bcikis.fill(0);
}

// ---------------------------------------------------------------------------
// Geri yayılım
// ---------------------------------------------------------------------------

/**
 * Tek bir örneğin gradyanını hesaplar ve `grad` üzerine EKLER.
 * Toplu (batch) eğitimde aynı kap üzerine birden fazla örnek biriktirilir,
 * güncelleme sırasında örnek sayısına bölünür.
 *
 * `iz` ileri geçişten gelen kayıttır — arayüzde gösterilen izin ta kendisi.
 */
export function geriYayilim(model: Model, iz: IleriIz, hedef: number, grad: Gradyan): void {
  const D = model.ayar.D;

  // --- Çıkış: dL/dlogits = olasılık - birSıcak(hedef) -----------------------
  // Sıcaklık logits'i böldüğü için türevi de bölünür. Eğitimde sıcaklık 1'dir,
  // ama formülü eksiksiz yazıyoruz ki başka sıcaklıkta da doğru kalsın.
  const V = iz.logits.length;
  const dlogits = new Float32Array(V);
  for (let j = 0; j < V; j++) dlogits[j] = iz.olasilik[j] / iz.sicaklik;
  dlogits[hedef] -= 1 / iz.sicaklik;

  // Çıkış katmanının girdisi: son bloğun çıktısı (blok yoksa h0).
  const sonH = iz.bloklar.length > 0 ? iz.bloklar[iz.bloklar.length - 1].cikti : iz.h0;

  const dh = new Float32Array(D);
  for (let i = 0; i < D; i++) {
    const satirBasi = i * V;
    const hi = sonH[i];
    let toplam = 0;
    for (let j = 0; j < V; j++) {
      grad.Wcikis.veri[satirBasi + j] += hi * dlogits[j];
      toplam += model.Wcikis.veri[satirBasi + j] * dlogits[j];
    }
    dh[i] = toplam;
  }
  for (let j = 0; j < V; j++) grad.bcikis[j] += dlogits[j];

  // --- Bloklar, tersten ----------------------------------------------------
  let dCikti = dh;
  for (let l = model.bloklar.length - 1; l >= 0; l--) {
    const blok = model.bloklar[l];
    const gb = grad.bloklar[l];
    const bi = iz.bloklar[l];
    const genis = blok.W1.sutun; // 4D

    // cikti = girdi + dal  ->  gradyan iki yola da aynen geçer.
    // Artık bağlantının bütün marifeti bu: gradyan blokları atlayarak
    // doğrudan geriye akabilir, bu yüzden derin ağlar eğitilebilir kalır.
    const dGirdi = new Float32Array(D);
    for (let i = 0; i < D; i++) dGirdi[i] = dCikti[i]; // doğrudan yol
    const dDal = dCikti;

    // dal = relu @ W2 + b2
    const dRelu = new Float32Array(genis);
    for (let i = 0; i < genis; i++) {
      const satirBasi = i * D;
      const ri = bi.relu[i];
      let toplam = 0;
      for (let j = 0; j < D; j++) {
        gb.W2.veri[satirBasi + j] += ri * dDal[j];
        toplam += blok.W2.veri[satirBasi + j] * dDal[j];
      }
      dRelu[i] = toplam;
    }
    for (let j = 0; j < D; j++) gb.b2[j] += dDal[j];

    // ReLU'nun türevi: giriş pozitifse 1, değilse 0.
    // Sıfırlanan nöronlar gradyanı da geçirmez — arayüzde "kaç nöron söndü"
    // sayısının önemi buradan gelir: sönen nöron o adımda hiç öğrenmez.
    const dOncesi = new Float32Array(genis);
    for (let i = 0; i < genis; i++) dOncesi[i] = bi.oncesi[i] > 0 ? dRelu[i] : 0;

    // oncesi = girdi @ W1 + b1
    for (let i = 0; i < D; i++) {
      const satirBasi = i * genis;
      const gi = bi.girdi[i];
      let toplam = 0;
      for (let j = 0; j < genis; j++) {
        gb.W1.veri[satirBasi + j] += gi * dOncesi[j];
        toplam += blok.W1.veri[satirBasi + j] * dOncesi[j];
      }
      dGirdi[i] += toplam; // dal yolundan gelen katkı
    }
    for (let j = 0; j < genis; j++) gb.b1[j] += dOncesi[j];

    dCikti = dGirdi;
  }

  if (model.dikkat && iz.dikkat && grad.dikkat && model.P && grad.P) {
    dikkatGeriYayilim(model, iz, grad, dCikti, D);
    return;
  }

  // --- Giriş projeksiyonu: h0 = birlesik @ Wgiris + bgiris -----------------
  if (!model.Wgiris || !model.bgiris || !grad.Wgiris || !grad.bgiris || !iz.birlesik) return;
  const CD = model.Wgiris.satir;
  const dBirlesik = new Float32Array(CD);
  for (let i = 0; i < CD; i++) {
    const satirBasi = i * D;
    const xi = iz.birlesik[i];
    let toplam = 0;
    for (let j = 0; j < D; j++) {
      grad.Wgiris.veri[satirBasi + j] += xi * dCikti[j];
      toplam += model.Wgiris.veri[satirBasi + j] * dCikti[j];
    }
    dBirlesik[i] = toplam;
  }
  for (let j = 0; j < D; j++) grad.bgiris[j] += dCikti[j];

  // --- Gömmeler ------------------------------------------------------------
  // Birleştirilmiş vektörün gradyanını parçalayıp ilgili karakterin satırına
  // dağıtırız. Aynı karakter bağlamda birden çok kez geçiyorsa katkılar
  // toplanır — bu yüzden += kullanıyoruz.
  for (let t = 0; t < iz.baglamIds.length; t++) {
    const satirBasi = iz.baglamIds[t] * D;
    const kaynak = t * D;
    for (let k = 0; k < D; k++) {
      grad.E.veri[satirBasi + k] += dBirlesik[kaynak + k];
    }
  }
}

/**
 * DİKKATİN GERİ YAYILIMI
 * ======================
 *
 * Tahmini son konum ürettiği için gradyan da oradan giriyor. Yol şu:
 *
 *   y_T = x_T + o_T                     artık bağlantı: iki yola da geçer
 *   o_T = c_T @ Wo                      alışılmış matris türevi
 *   c_T = toplam_s a[s] · v_s           hem ağırlıklara hem değerlere gider
 *   a   = softmax(skor)                 aşağıdaki tek satırlık kural
 *   skor[s] = (q_T · k_s) / √D          sorguya ve anahtarlara dağılır
 *   q,k,v   = x @ Wq, Wk, Wv            yine matris türevi
 *   x_t = E[id_t] + P[t]                toplam olduğu için ikisine de aynen
 *
 * SOFTMAX'IN TÜREVİ
 * Softmax'ın jakobiyeni tam yazılınca C×C'lik bir matris; ama kaybın
 * gradyanıyla çarpımı tek satıra iniyor:
 *     dskor[s] = a[s] · (da[s] − toplam_u a[u]·da[u])
 * Sezgisi: bir ağırlığı artırmak zorunlu olarak diğerlerini azaltır
 * (toplamları 1 olmak zorunda), o yüzden her terimden ortalama çıkarılıyor.
 *
 * NEDEN SADECE SON SATIR
 * Dikkat haritasının bütün satırları ileri geçişte hesaplanıyor ama tahmini
 * yalnızca son satır üretiyor; kalanı ekranda göstermek için. Dolayısıyla
 * sorgu gradyanı yalnızca son konuma gidiyor. Anahtar ve değer gradyanları
 * ise bütün konumlara ulaşıyor, çünkü son satır hepsine bakıyor.
 */
function dikkatGeriYayilim(
  model: Model,
  iz: IleriIz,
  grad: Gradyan,
  dY: Float32Array,
  D: number,
): void {
  const d = model.dikkat!;
  const gd = grad.dikkat!;
  const gP = grad.P!;
  const di = iz.dikkat!;
  const C = iz.baglamIds.length;
  const T = C - 1;
  const olcek = 1 / Math.sqrt(D);

  const dx: Float32Array[] = [];
  for (let t = 0; t < C; t++) dx.push(new Float32Array(D));

  // y_T = x_T + o_T
  for (let i = 0; i < D; i++) dx[T][i] += dY[i];

  // o_T = c_T @ Wo
  const dKarisim = new Float32Array(D);
  for (let i = 0; i < D; i++) {
    const satir = i * D;
    const ci = di.karisim[T][i];
    let toplam = 0;
    for (let j = 0; j < D; j++) {
      gd.Wo.veri[satir + j] += ci * dY[j];
      toplam += d.Wo.veri[satir + j] * dY[j];
    }
    dKarisim[i] = toplam;
  }

  // c_T = toplam_s a[s] * v_s
  const dAgirlik = new Float32Array(C);
  const dDeger: Float32Array[] = [];
  for (let t = 0; t < C; t++) dDeger.push(new Float32Array(D));
  for (let sVal = 0; sVal <= T; sVal++) {
    let ic = 0;
    const a = di.agirliklar[T][sVal];
    for (let i = 0; i < D; i++) {
      ic += dKarisim[i] * di.deger[sVal][i];
      dDeger[sVal][i] += a * dKarisim[i];
    }
    dAgirlik[sVal] = ic;
  }

  // softmax
  let ortalama = 0;
  for (let sVal = 0; sVal <= T; sVal++) ortalama += di.agirliklar[T][sVal] * dAgirlik[sVal];
  const dSkor = new Float32Array(C);
  for (let sVal = 0; sVal <= T; sVal++) {
    dSkor[sVal] = di.agirliklar[T][sVal] * (dAgirlik[sVal] - ortalama);
  }

  // skor[s] = (q_T · k_s) * olcek
  const dSorgu = new Float32Array(D);
  const dAnahtar: Float32Array[] = [];
  for (let t = 0; t < C; t++) dAnahtar.push(new Float32Array(D));
  for (let sVal = 0; sVal <= T; sVal++) {
    const g = dSkor[sVal] * olcek;
    if (g === 0) continue;
    for (let i = 0; i < D; i++) {
      dSorgu[i] += g * di.anahtar[sVal][i];
      dAnahtar[sVal][i] += g * di.sorgu[T][i];
    }
  }

  // q_T = x_T @ Wq
  for (let i = 0; i < D; i++) {
    const satir = i * D;
    const xi = di.x[T][i];
    let toplam = 0;
    for (let j = 0; j < D; j++) {
      gd.Wq.veri[satir + j] += xi * dSorgu[j];
      toplam += d.Wq.veri[satir + j] * dSorgu[j];
    }
    dx[T][i] += toplam;
  }

  // k_s = x_s @ Wk ve v_s = x_s @ Wv — her konum için
  for (let sVal = 0; sVal < C; sVal++) {
    for (let i = 0; i < D; i++) {
      const satir = i * D;
      const xi = di.x[sVal][i];
      let toplam = 0;
      for (let j = 0; j < D; j++) {
        gd.Wk.veri[satir + j] += xi * dAnahtar[sVal][j];
        gd.Wv.veri[satir + j] += xi * dDeger[sVal][j];
        toplam += d.Wk.veri[satir + j] * dAnahtar[sVal][j] + d.Wv.veri[satir + j] * dDeger[sVal][j];
      }
      dx[sVal][i] += toplam;
    }
  }

  // x_t = E[id_t] + P[t] — toplamın gradyanı iki tarafa da aynen geçer
  for (let t = 0; t < C; t++) {
    const eSatir = iz.baglamIds[t] * D;
    const pSatir = t * D;
    for (let i = 0; i < D; i++) {
      grad.E.veri[eSatir + i] += dx[t][i];
      gP.veri[pSatir + i] += dx[t][i];
    }
  }
}

// ---------------------------------------------------------------------------
// Güncelleme
// ---------------------------------------------------------------------------

/** Gradyanın toplam büyüklüğü (L2 normu). Patlamayı ölçmek için. */
export function gradyanNormu(g: Gradyan): number {
  let kare = 0;
  const tumu = gradyanDizileri(g);
  for (const dizi of tumu) {
    for (let i = 0; i < dizi.length; i++) kare += dizi[i] * dizi[i];
  }
  return Math.sqrt(kare);
}

/** Sıra agirlikDizileri() ile birebir aynı olmak zorunda. */
function gradyanDizileri(g: Gradyan): Float32Array[] {
  const liste: Float32Array[] = [g.E.veri];
  if (g.P) liste.push(g.P.veri);
  if (g.dikkat) liste.push(g.dikkat.Wq.veri, g.dikkat.Wk.veri, g.dikkat.Wv.veri, g.dikkat.Wo.veri);
  if (g.Wgiris) liste.push(g.Wgiris.veri);
  if (g.bgiris) liste.push(g.bgiris);
  for (const b of g.bloklar) liste.push(b.W1.veri, b.b1, b.W2.veri, b.b2);
  liste.push(g.Wcikis.veri, g.bcikis);
  return liste;
}

/**
 * Gradyan inişinin tek adımı:  ağırlık <- ağırlık - öğrenmeOranı * gradyan
 *
 * `olcek` toplu eğitimde 1/örnekSayısı olur (gradyanların ortalaması).
 * `kirpma` gradyan normunu sınırlar: öğrenme oranı yüksek seçildiğinde
 * tek bir kötü örnek modeli uçuruma atmasın diye. Kırpma yapılmazsa
 * kaydıracı 1.0'a çeken kullanıcı kaybın NaN'a gittiğini görür — ki bunu
 * göstermek de öğretici, o yüzden kırpmayı kapatılabilir bıraktık.
 */
export function gradyanUygula(
  model: Model,
  grad: Gradyan,
  ogrenmeOrani: number,
  olcek = 1,
  kirpma = 5,
): number {
  const norm = gradyanNormu(grad) * olcek;
  let katsayi = ogrenmeOrani * olcek;
  if (kirpma > 0 && norm > kirpma) katsayi *= kirpma / norm;

  const gDizi = gradyanDizileri(grad);
  const mDizi = agirlikDizileri(model);
  for (let d = 0; d < mDizi.length; d++) {
    const hedef = mDizi[d];
    const kaynak = gDizi[d];
    for (let i = 0; i < hedef.length; i++) hedef[i] -= katsayi * kaynak[i];
  }
  return norm;
}

// ---------------------------------------------------------------------------
// Veri ve eğitim döngüsü
// ---------------------------------------------------------------------------

export interface Veri {
  /** Metnin tamamı, token id'leri olarak. */
  ids: number[];
  /** Eğitimde kullanılacak aralık [0, ayrim). */
  ayrim: number;
}

/**
 * Metni id'lere çevirir ve sonunu doğrulama için ayırır.
 * Doğrulama kaybını da izliyoruz: metin küçük olduğu için model ezberlemeye
 * başladığında eğitim kaybı düşerken doğrulama kaybı düşmeyi bırakır.
 * Bunu gizlemek yerine göstermek istiyoruz.
 */
export function veriHazirla(metin: string, dogrulamaOrani = 0.1): Veri {
  const ids = encode(metin);
  const ayrim = Math.floor(ids.length * (1 - dogrulamaOrani));
  return { ids, ayrim };
}

export interface EgitimAyarlari {
  ogrenmeOrani: number;
  /** Her adımda kaç örnek üzerinden ortalama alınacak. */
  yigin: number;
  sicaklik: number;
  kirpma: number;
}

export const VARSAYILAN_EGITIM: EgitimAyarlari = {
  ogrenmeOrani: 0.1,
  yigin: 32,
  sicaklik: 1,
  kirpma: 5,
};

export interface Egitici {
  model: Model;
  veri: Veri;
  ayar: EgitimAyarlari;
  grad: Gradyan;
  rnd: () => number;
  adim: number;
  /** Her adımın kaybı; kayıp eğrisi bunu çizer. */
  gecmis: number[];
}

export function egiticiOlustur(
  model: Model,
  veri: Veri,
  ayar: EgitimAyarlari = VARSAYILAN_EGITIM,
  seed = 1234,
): Egitici {
  return {
    model,
    veri,
    ayar,
    grad: gradyanOlustur(model),
    rnd: rastgeleUretec(seed),
    adim: 0,
    gecmis: [],
  };
}

/** Metinden rastgele bir konum seçip (bağlam, hedef) çifti üretir. */
function ornekSec(e: Egitici, baslangic: number, bitis: number): { baglam: number[]; hedef: number } {
  const C = e.model.ayar.baglam;
  const konum = baslangic + Math.floor(e.rnd() * (bitis - baslangic));
  const bas = Math.max(0, konum - C);
  return { baglam: e.veri.ids.slice(bas, konum), hedef: e.veri.ids[konum] };
}

/**
 * Tek bir eğitim adımı: bir yığın örnek üzerinde ileri geçiş + geri yayılım,
 * sonra tek bir güncelleme. Ortalama kaybı döndürür.
 */
export function egitimAdimi(e: Egitici): number {
  gradyanSifirla(e.grad);
  let toplamKayip = 0;
  for (let n = 0; n < e.ayar.yigin; n++) {
    const { baglam, hedef } = ornekSec(e, 1, e.veri.ayrim);
    const iz = ileriGecis(e.model, baglam, 1); // eğitimde sıcaklık her zaman 1
    toplamKayip += kayip(iz.olasilik, hedef);
    geriYayilim(e.model, iz, hedef, e.grad);
  }
  gradyanUygula(e.model, e.grad, e.ayar.ogrenmeOrani, 1 / e.ayar.yigin, e.ayar.kirpma);
  e.adim++;
  const ortalama = toplamKayip / e.ayar.yigin;
  e.gecmis.push(ortalama);
  return ortalama;
}

/**
 * Ayrılmış doğrulama bölümünde kaybı ölçer. Ağırlıklara dokunmaz.
 * Rastgele değil, düzenli aralıklarla örnekler — ölçüm gürültüsü düşük olsun.
 */
export function dogrulamaKaybi(model: Model, veri: Veri, ornekSayisi = 200): number {
  const C = model.ayar.baglam;
  const bas = Math.max(veri.ayrim, 1);
  const son = veri.ids.length;
  if (son - bas < 2) return NaN;
  const adim = Math.max(1, Math.floor((son - bas) / ornekSayisi));
  let toplam = 0;
  let sayi = 0;
  for (let konum = bas; konum < son; konum += adim) {
    const baglam = veri.ids.slice(Math.max(0, konum - C), konum);
    const iz = ileriGecis(model, baglam, 1);
    toplam += kayip(iz.olasilik, veri.ids[konum]);
    sayi++;
  }
  return toplam / sayi;
}
