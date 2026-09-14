/**
 * TEST ANINDA EĞİTİM (TTT)
 * ========================
 *
 * Bugünkü yaygın kurulumda bir model eğitilir, ağırlıkları dondurulur ve
 * ondan sonra ne kadar metin görürse görsün bir daha değişmez. Okuduğu
 * şeyden öğrenmez; sadece bağlam penceresinde tutabildiği kadarını taşır.
 *
 * TTT bunu kısmen gevşetir: parametrelerin küçük bir bölümü çıkarım
 * sırasında da güncellenmeye devam eder. Burada güncellenen kısım son
 * projeksiyon matrisi — Wçıkış ve yanlılığı. Geri kalan her şey (gömmeler,
 * giriş projeksiyonu, bütün MLP blokları) donmuş kalır.
 *
 * GÜNCELLEME NEREDEN GELİYOR
 * Metni soldan sağa okurken her konumda bir sonraki karakteri tahmin
 * ediyoruz. Bir sonraki karaktere geçtiğimizde doğru cevabı öğrenmiş
 * oluyoruz — ve bu, ek bir etiket ya da insan emeği olmadan gradyan
 * hesaplamak için yeterli. Yani model kendi okuduğu metinden öğreniyor.
 *
 * AŞAĞIDAKİ KOD GERİ YAYILIMIN SON HALKASI
 * train.ts'teki tam geri yayılımın ilk birkaç satırıyla aynı işi yapıyor;
 * sadece zincir kuralını daha geriye taşımıyor, çünkü daha geride
 * güncellenecek bir şey yok. Tam geri yayılımı çağırıp gradyanların
 * çoğunu çöpe atmak da olurdu, ama o zaman "sadece şu matris değişiyor"
 * fikri kodda görünmezdi.
 */

import {
  type IleriIz,
  type Model,
  agirlikKopyasi,
  agirliklariYaz,
  modelOlustur,
} from "./model.ts";

/**
 * Tek bir çıkarım anı güncellemesi. Yalnızca Wçıkış ve bçıkış oynar.
 * Uygulanan değişikliğin büyüklüğünü (L2 normu) döndürür.
 */
export function hizliAgirlikAdimi(
  model: Model,
  iz: IleriIz,
  hedef: number,
  ogrenmeOrani: number,
): number {
  const V = iz.logits.length;
  const D = model.ayar.D;

  // Softmax + çapraz entropinin türevi: olasılık eksi olması gereken.
  const dlogits = new Float32Array(V);
  for (let j = 0; j < V; j++) dlogits[j] = iz.olasilik[j];
  dlogits[hedef] -= 1;

  // Çıkış katmanının girdisi: son bloğun çıktısı.
  const sonH = iz.bloklar.length > 0 ? iz.bloklar[iz.bloklar.length - 1].cikti : iz.h0;

  let kareToplam = 0;
  for (let i = 0; i < D; i++) {
    const satir = i * V;
    const hi = sonH[i];
    for (let j = 0; j < V; j++) {
      const degisim = ogrenmeOrani * hi * dlogits[j];
      model.Wcikis.veri[satir + j] -= degisim;
      kareToplam += degisim * degisim;
    }
  }
  for (let j = 0; j < V; j++) {
    const degisim = ogrenmeOrani * dlogits[j];
    model.bcikis[j] -= degisim;
    kareToplam += degisim * degisim;
  }

  return Math.sqrt(kareToplam);
}

/** Aynı ayarlarla, aynı ağırlıklarla bağımsız bir kopya. */
export function modelKopyala(model: Model): Model {
  const kopya = modelOlustur(model.ayar);
  agirliklariYaz(kopya, agirlikKopyasi(model));
  return kopya;
}

/** Çıkarım sırasında güncellenen parametre sayısı (hızlı ağırlıklar). */
export function hizliParametreSayisi(model: Model): number {
  return model.Wcikis.veri.length + model.bcikis.length;
}

/** İki modelin çıkış katmanı ne kadar ayrıştı — TTT'nin ne kadar biriktiği. */
export function cikisFarki(a: Model, b: Model): number {
  let kare = 0;
  for (let i = 0; i < a.Wcikis.veri.length; i++) {
    const d = a.Wcikis.veri[i] - b.Wcikis.veri[i];
    kare += d * d;
  }
  for (let i = 0; i < a.bcikis.length; i++) {
    const d = a.bcikis[i] - b.bcikis[i];
    kare += d * d;
  }
  return Math.sqrt(kare);
}
