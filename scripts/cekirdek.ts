/**
 * ÇEKİRDEK DOĞRULAMA BETİĞİ
 * =========================
 * Arayüz yok. Bu betik üç şeyi kanıtlar:
 *   1) Sözlük Türkçeyi temsil ediyor ve encode/decode tutarlı.
 *   2) Elle yazılan gradyanlar doğru — sayısal türevle karşılaştırılıyor.
 *   3) Eğitim gerçekten öğreniyor — kayıp düşüyor, üretilen metin değişiyor.
 *
 * Çalıştırmak için:  npm run cekirdek
 */

import { ALFABE, SOZLUK_BOYUTU, decode, encode, normalize } from "../src/lib/tokenizer.ts";
import {
  RASTGELE_KAYIP,
  type Model,
  ileriGecis,
  kayip,
  modelOlustur,
  parametreSayisi,
  rastgeleUretec,
  uret,
} from "../src/lib/model.ts";
import {
  type Gradyan,
  dogrulamaKaybi,
  egiticiOlustur,
  egitimAdimi,
  geriYayilim,
  gradyanOlustur,
  veriHazirla,
} from "../src/lib/train.ts";
import { carpimKonumu, carpimSonucu } from "../src/lib/carpim.ts";
import { aciDerece, birimleHale, izdusumCikar, kosinus } from "../src/lib/pca.ts";
import { EGITIM_METNI } from "../src/data/metin.ts";

const cizgi = (baslik: string) => console.log(`\n${"─".repeat(64)}\n${baslik}\n${"─".repeat(64)}`);
const say = (x: number, basamak = 4) => x.toFixed(basamak).padStart(basamak + 4);

// ---------------------------------------------------------------------------
cizgi("1. SÖZLÜK");

console.log(`Sözlük boyutu (V): ${SOZLUK_BOYUTU}`);
console.log(`Alfabe: ${ALFABE.join("")}`);

const ornekMetin = "Çınarın altında çay içtik, hava ılıktı!";
const temiz = normalize(ornekMetin);
const ids = encode(ornekMetin);
console.log(`\nGirdi      : ${ornekMetin}`);
console.log(`Normalize  : ${temiz}`);
console.log(`Token id'ler: ${ids.slice(0, 12).join(" ")} ...  (toplam ${ids.length})`);
console.log(`decode(encode(x)) === normalize(x) ? ${decode(ids) === temiz ? "EVET" : "HAYIR"}`);

const veri = veriHazirla(EGITIM_METNI);
console.log(`\nEğitim metni: ${veri.ids.length} token (${(EGITIM_METNI.length / 1024).toFixed(1)} KB ham)`);
console.log(`Eğitim/doğrulama ayrımı: ${veri.ayrim} / ${veri.ids.length - veri.ayrim}`);

// ---------------------------------------------------------------------------
cizgi("2. GRADYAN DOĞRULAMASI (sayısal türev ile karşılaştırma)");

/**
 * Analitik gradyanın doğruluğunu bağımsız biçimde sınar.
 * Yöntem: bir ağırlığı epsilon kadar artır, epsilon kadar azalt, kaybın
 * farkına bak. Bu merkezi fark, türevin tanımının ta kendisidir:
 *     dL/dw ≈ (L(w+e) - L(w-e)) / (2e)
 * Geri yayılım doğruysa iki sayı birbirini tutmalı.
 *
 * Not: ağırlıklar Float32 tutuluyor, bu yüzden sayısal türev kaçınılmaz
 * olarak gürültülüdür. Beklenen bağıl hata 1e-2 mertebesinde; asıl aranan
 * şey işaretin ve büyüklüğün tutması.
 */
function gradyanSina(model: Model, baglam: number[], hedef: number): void {
  const grad: Gradyan = gradyanOlustur(model);
  const iz = ileriGecis(model, baglam, 1);
  geriYayilim(model, iz, hedef, grad);

  const adaylar: Array<{ ad: string; agirlik: Float32Array; gradyan: Float32Array }> = [
    { ad: "E", agirlik: model.E.veri, gradyan: grad.E.veri },
    { ad: "Wgiriş", agirlik: model.Wgiris.veri, gradyan: grad.Wgiris.veri },
    { ad: "W1 (katman 1)", agirlik: model.bloklar[0].W1.veri, gradyan: grad.bloklar[0].W1.veri },
    { ad: "W2 (katman 1)", agirlik: model.bloklar[0].W2.veri, gradyan: grad.bloklar[0].W2.veri },
    { ad: "W1 (katman 2)", agirlik: model.bloklar[1].W1.veri, gradyan: grad.bloklar[1].W1.veri },
    { ad: "Wçıkış", agirlik: model.Wcikis.veri, gradyan: grad.Wcikis.veri },
    { ad: "bçıkış", agirlik: model.bcikis, gradyan: grad.bcikis },
  ];

  const eps = 1e-2;
  console.log("matris           indeks   analitik      sayısal    bağıl fark");
  let enKotu = 0;
  for (const aday of adaylar) {
    // Gradyanı sıfıra çok yakın olmayan bir hücre seç: sıfıra bölmeyelim.
    let indeks = 0;
    let enBuyuk = 0;
    for (let i = 0; i < aday.gradyan.length; i++) {
      const v = Math.abs(aday.gradyan[i]);
      if (v > enBuyuk) {
        enBuyuk = v;
        indeks = i;
      }
    }
    const analitik = aday.gradyan[indeks];
    const eski = aday.agirlik[indeks];

    aday.agirlik[indeks] = eski + eps;
    const artiKayip = kayip(ileriGecis(model, baglam, 1).olasilik, hedef);
    aday.agirlik[indeks] = eski - eps;
    const eksiKayip = kayip(ileriGecis(model, baglam, 1).olasilik, hedef);
    aday.agirlik[indeks] = eski;

    const sayisal = (artiKayip - eksiKayip) / (2 * eps);
    const fark = Math.abs(analitik - sayisal) / Math.max(1e-8, Math.abs(analitik) + Math.abs(sayisal));
    enKotu = Math.max(enKotu, fark);
    console.log(
      `${aday.ad.padEnd(16)} ${String(indeks).padStart(6)}  ${say(analitik, 6)}  ${say(sayisal, 6)}  ${fark.toExponential(2)}`,
    );
  }
  console.log(`\nEn kötü bağıl fark: ${enKotu.toExponential(2)} → ${enKotu < 0.02 ? "GEÇTİ" : "KALDI"}`);
}

const sinaModel = modelOlustur({ D: 16, katmanSayisi: 2, baglam: 8, seed: 7 });
console.log(`Model: D=16, 2 katman, bağlam=8 — ${parametreSayisi(sinaModel).toLocaleString("tr-TR")} parametre\n`);
gradyanSina(sinaModel, veri.ids.slice(100, 108), veri.ids[108]);

// ---------------------------------------------------------------------------
cizgi("3. EĞİTİM");

const model = modelOlustur({ D: 16, katmanSayisi: 2, baglam: 8, seed: 1 });
const rnd = rastgeleUretec(42);
const baslangic = encode("deniz ");

console.log(`Rastgele modelin beklenen kaybı: ln(V) = ${RASTGELE_KAYIP.toFixed(4)}`);
console.log(`\nEğitimden ÖNCE üretilen metin:`);
console.log(`  "${decode(baslangic)}${decode(uret(model, baslangic, 120, 0.8, rnd))}"`);

const egitici = egiticiOlustur(model, veri, { ogrenmeOrani: 0.2, yigin: 32, sicaklik: 1, kirpma: 5 });

console.log(`\n adım    eğitim kaybı   doğrulama    süre`);
const basla = Date.now();
const ADIM = 3000;
for (let i = 1; i <= ADIM; i++) {
  const k = egitimAdimi(egitici);
  if (i === 1 || i % 250 === 0) {
    const dogrulama = dogrulamaKaybi(model, veri, 150);
    console.log(`${String(i).padStart(5)}    ${say(k)}        ${say(dogrulama)}    ${((Date.now() - basla) / 1000).toFixed(1)}s`);
  }
}

const sonKayip = egitici.gecmis.slice(-50).reduce((a, b) => a + b, 0) / 50;
const ilkKayip = egitici.gecmis.slice(0, 50).reduce((a, b) => a + b, 0) / 50;
console.log(`\nİlk 50 adımın ortalaması : ${ilkKayip.toFixed(4)}`);
console.log(`Son 50 adımın ortalaması : ${sonKayip.toFixed(4)}`);
console.log(`Düşüş: ${(ilkKayip - sonKayip).toFixed(4)} → ${sonKayip < ilkKayip - 0.3 ? "ÖĞRENİYOR" : "ÖĞRENMİYOR"}`);

console.log(`\nEğitimden SONRA üretilen metin (sıcaklık 0.8):`);
console.log(`  "${decode(baslangic)}${decode(uret(model, baslangic, 200, 0.8, rnd))}"`);
console.log(`\nDüşük sıcaklık (0.4) — daha kararlı, daha tekrarlı:`);
console.log(`  "${decode(baslangic)}${decode(uret(model, baslangic, 120, 0.4, rnd))}"`);
console.log(`\nYüksek sıcaklık (1.5) — daha dağınık:`);
console.log(`  "${decode(baslangic)}${decode(uret(model, baslangic, 120, 1.5, rnd))}"`);

// ---------------------------------------------------------------------------
cizgi("4. TEK BİR TAHMİN (arayüzün 3. ekranında gösterilecek olan)");

const cumle = encode("bugün hava çok güz");
const iz = ileriGecis(model, cumle, 1);
console.log(`Bağlam (son 8): "${decode(iz.baglamIds)}"`);
console.log(`h0 (ilk 8 sayı): ${Array.from(iz.h0.slice(0, 8)).map((v) => v.toFixed(3)).join("  ")}`);
const sondurulen = iz.bloklar[0].relu.reduce((a, v) => a + (v === 0 ? 1 : 0), 0);
console.log(`Katman 1 ReLU: ${sondurulen}/${iz.bloklar[0].relu.length} nöron sıfırlandı`);

const sirali = Array.from(iz.olasilik)
  .map((p, i) => ({ p, i }))
  .sort((a, b) => b.p - a.p)
  .slice(0, 5);
console.log(`\nEn olası 5 karakter:`);
for (const { p, i } of sirali) {
  const ad = ALFABE[i] === " " ? "␣" : ALFABE[i];
  const cubuk = "█".repeat(Math.round(p * 40));
  console.log(`  ${ad}  ${(p * 100).toFixed(1).padStart(5)}%  ${cubuk}`);
}
console.log();

// ---------------------------------------------------------------------------
cizgi("5. ÇARPIM ANİMASYONU MODELLE AYNI SAYIYI ÜRETİYOR MU?");

/**
 * İleri geçiş ekranı matris çarpımını terim terim canlandırıyor. Bu
 * animasyonun "gösteri" olmadığını, modelin yaptığı işin aynısı olduğunu
 * kanıtlamak lazım: her çıkış için terimleri tek tek toplayıp sonucu modelin
 * ürettiği değerle karşılaştırıyoruz. Float32 yuvarlaması dahil, birebir
 * eşit olmalı — yaklaşık değil, eşit.
 */
{
  const iz2 = ileriGecis(model, encode("kasabada deniz"), 1);
  const sinamalar: Array<{ ad: string; x: Float32Array; W: typeof model.Wgiris; b: Float32Array; y: Float32Array }> = [
    { ad: "birleşik @ Wgiriş", x: iz2.birlesik, W: model.Wgiris, b: model.bgiris, y: iz2.h0 },
  ];
  model.bloklar.forEach((blok, l) => {
    sinamalar.push({ ad: `blok ${l + 1}: h @ W1`, x: iz2.bloklar[l].girdi, W: blok.W1, b: blok.b1, y: iz2.bloklar[l].oncesi });
    sinamalar.push({ ad: `blok ${l + 1}: relu @ W2`, x: iz2.bloklar[l].relu, W: blok.W2, b: blok.b2, y: iz2.bloklar[l].dal });
  });
  const sonH = iz2.bloklar[iz2.bloklar.length - 1].cikti;
  sinamalar.push({ ad: "h @ Wçıkış", x: sonH, W: model.Wcikis, b: model.bcikis, y: iz2.logits });

  let toplamCikis = 0;
  let hataliCikis = 0;
  let enBuyukFark = 0;
  for (const s of sinamalar) {
    let farkli = 0;
    let enKotu = 0;
    for (let j = 0; j < s.W.sutun; j++) {
      const animasyon = carpimSonucu(s.x, s.W, s.b, j);
      const modelDegeri = s.y[j];
      const fark = Math.abs(animasyon - modelDegeri);
      if (fark > 0) farkli++;
      enKotu = Math.max(enKotu, fark);
      toplamCikis++;
    }
    hataliCikis += farkli;
    enBuyukFark = Math.max(enBuyukFark, enKotu);
    console.log(
      `${s.ad.padEnd(22)} ${String(s.W.sutun).padStart(3)} çıkış · ${String(s.W.satir * s.W.sutun).padStart(5)} çarpma · ` +
        `ayrılan ${farkli} · en büyük fark ${enKotu === 0 ? "0" : enKotu.toExponential(2)}`,
    );
  }
  console.log(
    `\nToplam ${toplamCikis} çıkışın ${toplamCikis - hataliCikis} tanesi birebir aynı → ` +
      `${hataliCikis === 0 ? "GEÇTİ" : "KALDI"}`,
  );

  // Ara toplamlar da tutarlı ilerlemeli: son terimden sonraki değer sonuca eşit.
  const W = model.Wgiris;
  const sonAdim = W.satir * W.sutun - 1;
  const sonKonum = carpimKonumu(iz2.birlesik, W, model.bgiris, sonAdim);
  console.log(
    `Son terimden sonra y[${sonKonum.j}] = ${sonKonum.sonrakiToplam.toFixed(6)} · ` +
      `model: ${iz2.h0[sonKonum.j].toFixed(6)} → ${sonKonum.sonrakiToplam === iz2.h0[sonKonum.j] ? "AYNI" : "FARKLI"}`,
  );
}
console.log();

// ---------------------------------------------------------------------------
cizgi("6. GÖMME KÜRESİ: PCA VE ÖĞRENİLEN YAPI");

{
  const izdusum = izdusumCikar(model.E);

  // Bileşenler birbirine dik ve birim uzunlukta olmalı — güç yinelemesinin
  // doğru çalıştığının ölçüsü bu.
  const b = izdusum.bilesenler;
  const boylar = b.map((v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0)));
  const dikligi = (i: number, j: number) => Math.abs(b[i].reduce((a, x, k) => a + x * b[j][k], 0));
  console.log(`bileşen uzunlukları : ${boylar.map((x) => x.toFixed(6)).join("  ")}`);
  console.log(
    `diklik (iç çarpım)  : 1·2 ${dikligi(0, 1).toExponential(2)}  1·3 ${dikligi(0, 2).toExponential(2)}  2·3 ${dikligi(1, 2).toExponential(2)}`,
  );
  console.log(`ilk 3 bileşenin taşıdığı değişkenlik: ${(izdusum.aciklananOran * 100).toFixed(1)}%`);

  const kalanlar = izdusum.noktalar.map((n) => n.kalan);
  const ortKalan = kalanlar.reduce((a, x) => a + x, 0) / kalanlar.length;
  console.log(
    `izdüşümde hayatta kalan oran: ortalama ${(ortKalan * 100).toFixed(1)}% · ` +
      `en düşük ${(Math.min(...kalanlar) * 100).toFixed(1)}% · en yüksek ${(Math.max(...kalanlar) * 100).toFixed(1)}%`,
  );

  // Aynı girdi iki kez çalıştırıldığında aynı izdüşüm çıkmalı (link paylaşılabilir olmalı).
  const tekrar = izdusumCikar(model.E);
  const enBuyukSapma = Math.max(
    ...izdusum.noktalar.map((n, i) =>
      Math.max(
        Math.abs(n.x - tekrar.noktalar[i].x),
        Math.abs(n.y - tekrar.noktalar[i].y),
        Math.abs(n.z - tekrar.noktalar[i].z),
      ),
    ),
  );
  console.log(`iki çalıştırma arasındaki en büyük sapma: ${enBuyukSapma.toExponential(2)} → ${enBuyukSapma < 1e-9 ? "KARARLI" : "KARARSIZ"}`);

  // Eğitim sesli harfleri birbirine yaklaştırdı mı?
  const SESLI = "aeıioöuü";
  const sesliMi = (id: number) => SESLI.includes(ALFABE[id]);
  const harfIds = ALFABE.map((_, i) => i).filter((i) => /\p{L}/u.test(ALFABE[i]));

  function ortalamaBenzerlik(birimler: Float64Array[], ayni: boolean): number {
    let toplam = 0;
    let sayi = 0;
    for (const a of harfIds) {
      for (const bId of harfIds) {
        if (a >= bId) continue;
        const ikisiDeSesli = sesliMi(a) && sesliMi(bId);
        const ikisiDeSessiz = !sesliMi(a) && !sesliMi(bId);
        const eslesme = ayni ? ikisiDeSesli : sesliMi(a) !== sesliMi(bId);
        if (!eslesme) continue;
        void ikisiDeSessiz;
        toplam += kosinus(birimler[a], birimler[bId]);
        sayi++;
      }
    }
    return sayi === 0 ? NaN : toplam / sayi;
  }

  const egitilmemis = modelOlustur({ D: 16, katmanSayisi: 2, baglam: 8, seed: 1 });
  const oncekiBirimler = birimleHale(egitilmemis.E);
  const sonrakiBirimler = izdusum.birimler;

  const oncesiSesli = ortalamaBenzerlik(oncekiBirimler, true);
  const oncesiKarisik = ortalamaBenzerlik(oncekiBirimler, false);
  const sonrasiSesli = ortalamaBenzerlik(sonrakiBirimler, true);
  const sonrasiKarisik = ortalamaBenzerlik(sonrakiBirimler, false);

  console.log(`\n                       sesli-sesli   sesli-sessiz   fark`);
  console.log(
    `eğitimden önce        ${say(oncesiSesli)}       ${say(oncesiKarisik)}    ${say(oncesiSesli - oncesiKarisik)}`,
  );
  console.log(
    `eğitimden sonra       ${say(sonrasiSesli)}       ${say(sonrasiKarisik)}    ${say(sonrasiSesli - sonrasiKarisik)}`,
  );
  console.log(
    `\nSesli harfler birbirine ${sonrasiSesli - sonrasiKarisik > oncesiSesli - oncesiKarisik ? "YAKLAŞTI" : "yaklaşmadı"}` +
      ` — model sesli/sessiz ayrımını hiç görmedi, sadece metni okudu.`,
  );

  // Örnek bir çift
  const a = ALFABE.indexOf("a");
  const e = ALFABE.indexOf("e");
  const k = ALFABE.indexOf("k");
  console.log(
    `\nörnek: a–e kosinüs ${say(kosinus(sonrakiBirimler[a], sonrakiBirimler[e]))} (${aciDerece(kosinus(sonrakiBirimler[a], sonrakiBirimler[e])).toFixed(1)}°) · ` +
      `a–k ${say(kosinus(sonrakiBirimler[a], sonrakiBirimler[k]))} (${aciDerece(kosinus(sonrakiBirimler[a], sonrakiBirimler[k])).toFixed(1)}°)`,
  );
}
console.log();
