/// <reference lib="webworker" />
/**
 * EĞİTİM WORKER'I
 * ===============
 *
 * Eğitim burada, ayrı bir iş parçacığında döner. Sebebi basit: tek bir
 * eğitim adımı otuz iki örnek üzerinde ileri geçiş + geri yayılım demek ve
 * saniyede yüzlercesi atılıyor. Bunu ana iş parçacığında yapmak sayfayı
 * dondurur; kaydıraç oynatılamaz, küre döndürülemez, düğmeye basılamaz.
 *
 * Worker modelin sahibidir. Ekran modelin bir AYNASINI tutar ve worker
 * belirli aralıklarla ağırlıkların kopyasını gönderir. İki taraf aynı
 * belleği paylaşmıyor (SharedArrayBuffer güvenlik başlıkları istiyor,
 * statik barındırmada yok), ama 30 KB'lık bir kopyayı saniyede on beş kez
 * göndermek hiçbir maliyet değil.
 *
 * Eğitim döngüsü setTimeout(0) ile parçalara bölünüyor. Kesintisiz bir
 * while döngüsü worker'ı da kilitler ve "dur" mesajı hiç işlenemezdi.
 */

import {
  type ModelAyarlari,
  type Model,
  agirlikKopyasi,
  modelOlustur,
  parametreSayisi,
} from "../lib/model.ts";
import {
  type Egitici,
  type EgitimAyarlari,
  dogrulamaKaybi,
  egiticiOlustur,
  egitimAdimi,
  veriHazirla,
} from "../lib/train.ts";
import { EGITIM_METNI } from "../data/metin.ts";

/** Bir parçada en fazla ne kadar hesap yapılsın (ms). */
const PARCA_SURESI = 24;
/** Ağırlık kopyası en sık hangi aralıkla gönderilsin (ms). */
const AGIRLIK_ARALIGI = 70;
/** Doğrulama kaybı kaç adımda bir ölçülsün. */
const DOGRULAMA_ARALIGI = 40;

export type EgitimIstegi =
  | { tur: "kur"; ayar: ModelAyarlari; egitim: EgitimAyarlari }
  | { tur: "basla"; adimSayisi: number }
  | { tur: "dur" }
  | { tur: "egitimAyari"; egitim: Partial<EgitimAyarlari> };

export type EgitimCevabi =
  | { tur: "hazir"; agirliklar: Float32Array[]; parametre: number; tokenSayisi: number }
  | {
      tur: "ilerleme";
      adim: number;
      /** Bu parçada atılan adımların kayıpları — eğri bunlardan çiziliyor. */
      kayiplar: Float32Array;
      dogrulama: number | null;
      agirliklar: Float32Array[] | null;
      calisiyor: boolean;
    };

const veri = veriHazirla(EGITIM_METNI);
let model: Model | null = null;
let egitici: Egitici | null = null;
let calisiyor = false;
let hedefAdim = 0;
let sonAgirlikZamani = 0;
let sonDogrulama: number | null = null;

function gonder(mesaj: EgitimCevabi) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(mesaj);
}

function kur(ayar: ModelAyarlari, egitimAyar: EgitimAyarlari) {
  model = modelOlustur(ayar);
  egitici = egiticiOlustur(model, veri, egitimAyar, ayar.seed * 7919 + 13);
  calisiyor = false;
  hedefAdim = 0;
  sonDogrulama = null;
  gonder({
    tur: "hazir",
    agirliklar: agirlikKopyasi(model),
    parametre: parametreSayisi(model),
    tokenSayisi: veri.ids.length,
  });
}

function parca() {
  if (!calisiyor || !egitici || !model) return;

  const basla = performance.now();
  const kayiplar: number[] = [];
  while (performance.now() - basla < PARCA_SURESI && egitici.adim < hedefAdim) {
    kayiplar.push(egitimAdimi(egitici));
    if (egitici.adim % DOGRULAMA_ARALIGI === 0) {
      sonDogrulama = dogrulamaKaybi(model, veri, 120);
    }
  }

  if (egitici.adim >= hedefAdim) calisiyor = false;

  const simdi = performance.now();
  const agirlikZamani = simdi - sonAgirlikZamani > AGIRLIK_ARALIGI || !calisiyor;
  if (agirlikZamani) sonAgirlikZamani = simdi;

  gonder({
    tur: "ilerleme",
    adim: egitici.adim,
    kayiplar: new Float32Array(kayiplar),
    dogrulama: sonDogrulama,
    agirliklar: agirlikZamani ? agirlikKopyasi(model) : null,
    calisiyor,
  });

  if (calisiyor) setTimeout(parca, 0);
}

self.onmessage = (olay: MessageEvent<EgitimIstegi>) => {
  const istek = olay.data;
  switch (istek.tur) {
    case "kur":
      kur(istek.ayar, istek.egitim);
      break;

    case "basla":
      if (!egitici) return;
      hedefAdim = egitici.adim + istek.adimSayisi;
      if (!calisiyor) {
        calisiyor = true;
        sonAgirlikZamani = 0;
        setTimeout(parca, 0);
      }
      break;

    case "dur":
      calisiyor = false;
      break;

    case "egitimAyari":
      // Ayarlar eğitim sürerken de değişebilir: öğrenme oranını canlı
      // oynatıp kaybın tepkisini görmek bu uygulamanın amacına uygun.
      if (egitici) Object.assign(egitici.ayar, istek.egitim);
      break;
  }
};
