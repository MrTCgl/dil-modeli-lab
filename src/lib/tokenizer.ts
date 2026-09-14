/**
 * KARAKTER DÜZEYİNDE SÖZLÜK
 * =========================
 *
 * Bu model kelimelerle değil, tek tek harflerle çalışır. Neden?
 * Çünkü kullanıcı istediği cümleyi yazabilsin istiyoruz. Kelime sözlüğü
 * kullansaydık, sözlükte olmayan bir kelime yazıldığında model çaresiz
 * kalırdı. Harf düzeyinde ise her Türkçe cümle temsil edilebilir.
 *
 * Sözlüğümüz 32 simgeden oluşur: 29 Türkçe harf + boşluk + nokta + virgül.
 * Küçük bir sözlük, modelin çıkış katmanını da küçük tutar; bu da her
 * olasılığı ekranda yan yana gösterebilmemizi sağlar.
 */

/** Türk alfabesi, sıralı: 29 harf. */
export const HARFLER = "abcçdefgğhıijklmnoöprsştuüvyz";

/** Modelin tanıdığı noktalama: boşluk, nokta, virgül. */
export const NOKTALAMA = " .,";

/** Sözlüğün tamamı. Dizideki sıra = token id. */
export const ALFABE: readonly string[] = [...HARFLER, ...NOKTALAMA];

/** Sözlük boyutu (V). Modelin çıkış katmanı bu kadar skor üretir. */
export const SOZLUK_BOYUTU = ALFABE.length;

/** Harf -> id araması için tersine tablo. */
const ID_TABLOSU = new Map<string, number>(ALFABE.map((ch, i) => [ch, i]));

/**
 * Türkçeye özgü küçültme. JavaScript'in varsayılan toLowerCase'i
 * "I" harfini "i" yapar; Türkçede doğrusu "ı"dır. Aynı şekilde "İ" -> "i".
 */
function kucult(metin: string): string {
  return metin.replace(/I/g, "ı").replace(/İ/g, "i").toLowerCase();
}

/**
 * Serbest metni sözlüğe sığdırır:
 *   - küçük harfe çevirir (Türkçe kurallarıyla)
 *   - şapkalı harfleri sadeleştirir (â -> a)
 *   - sözlükte olmayan noktalamayı en yakın karşılığına eşler
 *     (! ? ; : -> nokta ya da virgül), kalanını atar
 *   - ardışık boşlukları teke indirir
 *
 * Normalleştirme kayıplıdır ve bunu kullanıcıdan gizlemiyoruz: arayüz
 * yazılan metnin modele hangi biçimde girdiğini olduğu gibi gösterir.
 */
export function normalize(metin: string): string {
  let s = kucult(metin);

  // şapkalı ve yabancı harfler -> en yakın Türkçe karşılık
  const esler: Record<string, string> = {
    â: "a", ä: "a", à: "a", á: "a",
    î: "i", ï: "i", í: "i",
    û: "u", ù: "u", ú: "u",
    ê: "e", é: "e", è: "e",
    ô: "o", ó: "o",
    q: "k", w: "v", x: "ks",
  };
  s = s.replace(/./gu, (ch) => esler[ch] ?? ch);

  // cümle sonu sayılan noktalama -> nokta; ayırıcılar -> virgül
  s = s.replace(/[!?…]/g, ".").replace(/[;:]/g, ",");

  // satır sonları ve tireler -> boşluk
  s = s.replace(/[\n\r\t\-–—]/g, " ");

  // sözlükte olmayan her şeyi at
  s = s.replace(/./gu, (ch) => (ID_TABLOSU.has(ch) ? ch : ""));

  // ardışık boşlukları teke indir
  return s.replace(/ {2,}/g, " ");
}

/** Tek bir karakterin id'si. Sözlükte yoksa -1. */
export function karakterId(ch: string): number {
  return ID_TABLOSU.get(ch) ?? -1;
}

/** id -> karakter. */
export function idKarakter(id: number): string {
  return ALFABE[id] ?? "";
}

/**
 * Metni token id dizisine çevirir. Normalleştirme burada uygulanır,
 * yani encode(x) her zaman geçerli id'ler döndürür.
 */
export function encode(metin: string): number[] {
  const temiz = normalize(metin);
  const ids: number[] = [];
  for (const ch of temiz) {
    const id = karakterId(ch);
    if (id >= 0) ids.push(id);
  }
  return ids;
}

/** id dizisini metne çevirir. */
export function decode(ids: readonly number[]): string {
  let s = "";
  for (const id of ids) s += idKarakter(id);
  return s;
}

/** Ekranda göstermek için: boşluk görünmez olduğundan ona bir ad veririz. */
export function gorunurAd(id: number): string {
  const ch = idKarakter(id);
  if (ch === " ") return "␣";
  return ch;
}
