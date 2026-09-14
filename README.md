# dil-modeli-lab

Bir dil modelinin içeride gerçekte ne yaptığını gösteren, tarayıcıda çalışan öğretici laboratuvar.

## Temel ilke

Bu bir animasyon gösterisi değil. Gerçek sayılar, gerçek matris çarpımları, gerçek gradyan inişi çalışır.
Hiçbir değer sahte değil, hiçbir değer gizli değil — ekranda görünen her sayı doğrulanabilir.

Sunucu yok, API yok, dış model yok. Tüm matematik tarayıcıda saf TypeScript ile çalışır.

## Model

Karakter düzeyinde, minik bir dil modeli:

- **Sözlük:** Türkçe alfabe (ç, ğ, ı, ö, ş, ü dahil) + boşluk, nokta, virgül → ~35 token
- **Gömme boyutu (D):** varsayılan 16, 4–64 arası ayarlanabilir
- **Bağlam:** son 8 karakter
- **Dikkat:** tek başlı nedensel öz-dikkat (açılıp kapatılabilir). Açıkken konumlar birbirine bakabilir; kapalıyken bağlam uç uca eklenip sabit bir projeksiyondan geçer.
- **Eğitim kipi:** dikkat açıkken her konum kendi sonraki karakterini tahmin edebilir — tek ileri geçişten sekiz tahmin, sekiz ayrı gradyan.
- **Katmanlar:** 1–4 MLP bloğu — `x → Linear(D, 4D) → ReLU → Linear(4D, D)` + artık bağlantı
- **Çıkış:** `Linear(D, V)` → softmax

Eğitim tarayıcıda, elle yazılmış saf gradyan inişiyle yapılır (autograd kütüphanesi yok — kod okunabilir olmalı).
Ağır iş Web Worker içinde çalışır, arayüz donmaz.

## Ekranlar

1. **Ağırlıklar** (`/`) — modelin yedi matrisi de ısı haritası olarak. Her hücreye tıklayıp elle değiştirebilirsiniz; üstteki tahmin anında bozulur. Uzun ve dar matrisler okunur kalsın diye devrik gösterilir.
2. **İleri geçiş** (`/ileri-gecis`) — bir tahmine giden aşamalar tek tek (dikkat açıkken 17, kapalıyken 13). Matris çarpımı hücre hücre canlandırılır: hangi giriş hangi ağırlıkla çarpılıyor, ara toplam nereye gidiyor.
3. **Gömme küresi** (`/kure`) — her karakter küre üzerinde bir nokta, benzerlik aradaki açı. İki nokta seçince gerçek 16 boyutlu kosinüs benzerliği ve küredeki görünen açı yan yana yazılır.
4. **Eğitim** (`/egitim`) — Web Worker'da eğitim, canlı kayıp eğrisi, bütün kaydıraçlar, eşzamanlı güncellenen küre ve ızgaralar, üretim paneli.
5. **TTT modu** (`/ttt`) — donmuş model ile çıkarım anında son projeksiyon matrisini güncelleyen modelin aynı metin üzerindeki karşılaştırması.

## Teknik

Astro 5 + React adaları + TypeScript + Tailwind. Izgaralar ve grafikler D3, küre Three.js.
Tamamen statik build. Durum URL'de saklanır (seed, D, katman sayısı, sıcaklık), böylece link paylaşılabilir.

## Kod düzeni

```
src/lib/model.ts      — mimari, ileri geçiş
src/lib/train.ts      — gradyanlar, eğitim döngüsü
src/lib/tokenizer.ts  — karakter sözlüğü
src/workers/          — eğitim worker'ı
src/components/       — Sphere, WeightGrid, ForwardPass, TrainPanel, TTTCompare
```

`model.ts` ve `train.ts` bol yorumlu; matematiği açıkça yazar. Bu dosyalar uygulamanın kendisi kadar öğretici olmalı.

## Çalıştırma

```
npm install
npm run dev        # geliştirme sunucusu
npm run build      # statik build
npm run cekirdek   # arayüzsüz doğrulama betiği
npm run kontrol    # astro check + tsc
```

## Doğrulama

`npm run cekirdek` arayüze hiç dokunmadan şunları ölçer ve yazdırır:

- **Gradyanlar doğru mu?** Elle yazılan her türev merkezi sayısal türevle karşılaştırılır. En kötü bağıl fark ~9e-6.
- **Eğitim öğreniyor mu?** 3000 adımda kayıp 3.25'ten 1.58'e iner (rastgele modelin kaybı ln(32) = 3.47).
- **Çarpım animasyonu modelle aynı sayıyı mı üretiyor?** 208 çıkışın tamamı, Float32 yuvarlaması dahil, birebir aynı.
- **PCA doğru mu?** Bileşenler birim uzunlukta ve dik (iç çarpımlar ~1e-13), izdüşüm iki çalıştırmada birebir aynı.
- **Eğitim gerçekten yapı kuruyor mu?** Sesli-sesli ile sesli-sessiz ortalama benzerlik farkı eğitimden önce −0.062, sonra +0.203. Model sesli/sessiz ayrımını hiç görmedi, sadece metni okudu.
- **Dikkat ne katıyor?** Aynı tohum, aynı ayarlar, aynı eğitim kipiyle: dikkatsiz modelin doğrulama kaybı 2.475, dikkatli modelinki 2.400 — küçük ama tutarlı bir fark, üstelik dikkatli model 912 parametre daha az kullanıyor. Dikkat dağılımının ortalama entropisi eğitimle 2.078'den 1.649'a iniyor (düz dağılım ln(8) = 2.079), yani eğitim dikkati keskinleştiriyor.
- **Her konum tahmin ederse?** Aynı adım sayısında doğrulama kaybı 0.050 daha iyi, ama bir adım 4.4 kat pahalı. Aynı sürede ölçünce fark (0.022) ölçümün kendi gürültüsünün (0.035) altında kalıyor: örnek başına daha verimli, hesap başına başabaş.
- **TTT kazandırıyor mu?** Tekrar eden yeni bir metinde evet (ortalama kayıp 2.50 → 2.03); tek turluk ya da modelin zaten bildiği metinde hayır. Öğrenme oranı büyütüldükçe önce kazanç artar, sonra model kendini bozar.

## Lisans

MIT
