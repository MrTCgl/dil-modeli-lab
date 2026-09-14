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
- **Katmanlar:** 1–4 MLP bloğu — `x → Linear(D, 4D) → ReLU → Linear(4D, D)` + artık bağlantı
- **Çıkış:** `Linear(D, V)` → softmax

Eğitim tarayıcıda, elle yazılmış saf gradyan inişiyle yapılır (autograd kütüphanesi yok — kod okunabilir olmalı).
Ağır iş Web Worker içinde çalışır, arayüz donmaz.

## Ekranlar

1. **Gömme küresi** — normalize edilmiş gömme vektörleri 3B kürede; eğitim ilerledikçe canlı yer değiştirir (D > 3 olduğunda PCA izdüşümü)
2. **Ağırlık ızgaraları** — her matris bir ısı haritası; hücreler elle değiştirilebilir, çıktının bozulması anında görülür
3. **İleri geçiş** — token id → gömme → matris çarpımı (hücre hücre) → ReLU → artık bağlantı → logits → softmax
4. **Üretim** — tahmin edilen karakter cümleye eklenerek döngü sürer
5. **Eğitim paneli** — canlı kayıp eğrisi, eşzamanlı güncellenen küre ve ızgaralar
6. **TTT modu** — donmuş ağırlıklar ile çıkarım anında güncellenen fast weight karşılaştırması

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

## Durum

Kurulum aşamasında. Sıra: çekirdek (tokenizer, model, train) → ağırlık ızgarası → ileri geçiş → küre → eğitim paneli → TTT modu.

## Lisans

MIT
