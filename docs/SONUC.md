# Yarışma Sonucu ve Değerlendirme

> RSNA Knee Abnormality Detection · bitiş 22 Ekim 2026 23:59 UTC · 6.425 takım
> Takım: theartificialteam (kadirov0 + miraysudemutlu)
> Kaynak: Kaggle submission kayıtları ve private LB (25 Ekim; ev sahibi doğrulaması sonrası kesinleşecek)

## 1. Sonuç

| | |
|---|---|
| Final (ön) sıra | **1.730 / 6.425** (public'te 1.565) |
| Finaller | 10sa ve 10t: public 0.958 · private **0.942** |
| Madalya sınırları (yaklaşık) | Bronz: ilk %10 (~642. sıra), 499. sıra 0.943 private ile bronz aldı |
| 1. sıra | private 0.960 (Team Kangaroo), 50. sıra 0.951 |

### Bütün gönderimler: public ve private

| Gönderim | İçerik | Public | Private |
|---|---|---|---|
| 10sa | heliosli 0.958 tarifi | 0.958 | **0.942** |
| 10t | heliosli 0.9586 tarifi (+2.5D okuyucu) | 0.958 | **0.942** |
| 10sh | 3 CoAtNet v3 seed (gizli testte süreye sığmadı → taban yazıldı) | 0.958 | 0.942 |
| 10sb | 0.958 + 12v3 %10 | 0.956 | 0.941 |
| public blend v1 | eski heliosli 0.957 kopyası | 0.956 | 0.940 |
| 10sc / 10se / 10a | + bizim model %15–20 | 0.955 | 0.939 |
| final v2 | public 0.80 + Raptor-44 0.20 | 0.953 | 0.938 |
| 10sd | 0.958 + 12v3 %30 | 0.953 | 0.937 |
| 10sf | yalnız 12v3 (etiket v3) | 0.929 | 0.915 |
| 10r | yalnız Raptor-80 (etiket v2) | 0.928 | 0.913 |
| 10sg | yalnız CoAtNet s303 (etiket v4) | 0.926 | 0.911 |
| ilk gönderim | eski pipeline (sessiz hata) | 0.500 | 0.500 |

**Gözlemler:**
- Public ve private sıralaması tutarlı. Hiçbir varyant private'ta tabanı geçmedi.
- Bizim modeli eklemek private'ta da her ağırlıkta zararlıydı (0.942 → 0.941 → 0.939 → 0.937).
- Public 0.958'de toplanan kalabalığın tamamı private'ta ~0.942'de kaldı. Bu yüzden sıramız public ile private arasında çok değişmedi.

## 2. Kazananlar ne yaptı (çözüm yazılarından)

| Sıra | Takım | Private | Asıl fark |
|---|---|---|---|
| 1 | Team Kangaroo (tascj, liushuzhi) | 0.960 | LLM **ajanlarıyla görüntü + rapor etiketleme** (gpt-6 serisi, Codex/Claude CLI). Her vaka için görüntülere bakan bir ajan oturumu; bulgu olasılığı, bulgu başına 3B ROI kutusu ve kesit bazında lezyon kutuları. ROI dedektörü + kutu havuzlamalı ConvNeXt-T. |
| 2 | azamat1ch | 0.960 | Rapor okuyucu LLM (0.899) → **görüntüleri de okuyan LLM (Astra): 0.942** → öğretmen OOF karışımı: **0.956** (altın 58). 2.5D MIL ConvNeXt-T / EfficientNetV2-S, 6 üye, elle sabit ağırlıklar. |
| 7 | BarryZhou | 0.957 | **Dış veri yok.** GPT-6 Astra ile üç okuma (yapılandırılmış, yalnız görüntü, doğrulama) → altın 0.949; hastane bazında Synovitis düzeltmesi; çapraz model öğretmen → 0.952. 2.5D label-query ağı, 4 omurga. |
| 499 | Jay Gautam | 0.943 (bronz) | Public 0.958 tabanı + S6 ConvNeXt %20–35 |

**Ortak nokta:** İlk sıralardaki fark modelden değil **etiketten** geliyor. Rapor etiketlerinin tavanı altın 58'de ~0.90. Görüntüleri de okuyan bir LLM bunu 0.94–0.95'e çıkarıyor. 2. sıradaki takım bunu tek başına "etiket uyumundaki en büyük iyileşme" (+0.043) olarak raporluyor. Sınırlayıcı faktör, rapordan görülemeyen hafif bulgular: altın 58'de uzmanlar 240 pozitif işaretlemiş, raporlar 154.

## 3. Bizim yaklaşımla karşılaştırma

| Konu | Biz | Kazananlar |
|---|---|---|
| Etiket kaynağı | Rapor okuyan LLM'ler (steven, Opus 4.5, JEV) + kendi görüntü modelimizin OOF'u | Görüntü + rapor okuyan güçlü LLM, sonra öğretmen OOF |
| Etiket kalitesi (altın 58) | v2 0.901 · v3 0.922 (OOF içeriyor; LB'ye yansımadı) | 0.947–0.956 |
| Model | Raptor CoAtNet-384 (public tabanla aynı aile) | 2.5D ConvNeXt-T / EfficientNetV2-S, ROI kırpma |
| Kendi modelin public skoru | 0.929 | 0.964–0.969 |

**Neden geri kaldık:**
1. **Görüntü okuyan LLM etiketlemesine geçmedik.** Bunu "yerel VLM ile etiketleme" olarak fikir havuzuna almıştık. Ücretli API kullanılamadığı ve yerel denemelerin (Qwen 7B / 27B, yalnız rapor) zayıf kalması nedeniyle öncelik vermedik. Sonradan anlaşıldı ki en büyük kaldıraç tam olarak buydu.
2. **Karar ölçütümüz yanıltıcıydı.** Pseudo etiket (v3) altın 58'de +0.021 gösterdi ama LB'de +0.001'de kaldı. Etiketin içinde kendi modelimizin tahminleri olduğu için altın ölçümü iyimser çıktı. 2. sıradaki takım da ±0.02 şans payı uyarısında bulunup altın vakaları ayarlama ve kontrol için ikiye bölmüş.
3. **Public tabanla aynı aileden model eğittik.** Raptor-CoAtNet hem tabanın içinde vardı hem de kazananların listesinde en zayıf mimariydi.
4. **Altyapıya zaman kaybettik.** Sunucuya veri taşıma, CLI sürümleri, kuyruk ve yükleme hataları, boşta kalan GPU saatleri.

**İşe yarayanlar:**
- 0.500 → 0.958: sessiz hataların bulunması ve en iyi public tarifin doğru kurulması.
- Doğru DICOM ön işleme (geometrik sıralama, taraf, mm ölçek).
- Çıkarım–altın doğrulaması: ön işlemede hata olmadığını kanıtladı.
- Blend formülü: kendi modelimizin neden katkı vermediğini sayısal olarak açıkladı.

## 4. Bir sonraki tıbbi görüntü yarışması için dersler

1. **Etiket kalitesine ilk haftada yatırım yap.** Raporlu verilerde, görüntüyü de gören bir okuyucu (insan ya da VLM) ile küçük bir pilot yapıp altın vakalara karşı ölç.
2. **Altın vakaları ikiye böl:** biri ayar için, biri dokunulmadan kontrol için. Kendi modelinin tahminlerini içeren etiketleri altın vakada ölçme.
3. **Erken ve sık LB ölçümü:** Kendi modelinin tek başına LB skorunu ilk günlerde öğren. Blend kararlarını ondan sonra ver.
4. **Public tabanla farklı aileden model kur.** Çeşitlilik ancak gücü yeterli olunca işe yarıyor (bkz. RAPOR §6 eşik tablosu).
5. **Altyapıyı baştan sağlam kur:** tek ortam, token dosyası, idempotent kuyruk, yükleme ile eğitimi ayır, uzun işleri bilgisayarın uykusuna bırakma.

## Kaynaklar
- 1. sıra çözümü (tascj): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution
- 1. sıra, ikinci yazar (sakaku): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/752474
- 2. sıra çözümü (azamat1ch): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an
- 7. sıra, dış veri yok (BarryZhou): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/7th-place-solution-with-no-external-training-data
- 499. sıra, bronz (Jay Gautam): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/499th-place-bronze-0-943-private-4-family-rank
