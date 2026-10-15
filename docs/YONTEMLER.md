# RSNA Knee Abnormality Detection — Teknik Rapor

Bu belge projeye yeni katılan birinin tek başına okuyup durumu anlayabilmesi için yazıldı:
problem nedir, hangi yöntem nerede uygulanıyor ve **nereden alındı**, neler denendi ve hangi
sonuçla, sırada ne var. Değişikliklerin tarih sıralı listesi: [CHANGELOG.md](../CHANGELOG.md).
Kurulum ve çalıştırma: [README.md](../README.md).

> Son güncelleme: 2026-10-14 · Takım: kadirov0 (+ birleşen takım arkadaşı) · Kaggle sırası: 723 / 5.954 (LB 0.956)

---

## Sözlük (teknik terimler)

| terim | anlamı |
|---|---|
| **ROC AUC** | Bir modelin pozitif vakaları negatiflerden yukarı sıralama başarısı (0.5 = rastgele, 1.0 = kusursuz). Yalnızca **sıralamaya** bakar, olasılığın ölçeğine bakmaz. |
| **Makro AUC** | 12 bulgunun AUC'lerinin basit ortalaması; yarışmanın skoru. |
| **LB (Leaderboard)** | Sıralama tablosu. *Public LB* test setinin bir kısmıyla hesaplanır ve yarışma boyunca görünür; *private LB* kalan kısımla hesaplanır ve final sıralamayı belirler. |
| **Altın vaka (gold)** | Radyologun 12 bulguyu doğrudan işaretlediği 58 eğitim vakası; elimizdeki tek "gerçek" ölçü. |
| **Yumuşak etiket (soft / pseudo label)** | Rapordan türetilen 0–1 arası olasılık hedefi (ör. "şüpheli yırtık" → 0.6). Etiketsiz 4.349 vaka böyle etiketlenir. |
| **LLM** | Büyük dil modeli (Claude, GPT, Gemini, Qwen…). Burada raporları okuyup etiket üretmek için kullanılır. |
| **P / A / N / U** | LLM etiket kodları: *Present* (var), *Absent* (yok), *Not mentioned* (raporda geçmiyor), *Uncertain* (belirsiz). |
| **DICOM** | Tıbbi görüntü dosya formatı; her `.dcm` bir kesit, başlığında konum/yön/piksel boyutu bilgisi taşır. |
| **Seri (series)** | Tek bir MR çekimi (ör. sagital PD yağ baskılı); bir vakada 3–14 seri olur. |
| **Kesit (slice)** | Serideki tek 2D görüntü. |
| **Sagital / Koronal / Aksiyal** | Görüntü düzlemleri: yandan / önden / yukarıdan kesit. |
| **Sıvı-duyarlı (FS, PD/T2) / T1** | MR sekans türleri. Sıvı-duyarlı sekanslarda sıvı ve ödem parlak görünür (efüzyon, kontüzyon için önemli); T1 anatomiyi gösterir. |
| **Slot** | Bir vakanın serilerini sabit yuvalara (ör. "sagital sıvı-duyarlı") yerleştirme; model hangi görüntünün ne olduğunu bilir. |
| **LPS koordinatları** | DICOM hasta koordinat sistemi: +x hastanın solu, +y arkası, +z başı. |
| **ImagePositionPatient / ImageOrientationPatient** | DICOM başlığında kesitin uzaydaki konumu ve yönü; doğru kesit sırası buradan çıkar. |
| **Laterality** | Taranan dizin sağ mı sol mu olduğu. Medial/lateral bulgular için kritik. |
| **FOV (görüş alanı)** | Görüntünün kapsadığı fiziksel genişlik (mm). Biz sabit 160 mm, Raptor 140 mm kullanır. |
| **Span** | Seri içinde kullanılan kesit aralığı (ör. 0.15–0.85 = yığının ortadaki %70'i). |
| **2.5D pencere** | Ardışık 3 kesitin 3 renk kanalı gibi verilmesi; 2D ağ biraz 3B bağlam görür. |
| **Omurga (backbone)** | Görüntüden öznitelik çıkaran ön eğitimli ağ (ConvNeXt, CoAtNet). |
| **ConvNeXt / CoAtNet** | Modern evrişimli (ConvNeXt) ve evrişim+dikkat karması (CoAtNet) görüntü ağları. |
| **ImageNet normalizasyonu** | Girdiyi ön eğitim verisinin ortalama/std'siyle ölçekleme; ön eğitimli ağlar bunu bekler. |
| **MIL / dikkat havuzlama** | Çok örnekli öğrenme: bir vakanın onlarca penceresinden bulguya göre önemli olanlara ağırlık vererek tek tahmin üretme. |
| **BCE** | İkili çapraz entropi kaybı; çok etiketli sınıflandırmanın standart kaybı. |
| **pos_weight** | BCE'de pozitif örneklere ek ağırlık. |
| **Fold / CV / OOF** | Çapraz doğrulama: veri k parçaya bölünür; her parça bir kez doğrulama olur. OOF (out-of-fold) = modelin eğitimde görmediği vakalardaki tahminleri. |
| **Bootstrap / GA** | Vakaları yeniden örnekleyerek ölçümün ne kadar oynak olduğunu bulma; GA = güven aralığı. |
| **SWA** | Son birkaç epoch'un ağırlıklarının ortalaması; genelde daha kararlı model. |
| **Epoch** | Tüm eğitim verisinin bir kez görülmesi. |
| **AMP / fp16 / bf16** | Karışık hassasiyet: hesapları 16 bitte yaparak hızlanma. T4 GPU fp16 destekler, bf16'yı donanımda desteklemez. |
| **GradScaler** | fp16 eğitimde küçük gradyanların sıfırlanmasını önleyen ölçekleyici. |
| **OOM** | Bellek yetmezliği (out of memory). |
| **Rank blend** | Modellerin tahminlerini bulgu bazında sıraya (yüzdeliğe) çevirip ağırlıklı ortalama; AUC sıralamaya baktığı için doğal birleştirme. |
| **TTA** | Test anı artırma: aynı görüntünün varyantlarıyla (ör. ayna) tahmin alıp ortalama. |
| **Cache** | DICOM'ların bir kez işlenip hızlı okunabilir biçimde (`.npz`) saklanması. |
| **Kernel / notebook** | Kaggle'da çalışan kod. *Code competition*: gönderilen notebook gizli test setinde Kaggle tarafından yeniden çalıştırılır (≤ 9 saat, internetsiz). |
| **GPU kotası** | Kaggle'ın haftalık ücretsiz GPU süresi (~30 saat); aynı anda en fazla 2 GPU oturumu. |
| **Raptor** | dreaddevelopment'ın public model ailesi: CoAtNet-384 + 2.5D + bulguya özgü dikkat havuzlama. |
| **R7 / JEV / steven / pilkwang / Opus** | Public model ve etiket kaynaklarının kısa adları (bkz. Kaynakça). |

---

## Yönetici özeti

**Başlangıç (2026-10-13):** Tek gönderim, LB **0.500**, sıra 5896. Sebep: eski submission kernel'i
modeli yükleyemeyip her vakaya aynı değeri yazıyordu (sessiz hata). Ayrıca eski pipeline kesitleri
dosya adına göre (rastgele) sıralıyor, sağ/sol dizi ayırt etmiyordu.

**Yapılanlar:**
1. Kod modüllere ayrıldı (`src/`), her adım Türkçe açıklamalı bir notebook oldu (02–10).
2. DICOM ön işlemesi doğru hâle getirildi ve gerçek veride doğrulandı (geometrik sıralama, yön,
   sağ/sol, mm ölçeği). 4.407 vaka sıfır hatayla cache'lendi.
3. Etiketler beş kaynaktan denetlendi; en iyi karışım (v2) altın vakada 0.893 → **0.901**.
4. Public 0.957 blend gönderildi: LB **0.956**, sıra **723**.
5. En güçlü public reçete (Raptor CoAtNet-384) kendi etiketlerimizle iki kez yeniden eğitildi.
6. Final notebook kuruldu: public blend + bizim modeller, korumalı ve süre bütçeli.

**Sırada:** 07b ve 05 eğitimlerinin bitmesi → final notebook ile gönderim → 2 final seçimi.

---

## Denenen yöntemler — özet tablo

| # | yöntem | nerede | kaynak [ref] | sonuç | durum |
|---|---|---|---|---|---|
| 1 | Rapor etiketlerinin altın vakaya karşı denetimi | `src/labels.py`, 02 | kendi | Kaynaklar 0.87–0.90; fark gürültü bandında | ✅ kullanılıyor |
| 2 | Etiket v1: steven + pilkwang ortalaması | 02 | [7][8] | 0.8927 | ⏸ yerine v2 |
| 3 | Etiket v2: steven + Opus 4.5 + JEV ortalaması | 02 | [7][9][10][11] | **0.9014** | ✅ kullanılıyor |
| 4 | Kendi LLM etiketlemesi (Qwen2.5-7B, P/A/N/U rubrik) | 08 | yöntem [16] (BTK) | 0.797 | ❌ elendi |
| 5 | Geometrik kesit sıralama (IPP · normal) | `src/dicom.py`, 03 | [6] | dosya adı sırası ρ 0.09–0.31 idi | ✅ |
| 6 | Sağ/sol tespiti (etiket, yoksa merkez x) + aynalama | `src/dicom.py` | [6][5] | etiketle %98.5 uyum | ✅ |
| 7 | Sabit FOV (mm) yeniden örnekleme, seri düzeyi yoğunluk | `src/dicom.py` | [6] | — | ✅ |
| 8 | 6 slotlu cache (16 kesit, 224 px, 160 mm) | `src/cache.py`, 04 | kendi | 4.407 vaka, 0 hata, ~12 GB | ✅ |
| 9 | ConvNeXt-T + slot/konum gömmesi + Transformer + 12 başlı dikkat | `src/model.py`, 05/05b | [12][13][4] | fold 0: pseudo 0.876, altın(12) 0.929 | 🔄 fold 1–4 |
| 10 | Raptor CoAtNet-384 yeniden eğitimi, etiket v1, norm yok | `third_party/raptor`, 07 | [3] | altın 0.888 (SWA 0.8883) | ✅ final aday |
| 11 | Raptor, etiket v2 + ImageNet norm | 07b | [3] | altın **0.8965** | ✅ final aday |
| 12 | Raptor corpus geometrisinin piksel korelasyonuyla doğrulanması | 09 | kendi | span 0.15–0.85, 140 mm → 1.000 | ✅ |
| 13 | Public blend (Raptor 96 kesit + R7) | public kopya | [2] | LB 0.956 | ✅ taban |
| 14 | Final: public + bizim Raptor (44 kesit), W = 0.20 rank blend | 10 v2 | [2][4] | **LB 0.953** (< 0.956) | ❌ bu ağırlıkla zararlı |
| 15 | Sessiz geri dönüşe karşı korumalar (strict yükleme, hata oranı, sabit tahmin) | `src/inference.py` | kendi (0.500 dersi) | — | ✅ |
| 16 | DICOM JPEG çözücüleri (internetsiz wheel) | `src/dicom.py` | [17] | Kaggle imajında gerekmedi | ✅ güvence |
| 17 | OAI harici veri | — | [5] | forumda +0.005 | ❌ kural riski |
| 18 | Anatomik ayna TTA | — | [5] | forumda +0.001 | 💡 fikir |
| 19 | Yardımcı kayıplar / ROI kırpma | — | [12][13] | — | 💡 fikir |
| 20 | Kendi 80 kesitlik corpus'umuz + Raptor eğitimi (etiket v2), 2 seed | 11, 12, 12b | [3] (geometri tarifi) | altın 0.9056 / 0.9070, birlikte **0.9097** | 🔄 LB ölçümü |

Durum: ✅ kullanılıyor · 🔄 çalışıyor/bekliyor · ⏸ yerine yenisi geldi · ❌ elendi · 💡 denenmedi

---

## İçindekiler

1. [Problem ve metrik](#1-problem-ve-metrik)
2. [Veri hakkında kritik gerçekler](#2-veri-hakkında-kritik-gerçekler)
3. [Genel strateji](#3-genel-strateji)
4. [Yöntemler (aşama aşama)](#4-yöntemler-aşama-aşama)
5. [Deney günlüğü](#5-deney-günlüğü)
6. [Bulunan hatalar ve dersler](#6-bulunan-hatalar-ve-dersler)
7. [Benzer yarışmalardan alınan fikirler](#7-benzer-yarışmalardan-alınan-fikirler)
8. [Fikir havuzu](#8-fikir-havuzu-denenmedi)
9. [Kaynakça](#9-kaynakça)

---

## 1. Problem ve metrik

Her diz MR incelemesi (study) için 12 bulgunun olasılığı tahmin edilir:
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture.

**Metrik:** 12 bulgunun ROC AUC'sinin ortalaması. AUC yalnızca **sıralamaya** bakar, yani:
- kalibrasyon (olasılığın "doğru" ölçekte olması) puan getirmez,
- modelleri birleştirirken olasılık yerine **sıra (rank)** ortalamak doğal yöntemdir,
- tüm vakalara aynı değeri yazmak AUC = 0.5 demektir.

**Kurallar:** günde 5 gönderim, 2 final seçimi, notebook ≤ 9 saat, internet kapalı, public
harici veri/model serbest. Son gönderim 22 Ekim 2026, takım birleştirme son günü 15 Ekim 2026.

## 2. Veri hakkında kritik gerçekler

| gerçek | sonucu |
|---|---|
| 4.407 eğitim vakasının yalnızca **58**'inde radyolog etiketi var | Etiketler raporlardan türetilmek zorunda; 58 vaka tek "gerçek" ölçü ama küçük farkları ayırt edemez (±0.006 AUC gürültü) |
| Test setinde (~1.300 vaka) **rapor yok** | Metin modele girdi olamaz; yalnızca eğitim hedefi üretir. "Metin + görüntü füzyonu" mümkün değil |
| Raporlar çok dilli (İngilizce, İspanyolca, Fransızca, Hollandaca, Almanca, Yunanca, Türkçe…) | Etiket çıkarıcı çok dilli olmalı |
| DICOM dosya adları rastgele UID | Kesit sırası geometriden çıkarılmalı |
| `InstanceNumber` bazı serilerde ters yönde | Sıralama için de güvenilmez |
| `Laterality` etiketi vakaların yaklaşık yarısında yok (2.207/4.407) | Sağ/sol diz geometriden çıkarılıyor; etiketli vakalarda %98.5 uyum |
| Seriler JPEG Lossless / JPEG 2000 ile sıkıştırılmış olabilir | Kaggle imajı bunları açabiliyor (cache v1'de 0 hata); `ensure_decoders()` yine de güvence olarak duruyor |
| Vaka başına 3–14 seri; düzlem × sekans (sıvı-duyarlı / T1) | 6 "slot"a indirgeniyor |

## 3. Genel strateji

Leaderboard'da 200–700. sıralar 0.956'da yığılmış durumda (public notebook'ların blend'leri).
İlk sıra 0.966. Öndekilerle aradaki fark büyük ölçüde **etiket kalitesinden** geliyor: en güçlü
public model ailesi (Raptor, CoAtNet-384) eğitim kodu ve görüntü corpus'u ile public, ama
etiketler public değil. 0.957 notebook'undaki "R7" modeli de aynı reçeteyi kendi etiketleriyle
yeniden eğitmiş.

Bu yüzden plan dört hattan oluşuyor:

| hat | iş | durum |
|---|---|---|
| A. Taban | Public 0.957 blend'in private kopyası | **LB 0.956**, sıra 723 |
| B. Etiketler | Kaynakları birleştirme; LLM ile yeniden etiketleme | Etiket v2 hazır (0.9014); Qwen-7B denendi, elendi |
| C. Model | Raptor reçetesini bizim etiketlerle eğitme | 07 (etiket v1) ve 07b (etiket v2 + imagenet norm) çalışıyor |
| D. Kendi pipeline'ımız | Doğru ön işleme + ConvNeXt modeli; çeşitlilik ve kontrol için | cache tamam; 05 (etiket v2, fold 0) eğitiliyor |
| E. Final | Public blend + bizim modeller, sabit ağırlıklı rank blend; 2 final seçimi | bekliyor |

## 4. Yöntemler (aşama aşama)

### 4.1 Etiketler — `src/labels.py`, `notebooks/02_etiketler.ipynb`

| yöntem | ayrıntı |
|---|---|
| Kaynaklar | v1: steven_v4 [7] + pilkwang_v2 [8]. **v2 (güncel):** steven_v4 [7] + Claude Opus 4.5 [9] + JEV HYBRID [10] |
| Birleştirme | Eşit ağırlıklı ortalama; bir kaynakta eksik vaka diğerleriyle doldurulur |
| Altın vakalar | Orijinal 0/1 etiket, `is_gold=True`, eğitimde 2× ağırlık |
| Denetim | Her kaynağın 58 vakadaki bulgu bazlı AUC'si; bootstrap ile fark güven aralığı |
| LLM ile yeniden etiketleme (08) | Qwen2.5-7B-Instruct; sistem prompt'unda bulgu tanımları ve belirsizlik kuralları; çıktı JSON, her bulgu için P/A/N/U. Kod → olasılık sabit tablo: P 0.95, U 0.6, N 0.08, A 0.03 (altın vakalara göre ayarlanmaz) |

Neden sabit `pos_weight` yok: kaynakların pozitiflik oranları altın vakalardan belirgin farklı
(ör. Fracture %7 vs %31), oran tabanlı ağırlıklar yanlı olur.

### 4.2 DICOM ön işleme — `src/dicom.py`, `notebooks/03_dicom_onisleme.ipynb`

Koordinat sistemi DICOM/LPS: +x hastanın solu, +y arka, +z baş.

| adım | yöntem |
|---|---|
| Kesit sırası | Kesit normalinin baskın ekseni boyunca `ImagePositionPatient` (sagital → x, koronal → y, aksiyal → z). Geometri yoksa `InstanceNumber` |
| Yön standardizasyonu | Satır yönü +x/+y, sütun yönü −z (baş yukarıda) veya +y olacak şekilde çevirme |
| Sağ / sol diz | Önce `Laterality`/`ImageLaterality`; yoksa görüntü merkezinin x koordinatının medyanı (x < 0 → sağ). \|x\| < 10 mm ise karar yok |
| Aynalama | Sağ diz → koronal/aksiyalde yatay çevirme, sagitalde kesit sırası ters |
| Ölçek | Sabit 160 mm görüş alanı → 224 px (0.71 mm/px), merkezden kırp/doldur |
| Yoğunluk | Seri düzeyinde %0.5–%99.5 yüzdelik pencere → uint8 |
| Kesit seçimi | Yığının ortadaki %80'inden eşit aralıklı 16 kesit |
| Sağlamlık | Okunamayan tek dosya atlanır; < 3 okunabilir kesit varsa seri düşer ve loglanır |
| Sıkıştırma | `ensure_decoders()` ile pylibjpeg + openjpeg + gdcm (internetsiz wheel) |

### 4.3 Cache — `src/cache.py`, `notebooks/04_cache_kaggle.ipynb`

Her vaka 6 slota indirgenir (sag_fs, sag_t1, cor_fs, cor_t1, ax_fs, ax_t1); bir slota birden çok
seri düşerse en çok kesitli seçilir. Çıktı: `<study>.npz` içinde `uint8 [slot, 16, 224, 224]` +
`meta.csv` (slot sayısı, taraf ve kaynağı, düzlem uyuşmazlığı, hatalar). Tahmini boyut ~12 GB,
süre ~1 saat (4 CPU). Vaka başına ortalama 4.8 slot dolu; `ax_t1` nadir (%19).

### 4.4 Kendi modelimiz — `src/model.py`, `src/dataset.py`, `src/engine.py`, `notebooks/05`

| bileşen | ayrıntı |
|---|---|
| Girdi | Her slottan 2.5D pencere: ardışık 3 kesit = 3 kanal. Eğitimde slot başına rastgele 6, değerlendirmede 14 pencere |
| Omurga | ConvNeXt-Tiny (ImageNet-22k), ImageNet normalizasyonu |
| Bağlam | Slot gömmesi (düzlem × sekans) + konum gömmesi + 1 katman Transformer |
| Havuzlama | 12 başlı, bulguya özgü maskeli dikkat havuzlama + ortalama havuzlama artığı |
| Kayıp | Yumuşak etiketlerle BCE, altın vakalar 2× |
| Optimizasyon | AdamW (omurga 1e-4, başlık 5e-4), ısınmalı kosinüs, fp16 AMP, gradyan kırpma 2.0 |
| Veri artırma | Slot bazında dönme ±10°, ölçek 0.9–1.1, kaydırma, parlaklık/kontrast. **Yatay çevirme yok** |
| Doğrulama | 5 fold (altın vaka + pozitif sayısına göre tabakalı); `pseudo_auc` ve `gold_auc` |

### 4.5 Raptor reçetesi — `third_party/raptor/train_knee.py`, `notebooks/07`

dreaddevelopment'ın public eğitim kodu: CoAtNet-RMLP-2 (384 px), 2.5D pencereler, bulguya
özgü dikkat havuzlama, OneCycle LR, en iyi 3 epoch'un ağırlık ortalaması (SWA). 58 altın vaka
yalnızca doğrulama. Görüntüler yazarın public corpus'undan (4.407 × 44 kesit × 336 px).
Bizim yamalar: T4 için fp16 + GradScaler, süre bütçesi, ilerleme logu.

**Not:** Bu checkpoint 44 kesitlik geometriyle eğitiliyor. Tahminde aynı geometriyi kullanan
yazarın 0.924 inference kodu kullanılmalı (public 0.957 blend'deki Raptor aşaması 96 kesitlik
farklı bir geometri kullanıyor).

### 4.5b Final submission — `notebooks/10_final_blend_kaggle.ipynb`, `third_party/raptor/infer_knee.py`

Tek notebook, üç aşama: (A) public 0.957 blend aynen → `submission.csv`; (B) bizim Raptor
checkpoint'leri (07, 07b) doğrulanmış 44 kesitlik geometriyle (`RAPTOR_CFG` JSON); (C) bulgu bazında
rank blend `(1 − W)·public + W·bizim`, W = 0.20 sabit. B hata verir, süreyi aşar ya da sabit tahmin
üretirse A'nın submission'ı korunur.

### 4.6 Inference ve birleştirme — `src/inference.py`, `notebooks/06`

- Test vakaları cache'siz, eğitimle **aynı** `process_study` fonksiyonuyla okunur.
- Korumalar: `strict=True` checkpoint; okunamayan vaka > %2 → hata; tüm tahminler aynıysa → hata;
  çözücü yoksa → hata.
- Okunamayan vakalar sıralamaya girmez, nötr 0.5 alır.
- Fold'lar ve modeller **rank blend** ile birleşir (bulgu bazında sıra yüzdesi, ağırlıklı ortalama).

## 5. Deney günlüğü

| tarih | deney | ayar | sonuç | karar |
|---|---|---|---|---|
| 10-06 | İlk submission (eski pipeline) | ResNet34, eski kernel | **LB 0.500** | Teşhis: kernel modeli yükleyemeyip sabit önsel yazmış |
| 10-13 | Etiket kaynağı denetimi (58 altın) | steven_v4 | makro AUC **0.8927** | — |
| 10-13 | 〃 | pilkwang_v2 | **0.8704** | — |
| 10-13 | 〃 | 50/50 ortalama | **0.8927**; fark %90 GA [−0.0068, +0.0054] | Ortalama kullanılıyor (fark gürültüde, ortalama daha sağlam) |
| 10-13 | Bulgu bazında zayıf noktalar | ortalama | Synovitis 0.79, Fracture 0.82, Lateral OA 0.81, Contusion 0.85 | LLM ile yeniden etiketleme önceliklendirildi |
| 10-13 | Kesit sırası testi (1 test vakası, 5 seri) | dosya adı vs geometri | Spearman ρ = 0.09–0.31; InstanceNumber ρ = ±1 | Geometrik sıralama zorunlu |
| 10-13 | Taraf tespiti (aynı vaka) | etiket yok | merkez x ≈ −83 / −87 mm → sağ diz | Geometrik yedek gerekli |
| 10-13 | Cache boyutu ölçümü | 16 kesit / 224 / 160 mm | slot başına 0.57 MB → ~11.6 GB | Bu ayar seçildi |
| 10-13 | Yerel uçtan uca test | resnet18, 1 vaka | geçti; koruma tetiklendi | — |
| 10-13 | Public 0.957 blend (private kopya) | heliosli v-latest | çalıştı, geçerli submission | 10-14'te gönderildi (aşağıda) |
| 10-14 | **Gönderim:** public blend v1 | `kadirov0/rsna-knee-public-blend-heliosli` v1 | **LB 0.956** → sıra 5896 → **723** / 5954 | Taban skor; 0.956'da yüzlerce takım berabere |
| 10-14 | Cache v1 (Kaggle CPU) | 16 kesit / 224 / 160 mm | 4.407/4.407 vaka, **0 okuma hatası**; ort. 4.84 slot (CSV ile birebir); taraf: 2.200 etiketten, 2.152 geometriden, 55 belirsiz; **geometri–etiket uyumu %98.5** (n=2.146); düzlem uyuşmazlığı 0 | v2 gerekmedi; geometrik taraf tespiti doğrulandı |
| 10-14 | Raptor yeniden eğitim 07 (Kaggle T4) | CoAtNet-384, etiket v1, norm yok, 8 epoch, bs 4, 9.5 saat | altın AUC epoch başına 0.718 → 0.803 → 0.834 → 0.861 → 0.868 → 0.887 → 0.887 → **0.888**; SWA (ep 5–7) **0.8883** (yazarın modeli ≈0.905) | Final blend'e aday; 07b'nin (v2 + imagenet) daha iyi olması bekleniyor |
| 10-14 | LLM etiketleme v1 (Kaggle 2×T4) | Qwen2.5-7B, P/A/N/U, sabit batch 24 | **OOM ile 1.464/4.407'de durdu** (en uzun raporlar). Kısmi sonuç: ayrıştırma hatası %1; mevcut etiketle uyum AUC 0.81–0.96; 13 altın vakada qwen 0.829 / steven 0.834 / pilkwang 0.841 (n çok küçük) | Kalite benzer, bağımsız 3. okuyucu olarak tamamlanıyor |
| 10-14 | LLM etiketleme v2 (tamamlandı) | kaldığı yerden devam, token bütçeli batch | 4.407/4.407 rapor; **Qwen-7B altın AUC 0.797** (kısa raporlarda iyi, uzun raporlarda zayıf); v2'ye eklenince 0.9014 → 0.8998 | **Kullanılmıyor.** 7B açık model bu görevde ticari modellerin (Opus, Gemini) gerisinde |
| 10-14 | Ek etiket kaynakları (58 altın) | Claude Opus 4.5 (NTejas-1, MIT) | **0.8919** (Fracture 0.89, PF OA 0.945 güçlü; Baker's 0.83 zayıf) | Bağımsız güçlü okuyucu |
| 10-14 | 〃 | JEV HYBRID (riadmohamed42) | **0.9015** (yazarı altın vakalara göre ayarlamış; iyimser) | — |
| 10-14 | Etiket v2 karşılaştırması | steven+opus / steven+opus+jev / 4 kaynak | 0.9011 / **0.9014** / 0.8997 | **Etiket v2 = steven + opus + JEV** (pilkwang JEV içinde zaten var) |
| 10-14 | Raptor eğitim hızı (T4) | CoAtNet-384, bs 4, k 12 | ~3.7 sn/adım, ~67 dk/epoch | 8 epoch ≈ 9–10 saat |
| 10-14 | Raptor normalizasyonu | script varsayılanı `--norm none` | 07 (v1) normalizasyonsuz eğitiliyor | 07b'de `--norm imagenet` (yazarın inference'ı bunu kullanıyor) |
| 10-14 | **Raptor corpus geometrisi** (09, 12 eğitim vakası, piksel korelasyonu) | slot düzeni maskeden: 12+10+8+6+8; span × kırpma ızgarası | **span 0.15–0.85 + 140 mm → 1.000**; inference varsayılanı 0.06–0.94 → 0.754; 120/160 mm → ≤0.64 | Final notebook (10) bu geometriyi kullanır; yanlış varsayılan sessizce skoru düşürecekti |
| 10-14 | Public blend puanlama süresi | gizli test (~1.300 vaka) | ~3.3 saat | 9 saatlik sınırda bizim aşamaya ~5 saat pay var |
| 10-14 | Final notebook 10 test (3 örnek vaka) | public + 07 (SWA), W = 0.20 | A, B, C aşamaları hatasız; bizim model vakalar arası farklı olasılıklar üretti; final = 0.8·rank(public) + 0.2·rank(bizim) elle doğrulandı | B'nin süre bütçesi 2 → 4.5 saat (gizli testte zaman aşımı riskine karşı) |
| 10-14 | Raptor 07b (Kaggle T4) | CoAtNet-384, **etiket v2**, **ImageNet norm**, 8 epoch | altın AUC 0.710 → 0.815 → 0.846 → 0.879 → 0.885 → 0.891 → 0.897 → **0.8965** (07: 0.888). Bulgu bazında güçlü: Effusion 0.98, Baker's 0.98, Medial OA 0.96; zayıf: Lateral OA 0.78, Lateral Meniscus 0.81, PF OA 0.82 | v2 etiket + normalizasyon +0.009; final notebook'a eklendi (10 v2) |
| 10-14 | Final notebook 10 v2 (public + 07 SWA + 07b SWA, W = 0.20) | 3 örnek vakada test | iki Raptor kolu tutarlı tahmin üretti; final değerlerin %83'ü public'ten farklı | **LB 0.953** (public tek başına 0.956) → **daha kötü**. Sebep: 44 kesit/336 px corpus'la eğitilmiş kollar (tahmini LB ~0.92) public'in 96 kesit/384 px modellerinden zayıf; %20 ağırlık ortalamayı aşağı çekti. Puanlama süresi ~9 saat |
| 10-14 | Kendi modelimiz 05 (ConvNeXt-T, etiket v2, fold 0) | 8 epoch, ~10.5 dk/epoch | pseudo AUC 0.759 → … → **0.876**; altın (n=12) 0.777 → … → 0.929 (ep 7'de 0.935) | 05b (fold 1–4) ile 5 fold tamamlanacak; final notebook Aşama B2 |
| 10-14 | Raptor kolları birlikte (58 altın) | 07 SWA + 07b, rank ortalaması | 07: 0.8883, 07b: 0.8965, **50/50: 0.8976** (bootstrap fark +0.0013, %90 GA [−0.005, +0.008]); bulgu bazında korelasyon 0.91–0.97 | Aynı reçetenin kopyaları az çeşitlilik katıyor; asıl çeşitlilik ConvNeXt'ten beklenir |
| 10-14 | ConvNeXt 5 fold OOF (58 altın, gerçek OOF) | 05 + 05b, etiket v2 | **0.890**; Raptor 07b ile korelasyon 0.84 (Raptor kolları arası 0.91–0.97); Raptor+ConvNeXt 50/50 **0.906** | Çeşitlilik var ama mutlak güç (~0.90 altın ≈ ~0.93 LB) blend için yetersiz; final'de ağırlık 0 |
| 10-14 | 80 kesitlik kendi corpus'umuz (11, 3 CPU kernel) | 22/18/15/10/15 slot, span 0.02–0.98, 140 mm, 336 px | parça 0/1: 1.469'ar vaka, 20–27 dk, **0 hata**; vaka başına 77/80 dolu kesit; slot doluluk sag-T1 %97, kor-T1 %74 (CSV ile tutarlı) | 12 (Raptor-80, etiket v2) eğitimine girdi |
| 10-14 | Raptor 07c (seed 7, etiket v2, ImageNet norm, 44 kesit) | 8 epoch | altın AUC 0.684 → … → **0.9053** (07b: 0.8965, yazarın modeli ≈0.905) | En iyi 44 kesitlik kol; 44 kesit geometrisi nedeniyle final ağırlığı şimdilik 0 |
| 10-14 | Kaggle GPU kotası | kadirov0 | **haftalık 30 saat doldu** (cumartesi yenilenir) | 12 (Raptor-80) takım arkadaşının hesabında (`miraysudemutlu`) çalıştırıldı; kaynaklar paylaşımla, src kopyası onun hesabında |
| 10-15 | **Raptor-80 (12)** — kendi 80 kesitlik corpus'umuz, etiket v2, ImageNet norm | miraysudemutlu hesabı, 8 epoch | altın AUC 0.680 → … → **0.9056** (SWA 0.9041) | — |
| 10-15 | **Raptor-80 seed 7 (12b)** | 〃 | **0.9070** (SWA 0.9053) | — |
| 10-15 | 12 + 12b birlikte (58 altın) | rank ortalaması | **0.9097** (bizim en iyi); 12–12b korelasyon 0.95; 12–07b (44 kesit) 0.91 | 58 vaka 44 ↔ 80 kesit farkını göstermiyor → LB'de ölçülecek: 10 (public 0.85 + r80 0.15) ve 10r (yalnızca r80) paralel test |

**Girdi araması:** `/kaggle/input` altında `rglob` kullanmayın; yarışmanın DICOM ağacını dolaşıp
dakikalar sürer (05'te ~25 dk). `src.config.find_input` / `find_inputs` DICOM klasörlerini atlar.

**Kaggle kısıtı (ölçüldü):** haftalık **30 saat** GPU kotası (dolunca `Maximum weekly GPU quota` hatası; takım üyesinin hesabından `scripts/kaggle_push.py ... --token-file .env.teammate --as KULLANICI` ile çalıştırılabilir); aynı anda en fazla **2 GPU** oturumu ("Maximum batch GPU session
count of 2"). CPU işleri bu sınıra dahil değil. Çok dosyalı kernel çıktısını indirirken
`--page-size 200` kullanılmalı, aksi hâlde API 429 (istek limiti) verir.

## 6. Bulunan hatalar ve dersler

0. **Blend'e eklenen model, blend'in bileşenlerine yakın güçte olmalı.** 0.956'lık blend'e ~0.92
   seviyesinde bir modeli %20 ağırlıkla eklemek skoru 0.953'e düşürdü. Altın vakadaki 0.89–0.90
   (bizim) ile public modellerin gücü arasındaki farkı LB'de görmeden ölçmek mümkün değildi;
   altın 58 vaka bu karşılaştırmayı yapamaz.

1. **Sessiz hatalar en tehlikelisi.** Eski pipeline hiç hata vermeden 0.500 aldı. Artık her
   geri dönüş (fallback) ya hata verir ya da sayılıp yazdırılır.
2. **Dosya adı ≠ sıra.** Tıbbi görüntüde kesit sırası her zaman geometriden alınmalı.
3. **58 vaka ile seçim yapmak sızıntı + gürültü.** Altın vakalar yalnızca büyük farkları
   doğrulamak için kullanılır.
4. **Donanım/format sözleşmeleri:** T4 bf16 desteklemez; RadImageNet ağırlıkları tek kanal
   bekler; Raptor checkpoint'i eğitildiği kesit geometrisini bekler. Bu sözleşmeler
   config'te görünmez, bozulunca "model çalışmıyor" gibi görünür.

## 7. Benzer yarışmalardan alınan fikirler

| kaynak | fikir | bizdeki karşılığı |
|---|---|---|
| RSNA 2024 Lumbar, 1. sıra | 2 aşama: önce ilgili kesit/koordinatı bul, sonra kırpıp sınıflandır | Fikir havuzunda (diz eklemi kırpma) |
| 〃 | Attention-MIL + bi-LSTM, yardımcı kayıplar; küçük modeller (ConvNeXt-S) büyüklerden iyi; 7–14 epoch yeterli | Dikkat havuzlama var; LSTM havuzda |
| RSNA 2023 Abdominal, 1. sıra | 2.5D + CNN + GRU; **yardımcı segmentasyon kaybı** (+0.01–0.03); kesit düzeyi yumuşak etiket | Yardımcı hedefler fikir havuzunda |
| Bu yarışmanın forumu (Raptor yazarı) | Tek CoAtNet-384 ≈ 0.936–0.938; kesit yoğunluğu önemli (64 → 80 kesit +0.004); 384 → 256 px −0.029; dikkat yerine ortalama havuzlama −0.014/−0.02 | Raptor reçetesi 07'de; kendi modelimiz 224 px olduğu için çeşitlilik amaçlı |
| BTK Datathon 2026 (kendi önceki çalışmamız) | LLM ile yapılandırılmış rubrik puanları, öğretmen-öğrenci damıtma, çok kaynaklı stack | 08'de rubrik etiketleme; burada metin test setinde olmadığı için özellik değil hedef olarak |

## 7b. Yarışma forumundan öğrenilenler (2026-10-14)

| kaynak | bulgu | etkisi |
|---|---|---|
| Ev sahibi (tartışma 733965) | Raporları ticari LLM API'leriyle (OpenAI, Anthropic, Google) etiketlemek **açıkça serbest** | API ile yeniden etiketleme kurallara uygun |
| Ev sahibi | Harici veri: kolay erişilebilen (tıkla-kabul et) setler serbest; ağır onay gerektirenler sorunlu olabilir | OAI gri alan → kullanmıyoruz |
| "0.949 with one CoAtNet" (746792) | Tek CoAtNet-384, 96 kesit, 94 pencere; etiket = kendi LLM (Gemini) + steven_v4 + JEV ortalaması; steven'da 0.25 = "raporda bahsedilmiyor" işareti (hücrelerin %19'u); anatomik ayna TTA +0.001; OAI harici veri +0.005 | Etiket v2 stratejisini doğruluyor; ayna TTA fikir havuzuna |
| "Tricks" (hengck23) | Altın etiketler bazen raporla çelişiyor (rapor "efüzyon yok", etiket 1) → altın etiketler görüntüden okunmuş olabilir; rapor tabanlı etiketin tavanı ~0.90; "altında 0.89 olan LLM etiketi ≈ 0.930 LB" | Etiket kalitesi tek başına yetmez; görüntü modeli bu tavanı aşabilir |
| NTejas-1 GitHub | Claude Opus 4.5 etiketleri public (MIT); OOF ile "etiket onarma" Synovitis'i 0.038 kötüleştirmiş | Opus etiketleri v2'de; self-distillation ile etiket onarmaktan kaçınıyoruz |

## 8. Fikir havuzu (denenmedi)

Öncelik sırasıyla:
1. **Etiket v3 (yalnızca güçlü modelle):** Qwen-7B yetersiz kaldı (0.797). Ticari API (Claude /
   Gemini; kurallara uygun [1]) ile yalnızca kaynakların anlaşamadığı raporları yeniden okutmak;
   steven'ın 0.25 "bahsedilmiyor" işaretinin ayrı ele alınması [5].
2. **Anatomik ayna TTA [5]:** sagitalde kesit sırası, koronal/aksiyalde genişlik çevrilmiş ikinci
   görünümün olasılıklarının ortalaması (forumda +0.001).
3. **Yardımcı hedefler [13]:** raporlardan 12'nin ötesinde ayrıntılar (yırtık yeri, kıkırdak
   derecesi, kemik iliği ödeminin yeri, efüzyon miktarı) çıkarıp modele ek başlık olarak öğretmek.
4. **Diz eklemi kırpma (ROI) [3][12]:** Raptor script'inde `--roi` seçeneği hazır.
5. **Farklı tohum (seed) ile ikinci Raptor:** blend çeşitliliği.

## 9. Kaynakça

Her yöntemin nereden alındığı yukarıdaki "Denenen yöntemler" tablosunda [n] ile gösterilmiştir.

| # | kaynak | tür | bizde ne için kullanıldı |
|---|---|---|---|
| [1] | RSNA Knee Abnormality Detection — yarışma sayfası, veri açıklaması ve kurallar; ev sahibi açıklaması "Use of Commercially Hosted LLMs" (tartışma 733965). https://www.kaggle.com/competitions/rsna-knee-abnormality-detection | yarışma | bulgu tanımları, kurallar, LLM API izni |
| [2] | heliosli — "≈0.957 Latest dataset score in one click" (Raptor 96 kesit + R7 blend). https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click | public notebook | taban gönderim (LB 0.956), final notebook Aşama A |
| [3] | dreaddevelopment — Raptor eğitim kodu ("Knee MRI: training the twelve-finding model"), inference kodu ("…twelve findings from a single model"), corpus `knee-raptor-corpus(-ext)` ve tartışma 737696. https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model | public kod + veri | 07/07b eğitimi, `third_party/raptor/*`, kesit yoğunluğu/çözünürlük bulguları |
| [4] | Danial Zakaria — "Reaching 0.949 with one CoAtNet" (tartışma 746792) | forum yazısı | etiket karışımı stratejisi, 0.25 işareti, ayna TTA, W ağırlığı için referans |
| [5] | Forum tartışmaları: hengck23 "Tricks" (745861); "external datasets" (743416); 746792 yorumları | forum | altın etiket–rapor çelişkisi, OAI riski, ayna TTA |
| [6] | prvsiyan — "RSNA Knee: read the report, then the knee". https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee | public notebook | geometrik sıralama ve sağ/sol tespiti fikri (kod bizim) |
| [7] | stevenleehans — `rsna-knee-llm-report-labels` (`llm_labels_v4_blend.csv`) | public etiket | etiket v1, v2 |
| [8] | pilkwang — `rsna-knee-llm-labels` (`report_labels_v2.csv`) | public etiket | etiket v1 |
| [9] | NTejas-1 — RSNA-Knee-Abnormality-Detection, Claude Opus 4.5 etiketleri (MIT). https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection | GitHub | etiket v2 |
| [10] | riadmohamed42 — `jev-knee-labels` (JEV HYBRID) ve "jev-knee-labels-verification" | public etiket | etiket v2 |
| [11] | Qwen2.5-7B-Instruct (Alibaba Qwen). https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | açık model | 08 LLM etiketleme denemesi |
| [12] | NANACHI — RSNA 2024 Lumbar Spine, 1. sıra çözümü. https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution | yarışma çözümü | attention-MIL, ROI kırpma, küçük modeller fikri |
| [13] | Team Oxygen — RSNA 2023 Abdominal Trauma, 1. sıra çözümü. https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen | yarışma çözümü | 2.5D + sıralı havuzlama, yardımcı kayıp fikri |
| [14] | timm (PyTorch Image Models) — ConvNeXt, CoAtNet ön eğitimli ağırlıkları | kütüphane | omurgalar |
| [15] | pydicom; pylibjpeg / openjpeg / python-gdcm | kütüphane | DICOM okuma ve çözme |
| [16] | Ozan Mohurcu — BTK Datathon 2026 (önceki çalışmamız). https://github.com/Ozan-Mohurcu/BTK-Datathon2026 | GitHub | LLM rubrik etiketleme yaklaşımı (08) |
| [17] | Kaggle notebook'larında internetsiz wheel kurulumu (ör. goodpjw2008 "CoAtNet + 2.5D ConvNeXt MIL") | public notebook | `ensure_decoders()` fikri |

Kopyalanan üçüncü taraf kodları (`third_party/raptor/`) dosya başlarında kaynakları ve yaptığımız
değişiklikler (`[KADIROV PATCH]`) ile birlikte belirtilmiştir.
