# 🏥 RSNA Knee Abnormality Detection

> **Kaggle Yarışma Bağlantısı:** [kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Düzenleyen:** Radiological Society of North America (RSNA)  
> **Ödül Havuzu:** $77,000 USD  
> **Kategori:** Research Code Competition  
> **Bitiş Tarihi:** 22 Ekim 2026  

---

## 📌 1. Yarışmaya Genel Bakış (Overview & Problem Tanımı)

Diz eklemi rahatsızlıkları ve travmaları, kas-iskelet sistemi hastalıkları arasında en yaygın olanlarındandır. Manyetik Rezonans Görüntüleme (MRG / MRI), kemik ve yumuşak doku hasarlarını değerlendirmede altın standarttır. 

**RSNA Knee Abnormality Detection**, katılımcılardan bir dizi çok kesitli ve çok sekanslı diz MR serilerini ve bunlara eşlik eden radyoloji raporlarını kullanarak **12 farklı anormallik/patoloji sınıfını** çalışma (Study) düzeyinde tespit etmelerini beklemektedir.

Bu yarışma, **RSNA tarihindeki ilk multimodal (tıbbi görüntüleme + çok dilli klinik metin raporları) veri kümesini sunan yarışmadır**.

---

## 🏆 2. Canlı Liderlik Tablosu (Leaderboard Top 5)

*Veriler resmi Kaggle API üzerinden canlı olarak çekilmiştir (Ekim 2026).*

| Sıra | Takım / Yarışmacı Adı | Skor (ROC AUC) | Son Gönderim Zamanı |
| :---: | :--- | :---: | :---: |
| 🥇 **1** | **РазвалиНа** | **0.963** | 2026-10-05 05:36 |
| 🥈 **2** | **rhoskeri** | **0.963** | 2026-10-04 23:54 |
| 🥉 **3** | **joinT fusion** | **0.961** | 2026-10-05 09:36 |
| 🎖️ **4** | **Wassim Dobbi** | **0.961** | 2026-10-04 16:31 |
| 🎖️ **5** | **🦵YOKKOISHO🦵** | **0.960** | 2026-10-05 03:34 |

> **Not:** Liderlik tablosundaki en yüksek skorlar `0.963` seviyelerinde olup, model füzyonu (görüntü + metin NLP) ve güçlü kros-validasyon stratejilerinin fark yarattığı görülmektedir.

---

## 🎯 3. Tahmin Edilecek 12 Hedef Sınıf (Target Classes)

Her bir hasta incelemesi (`StudyInstanceUID`) için aşağıdaki 12 bağımsız patolojinin varlık olasılığı `[0.0, 1.0]` aralığında tahmin edilir:

| # | Hedef Etiket | Türkçe Karşılığı / Açıklama | Eğitim Kümesindeki Pozitiflik Oranı |
|---|---|---|:---:|
| 1 | **Effusion** | Eklem içi sıvı toplanması (Efüzyon) | **%60.3** |
| 2 | **Synovitis** | Sinovyal zar iltihaplanması/kalınlaşması | **%46.6** |
| 3 | **Medial Meniscus** | İç menisküs yırtığı / dejenerasyonu | **%44.8** |
| 4 | **ACL** | Ön Çapraz Bağ (Anterior Cruciate Ligament) hasarı/yırtığı | **%41.4** |
| 5 | **Lateral Meniscus** | Dış menisküs yırtığı | **%39.7** |
| 6 | **PF OA** | Patellofemoral osteoartrit (Diz kapağı kireçlenmesi) | **%36.2** |
| 7 | **Contusion** | Kemik iliği ödemi / Kemik kontüzyonu (Kemik ezilmesi) | **%32.8** |
| 8 | **Fracture** | Kemik kırığı / gizli mikro-kırıklar | **%31.0** |
| 9 | **Medial OA** | Medial kompartman kireçlenmesi (İç eklem artrozu) | **%25.9** |
| 10 | **Baker's** | Baker kisti (Popliteal kist) | **%20.7** |
| 11 | **Lateral OA** | Lateral kompartman kireçlenmesi (Dış eklem artrozu) | **%19.0** |
| 12 | **MCL** | İç Yan Bağ (Medial Collateral Ligament) zorlanması/yırtığı | **%15.5** |

---

## 📐 4. Değerlendirme Metriği (Evaluation Metric)

Yarışma, tahmin edilen olasılık skorları ile gerçek etiketler arasındaki **Makro Ortalama ROC AUC (Macro-Averaged Area Under the ROC Curve)** metriğine göre sıralanır:

$$\text{Final Score} = \frac{1}{12} \sum_{k=1}^{12} \text{AUC}_k$$

* Her hedef etiket için ayrı ayrı ROC AUC hesaplanır ve 12 etiketin aritmetik ortalaması alınır.
* Bu metrik sınıf dengesizliklerinden (imbalance) doğrudan etkilenmez; modelin pozitif ve negatif örnekleri ne kadar doğru sıraladığına bakar.

---

## 📂 5. Veri Seti Yapısı ve Dosyalar (Dataset Anatomy)

Veri seti uluslararası, çok merkezli ve farklı MR cihazlarından toplanmıştır:

```text
rsna-knee-abnormality-detection/
├── .env                       # Kaggle API anahtarı
├── config.py                  # Kimlik doğrulama script'i
├── kaggle_cli.py              # Kaggle CLI köprüsü
├── data/
│   ├── train.csv              # 4,407 vaka: [StudyInstanceUID, Report, 12 Target Labels]
│   ├── train_series.csv       # Her StudyInstanceUID için mevcut MR serileri ve düzlemleri
│   ├── test.csv               # Test seti inceleme kimlikleri (Gizli test setine submission ile evaluate edilir)
│   ├── test_series.csv        # Test seti serileri
│   └── sample_submission.csv  # 12 etiket sütunlu örnek format
├── eda.py                     # İstatistik ve veri inceleme aracı
├── baseline.py                # İlk çalışan model & submission üretici
├── submission.csv             # Oluşturulan tahmin dosyası
└── README.md                  # Bu ana doküman
```

### Önemli Noktalar:
1. **Multimodal Raporlar (`Report`):** Eğitim setindeki vakaların radyoloji raporları mevcuttur (İspanyolca/Çok dilli). NLP modelleri (ClinicalBERT, RoBERTa vb.) bu raporları işlemek için doğrudan kullanılabilir.
2. **Görüntüleme Serileri (`.dcm`):** Her vaka için Sagittal, Koronal ve Aksiyel planlarda T1, T2, PD ve Proton Yoğunluğu Yağ Baskılı (PD-FS) MR sekansları mevcuttur.

---

## 🚀 6. Hızlı Başlangıç (Quickstart)

### 1. Ortamı Doğrulama ve Veri İnceleme
```bash
python eda.py
```

### 2. Baseline Modelini Çalıştırma ve Tahmin Üretme
```bash
python baseline.py
```
Bu komut, eğitim etiketlerinin frekans dağılımını kullanarak standart formata uygun `submission.csv` üretir.

### 3. Kaggle'a Gönderim (Submission)
```bash
python kaggle_cli.py competitions submit -c rsna-knee-abnormality-detection -f submission.csv -m "Baseline frequency submission"
```

---

## 💡 7. İleri Düzey Çözüm Stratejileri (Roadmap)

1. **Text Modeli (NLP Track):**
   * Rapor metinleri üzerinde multilingual dil modeli (örn. `xlm-roberta-base` veya `Bio_ClinicalBERT`) ile multi-label sınıflandırıcı eğitmek.
2. **Görüntü Modeli (Vision Track - DICOM):**
   * 2.5D ve 3D CNN / Vision Transformer (ConvNeXt, EfficientNet-v2, CoAtNet) modelleri ile MR kesitlerini öznitelik vektörlerine dönüştürmek.
3. **Multimodal Late/Early Fusion:**
   * Hem görüntü hem rapor özniteliklerini birleştirip (Ensemble / CatBoost / MLP) nihai olasılık tahminlerini harmanlamak.
