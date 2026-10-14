# Değişiklik Günlüğü (Changelog)

Projedeki önemli değişiklikler tarih sırasıyla (en yeni üstte) burada tutulur.
Biçim: [Keep a Changelog](https://keepachangelog.com/tr/1.1.0/).
Yöntemlerin ayrıntılı açıklaması ve deney sonuçları için: [docs/YONTEMLER.md](docs/YONTEMLER.md).

Kategoriler: **Eklendi** · **Değişti** · **Düzeltildi** · **Kaldırıldı** · **Deney** (Kaggle'da çalıştırılan işler) · **Bilinen sorunlar**

---

## [Yayınlanmadı] — 2026-10-14

### Eklendi
- `third_party/raptor/train_knee.py`: dreaddevelopment'ın public Raptor eğitim script'inin
  Kaggle T4 için yamalı kopyası. Yamalar: bf16 yerine fp16 + GradScaler (T4 bf16'yı
  donanımda desteklemiyor), `--time_budget_h` (oturum sınırından önce temiz bitiş),
  `--log_every`. Model ve veri mantığı orijinal; kaynak dosyanın başında belirtilmiş.
- `notebooks/07_raptor_yeniden_egitim_kaggle.ipynb`: Raptor reçetesini (CoAtNet-384) kendi
  etiketlerimizle yeniden eğitir.
- `notebooks/08_llm_etiketleme_kaggle.ipynb`: raporları Qwen2.5-7B-Instruct ile okuyup her bulgu
  için P/A/N/U (var / yok / bahsedilmiyor / belirsiz) kodu üretir; 58 altın vakaya karşı ölçer.
- `src/dicom.py → ensure_decoders()`: JPEG Lossless / JPEG 2000 DICOM çözücülerini
  (pylibjpeg, openjpeg, gdcm) internetsiz kurar ve doğrular.
- `third_party/wheels/` (git dışı): çözücülerin Linux wheel'leri (Python 3.11–3.13).
- `CHANGELOG.md` ve `docs/YONTEMLER.md`.

- **Etiket v2** (`data/train_labels_v2.csv`, `src/labels.py → SOURCES_V2`): steven_v4 + Claude Opus 4.5
  (NTejas-1, MIT) + JEV HYBRID (riadmohamed42) ortalaması; 58 altın vakada 0.8927 → **0.9014**.
- `notebooks/07b_raptor_etiket_v2_kaggle.ipynb`: Raptor reçetesi, etiket v2 ve `--norm imagenet` ile.
- `docs/YONTEMLER.md` 7b: yarışma forumundan öğrenilenler (LLM API izni, 0.949 tek model reçetesi,
  altın etiket–rapor çelişkileri).

- `third_party/raptor/infer_knee.py`: yazarın 0.924 inference kodu; geometri/normalizasyon/ARMS/çıktı
  `RAPTOR_CFG` JSON ile ayarlanabilir, ham olasılıklar da kaydedilir.
- `notebooks/09_raptor_geometri_kontrol_kaggle.ipynb`: corpus geometrisini piksel korelasyonuyla doğrular.
  Sonuç: span 0.15–0.85, 140 mm (korelasyon 1.000; inference varsayılanı 0.75 verirdi).
- `notebooks/10_final_blend_kaggle.ipynb`: public blend + bizim Raptor modelleri + sabit ağırlıklı
  rank blend; bizim aşama başarısızsa public submission korunur.

### Düzeltildi
- **Yavaş girdi araması:** `Path("/kaggle/input").rglob(...)` yarışmanın ~730 bin DICOM dosyasını
  dolaşıyordu (05 eğitimi dosya ararken ~25 dk GPU kaybetti; final notebook'ta gizli testte süre
  riski). `src/config.py → find_input / find_inputs` DICOM klasörlerine girmeden arar; tüm
  notebook'lar ve `labels.py` buna geçirildi.
- `.gitignore`: iki satır birleşmişti (`!experiments/.gitkeepexperiments/cache/`), 12 GB'lık cache
  git'e girebilirdi.
- Final notebook (10): Aşama B süre bütçesi 2 → 4.5 saat.

### Değişti
- `notebooks/02_etiketler.ipynb`: etiket v1 ve v2 birlikte; v2 için karışım karşılaştırma tablosu.
- `notebooks/05_egitim_kaggle.ipynb`: etiket v2 ile eğitir.
- `src/labels.py`: `macro_auc` boş tahmin hücrelerini medyanla doldurur; `load_source` `STUDY_UID`
  sütun adını da tanır.
- `notebooks/04_cache_kaggle.ipynb` ve `06_submission_kaggle.ipynb`: başa çözücü kurulum hücresi
  eklendi. Submission notebook'u çözücüler yoksa hata verir (sessizce seri atlamaz).
- `notebooks/05_egitim_kaggle.ipynb`: adım başına vaka sayısı 4 → 2, gradyan biriktirme 2 → 4
  (etkin batch aynı, T4 belleğine sığması için).
- `scripts/kaggle_push.py`: 07 ve 08 kernel'leri eklendi; `third_party/` src dataset'ine dahil;
  04 için internet açıldı.

### Deney
- Final notebook 10 testi (public + 07) başarılı: üç aşama hatasız, final blend formülü doğrulandı.
- Kendi modelimiz 05 (ConvNeXt-Tiny, etiket v2, fold 0) Kaggle'da eğitiliyor.
- Raptor 07 (etiket v1) tamamlandı: altın AUC 0.888, SWA 0.8883.
- Final notebook 10, yalnızca 07 checkpoint'i ile test çalıştırmasına gönderildi.
- LLM etiketleme (08) tamamlandı: Qwen-7B altın AUC 0.797, etiket v2'yi düşürdüğü için kullanılmıyor.
- `rsna-knee-07b-raptor-labels-v2` 08 biter bitmez otomatik başlatıldı.
- **Gönderim:** public blend v1 (`kadirov0/rsna-knee-public-blend-heliosli` v1): **LB 0.956**, sıra 5896 → 723;
  takım birleşmesi tamamlandı (günlük 5 hak artık takım ortak).
- Kaggle'da paralel çalışan işler (2 GPU + 1 CPU): `rsna-knee-04-cache`,
  `rsna-knee-07-raptor-retrain`, `rsna-knee-08-llm-labels`. Sonuçlar gelince
  `docs/YONTEMLER.md` deney tablosuna işlenecek.

- Cache v1 tamamlandı: 4.407 vaka, 0 okuma hatası; geometrik taraf tespiti etiketli vakalarda
  %98.5 uyumlu. Çözücü endişesi bu ortam için geçersiz çıktı, v2 gerekmedi.
- Kaggle'ın aynı anda en fazla 2 GPU oturumu sınırı ölçüldü; 05 eğitimi sıraya alındı.
- LLM etiketleme v1 bellek yetmezliği (OOM) ile 1.464/4.407 raporda durdu; v2 kısmi çıktıdan devam
  eder, batch boyutunu rapor uzunluğuna göre ayarlar ve OOM'da batch'i yarıya indirir.

---

## 2026-10-13 — Pipeline'ın yeniden yapılandırılması

### Eklendi
- Modüler kod (`src/`): `labels.py`, `dicom.py`, `cache.py`, `dataset.py`, `model.py`,
  `engine.py`, `inference.py`. Notebook'lar mantık içermez, bu modülleri çağırır; böylece eğitim
  ve tahmin aynı kodu kullanır.
- Açıklamalı notebook serisi:
  - `02_etiketler.ipynb` (yerelde çalıştırıldı): etiket kaynaklarının 58 altın vakaya karşı
    denetimi, `data/train_labels.csv` üretimi.
  - `03_dicom_onisleme.ipynb` (yerelde çalıştırıldı): ön işlemenin her adımının gerçek bir test
    vakasında görsel gösterimi.
  - `04_cache_kaggle.ipynb`, `05_egitim_kaggle.ipynb`, `06_submission_kaggle.ipynb`.
- `scripts/kaggle_push.py`: src dataset'ini ve kernel'leri Kaggle'a **private** gönderir.
- `tests/smoke_test.py`: yerel uçtan uca test (cache → dataset → eğitim → test inference).
- Harici etiket kaynağı: `pilkwang/rsna-knee-llm-labels` (`report_labels_v2.csv`).
- Kaggle'da private dataset `kadirov0/rsna-knee-src` (src + etiket tablosu).
- Kaggle'da public 0.957 blend notebook'unun private kopyası
  (`kadirov0/rsna-knee-public-blend-heliosli`): sorunsuz çalıştı, `submission.csv` üretti.

### Düzeltildi
- **Kesit sıralaması:** dosya adına göre sıralama yerine `ImagePositionPatient` ile geometrik
  sıralama. Dosya adları rastgele UID'dir (ölçülen korelasyon 0.09–0.31).
- **Sağ / sol diz:** `Laterality` etiketi, yoksa görüntü merkezinin hasta x koordinatı ile taraf
  tespiti; sağ dizler sol görünüme aynalanır.
- **Fiziksel ölçek:** sabit piksele sündürme yerine sabit 160 mm görüş alanı.
- **Yoğunluk normalizasyonu:** kesit başına değil seri başına.
- **Sessiz geri dönüş (LB 0.500'ün sebebi):** okunamayan vaka oranı %2'yi aşarsa ya da
  checkpoint uyuşmazsa submission hata verir; `strict=True` checkpoint yükleme.
- **Checkpoint anahtar uyuşmazlığı** (`model_state_dict` ≠ `model`): tek format.
- **Doğrulama:** LLM etiketine uyumdan ayrı olarak `gold_auc` (altın vaka) raporlanır.
- Boş (okunamayan) vakanın NaN üretip rank-blend'i bozması; tek bozuk DICOM dosyasının tüm seriyi
  düşürmesi.

### Kaldırıldı
- `scripts/train.py`, `scripts/inference_submission.py`, `scripts/audit_labels.py`,
  `scripts/baseline.py`, `scripts/eda.py`, `src/data/`, `src/models/`,
  `notebooks/02_kaggle_submission_kernel.ipynb`,
  `notebooks/rsna-knee-2-5d-multi-view-attention-training.ipynb`.
  Hepsi yukarıdaki hataları içeriyordu; git geçmişinde duruyor.

### Değişti
- `README.md` yeni yapıya göre baştan yazıldı.
- Etiket stratejisi: "altın vakada en yüksek skorlu kaynağı seç" yerine iki bağımsız kaynağın
  eşit ağırlıklı ortalaması (58 vaka kaynakları ayırt etmeye yetmiyor).

---

## 2026-10-08
### Değişti
- Eğitim çözünürlüğü 256; slayt ön-örnekleme ve alt-örneklenmiş yüzdelik ile disk darboğazının
  azaltılması; epoch başına checkpoint.
### Deney
- 2026-10-06 tarihli submission: **LB 0.500** (kernel modeli yükleyemeyip sabit önsellere düştü;
  2026-10-13'te teşhis edildi).

## 2026-10-06
### Eklendi
- İlk eğitim pipeline'ı: `KneeStudyDataset`, `KneeStudyModel` (12 başlı dikkat havuzlama),
  submission kernel'i, LLM etiket denetim script'i ve `train_annotated.csv`.

## 2026-10-05
### Eklendi
- Proje iskeleti, Kaggle CLI köprüsü, `01_exploratory_data_analysis.ipynb`.
