# RSNA Knee Abnormality Detection

> **Yarışma:** [kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · RSNA · $77.000 · Research Code Competition
> **Son gönderim:** 22 Ekim 2026 · **Takım birleştirme son günü:** 15 Ekim 2026
> **Kurallar:** günde 5 gönderim, 2 final seçimi, notebook ≤ 9 saat, internet kapalı, public harici veri/model serbest.

Her diz MR incelemesi (`StudyInstanceUID`) için 12 bulgunun olasılığı tahmin edilir.
Metrik: 12 bulgunun ROC AUC'sinin ortalaması (yalnızca **sıralama** önemlidir).

📘 **Yöntemler, deney sonuçları ve kararlar:** [docs/YONTEMLER.md](docs/YONTEMLER.md) ·
📝 **Değişiklik günlüğü:** [CHANGELOG.md](CHANGELOG.md)

---

## 1. Projenin mantığı (tek paragraf)

58 vaka dışında etiket yok; etiketler raporlardan türetiliyor (`02`). Test setinde rapor
olmadığı için model yalnızca görüntüden öğrenmek zorunda. Görüntüyü doğru okumak
(kesit sırası, yön, mm ölçeği, sağ/sol diz) bu yarışmanın en kritik ve en sessiz
hata kaynağı (`03`). Her vaka bir kez okunup cache'lenir (`04`), model cache'ten eğitilir
(`05`), submission notebook'u gizli test setini aynı ön işlemeyle okuyup tahmin eder (`06`).

```
train.csv (rapor) ──► 02 etiketler ──► train_labels.csv ─┐
                                                         ├──► 05 eğitim ──► model_fold*.pt ──► 06 submission ──► submission.csv
DICOM ──► 03 ön işleme (src/dicom.py) ──► 04 cache ──────┘                                        ▲
                                    └──────────────── aynı fonksiyonlar ───────────────────────────┘
```

## 2. Notebook'lar (sırayla okuyun)

| # | notebook | nerede çalışır | ne yapar |
|---|---|---|---|
| 01 | `01_exploratory_data_analysis.ipynb` | yerel | veri yapısı, etiket dağılımı, diller, sekanslar |
| 02 | `02_etiketler.ipynb` | yerel ✅ çalıştırıldı | etiket kaynaklarını 58 altın vakaya karşı ölçer, `data/train_labels.csv` üretir |
| 03 | `03_dicom_onisleme.ipynb` | yerel ✅ çalıştırıldı | ön işlemenin her adımını gerçek bir vaka üzerinde görsel olarak gösterir |
| 04 | `04_cache_kaggle.ipynb` | Kaggle CPU | tüm eğitim vakalarını kanonik `.npz` hacimlere çevirir (~12 GB) |
| 05 | `05_egitim_kaggle.ipynb` | Kaggle GPU | modeli fold bazında eğitir, OOF + checkpoint yazar |
| 06 | `06_submission_kaggle.ipynb` | Kaggle GPU, internet kapalı | test inference + korumalar + `submission.csv` |

04–06 yerelde "dry-run" modunda da çalışır (küçük model, tek test vakası) — Kaggle'a
göndermeden önce kodun uçtan uca çalıştığını doğrulamak için.

## 3. Kod modülleri (`src/`)

Notebook'lar mantığı içermez; hepsi bu modülleri çağırır. Böylece eğitim ve inference
**aynı** kodu kullanır.

| modül | sorumluluk |
|---|---|
| `config.py` | yollar (yerel / Kaggle otomatik), 12 hedef sütun, Kaggle API |
| `labels.py` | etiket kaynaklarını yükleme, altın vaka denetimi, eğitim etiket tablosu |
| `dicom.py` | DICOM başlığı, geometrik kesit sıralama, yön standardizasyonu, sağ/sol tespiti, mm'ye göre yeniden örnekleme |
| `cache.py` | vaka → 6 slot (düzlem × sekans) → `uint8 [slot,16,224,224]` npz; paralel cache üretimi |
| `dataset.py` | cache'ten 2.5D pencereler, veri artırma, maskeli batch birleştirme |
| `model.py` | ConvNeXt omurga + slot/konum gömmesi + Transformer bağlamı + 12 başlı dikkat havuzlama |
| `engine.py` | fold bölme, eğitim/tahmin döngüsü, `pseudo_auc` / `gold_auc` |
| `inference.py` | test inference (cache'siz), sessiz hata korumaları, rank blend, submission doğrulama |

Yardımcılar: `scripts/kaggle_cli.py` (`.env` token'ı ile Kaggle CLI),
`scripts/kaggle_push.py` (src dataset'i ve 04/05/06 kernel'lerini **private** gönderir),
`tests/smoke_test.py` (yerel uçtan uca test).

## 4. Çalıştırma

```bash
pip install -r requirements.txt
python tests/smoke_test.py                  # yerel uçtan uca kontrol (data/test_series altında 1 vaka gerekir)
python scripts/kaggle_push.py src           # src/ + train_labels.csv -> kadirov0/rsna-knee-src (private)
python scripts/kaggle_push.py kernel 04     # cache (CPU)  -> bitince:
python scripts/kaggle_push.py kernel 05     # eğitim (GPU) -> bitince:
python scripts/kaggle_push.py kernel 06     # submission (GPU)
python scripts/kaggle_push.py status 05
```

Kaggle'a gönderim (kernel versiyonu çalıştıktan sonra):
```bash
python scripts/kaggle_cli.py competitions submit -c rsna-knee-abnormality-detection -k kadirov0/rsna-knee-06-submission -v <versiyon> -f submission.csv -m "açıklama"
```

## 5. Eski pipeline'da bulunan ve düzeltilen hatalar

| hata | etkisi | düzeltme |
|---|---|---|
| Kesitler dosya adına göre sıralanıyordu (dosya adı = rastgele UID, ρ≈0.1–0.3) | 2.5D pencereler ilgisiz kesitlerdi | `ImagePositionPatient` ile geometrik sıralama |
| Sağ/sol diz ayrımı yok | medial/lateral 5 hedef öğrenilemiyordu | etiket, yoksa geometri ile taraf; sağ dizler aynalanıyor |
| Sabit piksele boyutlandırma | görüntülerin fiziksel ölçeği farklıydı | 160 mm sabit FOV |
| Kernel okuyamayınca sabit önsel yazıyordu | LB **0.500** | okunamayan oran > %2 ise hata; `strict=True` checkpoint |
| Checkpoint anahtarı uyuşmuyordu (`model_state_dict` ≠ `model`) | yerel inference rastgele ağırlıkla | tek format, `strict=True` |
| Doğrulama yalnızca LLM etiketine karşı | iyimser/yanlı metrik | `gold_auc` ayrı raporlanıyor |
| Her epoch DICOM okuma | çok yavaş eğitim | tek seferlik cache |

## 6. Strateji ve durum

Leaderboard (13 Ekim): 1. sıra 0.966, ilk 100 ≈ 0.957, 200–700. sıralar 0.956'da yığılmış
(public blend notebook'ları). Public en iyi notebook'lar birkaç public ağırlık setinin
rank-blend'idir.

1. **Taban:** public 0.956–0.957 blend notebook'unun private kopyası (`kadirov0/rsna-knee-public-blend-heliosli`).
2. **Kendi modelimiz:** doğru ön işlemeyle eğitilmiş, public modellerden farklı bir model.
   Tek başına ~0.94 bile olsa düşük korelasyonu sayesinde blend'i yukarı taşıyabilir.
3. **Blend:** public blend + bizim model, ağırlıklar önceden sabit (public LB'ye uydurulmaz).
4. **Final:** biri temkinli, biri agresif iki gönderim.
