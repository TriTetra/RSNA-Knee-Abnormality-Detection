# RSNA Knee Abnormality Detection

> **Yarışma:** [kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · RSNA · $77.000 · Research Code Competition
> **Son gönderim:** 22 Ekim 2026 · **Takım birleştirme son günü:** 15 Ekim 2026
> **Kurallar:** günde 5 gönderim, 2 final seçimi, notebook ≤ 9 saat, internet kapalı, public harici veri/model serbest.

Her diz MR incelemesi (`StudyInstanceUID`) için 12 bulgunun olasılığı tahmin edilir.
Metrik: 12 bulgunun ROC AUC'sinin ortalaması (yalnızca **sıralama** önemlidir).

📘 **Yöntemler ve sözlük:** [docs/YONTEMLER.md](docs/YONTEMLER.md) ·
🗓 **12–20 Ekim çalışma raporu:** [docs/RAPOR_12-20_EKIM.md](docs/RAPOR_12-20_EKIM.md) ·
🏁 **Sonuç ve kazanan çözümler:** [docs/SONUC.md](docs/SONUC.md) ·
📝 **Değişiklik günlüğü:** [CHANGELOG.md](CHANGELOG.md)

## Sonuç (yarışma 22 Ekim 2026'da bitti)

| | |
|---|---|
| Takım | theartificialteam (kadirov0 + miraysudemutlu) |
| Final (ön) sıra | **1.730 / 6.425** (private LB, kesinleşme bekleniyor) |
| En iyi skor | public **0.958** · private **0.942** (10sa, 10t, 10sh) |
| Kendi modelimizin en iyisi | Raptor-80 CoAtNet, etiket v3: public 0.929 · private 0.915 |
| 1. sıra | private 0.960 (Team Kangaroo) |

Ayrıntılı değerlendirme ve kazananların yöntemleri: [docs/SONUC.md](docs/SONUC.md).

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

## 2. Notebook'lar

| klasör | notebook'lar | ne yapar |
|---|---|---|
| `notebooks/` | 01–06, 05b | Ana pipeline: EDA, etiketler, DICOM ön işleme, cache, ConvNeXt eğitimi, submission |
| `notebooks/raptor/` | 07, 07b, 07c, 09, 12, 12b, 12c, 12v3, 17 | Raptor (CoAtNet MIL) yeniden eğitimleri, corpus geometri kontrolü, çıkarım–altın doğrulaması |
| `notebooks/veri_hazirlik/` | 11, 14, 15, 16 (×3 parça) | 80 kesitlik corpus üretimi ve sunucuya taşınması (npz → tar) |
| `notebooks/submission/` | 10, 10r, 10sa–10sh, 10t, 10v3, `_public/` | Final blend notebook'u, LB ölçüm varyantları, public 0.958 / 0.9586 tarifleri |
| `notebooks/arsiv/` | 08, 13 | Kullanılmayan denemeler (Qwen-7B etiketleme, Colab eğitimi) |

Final submission'lar: `submission/10sa_blend_kaggle.ipynb` ve `submission/10t_heliosli_09586_kaggle.ipynb`
(her ikisi public 0.958 / private 0.942). Varyantlar `scripts/make_blend_variants.py` ile üretilir.

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

## 6. Strateji ve sonuç

1. **Taban:** public blend (önce 0.956, sonra heliosli'nin 0.958 tarifi).
2. **Kendi modellerimiz:** Raptor-80 (CoAtNet, DINOv2, ConvNeXt), rapor etiketleri + pseudo etiket (v3, v4).
   Tek başına public 0.926–0.929 kaldı; tabana her ağırlıkta eklenmesi skoru düşürdü (0.958 → 0.953).
3. **Final:** iki 0.958 submission (biri bulgu başına ağırlıklı, biri sabit 50/50 + 2.5D okuyucu).

Neden kazananlardan geri kaldığımız ve dersler: [docs/SONUC.md](docs/SONUC.md).
