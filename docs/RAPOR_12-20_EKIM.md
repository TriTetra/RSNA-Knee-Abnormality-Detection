# RSNA Knee Abnormality Detection — Çalışma Raporu (12–20 Ekim 2026)

Bu belge 12 Ekim'den 20 Ekim'e kadar yapılan işlerin tamamını kronolojik olarak anlatır: hangi yöntem
neden denendi, ne çıktı, nerede kazanç sağlandı, nerede hata yapıldı, hangi zorluklar yaşandı, kimlerden
ilham alındı ve neler okundu. Yöntemlerin teknik ayrıntıları ve terim sözlüğü için
[YONTEMLER.md](YONTEMLER.md), dosya bazlı değişiklikler için [CHANGELOG.md](../CHANGELOG.md).

> Takım: kadirov0 + miraysudemutlu (15 Ekim'de birleşildi) · Yarışma bitişi: 22 Ekim 23:59 UTC
> 20 Ekim itibarıyla en iyi public LB: **0.958** (10sa) · 6.219 takım · 0.958'de 1.272 takım berabere

---

## 1. Özet

| | |
|---|---|
| **Başlangıç (12 Ekim)** | Tek gönderim, LB **0.500** (eski pipeline sessizce sabit tahmin yazıyordu) |
| **Şu an (20 Ekim)** | LB **0.958** (public 0.958 tarifi, bizim düzenimizde) |
| **Kendi modellerimizin en iyisi** | CoAtNet Raptor-80, etiket v3: altın (58 vaka) 0.92; gizli testte tek başına tahmini **~0.939** |
| **En önemli olumlu bulgu** | Öğretmen–öğrenci pseudo etiketleme (etiket v3) altın skoru 0.901 → 0.922 çıkardı; LB etkisi 20 Ekim gönderimleriyle ölçülüyor |
| **En önemli olumsuz bulgu** | Kendi modellerimizi 0.958 tabanına eklemek her ağırlıkta skoru düşürdü (0.956 / 0.955 / 0.953). Sebep: modelimiz tabandan ~0.02 zayıf ve aynı model ailesinden (hataları benzer) |
| **En önemli ders** | 58 altın vaka LB'yi tahmin etmedi. Karar ölçütü LB ve blend formülüne çevrildi (bkz. §6) |

---

## 2. Kronoloji

### 12–13 Ekim · Teşhis ve yeniden yapılanma
**Ne yapıldı:**
- Repo incelendi. İlk gönderimin 0.500 almasının üç sebebi bulundu:
  1. Submission kernel'i checkpoint anahtarlarını eşleştiremeyip her vakaya önsel olasılık yazıyordu; hata vermediği için fark edilmemişti.
  2. Kesitler dosya adına göre sıralanıyordu (dosya adları rastgele SOP UID; geometriyle korelasyon ρ = 0.09–0.31).
  3. Sağ/sol diz ayrımı yapılmıyordu.
- Kod `src/` altında modüllere ayrıldı; her adım Türkçe açıklamalı bir notebook oldu (01–06).
- DICOM ön işleme: kesit sırası ImagePositionPatient·normal ile; taraf DICOM etiketinden, yoksa görüntü merkezinin x koordinatından (etiketle %98.5 uyum); sabit mm görüş alanı.
- Rapor etiketleri 58 altın vakaya karşı denetlendi: steven_v4 0.8927, pilkwang 0.8704.

**Neden:** Sessiz hatalar düzeltilmeden hiçbir model ölçümü güvenilir olmazdı.

### 13–14 Ekim · Taban skor, etiket v2, Raptor reçetesi
| Çalışma | Sonuç |
|---|---|
| Public 0.957 blend'in (heliosli) private kopyası gönderildi | **LB 0.956**, sıra 5896 → 723 |
| Cache (4.407 vaka) | 0 okuma hatası |
| Etiket v2 = steven + Claude Opus 4.5 (NTejas-1) + JEV | altın 0.8927 → **0.9014** |
| Qwen2.5-7B ile kendi LLM etiketlememiz | altın 0.797 → kullanılmadı |
| Raptor (CoAtNet-384) yeniden eğitimi, 44 kesit: 07 / 07b / 07c | altın 0.888 / 0.8965 / 0.9053 |
| Raptor corpus geometrisinin piksel korelasyonuyla bulunması (09) | span 0.15–0.85, 140 mm → korelasyon 1.000 |
| ConvNeXt-T, kendi pipeline'ımız, 5 fold | OOF altın 0.890; Raptor ile korelasyon 0.84 |
| Final v2: public 0.80 + Raptor-44 0.20 | **LB 0.953** (taban 0.956) |
| Kendi 80 kesitlik corpus'umuz (11) | 4.407 vaka, 0 hata |

**Neden Raptor:** Forumdaki en güçlü public tek model tarifiydi (yazarı 0.924–0.938 bildirmişti).
**Zorluk:** kadirov0'ın haftalık 30 saatlik Kaggle GPU kotası doldu; eğitimler takım arkadaşının hesabına taşındı (`kaggle_push.py --token-file`).

### 15 Ekim · Raptor-80, LB ölçümleri, etiket v3, yerel sunucu
| Çalışma | Sonuç |
|---|---|
| Raptor-80 (kendi corpus'umuz), etiket v2: 12 / 12b / 12c | altın 0.9056 / 0.9070 / 0.9009 |
| 10r: yalnızca Raptor-80 (12+12b) | **LB 0.928** |
| 10a: public 0.85 + Raptor-80 0.15 | **LB 0.955** (taban 0.956) |
| **Etiket v3** = v2 %50 + ConvNeXt 5-fold OOF %50 (öğretmen–öğrenci) | altın 0.9014 → **0.9223** (bootstrap %90 GA [+0.007, +0.035]) |
| 12v3: Raptor-80, etiket v3 | altın **0.9205** |
| Yerel sunucuda qwen3.5:27b ile 0–100 puan etiketleme | altın 0.881; v2'ye katkı yok → kullanılmadı |
| Yerel sunucu (RTX 5090) eğitim altyapısı | corpus npz → tar → sunucu; gece kuyruğu |

**Neden etiket v3:** Rapor etiketlerinin altın vakaya uyumu ~0.90'da tavan yapıyordu. Görüntü modelinin OOF tahminleri rapordan farklı bilgi taşır. Öğretmen–öğrenci damıtma, benzer yarışmaların ve forumdaki üst takımların ortak tekniği.

**Zorluklar:**
- Eski Kaggle CLI çıktı sayfalaması yapamadı (dosyaların sadece bir kısmı indi).
- Sunucunun bağlantısı 3 GB'lık arşivlerde kopuyordu → corpus 30'ar küçük arşive bölündü (`server_get_tar30.sh`).
- tmux oturumu sunucudaki başka bir araç tarafından kapatıldı → işler `setsid nohup` ile başlatıldı.

### 16 Ekim · 5 fold, araştırma, 0.958 tabanı, etiket v4
| Çalışma | Sonuç |
|---|---|
| Sunucuda Raptor-80 v3, 5 fold (etiket v4 için OOF) | f0 0.9121 · f1 0.9084 · f2 0.9041 · f3 0.9078 · f4 tamamlandı |
| Fold'lar 12v3'e ek mi? (altın) | 12v3 0.9205 → 12v3 + f0 + f1 0.9187; fold'lar 12v3 ile r = 0.96 → katkı yok |
| Leaderboard ve forum araştırması | Kalabalık 0.956–0.958'de; 1. sıra 0.968; ilk 50 sınırı 0.961 |
| Analiz notebook'u (leoprovorov) | Public modellerin hata korelasyonu ρ ≈ 0.86–0.89, aile tavanı ~0.960; public LB gürültüsü ~±0.003 |
| **Taban heliosli'nin 0.958 tarifine geçirildi** (Raptor 384 + 224 + R6 flip TTA, bulgu bazında ağırlıklar) | submission varyantları `make_blend_variants.py` ile üretiliyor |
| OAI dış veri denemesi | Login.gov kaydı **LG22 (coğrafi engel)** → bırakıldı (§5) |
| **Etiket v4** = v2 %50 + ConvNeXt OOF %25 + Raptor-80 OOF %25 | `train_labels_v4.csv` |
| 2. tur sunucu eğitimi (etiket v4) | CoAtNet s202 0.9110 · ConvNeXt-S 0.8950 · CoAtNet s303 0.9186 · DINOv2 hata verdi |

**Sorunlar (ayrıntı §4):** venv'siz başlatılan kuyruk; yükleme hatası yüzünden her fold 3 kez eğitildi; etiket v4 izleyicisi f4'ü kaçırdı (GPU 3,5 saat boş kaldı); DINOv2'de çözünürlük hatası.

### 17–18 Ekim · Hafta sonu
Sunucudaki 2. tur kuyruğu 16'sını 17'sine bağlayan gece bitti (01:05). Kaggle kotaları 17 Ekim'de yenilendi. Yeni iş başlatılmadı.

### 19 Ekim · 0.958 tabanı üzerinde LB ölçümleri
| Gönderim | İçerik | LB |
|---|---|---|
| **10sa** | heliosli 0.958 tarifi, olduğu gibi | **0.958** |
| 10sb | + 12v3 %10 | 0.956 |
| 10sc | + 12v3 %20 | 0.955 |
| 10se | + (12v3 %70 + DINOv2 %30) %20 | 0.955 |
| 10sd | + 12v3 %30 | 0.953 |

Ek eğitimler (sunucu, etiket v3):
| Model | Altın |
|---|---|
| DINOv2 ViT-B, 336 px (s202 / s303) | 0.9045 / 0.9116 (12v3 ile r = 0.949) |
| CoAtNet s404 / s505 | 0.9177 / **0.9216** |
| 3 CoAtNet v3 seed birlikte | 0.9219 |

**Teşhis (17 numaralı notebook):** Modelimizin LB'de neden zayıf kaldığını anlamak için gizli testteki çıkarım boru hattı 58 altın vakaya uygulandı: 12v3 **0.9227** (eğitimde 0.9205), s505 **0.9199** (0.9216). Ön işlemede hata yok.

**Blend formülü:** İki modelin rank blend'i için ikili-normal model (bkz. §6) dünkü 4 ölçüme oturtuldu. Gözlemleri ±0.0002 içinde açıkladı; modelimizin gizli testte tek başına gücü **~0.939** çıktı.

### 20 Ekim · Forum ve tek model araştırması, doğrulama gönderimleri
- **Danial Zakaria'nın 0.949 tarifi:** Başlangıç noktası (80 kesit, 78 pencere, CoAtNet) **0.939**; bu, formülün bizim için verdiği değerle aynı. Yazarın 0.949'a çıkışının +0.008'i OAI dış veri ve 384 tam alan tarifinden gelmiş.
- **"Best single-model score" tartışması:** 2. sıradaki takım: ConvNeXt-T 5 fold 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952. Diğer üst takımlar: 2.5D ConvNeXt-T / DINOv2-S + pseudo etiket 0.951–0.953. Ortak tema: etiket kalitesi ve pseudo etiket; küçük omurga, 224 px.
- **Doğrulama gönderimleri** (puanlanıyor):

| Gönderim | Soru |
|---|---|
| 10t: heliosli 0.9586 tarifi (Miray) | 2. final adayı olabilir mi? |
| 10sf: yalnız 12v3 (etiket v3) | Pseudo etiketleme LB'de ne kazandırdı? (10r, v2: 0.928) |
| 10sg: yalnız CoAtNet s303 (etiket v4) | İkinci etiket turu ek kazanç getiriyor mu? |
| 10sh: yalnız 3 CoAtNet v3 seed | Seed ortalamasının LB etkisi |

---

## 3. Yöntemler ve kazanımlar

| Yöntem | Neden | Ölçüm | Kazanım |
|---|---|---|---|
| Geometrik kesit sıralama, taraf tespiti, mm ölçek | Eski pipeline'ın 0.500 hatası | 4.407 vaka 0 hata | Temel doğruluk |
| Sessiz hata korumaları | 0.500 dersi | — | Her geri dönüş görünür |
| Public blend'i taban almak | En güçlü mevcut sistem | 0.956 → 0.958 | **+0.458 → 0.958** |
| Etiket v2 (3 LLM kaynağının ortalaması) | Tek kaynağın gürültüsünü azaltmak | altın 0.8927 → 0.9014 | +0.009 altın |
| Etiket v3 (ConvNeXt OOF ile pseudo etiket) | Rapor tavanını görüntü bilgisiyle aşmak | altın 0.9014 → 0.9223 | +0.021 altın; LB ölçülüyor |
| Etiket v4 (+ Raptor OOF) | İkinci öğretmen | Raptor altın 0.911–0.919 | Altında fark yok; LB ölçülüyor |
| Kendi 80 kesitlik corpus | Daha yoğun kesit (forumda +0.004) | 44 → 80 kesit: altın +0.005–0.010 | Küçük |
| Yerel sunucu (RTX 5090) | Kaggle kotasından bağımsız eğitim | Bir CoAtNet-384 ~1–1,2 saat (Kaggle T4'te ~9 saat) | ~8× hız |
| Blend formülü (ikili-normal) | Gönderim öncesi kazanç tahmini | 4 LB ölçümünü ±0.0002 açıkladı | Karar ölçütü |

---

## 4. Hatalarımız

| # | Hata | Bedeli | Ders / düzeltme |
|---|---|---|---|
| 1 | Model seçimini 58 altın vakaya göre yaptık | Altında iyi görünen modeller LB'de blend'i düşürdü | Karar ölçütü LB + formül; altın yalnızca büyük farklar için |
| 2 | Bütün eğitim yatırımını Raptor-CoAtNet ailesine yaptık | Modellerimiz hem tabandan zayıf hem ona çok benzer | Üst takımların en iyi tek modeli ConvNeXt-T; çeşitlilik farklı aileden gelir |
| 3 | Sunucu kuyruğu venv etkinleştirilmeden başlatıldı | 15→16 Ekim gecesi hiç eğitim olmadı (`numpy` yok) | Kuyruk script'i venv'i kendisi etkinleştiriyor |
| 4 | Yükleme adımındaki hata eğitimi başarısız saydırdı | Her fold 3 kez eğitildi, ~3 saat GPU kaybı | Yükleme ayrı script'te; hatası eğitimi etkilemiyor |
| 5 | Etiket v4 için f4'ü izleyen kod dosyayı listede bulamadı | GPU ~3,5 saat boş kaldı | Varlık kontrolü listeleme yerine doğrudan indirme denemesiyle |
| 6 | DINOv2'yi 384 px ile başlattık (14'e bölünmüyor) | Bir eğitim hakkı | Çözünürlük 336 px; script'e `RES` değişkeni eklendi |
| 7 | Gönderim başına kota maliyetini 3–4 saat sandık | Gereksiz plan karmaşası | Kayıt çalışması yalnız 3 görünür test vakasıyla ~10 dk |
| 8 | Kaggle token'ı sohbete ve sunucu komut geçmişine düştü | Güvenlik riski | Token dosyada (`~/.kaggle/access_token`, 600 izin); **token'ın yenilenmesi bekliyor** |
| 9 | Doğrulama yapmadan plan önerdik (ör. 5 fold ensemble, OAI) | Zaman ve dikkat | Her adımın önceden tanımlı ölçüm ve eşik kuralı var |

---

## 5. Zorluklar

| Zorluk | Etkisi | Çözüm |
|---|---|---|
| Kaggle haftalık 30 saat GPU, aynı anda 2 oturum | 14–16 Ekim eğitimleri sıkıştı | Takım arkadaşının hesabı + yerel sunucu |
| Kaggle CLI sürümleri (1.7 sayfalama yapamıyor) | Corpus eksik indi | Sunucuda ayrı Python 3.11 ortamında CLI 2.2.4 |
| Sunucu ağında uzun indirmelerde kopma | 3 GB arşivler inmedi | 30 küçük arşiv + bütünlük kontrolü + yeniden deneme |
| Sunucuda tmux oturumlarının kapanması | Gece işi durdu | `setsid nohup`; ardından ayrı soketli tmux (`tmux -L rsna`) |
| Sunucu paylaşımlı (başka projeler, disk %98–99) | GPU/disk çakışması riski | Bellek eşikli kuyruk; gereksiz checkpoint'ler silindi |
| OAI erişimi coğrafi olarak engelli (LG22) | En büyük dış veri kaldıracı yok | Bırakıldı; engeli aşmak kurallara ve veri şartlarına aykırı |
| Kaggle puanlama kuyruğu (son hafta) | Gönderim sonuçları 4–9 saat sürdü | Gönderimler sabah toplu yapılıyor |
| 58 altın vakanın gürültüsü (±0.01–0.03) | Model karşılaştırmaları güvenilmez | §6 yöntemi |

---

## 6. Karar yöntemi: blend kazancının önceden hesabı

Bir modelin 0.958 tabanına ne katacağını iki sayı belirler: modelin gizli testteki tek başına gücü ve tabanla hata korelasyonu (ρ). İkili-normal model ile her bulgu için skor = d·y + gürültü alınır; tek modelin AUC'si Φ(d/√2), ağırlık w ile karışımın AUC'si:

AUC(w) = Φ( ((1−w)·d₁ + w·d₂) / √(2·((1−w)² + w² + 2w(1−w)ρ)) )

19 Ekim'deki 4 gönderime oturtulduğunda gözlemleri ±0.0002 içinde açıkladı. Bir adayın +0.002 getirmesi için gerekenler:

| Tabanla korelasyon ρ | Gereken tek başına LB |
|---|---|
| 0.60 | ≥ 0.900 |
| 0.70 | ≥ 0.921 |
| 0.80 | ≥ 0.940 |
| 0.86 | ≥ 0.949 |
| 0.90 | ≥ 0.954 |

Sınırlılık: 4 noktalık ölçümde güç ve korelasyon birbirinden tam ayrılamaz. Bu yüzden 20 Ekim'de modellerin tek başına gücü doğrudan ölçülüyor (10sf/10sg/10sh).

---

## 7. İlham kaynakları ve okumalar

| Kaynak | Ne öğrendik / ne aldık |
|---|---|
| dreaddevelopment — Raptor eğitim/çıkarım kodu ve corpus | CoAtNet + 2.5D + bulguya özgü dikkat havuzlama; `third_party/raptor/` |
| heliosli — 0.956 / 0.958 / 0.9586 public blend'ler | Taban sistem; bulgu bazında ağırlıklar |
| nartaa (Danial Zakaria) — "Reaching 0.949 with one CoAtNet" | Tarif tablosu (0.939 başlangıç), ayna TTA, OAI etkisi, etiket karışımı, "altında iyi, LB'de kötü" uyarısı |
| leoprovorov — "0.957 Explained" el kitabı | Hata korelasyonu ρ ≈ 0.86–0.89, aile tavanı ~0.960, LB gürültüsü |
| varshithakolupuri — "real scores of every public notebook", "what the top notebooks are made of" | Beraberlik blokları, tie-break kuralı, çeşitliliğin kaldıraç olması |
| goodpjw2008 — 2.5D ConvNeXt okuyucu ve ölçümleri | Altın vakalara göre bulgu ağırlığı ayarı LB'de zararlı; S6 + R6 zararlı |
| lavinwins — "Train your own model" | Public modelleri yeniden karıştırmanın getirisi tükendi |
| hengck23 — "Tricks for the competition" | Etiket kalitesi belirleyici; altın etiketler görüntüden; az parametreli 58 vaka yöntemi; ajan/VLM ile etiketleme fikri |
| "Best single-model score" (735304) | Üst takımların tek model skorları ve tarifleri (ConvNeXt-T 0.959) |
| "external datasets" (743416) ve ev sahibi kararı | Dış veri kuralı: herkesin eşit erişebildiği veri |
| stevenleehans, pilkwang, NTejas-1 (Claude Opus 4.5), riadmohamed42 (JEV) | Etiket v1/v2 kaynakları |
| RSNA 2024 Lumbar (Avengers), RSNA 2023 Abdominal (Team Oxygen) 1. sıra çözümleri | Attention-MIL, küçük omurgalar, yardımcı kayıplar, iki aşamalı ROI |
| RoS-KD (Jaiswal ve ark., arXiv 2210.08388) | Gürültülü etiketlerde çok-öğretmenli damıtma |
| BTK Datathon 2026 (önceki çalışmamız) | LLM rubrik etiketleme, öğretmen–öğrenci damıtma |

---

## 8. Açık konular ve sonraki adımlar

1. **20 Ekim doğrulama gönderimlerinin sonuçları** (10t, 10sf, 10sg, 10sh). Karar kuralı:
   - 10sf ≥ 0.938 ve 10sg > 10sf → etiket turları LB'de işe yarıyor → 2.5D ConvNeXt-T, 224 px, 5 fold bu etiketlerle eğitilir.
   - 10sf ≥ 0.938, 10sg ≤ 10sf → ConvNeXt-T etiket v3 ile.
   - 10sf < 0.935 → yeni eğitim yapılmaz.
2. **Final seçimi (22 Ekim 23:59 UTC):** 10sa (0.958) + (10t ya da ölçülmüş en iyi aday). Kaggle'da elle işaretlenmeli.
3. **Kaggle token'ının yenilenmesi.**
4. **OAI ile eğitilmiş public checkpoint'ler:** Ev sahibi net karar vermedi; sıralamamız için pratik risk düşük, ödül denetimine giren takımlar için önemli.
