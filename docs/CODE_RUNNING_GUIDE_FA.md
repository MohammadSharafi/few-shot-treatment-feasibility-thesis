# راهنمای اجرای کدها و توضیح فنی پایان‌نامه

این راهنما برای بخش فنی دفاع نوشته شده است: چه چیزی را اجرا کنی، چه خروجی ببینی، و هر خروجی چه چیزی از پایان‌نامه را نشان می‌دهد.

## سؤال اصلی پایان‌نامه

پرسش محوری این است:

> آیا می‌توان با دادهٔ محدود، حساس و غیرمتمرکز ICU، امکان‌پذیری درمان/ترخیص بیمار را با ترکیبی از مدل‌های کلاسیک، نمایش توکنی علائم، few-shot learning و یادگیری فدرال پیش‌بینی کرد؟

پایان‌نامه قرار نیست بگوید «few-shot خام همیشه بهتر از مدل کلاسیک است». نتیجهٔ درست و دفاع‌پذیر این است:

> few-shot خام روی دادهٔ جدولی بالینی از Random Forest ضعیف‌تر بود، اما وقتی embedding/few-shot با مدل‌های کلاسیک در یک pipeline ترکیبی/stacked ادغام شد، عملکرد به سقف جدولی نزدیک شد.

## اجرای خیلی ساده برای روز دفاع

از ریشهٔ پروژه اجرا کن:

```bash
cd /Users/moe/Programming/Thesis-curser
source .venv/bin/activate
python scripts/defense_demo.py
```

چیزی که می‌بینی:

- `PASS Processed thesis dataset`: دیتاست نهایی آماده است.
- `PASS Token tensor`: نمایش توکنی/عددی بیماران ساخته شده است.
- `PASS Final Persian PDF`: فایل PDF نهایی پایان‌نامه وجود دارد.
- `PASS Exp1 table`: جدول مقایسهٔ اصلی مدل‌ها وجود دارد.
- `PASS Few-shot BEST json`: خروجی pipeline نهایی few-shot/stacked وجود دارد.
- `PASS Critical figures folder`: شکل‌های مورد نیاز پایان‌نامه ساخته شده‌اند.

در پایان، عددهای اصلی چاپ می‌شود:

- `Tabular ceiling/final reference`: بهترین مرجع جدولی/کلاسیک.
- `Stacked BEST`: نتیجهٔ نهایی pipeline ترکیبی.
- `Few-shot-only BEST`: بهترین نتیجهٔ few-shot بدون stacking.
- `Exp1 strict`: مقایسهٔ سخت‌گیرانه RF در برابر Proposed_FewShot.
- `Federated`: تفاوت آموزش متمرکز و فدرال.

این دستور آموزش سنگین انجام نمی‌دهد؛ فقط خروجی‌های آماده و freeze شده را می‌خواند. برای روز دفاع همین بهترین گزینه است.

## اجرای کمی کامل‌تر قبل از دفاع

اگر شب قبل از دفاع می‌خواهی assetها و readiness دوباره ساخته و چک شوند:

```bash
python scripts/defense_demo.py --mode quick
```

این کارها را انجام می‌دهد:

- `results/validate_thesis_readiness.py`: چک می‌کند داده‌ها، جدول‌ها، شکل‌ها و خروجی‌های کلیدی حاضرند.
- `scripts/build_thesis_assets.py`: شکل‌ها، جدول‌های LaTeX و `KEY_FINDINGS.md` را از خروجی‌ها دوباره تولید می‌کند.
- در پایان همان summary دفاع را چاپ می‌کند.

اجرای کامل همهٔ آزمایش‌ها برای جلسه توصیه نمی‌شود، چون ممکن است زمان‌بر باشد:

```bash
python scripts/defense_demo.py --mode full
```

## نقشهٔ کدها

### مرحلهٔ داده‌سازی

- `extraction/01_cohort.py`: انتخاب cohort بیماران ICU.
- `extraction/02_symptoms.py`: استخراج آزمایش‌ها و علائم حیاتی.
- `extraction/03_labels.py`: ساخت برچسب امکان‌پذیری درمان/ترخیص.
- `extraction/04_tokenize.py`: تبدیل ویژگی‌ها به token/representation قابل استفاده برای مدل.
- `extraction/05_federated_split.py`: تقسیم داده به گره‌های فدرال disease-based.

خروجی این مرحله در `data/processed/` قرار می‌گیرد، مخصوصاً:

- `thesis_dataset.parquet`: دیتاست نهایی جدولی برای آزمایش‌ها.
- `tokens.npy`: نمایش عددی/توکنی بیماران.
- `node_1.parquet` تا `node_5.parquet`: گره‌های شبیه‌سازی فدرال.

### مرحلهٔ مدل‌ها

- `models/baselines.py`: مدل‌های کلاسیک مثل Logistic Regression و Random Forest.
- `models/tokenizer.py`: منطق توکن‌سازی علائم.
- `models/encoder.py`: encoder برای embedding.
- `models/few_shot.py`: منطق few-shot/prototypical learning.
- `models/federated.py`: FedAvg و منطق آموزش فدرال.

### مرحلهٔ آزمایش‌ها

- `experiments/exp1_main.py`: مقایسهٔ اصلی مدل‌ها. اینجا باید صادقانه بگویی RF بهتر از few-shot خام است.
- `experiments/exp2_shots.py`: بررسی اثر تعداد shotها، یعنی K=1,3,5,10,15.
- `experiments/exp3_federated.py`: مقایسهٔ centralized در برابر federated.
- `experiments/exp4_ablation.py`: حذف اجزا برای فهم حساسیت pipeline.
- `experiments/exp5_generalization.py`: بررسی انتقال بین گره‌ها/گروه‌های بیماری.
- `experiments/exp_fewshot_best.py`: pipeline نهایی و قوی‌تر few-shot/stacked.
- `experiments/exp_tabular_sota.py`: سقف عملکرد مدل‌های جدولی/کلاسیک.

### مرحلهٔ خروجی پایان‌نامه

- `scripts/build_thesis_assets.py`: شکل‌ها و جدول‌ها را برای پایان‌نامه می‌سازد.
- `results/generate_all_outputs.py`: جدول‌های LaTeX و `KEY_FINDINGS.md` را از CSVها تولید می‌کند.
- `scripts/defense_demo.py`: مسیر سادهٔ روز دفاع برای خواندن خروجی‌ها و چاپ عددهای نهایی.

## تفسیر عددهای اصلی

| بخش | عدد | معنی دفاعی |
|---|---:|---|
| RF در Exp1 | `0.762` | baseline کلاسیک در مقایسهٔ سخت‌گیرانه بهتر است. |
| Proposed_FewShot در Exp1 | `0.611` | few-shot خام به‌تنهایی کافی نیست. |
| Few-shot-only BEST | `0.670` | با تنظیمات بهتر، few-shot بهتر می‌شود اما هنوز سقف جدولی نیست. |
| Stacked BEST | `0.746` | ترکیب few-shot/embedding با مدل کلاسیک تقریباً به سقف جدولی می‌رسد. |
| Tabular ceiling | `0.748` (خام `0.7475`) | مرجع عملکرد قابل انتظار روی دادهٔ جدولی. منبع: `results/CANONICAL_METRICS.json`. |
| Federated gap | `0.0048` | در شبیه‌سازی، هزینهٔ فدرال کوچک است. |

## جملهٔ فنی آماده برای دفاع

اگر استاد پرسید «پس کدها چه چیزی را ثابت می‌کنند؟» بگو:

> کدها سه چیز را نشان می‌دهند: اول، دادهٔ MIMIC-IV به یک benchmark جدولی و توکنی قابل بازتولید تبدیل شده است. دوم، few-shot خام در مقایسهٔ سخت‌گیرانه از RF ضعیف‌تر است، پس ادعای اغراق‌آمیز ندارم. سوم، وقتی embedding/few-shot با مدل کلاسیک در stacked pipeline ترکیب می‌شود، AUROC به حدود 0.746 می‌رسد و به سقف جدولی نزدیک می‌شود.

## Notebook دفاعی

برای نمایش مرحله‌به‌مرحله، فایل زیر ساخته شده است:

```bash
notebooks/thesis_defense_walkthrough.ipynb
```

اگر Jupyter داری، اجرا کن:

```bash
source .venv/bin/activate
jupyter notebook notebooks/thesis_defense_walkthrough.ipynb
```

اگر Jupyter نصب نبود، روز دفاع لازم نیست درگیر نصب شوی. همان دستور زیر کافی است:

```bash
python scripts/defense_demo.py
```

Notebook برای تمرین و توضیح بهتر است؛ `defense_demo.py` برای اجرای امن روز دفاع.
