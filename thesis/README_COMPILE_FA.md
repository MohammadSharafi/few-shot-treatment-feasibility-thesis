# پایان‌نامهٔ فارسی (`thesis_fa.tex`)

## موتور کامپایل (اجباری)

| درست ✅ | غلط ⛔ |
|--------|--------|
| **XeLaTeX** (`xelatex`) | pdfLaTeX (`pdflatex`) |
| | LuaLaTeX (`lualatex`) — بستهٔ **xepersian** فقط روی XeTeX کار می‌کند |

- در **لاگ** باید ببینید: `This is XeTeX, Version ...`
- اگر با pdfLaTeX بسازید، قلم فارسی و جهت متن خراب می‌شود (مربع، چپ‌به‌راست اشتباه، …).

### ترمینال (توصیه‌شده)

یکجا (اسکریپت):

```bash
cd thesis && ./build_thesis_fa.sh
```

یا با **latexmk** (از همین پوشه؛ `latexmkrc` موتور را XeLaTeX می‌گذارد):

```bash
cd thesis
latexmk -xelatex thesis_fa.tex
```

یا دستی:

```bash
cd thesis
xelatex thesis_fa && bibtex thesis_fa && xelatex thesis_fa && xelatex thesis_fa
```

### Cursor / VS Code (LaTeX Workshop)

پوشهٔ `thesis/` تنظیم محلی دارد: **`.vscode/settings.json`** — دستور پیش‌فرض **xelatex** است. فایل باز برای ساخت باید **`thesis_fa.tex`** باشد (نه `thesis.tex` انگلیسی با pdfLaTeX).

قالب بدنه در `thesis_fa.tex`: **فاصلهٔ خط ۱٫۵**، **۱۲pt**، حاشیهٔ راست **۳cm** (صحافی)، کپشن‌های یکدست، جداول فشرده‌تر با `\AtBeginEnvironment{table}`.

## پیش‌نیاز

- **XeLaTeX** (TeX Live / MacTeX)
- کلاس **`extreport`** با گزینهٔ **[14pt]** (بستهٔ `extsizes` روی TeX Live)
- بسته‌ها: `fontspec`, `xepersian`, `etoolbox`, …

## قالب رایج آیین‌نامه (اعمال‌شده در `thesis_fa.tex`)

| مورد | تنظیم |
|------|--------|
| قلم فارسی | **B Nazanin** یا **B Lotus** (سیستم یا `fonts/BNazanin.ttf`)؛ حدود **۱۴pt** (پایهٔ سند) |
| قلم انگلیسی | **Times New Roman** حدود **۱۲pt** (مقیاس نسبت به ۱۴pt در `thesis_fa_fonts.tex`) |
| حاشیه‌ها | راست **۳cm** (صحافی)، چپ/بالا/پایین **۲٫۵cm** |
| فاصلهٔ خط | `\setstretch{1.2}` (بازهٔ متداول **۱٫۱۵–۱٫۵** را در همان فایل می‌توانید عوض کنید) |
| شمارهٔ مقدمات | حروف **الف، ب، پ، …** تا ۳۲ صفحه (`thesis_fa_frontmatter.tex`) |
| بدنهٔ اصلی | **۵ فصل** با شمارهٔ **عربی** از ۱: کلیات، پیشینه، روش، یافته‌ها (آزمایش+نتایج+تحلیل داده)، بحث و نتیجه‌گیری |
| مقدمات | جلد، عنوان فارسی (`\maketitle`)، عنوان انگلیسی، تعهدنامه، تقدیم؛ سپس شمارهٔ حروفی و قدردانی، چکیده فارسی، فهرست مطالب/جداول/اشکال |
| پایان سند | پیوست‌ها، **مراجع** (`apalike`، نویسنده--سال)، **چکیده انگلیسی** |

جزئیات فونت و RTL/LTR در **`thesis_fa_fonts.tex`** و **`fonts/README.md`**.

راهنمای کپی فونت: **`fonts/README.md`**.

### رفتار دو جهته (مهم)

- پاراگراف‌های فارسی: **RTL** (پیش‌فرض XePersian).
- کلمات/جملات انگلیسی داخل متن فارسی: حتماً داخل **`\lr{...}`** تا **LTR** و فونت لاتین درست اعمال شود.
- بخش‌های کاملاً انگلیسی: **`\begin{latin}...\end{latin}`** (مثل فصل‌های لاتین و مراجع).
- در سربرگ فایل اصلی (`thesis_fa.tex`) بلوک کامنت انگلیسی دربارهٔ همین قواعد آمده است.

## ترتیب کامپایل

از پوشهٔ `thesis/`:

```bash
xelatex thesis_fa
bibtex thesis_fa
xelatex thesis_fa
xelatex thesis_fa
```

## ساختار فایل‌ها

- `thesis_fa.tex` — سند اصلی؛ چکیده و قدردانی فارسی
- `thesis_fa_fonts.tex` — فونت‌ها و یادداشت RTL/LTR
- `chapters_fa/` — فصل‌ها و پیوست‌ها (بخشی از بدنه در `latin` است)
- `\bibliography{references}` — کلیدهای `\cite` بدون تغییر

## جداول خودکار پیوست

از ریشهٔ مخزن:

`python results/generate_all_outputs.py`

سپس دوباره `xelatex`.
