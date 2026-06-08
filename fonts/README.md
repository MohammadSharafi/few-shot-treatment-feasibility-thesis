# قلم‌های پایان‌نامهٔ فارسی (`thesis_fa.tex`)

## B Nazanin فقط (بدون Vazirmatn)

فایل `thesis_fa_fonts.tex` به‌ترتیب زیر را امتحان می‌کند — **هیچ بازگشتی به Vazirmatn وجود ندارد**:

1. **`thesis/fonts/BNazanin.ttf`** (+ اختیاری **`BNazaninBd.ttf`**)
2. **`thesis/fontes/BNazanin.ttf`** (+ همان **Bold** اختیاری در همان پوشه) — اگر قلم‌ها را در پوشهٔ `fontes` گذاشته‌اید
3. روی سیستم: **IRNazanin** → **IR Nazanin** → **B Nazanin** → **BNazanin** — اگر هیچ فایل محلی نبود

نام فایل‌ها را دقیقاً **`BNazanin.ttf`** بگذارید (حروف بزرگ/کوچک روی بعضی سیستم‌ها مهم است). برای Bold: **`BNazaninBd.ttf`**.

## انگلیسی

متن لاتین (`\lr`، محیط `latin`) روی **Times New Roman** (در نبود: TeX Gyre Termes).

## جداول

قلاب `\AtBeginEnvironment{table}` در `thesis_fa.tex` اندازهٔ `\footnotesize` و `\tabcolsep` را برای جداول تنظیم می‌کند.

## ریاضی

در `thesis_fa.tex` گزینه‌های **`mathdigits=default`** برای `xepersian` فعال‌اند تا قلم‌های قدیمی در حالت ریاضی خطای نویسهٔ ٫/٪ ندهند.

## فاصلهٔ پاراگراف (بدنه)

پس از بارگذاری قلم‌ها، **`emergencystretch`** کم‌تر از مقدار بسیار بالای قبلی تنظیم شده تا فاصلهٔ افراطی بین واژه‌ها در تراز RTL کمتر شود؛ در صورت نیاز همان مقدار را در `thesis_fa.tex` کمی بالا ببرید تا `Underfull \hbox` کمتر شود.
