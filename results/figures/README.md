# Thesis figures

PNG files here are **generated** — they are not required in git for the thesis to be reproducible.

From the repository root (with dependencies from `requirements.txt` installed, preferably in a virtualenv):

```bash
python scripts/build_thesis_assets.py
```

This runs `scripts/generate_thesis_figures.py`, then `generate_figures.py`, then `results/generate_all_outputs.py`.

If `matplotlib` is missing, create a venv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_thesis_assets.py
```
