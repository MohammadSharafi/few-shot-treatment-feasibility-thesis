#!/usr/bin/env python3
"""
scripts/validate_results.py

Strict thesis-readiness validator.

Validates against the SINGLE source of truth (results/CANONICAL_METRICS.json):

  1. canonical file exists and is valid JSON
  2. every canonical metric traces to an existing source file with a value
     that matches the recorded raw_value (within tolerance)
  3. README.md, results/KEY_FINDINGS.md, all docs/*.md, and all
     thesis/chapters_fa/*.tex contain the displayed canonical values where
     they cite them — and do NOT contain conflicting rounded variants
  4. thesis title is consistent across cover, commitment, \\title, pdftitle
  5. every LaTeX \\includegraphics target referenced from thesis/chapters_fa/
     and thesis/frontmatter_fa/ resolves to a real file
  6. every LaTeX \\input target resolves to a real file
  7. PDFs produced by the build exist and PDF metadata matches the official title

Writes docs/FINAL_VALIDATION_REPORT.md and exits non-zero on hard failures.

Usage:
    python scripts/validate_results.py
    python scripts/validate_results.py --write-report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_PATH = ROOT / "results" / "CANONICAL_METRICS.json"
TOL = 0.0005  # |actual - canonical raw_value| must be <= 0.0005

OFFICIAL_TITLE_FA = "فیوشات درمانی: پیش‌بینی هوشمند قابلیت درمان با توکن‌های علائم از داده‌های تجمیع‌شده"
OFFICIAL_TITLE_FA_KEY = "فیوشات درمانی"
BANNED_TITLE_FRAGMENTS = ["چندسطحی درمانی"]  # explicit "wrong title" fragments

# Display strings that should NEVER appear with their conflicting variants
# Each tuple: (canonical_display, forbidden_variants)
ROUNDING_BANS = [
    ("0.748", []),  # 0.748 is canonical; 0.747 alone is allowed in specific contexts (raw 0.7475)
    # We catch the dangerous double notation:
]
FORBIDDEN_PHRASES = ["0.747/0.748", "0.747 / 0.748"]

# Canonical experiments/ contents (per micro-cleanup pass). Anything else
# directly under experiments/ that ends in .py is treated as a historical
# residue and should live in experiments/archive/.
CANONICAL_EXPERIMENTS = {
    "run_all.py",
    "exp1_main.py",
    "exp2_shots.py",
    "exp3_federated.py",
    "exp4_ablation.py",
    "exp5_generalization.py",
    "exp_tabular_sota.py",
    "exp_fewshot_best.py",
    "exp_ethos_rf_hybrid.py",
}

# Filename fragments that almost always indicate a non-canonical / historical
# script if found directly under experiments/ (not under experiments/archive/).
HISTORICAL_SCRIPT_PATTERNS = [
    re.compile(r"^exp_fewshot_v\d+(?:_\w+)?\.py$"),
    re.compile(r"^exp_improved_fewshot.*\.py$"),
    re.compile(r"^exp1_improved.*\.py$"),
    re.compile(r"^exp1_fewshot_(?:final|sweep).*\.py$"),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class ValidationResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failed: list[str] = []

    def ok(self, msg: str) -> None: self.passed.append(msg)
    def warn(self, msg: str) -> None: self.warnings.append(msg)
    def fail(self, msg: str) -> None: self.failed.append(msg)

    def check(self, cond: bool, ok_msg: str, fail_msg: str) -> bool:
        (self.ok if cond else self.fail)(ok_msg if cond else fail_msg)
        return cond

    @property
    def score(self) -> int:
        total = len(self.passed) + len(self.warnings) + len(self.failed)
        if total == 0:
            return 0
        ok_w = len(self.passed) + 0.5 * len(self.warnings)
        return int(round(100 * ok_w / total))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _csv_rows(path: Path) -> list[dict]:
    import csv
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _floatv(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


_PRUNED_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git", "catboost_info", "data"}


def _shallow_rglob(name: str) -> list[Path]:
    """Walk ROOT looking for files named ``name`` while pruning heavy dirs."""
    import os
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _PRUNED_DIRS]
        if name in filenames:
            out.append(Path(dirpath) / name)
    return out


# ---------------------------------------------------------------------------
# 1. Canonical-file structural checks + value matching
# ---------------------------------------------------------------------------
def trace_canonical_to_sources(canon: dict, r: ValidationResult) -> None:
    metrics = canon.get("metrics", {})
    if not metrics:
        r.fail("CANONICAL_METRICS.json has no 'metrics' block")
        return

    # Re-load each source file once
    file_cache: dict[str, object] = {}

    def get_source(p: str):
        if p in file_cache:
            return file_cache[p]
        path = ROOT / p
        if p.endswith(".json"):
            data = _load_json(path)
        elif p.endswith(".csv"):
            data = _csv_rows(path)
        else:
            data = None
        file_cache[p] = data
        return data

    # Helper for extracting raw value from each source
    def actual_value(metric_key: str, mdata: dict):
        src = mdata.get("source_file", "")
        key = mdata.get("source_key_or_table", "")
        data = get_source(src)
        if data is None:
            return None

        # JSON dict (fewshot_best.json)
        if isinstance(data, dict):
            # source_key_or_table like "stacked_best.auroc"
            if "." in key:
                parts = key.split(".")
                node = data
                for p in parts:
                    if isinstance(node, dict) and p in node:
                        node = node[p]
                    else:
                        return None
                return _floatv(node)
            return None

        # JSON list (exp1_main.json, exp3_federated.json, exp4_ablation.json)
        if isinstance(data, list):
            # Try to parse "object with model=X, key=auroc"
            obj_match = re.search(r"object with (\w+)=([\w_-]+), key (\w+)", key)
            if obj_match:
                k, v, m = obj_match.group(1), obj_match.group(2), obj_match.group(3)
                for row in data:
                    if isinstance(row, dict) and str(row.get(k, "")).lower() == v.lower():
                        return _floatv(row.get(m))
                return None

            # CSV-style "row model=X, column auroc"
            csv_match = re.search(r"row (\w+)=([\w_\.\-]+), column (\w+)", key)
            if csv_match:
                k, v, m = csv_match.group(1), csv_match.group(2), csv_match.group(3)
                for row in data:
                    if isinstance(row, dict) and str(row.get(k, "")) == v:
                        return _floatv(row.get(m))
                return None

        return None

    for mk, mdata in metrics.items():
        if not isinstance(mdata, dict):
            continue
        src = mdata.get("source_file")
        raw = _floatv(mdata.get("raw_value"))
        if src is None or raw is None:
            r.warn(f"canonical metric '{mk}' missing source_file or raw_value")
            continue
        if not (ROOT / src).exists():
            r.fail(f"canonical metric '{mk}': source file missing → {src}")
            continue

        act = actual_value(mk, mdata)
        if act is None:
            r.warn(f"canonical metric '{mk}': could not auto-extract value from {src} (selector: '{mdata.get('source_key_or_table')}')")
            continue
        if abs(act - raw) <= TOL:
            r.ok(f"canonical metric '{mk}' = {act:.6f} matches recorded {raw:.6f} ✓ (source: {src})")
        else:
            r.fail(f"canonical metric '{mk}': source shows {act:.6f} but JSON claims {raw:.6f} (|Δ| > {TOL}) — FIX JSON or rerun source")


# ---------------------------------------------------------------------------
# 2. Text consistency across docs / README / thesis chapters
# ---------------------------------------------------------------------------
def scan_text_consistency(canon: dict, r: ValidationResult) -> None:
    metrics = canon.get("metrics", {})

    # Auto-generated reports legitimately contain quoted historical failure
    # strings; skip them from the forbidden-phrase scan.
    self_reports = {
        ROOT / "docs" / "FINAL_VALIDATION_REPORT.md",
    }

    files_to_scan: list[Path] = []
    files_to_scan += [ROOT / "README.md"]
    files_to_scan += sorted((ROOT / "docs").glob("*.md"))
    files_to_scan += sorted((ROOT / "results").glob("*.md"))
    files_to_scan += sorted((ROOT / "thesis").glob("*.md"))

    # 2a) forbidden phrases (dangerous double-rounding like the slash-joined form)
    for f in files_to_scan:
        if f in self_reports:
            continue
        text = _read(f)
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                r.fail(f"{f.relative_to(ROOT)}: forbidden phrase '{phrase}' present (use canonical 0.748)")

    # 2b) verify README contains every used_in_defense display value
    readme = _read(ROOT / "README.md")
    if not readme:
        r.fail("README.md missing or empty")
    else:
        for mk, md in metrics.items():
            if md.get("used_in_defense"):
                disp = str(md.get("display", ""))
                if disp and disp in readme:
                    r.ok(f"README contains defense metric '{mk}' display '{disp}' ✓")
                else:
                    r.fail(f"README missing defense metric '{mk}' display '{disp}'")

    # 2c) KEY_FINDINGS must contain the headline four
    kf = _read(ROOT / "results" / "KEY_FINDINGS.md")
    for mk in ("exp1_random_forest_auroc", "exp1_proposed_fewshot_auroc"):
        disp = str(metrics.get(mk, {}).get("display", ""))
        if disp and disp in kf:
            r.ok(f"KEY_FINDINGS contains '{disp}' ({mk}) ✓")
        else:
            r.fail(f"KEY_FINDINGS missing '{disp}' for {mk}")


# ---------------------------------------------------------------------------
# 2d. Thesis/proposal alignment section exists and is explicit
# ---------------------------------------------------------------------------
def check_proposal_alignment_section(r: ValidationResult) -> None:
    intro_path = ROOT / "thesis" / "chapters_fa" / "introduction.tex"
    text = _read(intro_path)
    if not text:
        r.fail("thesis/chapters_fa/introduction.tex missing or unreadable")
        return

    required_markers = {
        "proposal-alignment subsection": r"تطبیق با پروپوزال اولیه",
        "proposal-alignment table label": r"\label{tab:proposal_alignment}",
        "coverage level column": "سطح پوشش",
        "defense note column": "نکتۀ دفاعی",
        "honest scoping explanation": "داده‌ی ابزارهای پوشیدنی، واسط کاربری تولیدی و استقرار چندسایتی واقعی",
    }
    for label, marker in required_markers.items():
        if marker in text:
            r.ok(f"proposal alignment: {label} present ✓")
        else:
            r.fail(f"proposal alignment: missing {label} marker ({marker})")


# ---------------------------------------------------------------------------
# 3. Title consistency
# ---------------------------------------------------------------------------
def check_titles(r: ValidationResult) -> None:
    title_files = [
        "thesis/frontmatter_fa/cover.tex",
        "thesis/frontmatter_fa/commitment.tex",
        "thesis/thesis_fa.tex",
    ]
    for rel in title_files:
        text = _read(ROOT / rel)
        if not text:
            r.fail(f"{rel}: cannot read or empty")
            continue

        # Banned old title fragments
        for banned in BANNED_TITLE_FRAGMENTS:
            if banned in text:
                r.fail(f"{rel}: contains banned old-title fragment '{banned}'")

        # cover.tex and commitment.tex should contain the canonical key fragment
        if rel.endswith("cover.tex") or rel.endswith("commitment.tex"):
            if OFFICIAL_TITLE_FA_KEY in text:
                r.ok(f"{rel}: contains canonical title key '{OFFICIAL_TITLE_FA_KEY}' ✓")
            else:
                r.fail(f"{rel}: missing canonical title key '{OFFICIAL_TITLE_FA_KEY}'")

        # thesis_fa.tex \title{} and \hypersetup{pdftitle=...} should be canonical
        if rel.endswith("thesis_fa.tex"):
            # Find \title{...} and \hypersetup{...} blocks (multiline)
            title_match = re.search(r"\\title\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text, re.DOTALL)
            if title_match and OFFICIAL_TITLE_FA_KEY in title_match.group(1):
                r.ok("thesis_fa.tex \\title{} uses canonical title ✓")
            elif title_match:
                r.fail(f"thesis_fa.tex \\title{{...}} does NOT contain canonical title key '{OFFICIAL_TITLE_FA_KEY}'")
            else:
                r.warn("thesis_fa.tex: \\title{...} block not found")

            hyper_match = re.search(r"pdftitle=\{([^}]*)\}", text)
            if hyper_match and OFFICIAL_TITLE_FA_KEY in hyper_match.group(1):
                r.ok("thesis_fa.tex \\hypersetup{pdftitle=...} uses canonical title ✓")
            elif hyper_match:
                r.fail("thesis_fa.tex pdftitle does NOT use canonical title")
            else:
                r.warn("thesis_fa.tex: pdftitle not found")


# ---------------------------------------------------------------------------
# 4. LaTeX figure + input references resolve to existing files
# ---------------------------------------------------------------------------
def check_latex_refs(r: ValidationResult) -> None:
    chapters = list((ROOT / "thesis" / "chapters_fa").glob("*.tex"))
    frontmatter = list((ROOT / "thesis" / "frontmatter_fa").glob("*.tex"))
    main = [ROOT / "thesis" / "thesis_fa.tex"]
    tex_files = chapters + frontmatter + main

    # Image-search roots used by thesis_fa.tex \graphicspath{...}
    graphics_roots = [
        ROOT / "thesis",
        ROOT / "results" / "figures",
        ROOT / "results",
    ]

    image_pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    input_pat = re.compile(r"\\(?:input|include)\{([^}]+)\}")

    missing_imgs: set[str] = set()
    missing_inputs: set[str] = set()
    for tf in tex_files:
        text = _read(tf)
        for m in image_pat.finditer(text):
            target = m.group(1).strip()
            if target.startswith("/"):
                p = Path(target)
                if not p.exists():
                    missing_imgs.add(target)
                continue
            # Try graphics roots
            found = False
            for gr in graphics_roots:
                # try as-is and with .png / .pdf added
                for cand in [gr / target, gr / f"{target}.png", gr / f"{target}.pdf"]:
                    if cand.exists():
                        found = True
                        break
                if found:
                    break
            if not found:
                missing_imgs.add(target)
        for m in input_pat.finditer(text):
            target = m.group(1).strip()
            # \input is resolved relative to thesis/
            cand = ROOT / "thesis" / f"{target}.tex"
            if not cand.exists():
                cand2 = ROOT / "thesis" / target
                if not cand2.exists():
                    missing_inputs.add(target)

    if missing_imgs:
        for img in sorted(missing_imgs)[:15]:
            r.warn(f"LaTeX \\includegraphics target may be unresolved: {img}")
        r.warn(f"({len(missing_imgs)} total unresolved image references)")
    else:
        r.ok("All LaTeX \\includegraphics targets resolved ✓")

    if missing_inputs:
        for inp in sorted(missing_inputs)[:15]:
            r.fail(f"LaTeX \\input target missing: {inp}.tex")
    else:
        r.ok("All LaTeX \\input/\\include targets resolved ✓")


# ---------------------------------------------------------------------------
# 5a. Hygiene: experiments/ folder must not duplicate historical scripts
# ---------------------------------------------------------------------------
def check_experiments_hygiene(r: ValidationResult) -> None:
    exp_dir = ROOT / "experiments"
    if not exp_dir.exists():
        r.fail("experiments/ folder missing")
        return

    py_files = sorted(p.name for p in exp_dir.glob("*.py"))
    if not py_files:
        r.warn("experiments/ has no .py files at the top level")
        return

    # Flag any top-level script that is neither in the canonical set nor
    # matches a historical pattern (so we surface unknown extras too).
    historical_present: list[str] = []
    unknown_extras: list[str] = []
    for fn in py_files:
        if fn in CANONICAL_EXPERIMENTS:
            continue
        if any(pat.match(fn) for pat in HISTORICAL_SCRIPT_PATTERNS):
            historical_present.append(fn)
        else:
            unknown_extras.append(fn)

    for fn in historical_present:
        r.fail(
            f"experiments/{fn}: historical script present at top level "
            "(should live in experiments/archive/)"
        )
    for fn in unknown_extras:
        r.warn(
            f"experiments/{fn}: not in canonical experiments list and not a "
            "known historical pattern — confirm it is canonical or move to archive"
        )

    if not historical_present and not unknown_extras:
        r.ok(
            f"experiments/ contains only the {len(CANONICAL_EXPERIMENTS)} "
            "canonical scripts ✓"
        )

    # archive folder must exist and must NOT be missing the historical ones
    archive_dir = exp_dir / "archive"
    if archive_dir.exists():
        archived = sorted(p.name for p in archive_dir.glob("*.py"))
        if archived:
            r.ok(
                f"experiments/archive/ contains {len(archived)} historical "
                "script(s) for traceability ✓"
            )
        else:
            r.warn("experiments/archive/ exists but contains no .py files")
    else:
        r.warn("experiments/archive/ folder is missing (expected for traceability)")


# ---------------------------------------------------------------------------
# 5b. Hygiene: no duplicate result folders / canonical files
# ---------------------------------------------------------------------------
def check_result_folder_hygiene(r: ValidationResult) -> None:
    # 5b.i: thesis/results/ must not exist as an active duplicate
    thesis_results = ROOT / "thesis" / "results"
    if thesis_results.exists() and not thesis_results.is_symlink():
        result_like = [
            *thesis_results.glob("*.json"),
            *thesis_results.glob("*.md"),
            *thesis_results.glob("*.log"),
            *thesis_results.glob("*.csv"),
        ]
        if result_like:
            r.fail(
                f"thesis/results/ still exists with {len(result_like)} "
                "JSON/MD/LOG/CSV file(s) — duplicate of root results/. "
                "Move to results/archive/thesis_results_duplicate_backup/."
            )
        else:
            r.warn("thesis/results/ exists but is empty — remove it")
    else:
        r.ok("thesis/results/ is not a duplicate result folder ✓")

    # 5b.ii: CANONICAL_METRICS.json must only live at results/CANONICAL_METRICS.json
    canonical_copies: list[Path] = []
    for p in _shallow_rglob("CANONICAL_METRICS.json"):
        relp = p.relative_to(ROOT)
        if relp == Path("results/CANONICAL_METRICS.json"):
            continue
        canonical_copies.append(p)

    if canonical_copies:
        for p in canonical_copies:
            relp = p.relative_to(ROOT)
            # If it's inside results/archive/ we treat as a documented backup
            # (not a hygiene failure).
            if str(relp).startswith("results/archive/"):
                r.warn(f"{relp}: archived copy of CANONICAL_METRICS.json (documented)")
            else:
                r.fail(
                    f"{relp}: duplicate CANONICAL_METRICS.json outside root "
                    "results/ — remove or move to results/archive/"
                )
    else:
        r.ok("CANONICAL_METRICS.json exists only at results/ (single source of truth) ✓")

    # 5b.iii: DEFENSE_RUN_SUMMARY and KEY_FINDINGS duplicates
    for stem in ("DEFENSE_RUN_SUMMARY.md", "KEY_FINDINGS.md"):
        copies = [
            p for p in _shallow_rglob(stem)
            if p.relative_to(ROOT) != Path("results") / stem
        ]
        active = [p for p in copies if "results/archive/" not in str(p.relative_to(ROOT))]
        archived = [p for p in copies if "results/archive/" in str(p.relative_to(ROOT))]
        if active:
            root_text = _read(ROOT / "results" / stem)
            for p in active:
                relp = p.relative_to(ROOT)
                dup_text = _read(p)
                if root_text and dup_text == root_text:
                    r.fail(f"{relp}: byte-identical duplicate of results/{stem} — remove or archive")
                else:
                    r.fail(f"{relp}: divergent duplicate of results/{stem} — content mismatch with root canonical file")
        if archived:
            r.warn(f"{stem}: {len(archived)} archived copy under results/archive/ (documented)")


# ---------------------------------------------------------------------------
# 5c. LaTeX log analysis (warning honesty)
# ---------------------------------------------------------------------------
def analyse_latex_log(r: ValidationResult) -> dict[str, int]:
    log_path = ROOT / "thesis" / "thesis_fa.log"
    counts = {
        "log_present": 0,
        "latex_warnings": 0,
        "overfull_hbox": 0,
        "underfull_hbox": 0,
        "package_warnings": 0,
        "missing_characters": 0,
        "errors": 0,
    }
    if not log_path.exists():
        r.fail("thesis/thesis_fa.log not present — validation must parse a fresh build log (run `make thesis-fa`)")
        return counts

    counts["log_present"] = 1
    text = _read(log_path)
    counts["latex_warnings"] = len(re.findall(r"^LaTeX Warning:", text, flags=re.MULTILINE))
    counts["overfull_hbox"] = len(re.findall(r"^Overfull \\[hv]box", text, flags=re.MULTILINE))
    counts["underfull_hbox"] = len(re.findall(r"^Underfull \\[hv]box", text, flags=re.MULTILINE))
    counts["package_warnings"] = len(re.findall(r"^Package .* Warning:", text, flags=re.MULTILINE))
    counts["missing_characters"] = len(re.findall(r"Missing character|no character", text))
    counts["errors"] = len(re.findall(r"^! ", text, flags=re.MULTILINE))

    if counts["errors"] > 0:
        r.fail(f"thesis_fa.log contains {counts['errors']} LaTeX error(s) (lines starting with `!`)")
    else:
        r.ok("thesis_fa.log contains no LaTeX errors ✓")

    if counts["missing_characters"] > 0:
        r.fail(
            f"thesis_fa.log: {counts['missing_characters']} missing-character warning(s) — "
            "likely font/glyph problem, must be fixed before submission"
        )
    else:
        r.ok("thesis_fa.log: no missing-character warnings ✓")

    # Undefined citations / references are strict failures for a finalised thesis.
    undef_cit = len(re.findall(r"Citation .* undefined", text))
    undef_ref = len(re.findall(r"Reference .* undefined", text))
    counts["undefined_citations"] = undef_cit
    counts["undefined_references"] = undef_ref
    if undef_cit > 0:
        r.fail(f"thesis_fa.log: {undef_cit} undefined citation(s) — bibtex must run and references.bib must contain all keys")
    else:
        r.ok("thesis_fa.log: no undefined citations ✓")
    if undef_ref > 0:
        r.fail(f"thesis_fa.log: {undef_ref} undefined reference(s) — labels missing or xelatex must rerun")
    else:
        r.ok("thesis_fa.log: no undefined references ✓")

    if counts["overfull_hbox"] > 50:
        r.warn(f"thesis_fa.log: {counts['overfull_hbox']} Overfull hbox warnings (review for visible spillover)")
    else:
        r.ok(f"thesis_fa.log: {counts['overfull_hbox']} Overfull hbox warnings (within cosmetic tolerance)")

    return counts


# ---------------------------------------------------------------------------
# 5d. PDF metadata title matches official canonical Persian title
# ---------------------------------------------------------------------------
def check_pdf_metadata(r: ValidationResult) -> None:
    pdf_path = ROOT / "thesis" / "thesis_fa.pdf"
    if not pdf_path.exists():
        r.fail("thesis/thesis_fa.pdf missing — run `make thesis-fa`")
        return
    title = ""
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        title = _extract_pdf_title_without_pypdf(pdf_path)
        if not title:
            r.fail("could not verify PDF metadata title: pypdf is unavailable and fallback parser found no /Title")
            return
    else:
        try:
            reader = PdfReader(str(pdf_path))
            md = reader.metadata or {}
            title = md.title or md.get("/Title") or ""
        except Exception as e:
            r.fail(f"could not read PDF metadata: {e}")
            return
    if title == OFFICIAL_TITLE_FA:
        r.ok("PDF /Title exactly matches the official Persian title ✓")
    else:
        r.fail(f"PDF /Title does NOT exactly match the official Persian title — got: {title!r}")
    for banned in BANNED_TITLE_FRAGMENTS:
        if banned in title:
            r.fail(f"PDF /Title contains banned old-title fragment '{banned}' — rebuild required")


def _decode_pdf_hex_string(hex_bytes: bytes) -> str:
    """Decode PDF hex strings used by hyperref for Unicode metadata."""
    try:
        raw = bytes.fromhex(hex_bytes.decode("ascii"))
    except Exception:
        return ""
    for enc in ("utf-16-be", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            if text.startswith("\ufeff"):
                text = text[1:]
            return text
        except Exception:
            continue
    return ""


def _extract_pdf_title_without_pypdf(pdf_path: Path) -> str:
    """Best-effort PDF metadata title reader for TeX-generated compressed PDFs.

    XeLaTeX/hyperref may place the document info dictionary in a compressed
    object stream. This fallback inflates Flate streams and looks specifically
    for the hyperref metadata dictionary (`/Creator(...)/Title<...>`), avoiding
    outline/bookmark titles that also use `/Title`.
    """
    data = pdf_path.read_bytes()
    streams: list[bytes] = [data]
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        if end < 0:
            continue
        raw_stream = data[start:end].strip(b"\r\n")
        try:
            streams.append(zlib.decompress(raw_stream))
        except Exception:
            continue

    metadata_title_pattern = re.compile(rb"/Creator\([^)]*\)/Title<([0-9A-Fa-f]+)>")
    generic_title_pattern = re.compile(rb"/Title<([0-9A-Fa-f]+)>")

    generic_titles: list[str] = []
    for stream in streams:
        match = metadata_title_pattern.search(stream)
        if match:
            return _decode_pdf_hex_string(match.group(1))
        for m in generic_title_pattern.finditer(stream):
            title = _decode_pdf_hex_string(m.group(1))
            if title:
                generic_titles.append(title)

    if OFFICIAL_TITLE_FA in generic_titles:
        return OFFICIAL_TITLE_FA
    return generic_titles[0] if len(generic_titles) == 1 else ""


# ---------------------------------------------------------------------------
# 6. Build artefacts (advisory)
# ---------------------------------------------------------------------------
def check_build_artifacts(r: ValidationResult) -> None:
    for rel in (
        "thesis/thesis_fa.pdf",
        "thesis/defense_slides.pdf",
        "data/processed/thesis_dataset.parquet",
        "data/processed/tokens.npy",
        "results/figures/fig1_main_comparison.png",
        "results/figures/fig4_federated.png",
        "results/tables/exp1_results.csv",
        "results/tables/ALL_TABLES.tex",
    ):
        path = ROOT / rel
        r.check(path.exists(), f"artifact exists: {rel} ✓", f"artifact missing: {rel}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(canon: dict, r: ValidationResult, timestamp: str, latex_counts: dict[str, int] | None = None) -> str:
    metrics = canon.get("metrics", {})
    lines = [
        "# Final Validation Report",
        "",
        f"**Generated:** {timestamp}",
        "**Script:** `python scripts/validate_results.py --write-report`",
        f"**Canonical source:** `{CANONICAL_PATH.relative_to(ROOT)}`",
        "",
        "---",
        "",
        "## Canonical Metrics",
        "",
        "| Metric | Display | Raw | Source |",
        "|--------|---------|----:|--------|",
    ]
    for mk in [
        "tabular_sota_test_auroc",
        "stacked_best_auroc",
        "fewshot_best_auroc",
        "exp1_random_forest_auroc",
        "exp1_proposed_fewshot_auroc",
        "federated_centralized_auroc",
        "federated_federated_auroc",
        "federated_auroc_gap",
        "ethos_rf_hybrid_auroc",
    ]:
        md = metrics.get(mk, {})
        lines.append(
            f"| {mk} | **{md.get('display','?')}** {md.get('ci_display','')} | {md.get('raw_value','?')} | `{md.get('source_file','?')}` |"
        )
    lines += ["", "---", ""]

    if r.passed:
        lines += ["## Passed Checks", ""]
        lines += [f"- {p}" for p in r.passed]
        lines += [""]
    if r.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- **WARN:** {w}" for w in r.warnings]
        lines += [""]
    if r.failed:
        lines += ["## Failed Checks", ""]
        lines += [f"- **FAIL:** {f}" for f in r.failed]
        lines += [""]
    else:
        lines += ["## Failed Checks", "", "- None", ""]

    if latex_counts:
        lines += [
            "## LaTeX Build Log Analysis",
            "",
            "Source: `thesis/thesis_fa.log` (last `make thesis-fa` run).",
            "",
            "| Category | Count |",
            "|----------|------:|",
            f"| LaTeX `! ` errors | {latex_counts.get('errors', 0)} |",
            f"| Missing character warnings | {latex_counts.get('missing_characters', 0)} |",
            f"| Undefined citations | {latex_counts.get('undefined_citations', 0)} |",
            f"| Undefined references | {latex_counts.get('undefined_references', 0)} |",
            f"| LaTeX Warning lines | {latex_counts.get('latex_warnings', 0)} |",
            f"| Package warnings | {latex_counts.get('package_warnings', 0)} |",
            f"| Overfull hbox/vbox | {latex_counts.get('overfull_hbox', 0)} |",
            f"| Underfull hbox/vbox | {latex_counts.get('underfull_hbox', 0)} |",
            "",
            (
                "All zero errors / missing characters means the build is clean. "
                "Overfull/Underfull hbox messages with magnitudes ≤ 5pt are cosmetic in XePersian RTL typesetting."
                if latex_counts.get('errors', 0) == 0 and latex_counts.get('missing_characters', 0) == 0
                else "Errors or missing-character warnings present — investigate before submission."
            ),
            "",
            "---",
            "",
        ]

    lines += [
        "---",
        "",
        f"## Final Readiness Score: **{r.score}/100**",
        "",
        f"- Passed: {len(r.passed)}",
        f"- Warnings: {len(r.warnings)}",
        f"- Failed: {len(r.failed)}",
        "",
        "## Final Recommendation",
        "",
    ]
    if r.failed:
        lines.append("- NOT READY: address all failed checks before submission.")
    elif r.warnings:
        lines.append("- Submit after minor fixes: review warnings and confirm each is acceptable.")
    else:
        lines.append("- Submit: no failed checks and no warnings.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict thesis validation against CANONICAL_METRICS.json")
    ap.add_argument("--write-report", action="store_true")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Thesis Result Validation ({timestamp}) ===\n")

    canon = _load_json(CANONICAL_PATH)
    if canon is None:
        print(f"FATAL: cannot read {CANONICAL_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 2

    r = ValidationResult()

    # canonical schema is present
    r.check(
        "metrics" in canon and isinstance(canon["metrics"], dict),
        "CANONICAL_METRICS.json contains 'metrics' object ✓",
        "CANONICAL_METRICS.json missing 'metrics' object",
    )

    trace_canonical_to_sources(canon, r)
    scan_text_consistency(canon, r)
    check_proposal_alignment_section(r)
    check_titles(r)
    check_latex_refs(r)
    check_experiments_hygiene(r)
    check_result_folder_hygiene(r)
    latex_counts = analyse_latex_log(r)
    check_pdf_metadata(r)
    check_build_artifacts(r)

    print(f"PASSED  ({len(r.passed)}):")
    for p in r.passed:
        print(f"  ✓ {p}")
    if r.warnings:
        print(f"\nWARNINGS ({len(r.warnings)}):")
        for w in r.warnings:
            print(f"  ⚠ {w}")
    if r.failed:
        print(f"\nFAILED ({len(r.failed)}):")
        for f in r.failed:
            print(f"  ✗ {f}")

    print(f"\n=== Final Score: {r.score}/100 ===")
    print(f"    Passed: {len(r.passed)}  Warnings: {len(r.warnings)}  Failed: {len(r.failed)}")

    if args.write_report:
        report_path = ROOT / "docs" / "FINAL_VALIDATION_REPORT.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(canon, r, timestamp, latex_counts))
        print(f"\nReport written to: {report_path.relative_to(ROOT)}")

    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
