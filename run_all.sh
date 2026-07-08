#!/usr/bin/env bash
# =============================================================================
# run_all.sh — end-to-end pipeline for the Few-Shot Treatment-Feasibility thesis
#
# Runs: environment setup -> data extraction -> validation -> model training ->
#       canonical results -> statistics & figures -> (optional) thesis PDF.
#
# Features: live progress bar + per-stage spinner/timer, per-stage logs, and a
#           clear failure report. Re-runnable and flag-controlled.
#
# Usage:
#   ./run_all.sh                 # base pipeline (extraction + training + analysis)
#   ./run_all.sh --quick         # skip the slow hyper-opt / ensemble re-runs
#   ./run_all.sh --skip-extract  # reuse existing data/processed_full_cohort/*
#   ./run_all.sh --skip-train    # only rebuild canonical results, stats, figures
#   ./run_all.sh --full          # + robustness suite: label-sensitivity (5 label
#                                #   defs), tokenization-sensitivity (7 schemes),
#                                #   multi-seed + 5-fold CV, cohort-expansion (ICD-9)
#   ./run_all.sh --full --tune   # + hyperparameter search in the rigorous stage
#   ./run_all.sh --expanded      # + a full ICD-9-inclusive cohort pass (~2x data)
#   ./run_all.sh --build-pdf     # also compile thesis_fa.pdf (needs XeLaTeX)
#   ./run_all.sh --full --expanded --tune --build-pdf   # the strongest, longest run
#   ./run_all.sh --help
#
# Requires (for training): Python 3.10/3.11 with numpy, pandas, scikit-learn,
#   xgboost, lightgbm, catboost, pyarrow (see requirements.txt). Extraction also
#   needs the raw MIMIC-IV v3.1 files under ./mimic-iv-3.1/.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# ----------------------------- options ---------------------------------------
QUICK=0; SKIP_EXTRACT=0; SKIP_TRAIN=0; BUILD_PDF=0; FEWSHOT_EP=80
FULL=0; EXPANDED=0; TUNE=0; SEEDS=5; ANALYSES=0; READMIT=0; LLM=0
for arg in "$@"; do
  case "$arg" in
    --quick)        QUICK=1 ;;
    --skip-extract) SKIP_EXTRACT=1 ;;
    --skip-train)   SKIP_TRAIN=1; SKIP_EXTRACT=1 ;;
    --build-pdf)    BUILD_PDF=1 ;;
    --full)         FULL=1; ANALYSES=1 ;;  # robustness suite + deep analyses
    --analyses)     ANALYSES=1 ;;      # explainability, subgroups, learning curve, few-shot K, ROC/PR/calibration
    --expanded)     EXPANDED=1 ;;      # also run a full ICD-9-inclusive cohort pass (~2x data, heavy)
    --readmit)      READMIT=1 ;;       # also run a full pass with the readmission-aware label (ETHOS-inspired)
    --llm)          LLM=1 ;;           # benchmark open-source clinical/biomedical LLM backbones (downloads models)
    --tune)         TUNE=1 ;;          # hyperparameter search in the rigorous stage
    --seeds=*)      SEEDS="${arg#*=}" ;;
    --fewshot-episodes=*) FEWSHOT_EP="${arg#*=}" ;;
    -h|--help)
      sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (use --help)"; exit 2 ;;
  esac
done

# ----------------------------- colours ---------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
  YEL=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'; TTY=1
else
  BOLD=""; DIM=""; RED=""; GRN=""; YEL=""; CYN=""; RST=""; TTY=0
fi

LOGDIR="results/logs/run_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR" results/canonical results/full_cohort/figures

# ----------------------------- python detection ------------------------------
PY=""
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY="python3"
elif command -v python  >/dev/null 2>&1; then PY="python"
else echo "${RED}ERROR: no python found.${RST}"; exit 1; fi

# ----------------------------- progress bar ----------------------------------
TOTAL=0; STAGE_IDX=0; START_TS=$SECONDS
draw_bar() { # cur total label
  local cur=$1 total=$2 label=$3 width=28 j filled pct bar=""
  filled=$(( total>0 ? cur*width/total : 0 ))
  for ((j=0;j<width;j++)); do
    if [ "$j" -lt "$filled" ]; then bar+="█"; else bar+="░"; fi
  done
  pct=$(( total>0 ? cur*100/total : 0 ))
  if [ "$TTY" -eq 1 ]; then
    printf '\r\033[K%s[%s]%s %3d%% (%d/%d) %s' "$CYN" "$bar" "$RST" "$pct" "$cur" "$total" "$label"
  fi
}

run_stage() { # label -- command...
  local label="$1"; shift
  [ "$1" = "--" ] && shift
  STAGE_IDX=$((STAGE_IDX+1))
  local safe; safe=$(echo "$label" | tr ' /:' '___')
  local log="$LOGDIR/$(printf '%02d' "$STAGE_IDX")_${safe}.log"
  draw_bar "$((STAGE_IDX-1))" "$TOTAL" "▶ $label"
  ( "$@" ) >"$log" 2>&1 &
  local pid=$! start=$SECONDS spin='|/-\' i=0
  while kill -0 "$pid" 2>/dev/null; do
    i=$(( (i+1) % 4 ))
    draw_bar "$((STAGE_IDX-1))" "$TOTAL" "${spin:$i:1} $label  ${DIM}($((SECONDS-start))s)${RST}"
    sleep 0.2
  done
  wait "$pid"; local rc=$?
  local dur=$((SECONDS-start))
  if [ "$rc" -eq 0 ]; then
    draw_bar "$STAGE_IDX" "$TOTAL" "${GRN}✓${RST} $label ${DIM}(${dur}s)${RST}"
    [ "$TTY" -eq 1 ] && printf '\n' || echo "[ok] $label (${dur}s)"
  else
    draw_bar "$STAGE_IDX" "$TOTAL" "${RED}✗${RST} $label ${DIM}(exit $rc)${RST}"
    [ "$TTY" -eq 1 ] && printf '\n' || echo "[FAIL] $label (exit $rc)"
    echo "${RED}--- last 15 log lines ($log) ---${RST}"
    tail -15 "$log" | sed 's/^/  /'
    if [ "${STAGE_STRICT:-1}" = "1" ]; then
      echo "${RED}Pipeline stopped. Fix the error above and re-run (use --skip-extract to reuse data).${RST}"
      exit "$rc"
    fi
  fi
}

# soft stage = best-effort (won't abort the pipeline on failure)
soft_stage() { STAGE_STRICT=0 run_stage "$@"; STAGE_STRICT=1; }

# ----------------------------- preflight -------------------------------------
echo "${BOLD}Few-Shot Treatment-Feasibility — full pipeline${RST}"
echo "${DIM}python=$PY  quick=$QUICK  skip_extract=$SKIP_EXTRACT  skip_train=$SKIP_TRAIN  build_pdf=$BUILD_PDF${RST}"
if [ "$SKIP_EXTRACT" -eq 0 ] && [ ! -d "mimic-iv-3.1" ]; then
  echo "${YEL}WARNING: ./mimic-iv-3.1 not found — extraction will fail. Use --skip-extract to reuse processed data.${RST}"
fi

# ----------------------------- count stages ----------------------------------
TOTAL=1                                   # env setup
[ "$SKIP_EXTRACT" -eq 0 ] && TOTAL=$((TOTAL+5))
[ "$SKIP_TRAIN"   -eq 0 ] && TOTAL=$((TOTAL+1))   # validate
[ "$SKIP_TRAIN"   -eq 0 ] && TOTAL=$((TOTAL+1))   # full-cohort models
if [ "$SKIP_TRAIN" -eq 0 ] && [ "$QUICK" -eq 0 ]; then TOTAL=$((TOTAL+2)); fi  # opt + ensembles
TOTAL=$((TOTAL+5))                        # canonical, stats, table1, figures, assets
[ "$FULL" -eq 1 ] && TOTAL=$((TOTAL+5))   # label, token, rigorous, cohort-expansion, robustness-figs
[ "$ANALYSES" -eq 1 ] && TOTAL=$((TOTAL+9))  # +feature-imp, subgroups, learning, few-shot-K, curves, ETHOS-tokenizer, ETHOS-fewshot, GPT-pretrain, GPT-fewshot
[ "$EXPANDED" -eq 1 ] && TOTAL=$((TOTAL+7))  # ICD-9 extraction(5) + validate + models
[ "$READMIT" -eq 1 ] && TOTAL=$((TOTAL+7))    # readmission-aware extraction(5) + validate + models
[ "$LLM" -eq 1 ] && TOTAL=$((TOTAL+1))         # clinical/biomedical LLM backbone benchmark
[ "$BUILD_PDF" -eq 1 ] && TOTAL=$((TOTAL+1))

# ----------------------------- 1. environment --------------------------------
setup_env() {
  # Best-effort install; never abort here if core libs are already importable.
  if [ -f requirements.txt ]; then
    "$PY" -m pip install -r requirements.txt --quiet --disable-pip-version-check 2>/dev/null \
      || "$PY" -m pip install -r requirements.txt --quiet --break-system-packages --disable-pip-version-check 2>/dev/null \
      || echo "(pip install skipped — offline or restricted; relying on existing environment)"
  fi
  "$PY" -c "import numpy, pandas" || { echo "ERROR: numpy/pandas not importable — cannot continue"; exit 1; }
}
run_stage "Environment & dependencies" -- setup_env

# ----------------------------- 2. extraction ---------------------------------
if [ "$SKIP_EXTRACT" -eq 0 ]; then
  run_stage "Extract cohort (01)"          -- "$PY" extraction/01_cohort.py
  run_stage "Extract symptoms (02)"        -- "$PY" extraction/02_symptoms.py
  run_stage "Apply labels (03)"            -- "$PY" extraction/03_labels.py
  run_stage "Build symptom tokens (04)"    -- "$PY" extraction/04_tokenize.py
  run_stage "Federated disease split (05)" -- "$PY" extraction/05_federated_split.py
fi

# ----------------------------- 3. training -----------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
  run_stage "Validate full cohort"         -- "$PY" scripts/validate_full_cohort.py
  run_stage "Train full-cohort models"     -- "$PY" scripts/run_full_cohort_models.py --fewshot-episodes "$FEWSHOT_EP"
  if [ "$QUICK" -eq 0 ]; then
    run_stage "Final optimized experiments" -- "$PY" scripts/run_final_optimized_experiments.py
    run_stage "Ensembles + federated"       -- "$PY" scripts/run_final_ensembles_federated.py
  fi
fi

# ----------------------------- 4. analysis & figures -------------------------
run_stage "Build canonical results (1 source)" -- "$PY" scripts/build_canonical_results.py
run_stage "Statistics: CIs, DeLong, DCA"        -- "$PY" scripts/advanced_stats_and_table1.py
soft_stage "Table 1 + severity baseline"        -- "$PY" scripts/generate_table1_and_severity.py
soft_stage "Generate thesis figures"            -- "$PY" scripts/generate_thesis_figures.py
soft_stage "Build thesis assets"                -- "$PY" scripts/build_thesis_assets.py

# ----------------------------- 4b. robustness suite (--full) -----------------
if [ "$FULL" -eq 1 ]; then
  run_stage "Label-sensitivity (5 label defs)"  -- "$PY" scripts/exp_label_sensitivity.py
  run_stage "Tokenization-sensitivity (7 schemes)" -- "$PY" scripts/exp_token_sensitivity.py
  if [ "$TUNE" -eq 1 ]; then
    run_stage "Rigorous multi-seed + tuning + CV" -- "$PY" scripts/run_rigorous_models.py --seeds "$SEEDS" --tune
  else
    run_stage "Rigorous multi-seed + CV"          -- "$PY" scripts/run_rigorous_models.py --seeds "$SEEDS"
  fi
  soft_stage "Cohort-expansion analysis (ICD-9)" -- "$PY" scripts/exp_cohort_expansion.py
fi

# ----------------------------- 4c. expanded ICD-9 cohort (--expanded) --------
if [ "$EXPANDED" -eq 1 ]; then
  export THESIS_CONFIG=config_expanded.yaml
  # best-effort: a hiccup in this heavy optional pass must not abort the long run
  soft_stage "[ICD9] Extract cohort"        -- "$PY" extraction/01_cohort.py
  soft_stage "[ICD9] Extract symptoms"      -- "$PY" extraction/02_symptoms.py
  soft_stage "[ICD9] Apply labels"          -- "$PY" extraction/03_labels.py
  soft_stage "[ICD9] Build tokens"          -- "$PY" extraction/04_tokenize.py
  soft_stage "[ICD9] Federated split"      -- "$PY" extraction/05_federated_split.py
  soft_stage "[ICD9] Validate cohort"       -- "$PY" scripts/validate_full_cohort.py
  soft_stage "[ICD9] Train models"          -- "$PY" scripts/run_full_cohort_models.py --fewshot-episodes "$FEWSHOT_EP"
  unset THESIS_CONFIG
fi

# ----------------------------- 4d. deep analyses (--analyses / --full) -------
if [ "$ANALYSES" -eq 1 ]; then
  soft_stage "Feature importance + SHAP"   -- "$PY" scripts/exp_feature_importance.py
  soft_stage "Subgroup / fairness analysis" -- "$PY" scripts/exp_subgroups.py
  soft_stage "Learning curve (data efficiency)" -- "$PY" scripts/exp_learning_curve.py
  soft_stage "Few-shot K-shot sweep"        -- "$PY" scripts/exp_fewshot_kshot.py
  soft_stage "ROC / PR / calibration curves" -- "$PY" scripts/exp_curves.py
  soft_stage "ETHOS-aligned tokenizer (quantile+temporal)" -- "$PY" extraction/04_tokenize_ethos.py
  soft_stage "Few-shot + hybrid on ETHOS tokens" -- "$PY" scripts/exp_fewshot_ethos.py
  soft_stage "GPT-2 pretraining (ETHOS autoregressive)" -- "$PY" scripts/pretrain_ethos_gpt.py --epochs 15
  soft_stage "Few-shot/probe on pretrained ETHOS embeddings" -- "$PY" scripts/exp_fewshot_pretrained.py
fi

# ----------------------------- 4e. readmission-aware label (--readmit) -------
if [ "$READMIT" -eq 1 ]; then
  export THESIS_CONFIG=config_readmit.yaml
  # best-effort: optional stricter-label pass must not abort the long run
  soft_stage "[Readmit] Extract cohort"   -- "$PY" extraction/01_cohort.py
  soft_stage "[Readmit] Extract symptoms" -- "$PY" extraction/02_symptoms.py
  soft_stage "[Readmit] Apply labels"     -- "$PY" extraction/03_labels.py
  soft_stage "[Readmit] Build tokens"     -- "$PY" extraction/04_tokenize.py
  soft_stage "[Readmit] Federated split" -- "$PY" extraction/05_federated_split.py
  soft_stage "[Readmit] Validate cohort"  -- "$PY" scripts/validate_full_cohort.py
  soft_stage "[Readmit] Train models"     -- "$PY" scripts/run_full_cohort_models.py --fewshot-episodes "$FEWSHOT_EP"
  unset THESIS_CONFIG
fi

# ----------------------------- 4f. LLM backbone benchmark (--llm) ------------
if [ "$LLM" -eq 1 ]; then
  soft_stage "Benchmark clinical/biomedical LLM backbones" -- "$PY" scripts/exp_llm_backbones.py
fi

# robustness figures last, so the ICD-9 comparison can use the expanded results
[ "$FULL" -eq 1 ] && soft_stage "Robustness figures" -- "$PY" scripts/make_robustness_figures.py

# ----------------------------- 5. optional PDF -------------------------------
if [ "$BUILD_PDF" -eq 1 ]; then
  if command -v xelatex >/dev/null 2>&1; then
    run_stage "Compile thesis_fa.pdf (XeLaTeX)" -- bash thesis/build_thesis_fa.sh
  else
    soft_stage "Compile thesis_fa.pdf (XeLaTeX)" -- bash -c 'echo "xelatex not found — install MacTeX/TeX Live"; exit 1'
  fi
fi

# ----------------------------- summary ---------------------------------------
echo
echo "${BOLD}${GRN}Pipeline complete${RST} in $((SECONDS-START_TS))s."
echo "Logs:            $LOGDIR/"
echo "Canonical table: results/canonical/canonical_model_results.csv"
echo "Figures:         results/full_cohort/figures/ and results/canonical/figures/"
if [ "$FULL" -eq 1 ]; then
  echo "Robustness:      results/canonical/{label_sensitivity,token_sensitivity,rigorous_multiseed,rigorous_cv,cohort_expansion}.csv"
fi
[ "$EXPANDED" -eq 1 ] && echo "ICD-9 cohort:    results/full_cohort_icd9/  (data/processed_full_cohort_icd9/)"
[ "$BUILD_PDF" -eq 1 ] && echo "Thesis PDF:      thesis/thesis_fa.pdf"
echo "${DIM}Tip: re-run with --skip-extract --skip-train to only refresh analysis & figures.${RST}"
