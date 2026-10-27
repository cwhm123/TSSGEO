#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
D=${TSGEO_DATA_ROOT:-data}
N=${TSGEO_N:-100}
NCASES=${TSGEO_N_CASES:-1200}
W=${TSGEO_WORKERS:-8}
CC="$D/researchy/bridge/casecmp"

$PY -m tsgeo.case.build_library --n-cases "$NCASES" --workers "$W" --resume

$PY -m tsgeo.rewrite.bridged_metered --probe llama --n "$N" --workers "$W" --resume

$PY -m tsgeo.case.rewrite_cases --probe llama --n "$N" --workers "$W" --resume

$PY -m tsgeo.rewrite.eval_metrics --rewrites "$CC/rewrites_cond4_llama.jsonl" \
    --gen-before --workers 6 --resume
$PY -m tsgeo.rewrite.eval_metrics --rewrites "$CC/rewrites_case_llama_test_sim_k3.jsonl" \
    --gen-before --workers 6 --resume

$PY -m tsgeo.case.compare_analysis --conds cond4_llama llama_test_sim_k3

echo "[5_case_reuse] done."
