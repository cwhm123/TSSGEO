#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
D=${TSGEO_DATA_ROOT:-data}
N=${TSGEO_N:-100}
W=${TSGEO_WORKERS:-6}

$PY -m tsgeo.rewrite.bridged_llama --probe llama --n "$N" --workers "$W" \
    --target-mode random --seed 42 --resume

$PY -m tsgeo.rewrite.eval_metrics \
    --rewrites "$D/researchy/bridge/rewrites_bridge_llama.jsonl" \
    --gen-before --workers "$W" --resume

echo "[4_rewrite_llama] done."
