#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
D=${TSGEO_DATA_ROOT:-data}
GPU=${TSGEO_GPU:-0}
QWEN=${TSGEO_QWEN_MODEL:-models/Qwen3-8B}

$PY -m tsgeo.space.convert_scores \
    --labels      "$D/researchy/labels/glm_labels_1000.jsonl" \
    --queries-dir "$D/researchy/queries_1000" \
    --output      "$D/researchy/samples_1000.jsonl"

$PY -m tsgeo.space.extract_activations \
    --model "$QWEN" --output "$D/researchy/acts_A/all.pt" --gpu "$GPU"

$PY -m tsgeo.space.build_dimension_dict

$PY -m tsgeo.space.build_accumulated_dict

echo "[2_build_space] done."
