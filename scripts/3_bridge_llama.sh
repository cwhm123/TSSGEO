#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
D=${TSGEO_DATA_ROOT:-data}
GPU=${TSGEO_GPU:-0}
QWEN=${TSGEO_QWEN_MODEL:-models/Qwen3-8B}
LLAMA=${TSGEO_LLAMA_MODEL:-models/Meta-Llama-3.1-8B}

$PY -m tsgeo.space.convert_scores \
    --labels      "$D/researchy/labels/glm_labels_test.jsonl" \
    --queries-dir "$D/researchy/test_queries" \
    --output      "$D/researchy/samples_test.jsonl"

$PY -m tsgeo.space.extract_activations \
    --model "$LLAMA" --output "$D/researchy/bridge/acts_llama/all.pt" --gpu "$GPU"

$PY -m tsgeo.space.extract_activations \
    --model "$LLAMA" --samples "$D/researchy/samples_test.jsonl" \
    --output "$D/researchy/bridge/test_acts_llama/all.pt" --gpu "$GPU"

$PY -m tsgeo.space.extract_activations \
    --model "$QWEN" --samples "$D/researchy/samples_test.jsonl" \
    --output "$D/researchy/test_acts_A/all.pt" --gpu "$GPU"

$PY -m tsgeo.bridge.fit --target llama

echo "[3_bridge_llama] done."
