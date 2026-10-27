#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
D=${TSGEO_DATA_ROOT:-data}

$PY -m tsgeo.space.annotate_citations \
    --documents-dir "$D/researchy/queries_1000" \
    --output-jsonl   "$D/researchy/labels/glm_labels_1000.jsonl" \
    --resume

$PY -m tsgeo.space.annotate_citations \
    --documents-dir "$D/researchy/test_queries" \
    --output-jsonl   "$D/researchy/labels/glm_labels_test.jsonl" \
    --resume

echo "[1_annotate] done."
