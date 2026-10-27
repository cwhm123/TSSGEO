import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(os.environ.get("TSGEO_DATA_ROOT", _REPO_ROOT / "data"))
MODEL_ROOT = str(os.environ.get("TSGEO_MODEL_ROOT", _REPO_ROOT / "models"))

RESEARCHY = f"{DATA_ROOT}/researchy"
BRIDGE_DIR = f"{RESEARCHY}/bridge"
CASECMP_DIR = f"{BRIDGE_DIR}/casecmp"


QUERIES_DIR = f"{RESEARCHY}/queries_1000"
LABELS_TRAIN = f"{RESEARCHY}/labels/glm_labels_1000.jsonl"
CONSTRUCTION_SAMPLES = f"{RESEARCHY}/samples_1000.jsonl"
QWEN_ACTS = f"{RESEARCHY}/acts_A/all.pt"
DIMENSION_DICT = f"{DATA_ROOT}/dict/dimension_dict.json"
DICT_PATH = f"{DATA_ROOT}/dict/accumulated_dict_v2.json"


TEST_QUERIES = f"{RESEARCHY}/test_queries"
TEST_LABELS = f"{RESEARCHY}/labels/glm_labels_test.jsonl"
TEST_ACTS_A = f"{RESEARCHY}/test_acts_A/all.pt"


GLM_API = {
    "api_key": os.environ.get("GLM_API_KEY", ""),
    "api_base": os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/anthropic"),
    "model": os.environ.get("GLM_MODEL", "glm-5"),
}


def require_glm_key():
    if not GLM_API["api_key"]:
        raise SystemExit(
            "ERROR: GLM_API_KEY is not set. "
            "export GLM_API_KEY=<your zhipu key> "
            "(or pass --api-key where supported).")
