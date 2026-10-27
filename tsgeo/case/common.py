import difflib
import json
import re
import time

import numpy as np
import torch

from tsgeo.config import (RESEARCHY, BRIDGE_DIR, CASECMP_DIR, QWEN_ACTS,
                          CONSTRUCTION_SAMPLES, DICT_PATH, GLM_API)

SAMPLES = CONSTRUCTION_SAMPLES


PROBE_LAYER = 21

N_ANCHORS = 128


DATASETS = {
    "researchy": {
        "labels_test": f"{RESEARCHY}/labels/glm_labels_test.jsonl",
        "queries_test": f"{RESEARCHY}/test_queries",
        "test_acts": {p: f"{BRIDGE_DIR}/test_acts_{p}/all.pt" for p in ("llama", "gemma")},
        "dict": DICT_PATH,
    },
}


def probe_split():
    qa = torch.load(QWEN_ACTS, map_location="cpu", weights_only=False)
    meta = {}
    with open(SAMPLES) as f:
        for l in f:
            r = json.loads(l)
            if r["id"] in qa:
                meta[r["id"]] = r["score"]
    common = [s for s in sorted(qa) if s in meta and qa[s].float().abs().sum() > 0]
    qs = sorted({s.rsplit("_doc_", 1)[0] for s in common})
    rng = np.random.RandomState(0)
    rng.shuffle(qs)
    held_qs = set(qs[:int(len(qs) * 0.2)])
    train_ids = [s for s in common if s.rsplit("_doc_", 1)[0] not in held_qs]
    held_ids = [s for s in common if s.rsplit("_doc_", 1)[0] in held_qs]
    return qa, meta, train_ids, held_ids


def anchor_ids(train_ids):
    idx = np.random.RandomState(0).choice(len(train_ids), N_ANCHORS, replace=False)
    return [train_ids[i] for i in idx]


def cosmat(X, A):
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    An = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    return Xn @ An.T


def rel_matrix_probe(acts, ids, anchors, layer=PROBE_LAYER):
    mu = torch.stack([acts[s][layer].float() for s in ids]).mean(0, keepdim=True)
    X = torch.stack([acts[s][layer].float() for s in ids]) - mu
    A = torch.stack([acts[s][layer].float() for s in anchors]) - mu
    return cosmat(X, A), mu


def rel_vec_target(acts, sid, anchors, mu, layer):
    h = acts[sid][layer].float().unsqueeze(0)
    A = torch.stack([acts[a][layer].float() for a in anchors]) - mu
    return cosmat(h - mu, A)[0]


ITEM_RE = re.compile(
    r"Location:\s*(?P<loc>.*?)\s*\|\s*Problem:\s*(?P<prob>.*?)\s*\|\s*"
    r"Remove:\s*(?P<rem>.*?)\s*\|\s*Replace with:\s*(?P<rep>.*)",
    re.DOTALL)


def parse_grounding_items(text, max_items=6):
    items = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        m = re.sub(r"^\d+[\.\)]\s*", "", line)
        g = ITEM_RE.search(m)
        if g:
            items.append({k: g.group(k).strip() for k in ("loc", "prob", "rem", "rep")})
        if len(items) >= max_items:
            break
    return items


def _clip(s, n=300):
    return s if len(s) <= n else s[:n] + "..."


# a grounding item matches a target sentence when the sequence-similarity
# ratio reaches the threshold
def match_span_to_target(loc_prob_text, target_sents, threshold=0.30):
    q = (loc_prob_text or "").lower()
    best, best_r = None, 0.0
    for s in target_sents:
        r = difflib.SequenceMatcher(None, q, s.lower()).ratio()
        if r > best_r:
            best, best_r = s, r
    if best_r >= threshold:
        return _clip(best), best_r
    return None, best_r


def text_nDup_flag(case_doc, target_doc, n=2000):
    if not case_doc or not target_doc:
        return False
    return difflib.SequenceMatcher(None, case_doc[:n], target_doc[:n]).quick_ratio() > 0.8


def metered_call(client, prompt, model, max_tokens=4096, temperature=0.4):
    last_err = None
    t0 = time.time()
    retries = 0
    for attempt in range(8):
        try:
            kw = {"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}

            if temperature is not None:
                kw["temperature"] = temperature
            resp = client.messages.create(**kw)
            u = resp.usage
            usage = {}
            if u is not None:
                for f in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                          "cache_creation_input_tokens"):
                    v = getattr(u, f, None)
                    if v:
                        usage[f] = v
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            if not text and resp.content:
                text = resp.content[0].text
            info = {"in": u.input_tokens if u else 0, "out": u.output_tokens if u else 0,
                    "calls": 1, "retries": retries, "latency_s": round(time.time() - t0, 3),
                    "usage": usage}
            return text, info
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = ("429" in msg or "529" in msg or "overloaded" in msg.lower()
                         or "rate" in msg.lower())
            if not transient:
                break
            retries += 1
            time.sleep(min(10 * (2 ** attempt), 120))
    return (f"[ERROR: {last_err}]",
            {"in": 0, "out": 0, "calls": 1, "retries": retries,
             "latency_s": round(time.time() - t0, 3), "usage": {}})
