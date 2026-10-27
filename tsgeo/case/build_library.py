import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from anthropic import Anthropic

from tsgeo.config import require_glm_key
from tsgeo.core.axis import build_axis_and_prototype
from tsgeo.core.diagnose import diagnose
from tsgeo.case.common import (RESEARCHY, CASECMP_DIR, QWEN_ACTS, SAMPLES,
                               DICT_PATH, GLM_API, probe_split, anchor_ids,
                               rel_matrix_probe, parse_grounding_items,
                               metered_call)

CONSTRUCTION_ACTS = QWEN_ACTS
CONSTRUCTION_SAMPLES = SAMPLES
POOL_SEED = 777
MIN_DOC_CHARS = 200


def _ground_prompt(row, add_f, red_f):
    add_str = "; ".join(add_f) if add_f else "(none)"
    red_str = "; ".join(red_f) if red_f else "(none)"
    return f"""You are diagnosing a web document that a generative engine cited little or not at all.

Query the document should serve: {row['query']}

A representation-space analysis found this document:
- LACKS these features: {add_str}
- HAS these negative features: {red_str}

Read the document carefully and identify CONCRETELY where these problems occur.

CRITICAL: For every piece of low-quality content that should be REMOVED, you MUST specify what useful content should REPLACE it. The document must not get shorter — every deletion must be compensated.

Output a numbered list of specific findings (3-6 items), each in the form:
- Location: <which part> | Problem: <what's wrong> | Remove: <what to delete> | Replace with: <what useful content to add instead>

Document:
{row['doc'][:6000]}

Output ONLY the numbered list."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--topk-dims", type=int, default=8)
    ap.add_argument("--proto-quantile", type=int, default=75)
    ap.add_argument("--out-dir", default=CASECMP_DIR)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    require_glm_key()

    os.makedirs(args.out_dir, exist_ok=True)
    lib_path = os.path.join(args.out_dir, "case_library.jsonl")
    rmat_path = os.path.join(args.out_dir, "case_r.pt")

    t0 = time.time()

    w, prototype = build_axis_and_prototype(
        args.proto_quantile,
        construction_acts=CONSTRUCTION_ACTS,
        construction_samples=CONSTRUCTION_SAMPLES)
    dim_dict = json.load(open(DICT_PATH))

    qa, meta, train_ids, held_ids = probe_split()
    anchors = anchor_ids(train_ids)
    meta_rows = {}
    with open(SAMPLES) as f:
        for l in f:
            r = json.loads(l)
            meta_rows[r["id"]] = r
    eligible = [s for s in train_ids
                if len((meta_rows.get(s) or {}).get("doc", "") or "") >= MIN_DOC_CHARS]
    pool_rng = np.random.RandomState(POOL_SEED)
    pool = [eligible[i] for i in pool_rng.choice(len(eligible),
                                                 min(args.n_cases, len(eligible)),
                                                 replace=False)]
    print(f"[pool] eligible={len(eligible)} (train={len(train_ids)}, held={len(held_ids)}), "
          f"sampled {len(pool)} cases seed={POOL_SEED}", flush=True)

    R_pool, p_mu = rel_matrix_probe(qa, train_ids, anchors)
    rmap = {sid: R_pool[i].clone() for i, sid in enumerate(train_ids)}
    torch.save({"ids": pool, "R": torch.stack([rmap[s] for s in pool]),
                "anchors": anchors, "p_mu": p_mu,
                "protocol": "K=128 RandomState(0) anchors, Qwen L21, train-mean centered"},
               rmat_path)
    print(f"[rel] case R matrix {torch.stack([rmap[s] for s in pool]).shape} -> {rmat_path}",
          flush=True)

    done = set()
    if args.resume and os.path.exists(lib_path):
        for l in open(lib_path):
            try:
                r = json.loads(l)
                if not str(r.get("specific_diagnosis", "")).startswith("[ERROR"):
                    done.add(r["id"])
            except Exception:
                pass
        print(f"[resume] {len(done)} cases done", flush=True)

    client_cfg = GLM_API
    tot_in = tot_out = tot_calls = 0

    def ground_one(sid):
        row = meta_rows[sid]
        client = Anthropic(api_key=client_cfg["api_key"], base_url=client_cfg["api_base"])
        act = qa[sid][21].float().numpy()
        add_f, red_f, details = diagnose(act, w, prototype, dim_dict, args.topk_dims)
        text, info = metered_call(client, _ground_prompt(row, add_f, red_f),
                                  client_cfg["model"], max_tokens=1024)
        rec = {"id": sid, "qid": sid.rsplit("_doc_", 1)[0], "query": row["query"],
               "score": row["score"], "doc_len": len(row["doc"]), "doc": row["doc"][:2500],
               "add_features": add_f, "reduce_features": red_f,
               "axis_diagnosis": details, "specific_diagnosis": text,
               "items": parse_grounding_items(text),
               "grounding_tokens": [info["in"], info["out"]],
               "grounding_latency_s": info["latency_s"], "grounding_retries": info["retries"]}
        return rec, info

    todo = [s for s in pool if s not in done]
    n_ok = 0
    fout = open(lib_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(ground_one, s): s for s in todo}
            for fut in as_completed(futs):
                rec, info = fut.result()
                if str(rec["specific_diagnosis"]).startswith("[ERROR"):
                    print(f"  [ERR] {rec['id']}: {rec['specific_diagnosis'][:80]}", flush=True)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                tot_in += info["in"]; tot_out += info["out"]; tot_calls += 1
                n_ok += 1
                if n_ok % 20 == 0 or n_ok == len(todo):
                    print(f"  [{n_ok}/{len(todo)}] tot in/out={tot_in}/{tot_out} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        fout.close()

    manifest = {"n_cases": len(pool), "pool_seed": POOL_SEED,
                "n_anchors": len(anchors), "topk_dims": args.topk_dims,
                "proto_quantile": args.proto_quantile,
                "offline_cost": {"grounding_calls": tot_calls,
                                 "in_tokens": tot_in, "out_tokens": tot_out,
                                 "wall_s": round(time.time() - t0, 1)},
                "train_held": [len(train_ids), len(held_ids)]}
    json.dump(manifest, open(os.path.join(args.out_dir, "case_lib_manifest.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[saved] {lib_path} ({n_ok} new) + {rmat_path} + manifest")
    print(f"offline grounding cost: calls={tot_calls} in={tot_in} out={tot_out} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
