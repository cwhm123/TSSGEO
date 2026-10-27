import argparse
import json
import os
import random

import numpy as np
import torch
from anthropic import Anthropic

from tsgeo.config import (GLM_API, QWEN_ACTS, CONSTRUCTION_SAMPLES, TEST_ACTS_A,
                          TEST_LABELS, TEST_QUERIES, DICT_PATH, RESEARCHY,
                          require_glm_key)
from tsgeo.core.axis import build_axis_and_prototype
from tsgeo.core.diagnose import diagnose, diagnose_specific
from tsgeo.core.llm import rewrite

API_KEY = GLM_API["api_key"]
API_BASE = GLM_API["api_base"]
API_MODEL = GLM_API["model"]

CONSTRUCTION_ACTS = QWEN_ACTS

DEFAULT_TEST_ACTS = TEST_ACTS_A
DEFAULT_TEST_LABELS = TEST_LABELS
DEFAULT_TEST_QUERIES = TEST_QUERIES
DEFAULT_DICT = DICT_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--topk-dims", type=int, default=8)
    ap.add_argument("--rho", type=float, default=None,
                    help="diagnosis ratio rho=M/K; if given, topk=max(1,round(rho*K)) overrides --topk-dims")
    ap.add_argument("--proto-quantile", type=int, default=75,
                    help="prototype percentile q=(100-quantile)/100; 75=top25%% (default/main)")
    ap.add_argument("--output", default=f"{RESEARCHY}/rewrites_test.jsonl")
    ap.add_argument("--resume", action="store_true", help="resume: if output exists, skip written ids, append rest (crash-safe)")

    ap.add_argument("--test-acts", default=DEFAULT_TEST_ACTS)
    ap.add_argument("--test-labels", default=DEFAULT_TEST_LABELS)
    ap.add_argument("--test-queries", default=DEFAULT_TEST_QUERIES)
    ap.add_argument("--dict", default=DEFAULT_DICT)
    ap.add_argument("--construction-acts", default=CONSTRUCTION_ACTS,
                    help="training activations for axis (default researchy/acts_A/all.pt)")
    ap.add_argument("--construction-samples", default=CONSTRUCTION_SAMPLES,
                    help="training samples for axis (default researchy/samples_1000.jsonl)")
    ap.add_argument("--model", default=API_MODEL, help="rewrite/diagnosis LLM model name (default glm-5)")
    ap.add_argument("--api-base", default=API_BASE)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--workers", type=int, default=1, help="concurrent rewrite threads")
    ap.add_argument("--target-mode", default="weakest", choices=["weakest", "random", "original"],
                    help="weakest=prefer uncited/lowest contrib; random=random; original=dataset's target_id")
    ap.add_argument("--seed", type=int, default=42, help="random mode random seed")
    ap.add_argument("--target-id-parquet", default=None,
                    help="original mode required: path to main_test.parquet")
    args = ap.parse_args()
    require_glm_key()

    w, prototype = build_axis_and_prototype(
        args.proto_quantile,
        construction_acts=args.construction_acts,
        construction_samples=args.construction_samples)
    dim_dict = json.load(open(args.dict))
    if args.rho is not None:
        K = len(dim_dict["dimensions"])
        args.topk_dims = max(1, round(args.rho * K))
        print(f"[rho] ρ={args.rho:.1%} -> topk={args.topk_dims} (K={K})", flush=True)
    wn = w / (np.linalg.norm(w) + 1e-8)
    test_acts = torch.load(args.test_acts, map_location="cpu", weights_only=False)

    qdata, qid2query, doc_text = {}, {}, {}
    import re, math
    import nltk
    nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
    for l in open(args.test_labels):
        r = json.loads(l)
        qid = r["query_id"]
        qid2query[qid] = r["query"]
        sents = []
        for p in re.split(r"\n\n", r["generated_answer"]):
            for s in nltk.sent_tokenize(p):
                cits = [int(re.findall(r"\d+", c)[0]) for c in re.findall(r"\[[^\w\s]*\d+[^\w\s]*\]", s)]
                sents.append((nltk.word_tokenize(s), s, cits))
        scores = [0.0] * r["n_docs"]
        L = len(sents)
        for i, (toks, s, cits) in enumerate(sents):
            for c in cits:
                if 1 <= c <= r["n_docs"]:
                    sc = len([x for x in toks if len(x) > 2]) * (math.exp(-i/(L-1)) if L > 1 else 1.0) / max(len(cits), 1)
                    scores[c-1] += sc
        tot = sum(scores)
        scores = [x/tot if tot else 0 for x in scores]
        qdata[qid] = [(idx, d["is_cited"], scores[idx]) for idx, d in enumerate(r["documents"])]
    for qid in qdata:
        p = os.path.join(args.test_queries, f"{qid}.json")
        if os.path.exists(p):
            qd = json.load(open(p))
            for d in qd["documents"]:
                doc_text[(qid, d["document_idx"])] = d.get("text", "") or ""

    rng = random.Random(args.seed)
    qid2target = {}
    if args.target_mode == "original":
        import pandas as pd
        assert args.target_id_parquet, "original mode needs --target-id-parquet"
        qid2target = {row["query"]: int(row["target_id"]) for _, row in pd.read_parquet(args.target_id_parquet).iterrows()}
        print(f"[original] parquet target_id mapped {len(qid2target)} entries", flush=True)
    targets = []
    for qid, docs in qdata.items():
        def proj_of(idx):
            sid = f"{qid}_doc_{idx}"
            return float(test_acts[sid][21].float().numpy() @ wn) if sid in test_acts else 0
        if args.target_mode == "random":
            pool = [(idx, c, s) for idx, c, s in docs if f"{qid}_doc_{idx}" in test_acts]
            if not pool: continue
            idx, was_cited, score = rng.choice(pool)
        elif args.target_mode == "original":
            tid = qid2target.get(qid2query.get(qid, ""))
            if tid is None: continue
            match = [(idx, c, s) for idx, c, s in docs if idx == tid]
            if not match: continue
            idx, was_cited, score = match[0]
        else:
            uncited = [(idx, c, s) for idx, c, s in docs if not c]
            pool = uncited if uncited else docs
            idx, was_cited, score = min(pool, key=lambda x: proj_of(x[0]))
        sid = f"{qid}_doc_{idx}"
        if sid not in test_acts: continue
        targets.append((sid, qid, idx, was_cited, score, proj_of(idx)))
    if args.target_mode == "weakest":
        targets.sort(key=lambda t: t[5])
    targets = targets[:args.n]
    n_uncited = sum(1 for t in targets if not t[3])
    print(f"\n[target-mode={args.target_mode}] selected {len(targets)} targets ({n_uncited} uncited)\n", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def process_one(i, target):
        sid, qid, idx, was_cited, score, proj = target
        client = Anthropic(api_key=args.api_key, base_url=args.api_base)
        doc_act = test_acts[sid][21].float().numpy()
        add_f, red_f, details = diagnose(doc_act, w, prototype, dim_dict, args.topk_dims)
        orig = doc_text.get((qid, idx), "")
        qpath = os.path.join(args.test_queries, f"{qid}.json")
        query = json.load(open(qpath))["query"] if os.path.exists(qpath) else ""
        c_prompt, spec_diag, ci, co = diagnose_specific(orig, query, add_f, red_f, client, model=args.model)
        if spec_diag.startswith("[ERROR"):
            r_prompt, rewritten, ri, ro = "", spec_diag, 0, 0
        else:
            r_prompt, rewritten, ri, ro = rewrite(orig, query, spec_diag, client, model=args.model)
        rec = {"id": sid, "qid": qid, "doc_idx": idx, "query": query,
               "was_cited": was_cited, "orig_contribution": score, "projection": proj,
               "axis_diagnosis": details, "add_features": add_f, "reduce_features": red_f,
               "specific_diagnosis": spec_diag,
               "concretize_prompt": c_prompt, "revision_prompt": r_prompt,
               "concretize_tokens": [ci, co], "revision_tokens": [ri, ro],
               "original": orig, "rewritten": rewritten,
               "orig_len": len(orig), "new_len": len(rewritten)}
        return i, rec, (ci + ri, co + ro, 2)

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        for l in open(args.output):
            try: done_ids.add(json.loads(l)["id"])
            except: pass
        print(f"[resume] done {len(done_ids)}/{len(targets)}, skipping these", flush=True)
    todo = [(i, t) for i, t in enumerate(targets) if t[0] not in done_ids]
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done_cnt = len(done_ids); n_new = 0
    fout = open(args.output, "a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, i, t): (i, t) for i, t in todo}
            for fut in as_completed(futs):
                i, rec, (ti, to, calls) = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                done_cnt += 1; n_new += 1
                if n_new % 5 == 0 or n_new == len(todo):
                    print(f"  [{done_cnt}/{len(targets)}] {rec['id']}  done (in/out/calls={ti}/{to}/{calls})", flush=True)
    finally:
        fout.close()
    print(f"\nDONE: total {done_cnt}/{len(targets)} (new {n_new}) -> {args.output}")


if __name__ == "__main__":
    main()
