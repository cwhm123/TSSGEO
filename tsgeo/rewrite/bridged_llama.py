import argparse
import json
import os
import random
import re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from anthropic import Anthropic

from tsgeo.config import (GLM_API, QWEN_ACTS, CONSTRUCTION_SAMPLES, TEST_LABELS,
                          TEST_QUERIES, DICT_PATH, BRIDGE_DIR)
from tsgeo.core.axis import build_axis_and_prototype
from tsgeo.core.diagnose import diagnose, diagnose_specific
from tsgeo.core.llm import rewrite

API_KEY = GLM_API["api_key"]
API_BASE = GLM_API["api_base"]
API_MODEL = GLM_API["model"]

CONSTRUCTION_ACTS = QWEN_ACTS
DEFAULT_TEST_LABELS = TEST_LABELS
DEFAULT_TEST_QUERIES = TEST_QUERIES
DEFAULT_DICT = DICT_PATH

PROBES = {
    "llama": {"W": f"{BRIDGE_DIR}/bridge_W_llama.pt"},
    "gemma": {"W": f"{BRIDGE_DIR}/bridge_W_gemma.pt"},
}


REWRITER_ENGINES = {
    "glm":    {"api_key": os.environ.get("GLM_API_KEY", ""),
               "api_base": "https://open.bigmodel.cn/api/anthropic",
               "model":    "glm-5"},
    "gpt":    {"api_key": os.environ.get("GPT_API_KEY", ""),
               "api_base": os.environ.get("GPT_API_BASE", "https://api.sbbbbbbbbb.xyz"),
               "model":    "gpt-5.6-terra"},
    "claude": {"api_key": os.environ.get("CLAUDE_API_KEY", ""),
               "api_base": os.environ.get("CLAUDE_API_BASE", "https://api.sbbbbbbbbb.xyz"),
               "model":    "claude-opus-4-6"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=list(PROBES))
    ap.add_argument("--rewriter-engine", choices=list(REWRITER_ENGINES), default="glm",
                    help="改写引擎: glm (默认, GLM-5) / gpt (GPT-5.4) / claude (Claude Opus 4.6). "
                         "判定引擎 = rewriter-engine (tsgeo.rewrite.glm_native 同协议)。")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topk-dims", type=int, default=8)
    ap.add_argument("--proto-quantile", type=int, default=75)
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test-labels", default=DEFAULT_TEST_LABELS)
    ap.add_argument("--test-queries", default=DEFAULT_TEST_QUERIES)
    ap.add_argument("--test-acts", default=None,
                    help="目标模型 test 激活路径 (默认用 researchy/bridge/test_acts_<probe>/all.pt)")
    ap.add_argument("--dict", default=DEFAULT_DICT)
    ap.add_argument("--model", default=None,
                    help="覆盖 model name (默认按 --rewriter-engine 自动选 glm-5/gpt-5.4/claude-opus-4-6)")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-mode", default="random", choices=["weakest", "random"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.model is None:
        args.model = REWRITER_ENGINES[args.rewriter_engine]["model"]
    if args.api_base is None:
        args.api_base = REWRITER_ENGINES[args.rewriter_engine]["api_base"]
    if args.api_key is None:
        args.api_key = REWRITER_ENGINES[args.rewriter_engine]["api_key"]
    if not args.api_key:
        env_hint = {"glm": "GLM_API_KEY", "gpt": "GPT_API_KEY",
                    "claude": "CLAUDE_API_KEY"}[args.rewriter_engine]
        raise SystemExit(f"ERROR: no API key for engine '{args.rewriter_engine}'. "
                         f"export {env_hint}=<key> or pass --api-key.")
    if args.output is None:
        if args.rewriter_engine == "glm":

            args.output = f"{BRIDGE_DIR}/rewrites_bridge_{args.probe}.jsonl"
        else:
            args.output = f"{BRIDGE_DIR}/rewrites_bridge_{args.probe}_{args.rewriter_engine}.jsonl"
    if args.test_acts is None:
        args.test_acts = f"{BRIDGE_DIR}/test_acts_{args.probe}/all.pt"

    w, prototype = build_axis_and_prototype(
        args.proto_quantile,
        construction_acts=CONSTRUCTION_ACTS,
        construction_samples=CONSTRUCTION_SAMPLES)
    dim_dict = json.load(open(args.dict))
    wn = w / (np.linalg.norm(w) + 1e-8)

    bridge = torch.load(PROBES[args.probe]["W"], map_location="cpu", weights_only=False)
    W, mu, y_mean = bridge["W"], bridge["mu"], bridge["y_mean"]
    tgt_acts = torch.load(args.test_acts, map_location="cpu", weights_only=False)
    print(f"[bridge] {args.probe}: L{bridge['layer']} lam={bridge['lam']:.0f}, "
          f"{len(tgt_acts)} test docs loaded from {args.test_acts}", flush=True)
    print(f"[rewriter] engine={args.rewriter_engine} model={args.model} "
          f"api_base={args.api_base}", flush=True)

    # project Llama test activations through the bridge into the frozen Qwen
    # diagnostic space
    def bridged_act(sid):
        h = tgt_acts[sid][bridge["layer"]].float()
        return (h - mu.squeeze(0)) @ W + y_mean

    import nltk
    for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try: nltk.data.find(res)
        except LookupError:
            try: nltk.download(res.split("/")[-1], quiet=True)
            except: pass

    qdata, qid2query, doc_text = {}, {}, {}
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
    targets = []
    for qid, docs in qdata.items():
        if args.target_mode == "random":
            pool = [(idx, c, s) for idx, c, s in docs if f"{qid}_doc_{idx}" in tgt_acts]
            if not pool: continue
            idx, was_cited, score = rng.choice(pool)
        else:
            uncited = [(idx, c, s) for idx, c, s in docs if not c]
            pool = uncited if uncited else docs
            def proj_of(ix):
                sid = f"{qid}_doc_{ix}"
                return float(bridged_act(sid).numpy() @ wn) if sid in tgt_acts else 0
            idx, was_cited, score = min(pool, key=lambda x: proj_of(x[0]))
        sid = f"{qid}_doc_{idx}"
        if sid not in tgt_acts: continue
        proj = float(bridged_act(sid).numpy() @ wn)
        targets.append((sid, qid, idx, was_cited, score, proj))

    targets = targets[:args.n]
    n_uncited = sum(1 for t in targets if not t[3])
    print(f"\n[target-mode={args.target_mode}] selected {len(targets)} targets ({n_uncited} uncited)\n", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def process_one(i, target):
        sid, qid, idx, was_cited, score, proj = target
        client = Anthropic(api_key=args.api_key, base_url=args.api_base)
        doc_act = bridged_act(sid).numpy()
        add_f, red_f, details = diagnose(doc_act, w, prototype, dim_dict, args.topk_dims)
        orig = doc_text.get((qid, idx), "")
        qpath = os.path.join(args.test_queries, f"{qid}.json")
        query = json.load(open(qpath))["query"] if os.path.exists(qpath) else ""
        spec_diag, ci, co = diagnose_specific(orig, query, add_f, red_f, client, model=args.model)
        if spec_diag.startswith("[ERROR"):
            r_prompt, rewritten, ri, ro = "", spec_diag, 0, 0
        else:
            rewritten, ri, ro = rewrite(orig, query, spec_diag, client, model=args.model)
            r_prompt = ""
        rec = {"id": sid, "qid": qid, "doc_idx": idx, "query": query,
               "probe": args.probe,
               "was_cited": was_cited, "orig_contribution": score, "projection": proj,
               "axis_diagnosis": details, "add_features": add_f, "reduce_features": red_f,
               "specific_diagnosis": spec_diag,
               "concretize_prompt": "", "revision_prompt": r_prompt,
               "concretize_tokens": [ci, co], "revision_tokens": [ri, ro],
               "original": orig, "rewritten": rewritten,
               "orig_len": len(orig), "new_len": len(rewritten)}
        return i, rec, (ci + ri, co + ro, 2)

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        kept_lines = []
        for l in open(args.output):
            try:
                r = json.loads(l)
                if r["rewritten"].startswith("[ERROR"):
                    continue
                done_ids.add(r["id"]); kept_lines.append(l)
            except: pass
        if len(kept_lines) < sum(1 for _ in open(args.output)):
            with open(args.output, "w") as f:
                f.writelines(kept_lines)
        print(f"[resume] done {len(done_ids)}/{len(targets)} (剔除失败行)", flush=True)
    todo = [(i, t) for i, t in enumerate(targets) if t[0] not in done_ids]
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
