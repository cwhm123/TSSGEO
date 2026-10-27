import argparse
import json
import os
import random
import re
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local as thread_local

import numpy as np
import torch
from anthropic import Anthropic

import tsgeo.core.llm as llm_mod
import tsgeo.core.diagnose as diagnose_mod
from tsgeo.core.axis import build_axis_and_prototype
from tsgeo.case.common import (GLM_API, RESEARCHY, QWEN_ACTS, SAMPLES,
                               DICT_PATH, DATASETS)

BRIDGE_DIR = f"{RESEARCHY}/bridge"
OUT_DIR = f"{BRIDGE_DIR}/casecmp"
PROBES = {"llama": f"{BRIDGE_DIR}/bridge_W_llama.pt",
          "gemma": f"{BRIDGE_DIR}/bridge_W_gemma.pt"}


_tls = thread_local()


def metered_call_llm(client, prompt, model="glm-5", max_tokens=4096, temperature=0.4):
    trace = getattr(_tls, "trace", None)
    last_err = None
    t0 = time.time()
    retries = 0
    for attempt in range(8):
        try:
            resp = client.messages.create(model=model, max_tokens=max_tokens,
                                          temperature=temperature,
                                          messages=[{"role": "user", "content": prompt}])
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
            rec = {"in": u.input_tokens if u else 0, "out": u.output_tokens if u else 0,
                   "latency_s": round(time.time() - t0, 3), "retries": retries,
                   "prompt_chars": len(prompt), "usage": usage}
            if trace is not None:
                trace.append(rec)
            return text, rec["in"], rec["out"]
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = ("429" in msg or "529" in msg or "overloaded" in msg.lower()
                         or "rate" in msg.lower())
            if not transient:
                break
            retries += 1
            time.sleep(min(10 * (2 ** attempt), 120))
    rec = {"in": 0, "out": 0, "latency_s": round(time.time() - t0, 3), "retries": retries,
           "prompt_chars": len(prompt), "usage": {}}
    if trace is not None:
        trace.append(rec)
    return f"[ERROR: {last_err}]", 0, 0


llm_mod.call_llm = metered_call_llm
diagnose_mod.call_llm = metered_call_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=list(PROBES))
    ap.add_argument("--dataset", default="researchy", choices=list(DATASETS),
                    help="test 侧数据集 (桥 W/轴始终是 researchy 冻结件)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--topk-dims", type=int, default=8)
    ap.add_argument("--proto-quantile", type=int, default=75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-mode", default="random", choices=["weakest", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    ds_tag = "" if args.dataset == "researchy" else f"_{args.dataset}"
    out_path = args.output or os.path.join(OUT_DIR, f"rewrites_cond4_{args.probe}{ds_tag}.jsonl")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(vars(args), open(os.path.join(OUT_DIR, f"config_cond4_{args.probe}{ds_tag}.json"), "w"),
              indent=2, ensure_ascii=False)

    w, prototype = build_axis_and_prototype(
        args.proto_quantile, construction_acts=QWEN_ACTS,
        construction_samples=SAMPLES)
    dim_dict = json.load(open(ds["dict"]))

    bridge = torch.load(PROBES[args.probe], map_location="cpu", weights_only=False)
    W, mu, y_mean = bridge["W"], bridge["mu"], bridge["y_mean"]
    tgt_acts = torch.load(ds["test_acts"][args.probe],
                          map_location="cpu", weights_only=False)
    print(f"[bridge] {args.probe}: L{bridge['layer']} lam={bridge['lam']:.0f}, "
          f"{len(tgt_acts)} test docs", flush=True)

    def bridged_act(sid):
        h = tgt_acts[sid][bridge["layer"]].float()
        return (h - mu.squeeze(0)) @ W + y_mean

    import nltk
    for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try: nltk.data.find(res)
        except LookupError:
            try: nltk.download(res.split("/")[-1], quiet=True)
            except: pass

    qdata, doc_text = {}, {}
    for l in open(ds["labels_test"]):
        r = json.loads(l)
        qid = r["query_id"]
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
        p = os.path.join(ds["queries_test"], f"{qid}.json")
        if os.path.exists(p):
            qd = json.load(open(p))
            for d in qd["documents"]:
                doc_text[(qid, d["document_idx"])] = d.get("text", "") or ""

    rng = random.Random(args.seed)
    targets = []
    for qid, docs in qdata.items():
        pool_ = [(idx, c, s) for idx, c, s in docs if f"{qid}_doc_{idx}" in tgt_acts]
        if not pool_: continue
        idx, was_cited, score = rng.choice(pool_)
        sid = f"{qid}_doc_{idx}"
        if sid not in tgt_acts: continue
        targets.append((sid, qid, idx, was_cited, score))
    targets = targets[:args.n]
    print(f"[targets] {len(targets)} (mode={args.target_mode}, seed={args.seed})", flush=True)

    def process_one(i, target):
        t0 = time.time()
        sid, qid, idx, was_cited, score = target
        client = Anthropic(api_key=GLM_API["api_key"], base_url=GLM_API["api_base"])
        _tls.trace = []
        t_diag = time.time()
        doc_act = bridged_act(sid).numpy()
        add_f, red_f, details = diagnose_mod.diagnose(doc_act, w, prototype, dim_dict,
                                                      args.topk_dims)
        diag_ms = round((time.time() - t_diag) * 1000, 1)
        orig = doc_text.get((qid, idx), "")
        query = json.load(open(os.path.join(ds["queries_test"], f"{qid}.json")))["query"] \
            if os.path.exists(os.path.join(ds["queries_test"], f"{qid}.json")) else ""
        spec_diag, ci, co = diagnose_mod.diagnose_specific(orig, query, add_f, red_f,
                                                           client, model=GLM_API["model"])
        if spec_diag.startswith("[ERROR"):
            rewritten = spec_diag
        else:
            rewritten, ri, ro = llm_mod.rewrite(orig, query, spec_diag, client,
                                               model=GLM_API["model"])
        trace = getattr(_tls, "trace", [])
        _tls.trace = None
        tot_in = sum(t["in"] for t in trace)
        tot_out = sum(t["out"] for t in trace)
        rec = {"id": sid, "qid": qid, "doc_idx": idx, "query": query,
               "probe": args.probe, "condition": "cond4_bridge_grounding",
               "was_cited": was_cited, "orig_contribution": score,
               "add_features": add_f, "reduce_features": red_f,
               "axis_diagnosis": details, "specific_diagnosis": spec_diag,
               "original": orig, "rewritten": rewritten,
               "orig_len": len(orig), "new_len": len(rewritten),
               "cost": {"llm_calls": len(trace), "grounding_calls": 1 if len(trace) > 0 else 0,
                        "in_tokens": tot_in, "out_tokens": tot_out,
                        "trace": trace, "diag_local_ms": diag_ms,
                        "llm_latency_s": round(sum(t["latency_s"] for t in trace), 3),
                        "total_s": round(time.time() - t0, 3)}}
        return i, rec

    done = set()
    if args.resume and os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                if not r["rewritten"].startswith("[ERROR"):
                    done.add(r["id"])
            except Exception:
                pass
        print(f"[resume] {len(done)} done", flush=True)
    todo = [(i, t) for i, t in enumerate(targets) if t[0] not in done]
    fout = open(out_path, "a")
    n_new = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, i, t): (i, t) for i, t in todo}
            for fut in as_completed(futs):
                i, rec = fut.result()
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_new += 1
                c = rec["cost"]
                if n_new % 10 == 0 or n_new == len(todo):
                    print(f"  [{n_new}/{len(todo)}] {rec['id']} calls={c['llm_calls']} "
                          f"in/out={c['in_tokens']}/{c['out_tokens']}", flush=True)
    finally:
        fout.close()
    print(f"\nDONE {n_new} new -> {out_path}")


if __name__ == "__main__":
    main()
