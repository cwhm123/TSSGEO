import argparse
import json
import os
import random
import re
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from anthropic import Anthropic

from tsgeo.core.axis import build_axis_and_prototype
from tsgeo.core.diagnose import diagnose
from tsgeo.rewrite.bridged_llama import REWRITER_ENGINES
from tsgeo.case.common import (RESEARCHY, CASECMP_DIR, QWEN_ACTS, SAMPLES,
                               DICT_PATH, GLM_API, DATASETS, probe_split,
                               anchor_ids, cosmat, match_span_to_target,
                               text_nDup_flag, metered_call, _clip)

CONSTRUCTION_ACTS = QWEN_ACTS
BRIDGE_DIR = f"{RESEARCHY}/bridge"
PROBES = {"llama": {"W": f"{BRIDGE_DIR}/bridge_W_llama.pt",
                    "train_acts": f"{BRIDGE_DIR}/acts_llama/all.pt"},
          "gemma": {"W": f"{BRIDGE_DIR}/bridge_W_gemma.pt",
                    "train_acts": f"{BRIDGE_DIR}/acts_gemma/all.pt"}}


MATCH_THRESHOLD = 0.30


def single_rewrite_prompt(query, doc_text, plan_text, n_cases):
    return f"""You are a document optimization expert. Revise the document based on the edit plan below, under strict fact constraints.

Query the document should serve: {query}

Edit plan reused from {n_cases} similar source documents (editing experience, not facts to copy):
{plan_text}

Apply the ANALOGOUS type of edit to the indicated parts of THIS document, adapted to its own facts.
Do NOT copy names, numbers, facts, or conclusions from the edit plan.

Editing rules:
1. FACTS ARE FIXED. Keep every fact, name, number, date, and claim of the original document as given. Do not add facts, statistics, examples, or sources that do not appear in the original.
2. EDIT ONLY DIAGNOSED SPANS. Modify only the parts flagged by the edit plan; leave every other passage unchanged.
3. FILL GAPS FROM THE DOCUMENT ITSELF. When the plan asks you to remove a passage, fill the gap by reorganizing or rephrasing material that already exists elsewhere in the document, not by writing new factual content.
4. IMPROVE FORM, NOT CONTENT. You may restructure the flagged spans, add headings or transitions, and make them directly responsive to the query; you may not change what the document claims.
5. KEEP LENGTH STABLE. The revised document should be about the same length as the original (~{len(doc_text)} chars); do not expand it into a new article and do not shrink it.

If a plan item cannot be satisfied without inventing new factual content, satisfy it structurally (ordering, phrasing, emphasis) instead.

Output MUST be in English.

Original document:
{doc_text[:6000]}

Output the revised document directly, no explanations."""


# edit plan composed from the retrieved cases' grounding items, each mapped
# to the closest sentence of the target document
def compose_plan(cases, target_sents, ipc, mode, seed_i):
    raw = []
    for rank, c in enumerate(cases):
        for it in c["items"][:ipc]:
            matched, ratio = match_span_to_target(
                f"{it['loc']} {it['prob']}", target_sents, MATCH_THRESHOLD)
            raw.append({"case_id": c["id"], "case_rank": rank, "cos": c["cos"],
                        "prob": it["prob"], "rem": it["rem"], "rep": it["rep"],
                        "matched": matched, "match_ratio": round(ratio, 3)})
    if mode == "shuffle" and len(raw) > 1:
        rng = random.Random(123 + seed_i)
        triples = [(x["prob"], x["rem"], x["rep"]) for x in raw]
        shuffled = triples.copy()
        while shuffled == triples and len(set(map(str, triples))) > 1:
            rng.shuffle(shuffled)
        for x, (p, r0, r1) in zip(raw, shuffled):
            x["prob"], x["rem"], x["rep"] = p, r0, r1
    lines = []
    for i, x in enumerate(raw, 1):
        loc = (f'"{x["matched"]}"' if x["matched"]
               else "no close text match — locate the analogous part yourself")
        lines.append(
            f'{i}. Problem found in a similar document: {_clip(x["prob"], 220)}\n'
            f'   Edit applied there: Remove "{_clip(x["rem"], 200)}" -> Replace with "{_clip(x["rep"], 220)}"\n'
            f'   Closest part of THIS document: {loc}')
    return "\n".join(lines), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=list(PROBES))
    ap.add_argument("--dataset", default="researchy", choices=list(DATASETS),
                    help="test 侧数据集 (桥 W/轴/案例库始终是 researchy 冻结件)")
    ap.add_argument("--targets", default="test", choices=["test", "held"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--case-mode", default="sim", choices=["sim", "random", "shuffle", "none"])
    ap.add_argument("--topk-cases", type=int, default=3)
    ap.add_argument("--items-per-case", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lib-dir", default=CASECMP_DIR)
    ap.add_argument("--out-dir", default=CASECMP_DIR)
    ap.add_argument("--engine", default="glm", choices=list(REWRITER_ENGINES),
                    help="改写器引擎; 跨引擎表需要 gpt / claude")
    args = ap.parse_args()
    eng = REWRITER_ENGINES[args.engine]
    if not eng["api_key"]:
        env_hint = {"glm": "GLM_API_KEY", "gpt": "GPT_API_KEY",
                    "claude": "CLAUDE_API_KEY"}[args.engine]
        raise SystemExit(f"ERROR: no API key for engine '{args.engine}'. "
                         f"export {env_hint}=<key>.")

    ds = DATASETS[args.dataset]

    ds_tag = "" if args.dataset == "researchy" else f"{args.dataset}_"

    eng_tag = "" if args.engine == "glm" else f"_{args.engine}"
    tag = f"{args.probe}_{ds_tag}{args.targets}_{args.case_mode}_k{args.topk_cases}{eng_tag}"
    out_path = os.path.join(args.out_dir, f"rewrites_case_{tag}.jsonl")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, f"config_{tag}.json"), "w"),
              indent=2, ensure_ascii=False)

    t_start = time.time()

    w, prototype = build_axis_and_prototype(
        75, construction_acts=CONSTRUCTION_ACTS, construction_samples=CONSTRUCTION_SAMPLES)
    dim_dict = json.load(open(ds["dict"]))
    bridge = torch.load(PROBES[args.probe]["W"], map_location="cpu", weights_only=False)
    W, mu, y_mean = bridge["W"], bridge["mu"], bridge["y_mean"]
    qa, meta, train_ids, held_ids = probe_split()
    anchors = anchor_ids(train_ids)

    lib = {}
    for l in open(os.path.join(args.lib_dir, "case_library.jsonl")):
        r = json.loads(l)
        if not str(r.get("specific_diagnosis", "")).startswith("[ERROR"):
            lib[r["id"]] = r
    rmat = torch.load(os.path.join(args.lib_dir, "case_r.pt"), map_location="cpu",
                      weights_only=False)

    keep = [i for i, sid in enumerate(rmat["ids"]) if sid in lib]
    case_ids = [rmat["ids"][i] for i in keep]
    case_R = rmat["R"][torch.tensor(keep)]
    print(f"[lib] {len(case_ids)}/{len(rmat['ids'])} usable cases, "
          f"R {tuple(case_R.shape)}, n_anchors={len(rmat['anchors'])}", flush=True)

    tgt_train_acts = torch.load(PROBES[args.probe]["train_acts"], map_location="cpu",
                                weights_only=False)
    A_tgt = torch.stack([tgt_train_acts[a][bridge["layer"]].float() for a in anchors]) - mu

    import nltk
    for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try: nltk.data.find(res)
        except LookupError:
            try: nltk.download(res.split("/")[-1], quiet=True)
            except: pass

    meta_rows = {}
    with open(SAMPLES) as f:
        for l in f:
            r = json.loads(l)
            meta_rows[r["id"]] = r
    nltk.sent_tokenize("Prewarm. Sentence tokenizer.")

    if args.targets == "held":

        tgt_acts = tgt_train_acts
        rng = random.Random(args.seed)
        held_eligible = [s for s in held_ids
                         if s in tgt_acts and len((meta_rows.get(s) or {}).get("doc", "") or "") >= 200]
        picked = rng.sample(held_eligible, min(args.n, len(held_eligible)))
        targets = [(s, meta_rows[s]["query"], meta_rows[s]["doc"]) for s in picked]
        print(f"[val] {len(targets)} held construction docs as targets", flush=True)
    else:
        tgt_acts = torch.load(ds["test_acts"][args.probe],
                              map_location="cpu", weights_only=False)

        qdata, qid2query, doc_text = {}, {}, {}
        for l in open(ds["labels_test"]):
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
        print(f"[test] {len(targets)} targets (random seed {args.seed})", flush=True)

    Rn = case_R / case_R.norm(dim=1, keepdim=True)

    def process_one(i, target):
        t0 = time.time()
        if args.targets == "held":
            sid, query, doc = target
            qid, idx = sid.rsplit("_doc_", 1)[0], -1
            was_cited, score, proj = None, meta_rows[sid]["score"], None
        else:
            sid, qid, idx, was_cited, score = target
            doc = doc_text.get((qid, idx), "")
            qpath = os.path.join(ds["queries_test"], f"{qid}.json")
            query = json.load(open(qpath))["query"] if os.path.exists(qpath) else ""
            proj = None

        t_enc = time.time()
        h = tgt_acts[sid][bridge["layer"]].float()
        doc_act = ((h - mu.squeeze(0)) @ W + y_mean).numpy()
        add_f, red_f, details = diagnose(doc_act, w, prototype, dim_dict, 8)

        hh = (h - mu.squeeze(0)).unsqueeze(0)
        r_t = cosmat(hh, A_tgt)[0]
        t_retr = time.time()

        sims = (Rn @ (r_t / (r_t.norm() + 1e-8)))
        order = torch.argsort(sims, descending=True).tolist()
        chosen, skipped_ndup = [], 0
        if args.case_mode == "random":
            rr = random.Random(args.seed * 1000 + i)
            cand = [case_ids[j] for j in range(len(case_ids))]
            rr.shuffle(cand)
            for cid in cand:
                cdoc = lib[cid].get("doc", "")
                if text_nDup_flag(cdoc, doc): skipped_ndup += 1; continue
                chosen.append({"id": cid, "cos": None})
                if len(chosen) >= args.topk_cases: break
        elif args.case_mode != "none":
            for j in order:
                cid = case_ids[j]
                cdoc = lib[cid].get("doc", "")
                if text_nDup_flag(cdoc, doc): skipped_ndup += 1; continue
                chosen.append({"id": cid, "cos": round(float(sims[j]), 4)})
                if len(chosen) >= args.topk_cases: break
        for c in chosen:
            c["items"] = lib[c["id"]]["items"]
            c["query"] = lib[c["id"]]["query"]
        t_match = time.time()

        target_sents = nltk.sent_tokenize(doc)
        if args.case_mode == "none":
            plan_text = ("(no case guidance available — improve the document using GEO "
                         "best practices: make it more informative, factual, well-structured, "
                         "and directly responsive to the query.)")
            raw_items = []
        else:
            plan_text, raw_items = compose_plan(chosen, target_sents,
                                                args.items_per_case, args.case_mode, i)
        t_llm = time.time()

        client = Anthropic(api_key=eng["api_key"], base_url=eng["api_base"])
        prompt = single_rewrite_prompt(query, doc, plan_text, len(chosen))
        rewritten, info = metered_call(client, prompt, eng["model"], max_tokens=4096,
                                          temperature=None if args.engine != "glm" else 0.4)
        rec = {"id": sid, "qid": qid, "doc_idx": idx, "query": query,
               "probe": args.probe, "case_mode": args.case_mode, "topk": args.topk_cases,
               "was_cited": was_cited, "orig_contribution": score, "projection": proj,
               "add_features": add_f, "reduce_features": red_f,
               "axis_diagnosis": details,
               "retrieved_cases": [{"id": c["id"], "cos": c["cos"], "n_items": len(c["items"])}
                                    for c in chosen],
               "skipped_ndup": skipped_ndup,
               "plan_items": raw_items,
               "prompt_len_chars": len(prompt),
               "original": doc, "rewritten": rewritten,
               "orig_len": len(doc), "new_len": len(rewritten),
               "cost": {"llm_calls": info["calls"], "grounding_calls": 0,
                        "in_tokens": info["in"], "out_tokens": info["out"],
                        "usage": info["usage"], "retries": info["retries"],
                        "llm_latency_s": info["latency_s"],
                        "local_ms": {"encode_diag": round((t_retr - t_enc) * 1000, 1),
                                     "retrieve": round((t_match - t_retr) * 1000, 1),
                                     "match_compose": round((t_llm - t_match) * 1000, 1)},
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
                          f"in/out={c['in_tokens']}/{c['out_tokens']} "
                          f"llm={c['llm_latency_s']}s local={c['local_ms']}", flush=True)
    finally:
        fout.close()
    print(f"\nDONE {n_new} new -> {out_path} ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
