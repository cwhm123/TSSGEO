import argparse, json, os, re, time
from collections import Counter, defaultdict

import numpy as np
import torch
from anthropic import Anthropic

from tsgeo.config import (GLM_API, QWEN_ACTS, CONSTRUCTION_SAMPLES,
                          DIMENSION_DICT, DICT_PATH, require_glm_key)

API_KEY = GLM_API["api_key"]
API_BASE = GLM_API["api_base"]
API_MODEL = GLM_API["model"]
LAYER = 21
ACTS = QWEN_ACTS
SAMPLES = CONSTRUCTION_SAMPLES
EXISTING_DICT = DIMENSION_DICT
OUT = DICT_PATH


# accumulated dictionary: per-query top-K discriminative dims, kept when
# frequent in >= min-freq queries (cap max-dims), GLM-labeled
def main():
    global ACTS, SAMPLES, EXISTING_DICT, OUT, LAYER
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=ACTS, help="构造集激活 all.pt (文件路径)")
    ap.add_argument("--samples", default=SAMPLES, help="构造集 samples jsonl")
    ap.add_argument("--existing-dict", default=EXISTING_DICT, help="已有维度字典, 复用其中已标注维度")
    ap.add_argument("--output", default=OUT)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--topk-per-query", type=int, default=20)
    ap.add_argument("--min-freq", type=int, default=20, help="维度至少出现在多少个query的top-K里才纳入")
    ap.add_argument("--max-dims", type=int, default=200)
    args = ap.parse_args()
    require_glm_key()
    ACTS, SAMPLES, EXISTING_DICT, OUT, LAYER = args.acts, args.samples, args.existing_dict, args.output, args.layer

    acts = torch.load(ACTS, map_location="cpu", weights_only=False)
    meta = {}
    with open(SAMPLES) as f:
        for l in f:
            r = json.loads(l)
            if r["id"] in acts: meta[r["id"]] = r["score"]

    by_q = defaultdict(list)
    for sid in meta: by_q[sid.rsplit("_doc_", 1)[0]].append(sid)
    print(f"{len(by_q)} 个 query", flush=True)

    existing = {}
    if os.path.exists(EXISTING_DICT):
        for k, v in json.load(open(EXISTING_DICT))["dimensions"].items():
            llm = v.get("llm", v) if isinstance(v.get("llm"), dict) else v
            if "semantic_label" not in llm: llm = v.get("llm", {})
            existing[int(k)] = llm

    dim_freq = Counter()
    n_queries_used = 0
    for qi, (qid, sids) in enumerate(by_q.items()):
        if len(sids) < 2: continue
        xq = torch.stack([acts[s][LAYER].float() for s in sids]).numpy()
        sq = np.array([meta[s] for s in sids])
        if np.std(sq) == 0: continue

        x_demean = xq - xq.mean(0, keepdims=True)
        s_demean = sq - sq.mean()
        w_q = (x_demean * s_demean[:, None]).sum(0)

        top_dims = np.argsort(np.abs(w_q))[::-1][:args.topk_per_query]
        for d in top_dims:
            dim_freq[int(d)] += 1
        n_queries_used += 1
        if (qi+1) % 200 == 0:
            print(f"  处理 {qi+1}/{len(by_q)} query, 累积 {len(dim_freq)} 维", flush=True)

    print(f"\n用了 {n_queries_used} 个 query, 累积到 {len(dim_freq)} 个唯一维度", flush=True)
    for mf in [500, 200, 100, 50, 20]:
        n = sum(1 for f in dim_freq.values() if f >= mf)
        print(f"  freq>={mf}: {n} 维")

    ranked = [(d, f) for d, f in dim_freq.most_common() if f >= args.min_freq][:args.max_dims]
    new_dims = [d for d, _ in ranked if d not in existing]
    print(f"\n纳入字典: {len(ranked)} 维 (freq>={args.min_freq})")
    print(f"  已有标签: {len(ranked)-len(new_dims)}, 需新标注: {len(new_dims)}", flush=True)

    client = Anthropic(api_key=API_KEY, base_url=API_BASE)
    sids_all = sorted(meta.keys())
    Xall = np.stack([acts[s][LAYER].float().numpy() for s in sids_all])

    dimensions = {}
    for i, (dim, freq) in enumerate(ranked):
        if dim in existing:
            entry = existing[dim]
        else:
            entry = label_dim(dim, Xall, sids_all, client)
            time.sleep(0.3)
        lab = entry.get("semantic_label", "?") if isinstance(entry, dict) else "?"
        tag = "(已有)" if dim in existing else "(新标)"
        print(f"  [{i+1}/{len(ranked)}] dim {dim} freq={freq} {tag} {lab}", flush=True)
        dimensions[str(dim)] = {"dim": dim, "freq": freq, "llm": entry}

    out = {"layer": LAYER, "method": "accumulated_v2_perquery_score",
           "topk_per_query": args.topk_per_query, "min_freq": args.min_freq,
           "n_queries_used": n_queries_used, "dimensions": dimensions}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n累积字典 v2: {len(dimensions)} 维 -> {OUT}")


def label_dim(dim, Xall, sids, client, n_ex=5):
    order = np.argsort(Xall[:, dim])
    high = order[-n_ex:][::-1]; low = order[:n_ex]
    meta_map = {}
    with open(SAMPLES) as f:
        for l in f:
            r = json.loads(l); meta_map[r["id"]] = r
    def fmt(idx):
        rs = []
        for i in idx:
            s = sids[i]; r = meta_map.get(s, {})
            rs.append(f"[score={r.get('score',0):.2f}] q: {r.get('query','')}\ndoc: {(r.get('doc','') or '')[:450]}")
        return "\n---\n".join(rs)
    prompt = f"""Analyze what hidden dimension #{dim} of a language model encodes (generative-engine citation contribution context).
HIGH activation docs:
{fmt(high)}
LOW activation docs:
{fmt(low)}
Output STRICT JSON: {{"semantic_label":"<3-6 words>","text_features":["<f1>","<f2>",...]}}
Use concise snake_case tags. ONLY JSON."""
    try:
        r = client.messages.create(model=API_MODEL, max_tokens=250, temperature=0.3,
                                   messages=[{"role": "user", "content": prompt}])
        m = re.search(r"\{.*\}", r.content[0].text, re.DOTALL)
        return json.loads(m.group()) if m else {"semantic_label": f"dim_{dim}"}
    except Exception as e:
        return {"semantic_label": f"dim_{dim}", "error": str(e)[:80]}


if __name__ == "__main__":
    main()
