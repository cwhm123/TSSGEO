import argparse
import json
import os
import time

import numpy as np
import torch
from anthropic import Anthropic

from tsgeo.config import (RESEARCHY, CONSTRUCTION_SAMPLES, DIMENSION_DICT,
                          require_glm_key)

API_KEY = os.environ.get("GLM_API_KEY", "")
API_BASE = os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/anthropic")
API_MODEL = os.environ.get("GLM_MODEL", "glm-5")


def call_llm(client, prompt, max_tokens=600):
    resp = client.messages.create(
        model=API_MODEL, max_tokens=max_tokens, temperature=0.3,
        messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text


# per-query top dims by axis projection, GLM-labeled from high/low examples
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=f"{RESEARCHY}/acts_A")
    ap.add_argument("--samples", default=CONSTRUCTION_SAMPLES)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--top-dims", type=int, default=20)
    ap.add_argument("--n-examples", type=int, default=5, help="每个维度取几个高/低激活样例")
    ap.add_argument("--output", default=DIMENSION_DICT)
    args = ap.parse_args()
    require_glm_key()
    L = args.layer

    acts = torch.load(os.path.join(args.acts, "all.pt"), map_location="cpu", weights_only=False)
    meta = {}
    with open(args.samples) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in acts:
                meta[r["id"]] = r
    sids = sorted(meta.keys())
    print(f"加载 {len(sids)} 样本, 层 {L}, top {args.top_dims} 维度", flush=True)

    by_q = {}
    for sid in sids:
        qid = sid.rsplit("_doc_", 1)[0]
        by_q.setdefault(qid, []).append(sid)
    Xlist, Slist = [], []
    for qid, ids in by_q.items():
        if len(ids) < 2:
            continue
        xq = torch.stack([acts[s][L].float() for s in ids])
        sq = np.array([meta[s]["score"] for s in ids])
        Xlist.append(xq - xq.mean(0, keepdim=True))
        Slist.append(sq - sq.mean())
    X = torch.cat(Xlist, 0).numpy()
    S = np.concatenate(Slist)
    w = (X * S[:, None]).sum(0)
    print(f"轴 w norm={np.linalg.norm(w):.2f}", flush=True)

    Xall = np.stack([acts[s][L].float().numpy() for s in sids])
    sid_arr = np.array(sids)

    topk = np.argsort(np.abs(w))[::-1][:args.top_dims]
    client = Anthropic(api_key=API_KEY, base_url=API_BASE)

    dim_dict = {"layer": L, "axis_norm": float(np.linalg.norm(w)),
                "n_samples": len(sids), "dimensions": {}}

    for di, dim in enumerate(topk):

        dim_act = Xall[:, dim]
        order = np.argsort(dim_act)
        high_idx = order[-args.n_examples:][::-1]
        low_idx = order[:args.n_examples]

        def fmt(idx):
            rows = []
            for i in idx:
                s = sid_arr[i]
                rows.append(f"[score={meta[s]['score']:.2f}] query: {meta[s]['query']}\n"
                            f"doc: {(meta[s]['doc'] or '')[:600]}")
            return "\n---\n".join(rows)

        prompt = f"""You are analyzing what a single hidden dimension of a language model encodes, in the context of generative-engine citations.

Dimension #{dim} is one of the most important dimensions for predicting how much a document CONTRIBUTES to a generative engine's answer.

Below are the {args.n_examples} documents with the HIGHEST activation on this dimension:
{fmt(high_idx)}

And the {args.n_examples} documents with the LOWEST activation on this dimension:
{fmt(low_idx)}

Based on the differences between the high-activation and low-activation documents, answer in STRICT JSON:
{{
  "semantic_label": "<3-6 word label for what this dimension captures>",
  "explanation": "<1-2 sentences: what textual/content feature distinguishes high from low activation docs>",
  "text_features": ["<feature1>", "<feature2>", ...]
}}
text_features should be short tags like "evidence_density", "query_alignment", "authoritative_tone", "direct_answer", "structure_clarity", etc.
Output ONLY the JSON."""

        try:
            raw = call_llm(client, prompt)

            import re
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {"raw": raw}
        except Exception as e:
            parsed = {"error": str(e)}
            print(f"  dim {dim} error: {e}", flush=True)

        entry = {
            "dim": int(dim),
            "abs_weight": float(abs(w[dim])),
            "signed_weight": float(w[dim]),
            "llm": parsed,
            "high_activation_examples": [str(sid_arr[i]) for i in high_idx],
            "low_activation_examples": [str(sid_arr[i]) for i in low_idx],
        }
        dim_dict["dimensions"][str(int(dim))] = entry
        label = parsed.get("semantic_label", "?") if isinstance(parsed, dict) else "?"
        print(f"  [{di+1}/{len(topk)}] dim {dim} (|w|={abs(w[dim]):.2f}): {label}", flush=True)
        time.sleep(0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dim_dict, f, ensure_ascii=False, indent=2)
    print(f"\n维度字典已保存: {args.output}")
    print(f"共 {len(dim_dict['dimensions'])} 个维度")


if __name__ == "__main__":
    main()
