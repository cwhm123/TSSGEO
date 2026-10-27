import json
import os
from collections import defaultdict

import numpy as np
import torch

LAYER = 21


# w: cross-covariance of within-query-centered activations with centered
# contribution scores; prototype = mean of top-quartile documents.
def build_axis_and_prototype(quantile=75,
                             construction_acts=None,
                             construction_samples=None,
                             layer=None):
    if layer is None: layer = LAYER
    acts = torch.load(construction_acts, map_location="cpu", weights_only=False)
    meta = {}
    with open(construction_samples) as f:
        for l in f:
            r = json.loads(l)
            if r["id"] in acts: meta[r["id"]] = r["score"]
    by_q = defaultdict(list)
    for sid in meta: by_q[sid.rsplit("_doc_", 1)[0]].append(sid)
    Xlist, Slist = [], []
    for qid, ids in by_q.items():
        if len(ids) < 2: continue
        xq = torch.stack([acts[s][layer].float() for s in ids])
        sq = np.array([meta[s] for s in ids])
        Xlist.append(xq - xq.mean(0, keepdim=True))
        Slist.append(sq - sq.mean())
    X = torch.cat(Xlist, 0).numpy()
    S = np.concatenate(Slist)
    w = (X * S[:, None]).sum(0)
    all_sids = sorted(meta.keys())
    Xall = np.stack([acts[s][layer].float().numpy() for s in all_sids])
    scores = np.array([meta[s] for s in all_sids])
    thr = np.percentile(scores, quantile)
    prototype = Xall[scores >= thr].mean(0)
    print(f"axis norm={np.linalg.norm(w):.1f}, prototype from {(scores>=thr).sum()} high-contribution docs "
          f"(q={(100-quantile)}%, quantile={quantile})", flush=True)
    return w, prototype
