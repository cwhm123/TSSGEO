import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from tsgeo.config import QWEN_ACTS, CONSTRUCTION_SAMPLES, DICT_PATH, BRIDGE_DIR

SAMPLES = CONSTRUCTION_SAMPLES

TARGETS = {
    "llama": f"{BRIDGE_DIR}/acts_llama/all.pt",
    "gemma": f"{BRIDGE_DIR}/acts_gemma/all.pt",
}

QWEN_LAYER = 21
QUANTILE = 75


def _chol_solve(G, B):

    try:
        return torch.linalg.cholesky(G).cholesky_solve(B)
    except Exception:
        return torch.linalg.lstsq(G, B).solution


def ridge_solve(X, Y, lam):

    d = X.shape[1]
    G = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    return _chol_solve(G, X.T @ Y)


def source_axis(acts, meta, layer):
    by_q = defaultdict(list)
    for sid in meta:
        by_q[sid.rsplit("_doc_", 1)[0]].append(sid)
    Xlist, Slist = [], []
    for qid, ids in by_q.items():
        if len(ids) < 2:
            continue
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
    thr = np.percentile(scores, QUANTILE)
    prototype = Xall[scores >= thr].mean(0)
    return w, prototype


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


# pick (layer, lambda) by held-out R^2 on a query-level split, then refit on
# the full training side
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--lams", default="1e2,1e3,1e4,1e5")
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    qwen = torch.load(QWEN_ACTS, map_location="cpu", weights_only=False)
    tgt = torch.load(TARGETS[args.target], map_location="cpu", weights_only=False)
    common = sorted(set(qwen) & set(tgt))

    common = [s for s in common if qwen[s].float().abs().sum() > 0 and tgt[s].float().abs().sum() > 0]
    meta = {}
    with open(SAMPLES) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in qwen:
                meta[r["id"]] = r["score"]
    common = [s for s in common if s in meta]

    qs = sorted({s.rsplit("_doc_", 1)[0] for s in common})
    rng = np.random.RandomState(args.seed)
    rng.shuffle(qs)
    n_held = int(len(qs) * args.heldout_frac)
    held_qs = set(qs[:n_held])
    train_ids = [s for s in common if s.rsplit("_doc_", 1)[0] not in held_qs]
    held_ids = [s for s in common if s.rsplit("_doc_", 1)[0] in held_qs]
    print(f"[data] common={len(common)} train={len(train_ids)} held={len(held_ids)} "
          f"({len(held_qs)}/{len(qs)} queries held out)")

    Y21 = torch.stack([qwen[s][QWEN_LAYER].float() for s in train_ids])
    Y21_h = torch.stack([qwen[s][QWEN_LAYER].float() for s in held_ids])
    Y21c = Y21 - Y21.mean(0, keepdim=True)
    Y21c_h = Y21_h - Y21.mean(0, keepdim=True)
    n_tgt_layers = tgt[train_ids[0]].shape[0]

    results = {"per_layer": []}
    lams = [float(x) for x in args.lams.split(",")]
    best = None
    eye = None
    for li in range(n_tgt_layers):
        Xtr = torch.stack([tgt[s][li].float() for s in train_ids])
        mu = Xtr.mean(0, keepdim=True)
        Xtr_c = Xtr - mu

        G0 = Xtr_c.T @ Xtr_c
        B0 = Xtr_c.T @ Y21c
        if eye is None:
            eye = torch.eye(Xtr_c.shape[1], dtype=Xtr_c.dtype)

        Xhe = torch.stack([tgt[s][li].float() for s in held_ids]) - mu
        for lam in lams:
            W = _chol_solve(G0 + lam * eye, B0)
            pred = Xhe @ W
            err = (pred - Y21c_h).pow(2).sum()
            tot = Y21c_h.pow(2).sum()
            r2 = 1 - (err / tot).item()
            cos = torch.nn.functional.cosine_similarity(pred, Y21c_h, dim=1).mean().item()
            results["per_layer"].append({"layer": li, "lam": lam, "r2": r2, "cos": cos})
            if best is None or r2 > best["r2"]:
                best = {"layer": li, "lam": lam, "r2": r2, "cos": cos}
        if (li + 1) % 6 == 0:
            print(f"[scan] {li+1}/{n_tgt_layers} layers, best={best}", flush=True)

    print(f"[best] {args.target} L{best['layer']} lam={best['lam']:.0f} "
          f"R2={best['r2']:.4f} cos={best['cos']:.4f}")
    results["best"] = best

    li, lam = best["layer"], best["lam"]
    Xtr = torch.stack([tgt[s][li].float() for s in train_ids])
    mu = Xtr.mean(0, keepdim=True)
    W = ridge_solve(Xtr - mu, Y21c, lam)

    Xhe = torch.stack([tgt[s][li].float() for s in held_ids]) - mu
    recon = {s: (Xhe[i] @ W + Y21.mean(0)) for i, s in enumerate(held_ids)}
    real = {s: qwen[s][QWEN_LAYER].float() for s in held_ids}

    acts_recon = {s: recon[s].unsqueeze(0) for s in held_ids}
    acts_real = {s: real[s].unsqueeze(0) for s in held_ids}
    meta_h = {s: meta[s] for s in held_ids}
    w_recon, proto_recon = source_axis(acts_recon, meta_h, layer=0)
    w_real, _ = source_axis(acts_real, meta_h, layer=0)
    results["axis_cos"] = float(
        torch.nn.functional.cosine_similarity(torch.tensor(w_recon), torch.tensor(w_real), dim=0))

    by_q = defaultdict(list)
    for s in held_ids:
        by_q[s.rsplit("_doc_", 1)[0]].append(s)
    wn_r = torch.tensor(w_real / (np.linalg.norm(w_real) + 1e-8)).float()
    wn_c = torch.tensor(w_recon / (np.linalg.norm(w_recon) + 1e-8)).float()
    rhos_r, rhos_c = [], []
    for qid, ids in by_q.items():
        if len(ids) < 2:
            continue
        sc = np.array([meta[s] for s in ids])
        pj_real = np.array([float(real[s] @ wn_r) for s in ids])
        pj_rec = np.array([float(recon[s] @ wn_c) for s in ids])
        rhos_r.append(spearman(sc, pj_real))
        rhos_c.append(spearman(sc, pj_rec))
    results["spearman_real_axis"] = float(np.mean(rhos_r))
    results["spearman_bridge_axis"] = float(np.mean(rhos_c))

    dim_dict = json.load(open(DICT_PATH))
    dims = [int(k) for k in dim_dict["dimensions"].keys()]

    w_tr, proto_tr = source_axis({s: qwen[s][QWEN_LAYER].float().unsqueeze(0) for s in train_ids},
                                 {s: meta[s] for s in train_ids}, layer=0)
    wn_tr = w_tr / (np.linalg.norm(w_tr) + 1e-8)

    def diagnose_ids(doc_vec, w, proto, m=8):
        deficit = []
        for d in dims:
            wd, dv, pv = w[d], doc_vec[d], proto[d]
            if (wd > 0 and dv < pv) or (wd < 0 and dv > pv):
                deficit.append((d, abs(dv - pv) * abs(wd)))
        deficit.sort(key=lambda x: -x[1])
        return set(d for d, _ in deficit[:m]), deficit

    jac, jw = [], []
    for s in held_ids:
        ids_r, def_r = diagnose_ids(real[s], w_tr, proto_tr)
        ids_c, def_c = diagnose_ids(recon[s], w_tr, proto_tr)
        if ids_r or ids_c:
            jac.append(len(ids_r & ids_c) / len(ids_r | ids_c))
    results["topm_jaccard_mean"] = float(np.mean(jac)) if jac else None
    results["n_diag_docs"] = len(jac)

    out = args.out or f"{BRIDGE_DIR}/bridge_fit_{args.target}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "per_layer"}, indent=2))

    torch.save({"W": W, "mu": mu, "y_mean": Y21.mean(0), "layer": li, "lam": lam,
                "train_ids": train_ids},
               f"{BRIDGE_DIR}/bridge_W_{args.target}.pt")
    print(f"[saved] {out}, bridge_W_{args.target}.pt")


if __name__ == "__main__":
    main()
