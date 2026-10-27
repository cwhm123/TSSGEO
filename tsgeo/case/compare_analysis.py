import argparse
import json
import os

import numpy as np
from scipy import stats

from tsgeo.config import CASECMP_DIR

CASECMP = CASECMP_DIR


def load_rewrites(path):
    recs = {}
    for l in open(path):
        r = json.loads(l)
        recs[r["id"]] = r
    return recs


def load_eval(path):
    ev = {}
    eval_path = path.rsplit(".", 1)[0] + "_eval.jsonl"
    if os.path.exists(eval_path):
        for l in open(eval_path):
            d = json.loads(l)
            ev[d["id"]] = d
    return ev


def cost_stats(recs):
    rows = [r["cost"] for r in recs if "cost" in r]
    ok = [r for r in rows if r.get("llm_calls", 0) > 0]
    def m(f):
        return float(np.mean([f(r) for r in ok])) if ok else 0.0
    return {
        "n": len(rows),
        "llm_calls_mean": m(lambda r: r["llm_calls"]),
        "in_tokens_mean": m(lambda r: r["in_tokens"]),
        "out_tokens_mean": m(lambda r: r["out_tokens"]),
        "llm_latency_s_mean": m(lambda r: r.get("llm_latency_s", 0)),
        "total_s_mean": m(lambda r: r.get("total_s", 0)),
        "retries_total": int(sum(r.get("retries", 0) if "retries" in r
                                 else sum(t.get("retries", 0) for t in r.get("trace", [{}]))
                                 for r in rows)),
        "local_ms_mean": m(lambda r: sum(r.get("local_ms", {}).values())
                           if r.get("local_ms") else r.get("diag_local_ms", 0)),
        "n_error": sum(1 for r in recs if str(r.get("rewritten", "")).startswith("[ERROR")),
    }


def effect_stats(recs, ev):
    ids = [i for i in recs if i in ev and not str(recs[i]["rewritten"]).startswith("[ERROR")]
    if not ids:
        return None
    out = {"n": len(ids)}
    for k in ("word", "pos", "overall"):
        b = np.array([ev[i]["before"][k] for i in ids])
        a = np.array([ev[i]["after"][k] for i in ids])
        out[f"{k}_before"], out[f"{k}_after"] = float(b.mean()), float(a.mean())
        out[f"{k}_margin"] = float((a - b).mean())
    out["cite_before"] = float(np.mean([ev[i]["before"]["cited"] for i in ids]) * 100)
    out["cite_after"] = float(np.mean([ev[i]["after"]["cited"] for i in ids]) * 100)
    out["ids"] = ids
    return out


def paired(c1, c2, ev1, ev2, key="overall_after"):
    ids = [i for i in c1["ids"] if i in c2["ids"]]
    if not ids:
        return None
    d = np.array([ev1[i]["after"]["overall"] - ev2[i]["after"]["overall"] for i in ids])
    rng = np.random.RandomState(0)
    boots = [float(np.mean(d[rng.randint(0, len(d), len(d))])) for _ in range(20000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        stat, p = stats.wilcoxon(d)
    except Exception:
        stat, p = float("nan"), float("nan")
    return {"n_pairs": len(ids), "mean_diff_c1_minus_c2": float(d.mean()),
            "ci95": [float(lo), float(hi)], "wilcoxon_p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", required=True,
                    help="文件名主干列表, 如 cond4_llama llama_test_sim_k3")
    ap.add_argument("--dir", default=CASECMP)
    args = ap.parse_args()

    results = {}
    for c in args.conds:
        rw_path = os.path.join(args.dir, f"rewrites_{c}.jsonl")
        recs = load_rewrites(rw_path)
        cs = cost_stats(list(recs.values()))
        es = effect_stats(recs, load_eval(rw_path))
        results[c] = {"cost": cs, "effect": ({k: v for k, v in es.items() if k != "ids"} if es else None)}
        if es:
            results[c]["ids"] = es["ids"]

    base = args.conds[0]
    if results.get(base, {}).get("ids"):
        ev_base = load_eval(os.path.join(args.dir, f"rewrites_{base}.jsonl"))
        for c in args.conds[1:]:
            if results.get(c, {}).get("ids"):
                ev_c = load_eval(os.path.join(args.dir, f"rewrites_{c}.jsonl"))
                results[f"paired_{base}_vs_{c}"] = paired(
                    results[base], results[c], ev_base, ev_c)

    json.dump(results, open(os.path.join(args.dir, "results.json"), "w"),
              indent=2, ensure_ascii=False)

    lines = ["# 案例复用实验结果 (自动生成)", ""]
    lines.append("## 效果 (GLM 评估, before/after Word/Pos/Overall/Cite)")
    lines.append("| 条件 | n | Word | Pos | Overall | margin | Cite% |")
    lines.append("|---|---|---|---|---|---|---|")
    for c, r in results.items():
        if not c.startswith("paired") and r.get("effect"):
            e = r["effect"]
            lines.append(f"| {c} | {e['n']} | {e['word_after']:.2f} | {e['pos_after']:.2f} "
                         f"| {e['overall_after']:.2f} | {e['overall_margin']:+.2f} "
                         f"| {e['cite_before']:.0f}→{e['cite_after']:.0f} |")
    lines.append("")
    lines.append("## 在线成本 (每文档均值)")
    lines.append("| 条件 | n | LLM调用 | in tok | out tok | LLM延迟s | 总耗时s | 本地ms | 重试 | ERROR |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c, r in results.items():
        if not c.startswith("paired") and r.get("cost"):
            k = r["cost"]
            lines.append(f"| {c} | {k['n']} | {k['llm_calls_mean']:.2f} | {k['in_tokens_mean']:.0f} "
                         f"| {k['out_tokens_mean']:.0f} | {k['llm_latency_s_mean']:.1f} "
                         f"| {k['total_s_mean']:.1f} | {k['local_ms_mean']:.0f} "
                         f"| {k['retries_total']} | {k['n_error']} |")
    lines.append("")
    lines.append("## 配对比较 (overall after, c1 − c2)")
    for c, r in results.items():
        if c.startswith("paired") and r:
            lo, hi = r["ci95"]
            lines.append(f"- **{c}**: n={r['n_pairs']}, "
                         f"Δ={r['mean_diff_c1_minus_c2']:+.2f} "
                         f"(95% CI [{lo:+.2f}, {hi:+.2f}]), "
                         f"Wilcoxon p={r['wilcoxon_p']:.4f}")
    txt = "\n".join(lines)
    open(os.path.join(args.dir, "RESULTS.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
