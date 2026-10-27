import argparse, json, os, re, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import nltk
from anthropic import Anthropic

from tsgeo.config import RESEARCHY, TEST_LABELS, TEST_QUERIES
for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
    try: nltk.data.find(res)
    except LookupError:
        try: nltk.download(res.split("/")[-1], quiet=True)
        except: pass


EVAL_ENGINES = {
    "glm":    {"api_key": os.environ.get("GLM_API_KEY", ""),
               "api_base": "https://open.bigmodel.cn/api/anthropic",
               "model":    "glm-5"},
    "gpt":    {"api_key": os.environ.get("GPT_API_KEY", ""),
               "api_base": os.environ.get("GPT_API_BASE", "https://api.sbbbbbbbbb.xyz"),
               "model":    "gpt-5.4"},
    "claude": {"api_key": os.environ.get("CLAUDE_API_KEY", ""),
               "api_base": os.environ.get("CLAUDE_API_BASE", "https://api.sbbbbbbbbb.xyz"),
               "model":    "claude-opus-4-6"},
}

REWRITES = f"{RESEARCHY}/rewrites_test.jsonl"
MAX_CHARS = 15000
GEO_PROMPT = """Write an accurate and concise answer for the given user question, using only the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Francais, Espanol, Deutsch, or should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be immediately followed by an in-line citation to the search result(s). The cited search result(s) should fully support all the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You should select and cite AT MOST the 5 most relevant and informative sources from the provided search results, and ignore the rest. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}"""


def _call(client, prompt, max_tokens=4096):
    last = None
    for a in range(8):
        try:
            r = client.messages.create(model=API_MODEL, max_tokens=max_tokens, temperature=0.7,
                                       messages=[{"role": "user", "content": prompt}])
            u = r.usage
            text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            if not text:
                text = r.content[0].text
            return text, (u.input_tokens if u else 0), (u.output_tokens if u else 0)
        except Exception as e:
            last = e
            msg = str(e)
            transient = ("429" in msg or "529" in msg or "overloaded" in msg.lower() or "rate" in msg.lower())
            if not transient:
                return None, 0, 0
            wait = min(10 * (2 ** a), 120)
            print(f"    err({a+1}/8): {msg[:60]}, 等 {wait}s", flush=True); time.sleep(wait)
    return None, 0, 0


def sent_cites(text):
    out = []
    for p in re.split(r"\n\n", text):
        for s in nltk.sent_tokenize(p):
            cits = [int(re.findall(r"\d+", c)[0]) for c in re.findall(r"\[[^\w\s]*\d+[^\w\s]*\]", s)]
            out.append((nltk.word_tokenize(s), cits))
    return out

def geo_metrics(answer, n_docs, target_idx):
    sents = sent_cites(answer)
    L = len(sents)
    word_scores = [0.0]*n_docs
    pos_scores = [0.0]*n_docs
    overall_scores = [0.0]*n_docs
    for i, (toks, cits) in enumerate(sents):
        wc = len([x for x in toks if len(x) > 2])
        decay = math.exp(-1.0 * i / (L - 1)) if L > 1 else 1.0
        for c in cits:
            if 1 <= c <= n_docs:
                share = wc / len(cits)
                word_scores[c-1] += share
                pos_scores[c-1] += 1.0 / len(cits) * decay
                overall_scores[c-1] += share * decay

    def norm(xs):
        t = sum(xs)
        return [x/t*100 for x in xs] if t > 0 else [0]*len(xs)
    w = norm(word_scores); p = norm(pos_scores); o = norm(overall_scores)
    ti = target_idx
    cited = any((ti+1) in cits for _, cits in sents)
    return {"word": w[ti], "pos": p[ti], "overall": o[ti], "cited": cited}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--rewrites", default=REWRITES)
    ap.add_argument("--test-labels", default=TEST_LABELS)
    ap.add_argument("--test-queries", default=TEST_QUERIES)
    ap.add_argument("--engine", choices=list(EVAL_ENGINES), default="glm",
                    help="评估引擎: glm / gpt / claude. 同时控制改写 + 判分引擎. "
                         "不传 --model/--api-base/--api-key 时按此选默认配置.")
    ap.add_argument("--model", default=None, help="覆盖 model name (默认按 --engine 自动选 glm-5/gpt-5.4/claude-opus-4-6)")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--gen-before", action="store_true",
                    help="用评估模型重新生成 before (跨引擎评估时必用)")
    ap.add_argument("--workers", type=int, default=1, help="并发评估线程数")
    ap.add_argument("--resume", action="store_true",
                    help="续跑: 若 eval 文件已存在, 跳过已写 id, 追加写其余(崩了不丢已完成)")
    args = ap.parse_args()
    if args.model is None:
        args.model = EVAL_ENGINES[args.engine]["model"]
    if args.api_base is None:
        args.api_base = EVAL_ENGINES[args.engine]["api_base"]
    if args.api_key is None:
        args.api_key = EVAL_ENGINES[args.engine]["api_key"]
    globals().update(API_MODEL=args.model, API_BASE=args.api_base, API_KEY=args.api_key)
    if not args.api_key:
        env_hint = {"glm": "GLM_API_KEY", "gpt": "GPT_API_KEY",
                    "claude": "CLAUDE_API_KEY"}[args.engine]
        raise SystemExit(f"ERROR: no API key for engine '{args.engine}'. "
                         f"export {env_hint}=<key> or pass --api-key.")
    recs = [json.loads(l) for l in open(args.rewrites)]
    recs = [r for r in recs if not r["rewritten"].startswith("[ERROR")][:args.n]
    orig_ans = {}
    with open(args.test_labels) as f:
        for l in f:
            r = json.loads(l); orig_ans[r["query_id"]] = r["generated_answer"]
    def process_one(i, r):
        qid, idx, query = r["qid"], r["doc_idx"], r["query"]
        client = Anthropic(api_key=API_KEY, base_url=API_BASE)
        qd = json.load(open(os.path.join(args.test_queries, f"{qid}.json")))
        docs = [d.get("text", "") or "" for d in qd["documents"]]
        n_docs = len(docs)

        if args.gen_before:
            sources_orig = "\n\n".join([f"[{j+1}] {d[:MAX_CHARS]}" for j, d in enumerate(docs)])
            oa, bi, bo = _call(client, GEO_PROMPT.format(query=query, source_text=sources_orig))
        else:
            oa, bi, bo = orig_ans.get(qid, ""), 0, 0
        bm = geo_metrics(oa, n_docs, idx) if oa else {"word": 0, "pos": 0, "overall": 0, "cited": False}

        docs_rw = docs.copy(); docs_rw[idx] = r["rewritten"]
        sources = "\n\n".join([f"[{j+1}] {d[:MAX_CHARS]}" for j, d in enumerate(docs_rw)])
        ans, ei, eo = _call(client, GEO_PROMPT.format(query=query, source_text=sources))
        if ans is None:
            return i, None
        am = geo_metrics(ans, n_docs, idx)
        return i, {"id": r["id"], "qid": qid, "doc_idx": idx,
                   "before": bm, "after": am, "eval_tokens": [ei, eo], "_btok": (bi, bo)}

    eval_path = args.rewrites.rsplit(".", 1)[0] + "_eval.jsonl"

    done_ids = set()
    results = []
    if args.resume and os.path.exists(eval_path):
        with open(eval_path) as f:
            for l in f:
                try:
                    d = json.loads(l); results.append(d); done_ids.add(d["id"])
                except Exception: pass
        print(f"[resume] 已完成 {len(done_ids)}/{len(recs)}, 跳过这些, 追加写", flush=True)
    todo = [(i, r) for i, r in enumerate(recs) if r["id"] not in done_ids]
    tot_in = tot_out = tot_calls = 0
    t0 = time.time(); done_cnt = len(done_ids)
    fout = open(eval_path, "a")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, i, r): (i, r) for i, r in todo}
            for fut in as_completed(futs):
                i, rec = fut.result()
                if rec is None:
                    print(f"  [{done_cnt}/{len(recs)}] {recs[i]['id']} API失败跳过", flush=True)
                    continue
                bi, bo = rec.pop("_btok")
                ei, eo = rec["eval_tokens"]
                tot_in += bi + ei; tot_out += bo + eo
                tot_calls += 1 + (1 if args.gen_before else 0)
                results.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                done_cnt += 1
                bm, am = rec["before"], rec["after"]
                print(f"[{done_cnt}/{len(recs)}] {rec['id']} cited {bm['cited']}->{am['cited']} "
                      f"overall {bm['overall']:.2f}->{am['overall']:.2f} tok(in/out)={tot_in}/{tot_out}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        fout.close()
    before = [r["before"] for r in results]
    after = [r["after"] for r in results]
    print(f"\nper-doc -> {eval_path} (累计 {len(results)} 条)")

    def summ(rs, key):
        return np.mean([r[key] for r in rs]) if rs else 0
    print("\n" + "="*60)
    print(f"AutoGEO 指标 (n={len(after)}, 改写前 vs 改写后)")
    print("="*60)
    print(f"{'指标':<12} {'Before':>10} {'After':>10} {'相对提升':>10}")
    for k, name in [("word","Word Count"), ("pos","Position"), ("overall","Overall(PA-WC)")]:
        b, a = summ(before, k), summ(after, k)
        imp = (a-b)/b*100 if b > 0 else float('inf')
        print(f"{name:<12} {b:>10.2f} {a:>10.2f} {imp:>+9.1f}%")

    cb = np.mean([r["cited"] for r in before])*100
    ca = np.mean([r["cited"] for r in after])*100
    print(f"{'Cite Rate':<12} {cb:>9.1f}% {ca:>9.1f}% {(ca-cb):>+9.1f}pp")

    uncited = [i for i,(b,a) in enumerate(zip(before,after)) if not b["cited"]]
    became = sum(1 for i in uncited if after[i]["cited"])
    print(f"\n原未引用 {len(uncited)} 个 -> 改写后变引用 {became} ({became/max(len(uncited),1)*100:.0f}%)")

    metrics_path = args.rewrites.rsplit(".", 1)[0] + "_evalmetrics.json"
    out = {"manifest": {"model": API_MODEL, "rewrites": args.rewrites,
                        "test_labels": args.test_labels, "test_queries": args.test_queries,
                        "n": len(after), "max_chars": MAX_CHARS},
           "word_before": summ(before, "word"), "word_after": summ(after, "word"),
           "pos_before": summ(before, "pos"), "pos_after": summ(after, "pos"),
           "overall_before": summ(before, "overall"), "overall_after": summ(after, "overall"),
           "cite_before": cb, "cite_after": ca,
           "uncited_became_cited": f"{became}/{len(uncited)}",
           "eval_tokens": {"in": tot_in, "out": tot_out, "calls": tot_calls}}
    json.dump(out, open(metrics_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {eval_path} (per-doc) + {metrics_path} (metrics+tokens+manifest)")
    print(f"eval token (in/out/calls) = {tot_in} / {tot_out} / {tot_calls}")


if __name__ == "__main__":
    main()
