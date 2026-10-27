import argparse
import json
import os
import re
import math
import glob

import nltk
for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
    try:
        nltk.data.find(res)
    except LookupError:
        try:
            nltk.download(res.split("/")[-1], quiet=True)
        except Exception:
            pass


def num_words(toks):
    return len([x for x in toks if len(x) > 2])


def extract_cites(text):
    out = []
    for p in re.split(r"\n\n", text):
        for s in nltk.sent_tokenize(p):
            cits = [int(re.findall(r"\d+", c)[0])
                    for c in re.findall(r"\[[^\w\s]*\d+[^\w\s]*\]", s)]
            out.append((nltk.word_tokenize(s), s, cits))
    return out


# per-query contribution: word count of each citing sentence, decayed by
# position, split across its citations
def contribution(answer, n_docs):
    sents = extract_cites(answer)
    scores = [0.0] * n_docs
    L = len(sents)
    for i, (toks, s, cits) in enumerate(sents):
        for c in cits:
            if 1 <= c <= n_docs:
                sc = num_words(toks)
                sc *= math.exp(-1.0 * i / (L - 1)) if L > 1 else 1.0
                sc /= len(cits)
                scores[c - 1] += sc
    tot = sum(scores)
    return [x / tot for x in scores] if tot > 0 else [0.0] * n_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--queries-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    qtext = {}
    for p in glob.glob(os.path.join(args.queries_dir, "*.json")):
        d = json.load(open(p))
        qtext[d["query_id"]] = [doc.get("text", "") or "" for doc in d["documents"]]

    n_q = n_doc = 0
    with open(args.labels) as fin, open(args.output, "w") as fout:
        for line in fin:
            r = json.loads(line)
            qid = r["query_id"]
            if qid not in qtext:
                continue
            texts = qtext[qid]
            n_docs = r["n_docs"]
            scores = contribution(r["generated_answer"], n_docs)
            for idx in range(n_docs):
                sid = f"{qid}_doc_{idx}"
                fout.write(json.dumps({
                    "id": sid,
                    "label": "all",
                    "query": r["query"],
                    "doc": texts[idx] if idx < len(texts) else "",
                    "answer": "",
                    "score": scores[idx],
                }, ensure_ascii=False) + "\n")
                n_doc += 1
            n_q += 1
    print(f"Done. {n_q} queries -> {n_doc} samples")
    print(f"  -> {args.output}")

    import numpy as np
    sc = []
    with open(args.output) as f:
        for l in f:
            sc.append(json.loads(l)["score"])
    sc = np.array(sc)
    print(f"  贡献分数: ==0 的 {(sc==0).mean()*100:.1f}%, >0 的 {(sc>0).mean()*100:.1f}%, "
          f"范围 [{sc.min():.3f}, {sc.max():.3f}]")


if __name__ == "__main__":
    main()
