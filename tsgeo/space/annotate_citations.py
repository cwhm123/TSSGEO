import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


API_KEY = os.environ.get("GLM_API_KEY", "")
API_BASE = "https://open.bigmodel.cn/api/anthropic"
API_MODEL = "glm-5"


MAX_CITED = 5


GENERATION_PROMPT = """Write an accurate and concise answer for the given user question, using only the provided summarized web search results. The answer should be correct, high-quality, and written by an expert using an unbiased and journalistic tone. The user's language of choice such as English, Francais, Espanol, Deutsch, or should be used. The answer should be informative, interesting, and engaging. The answer's logic and reasoning should be rigorous and defensible. Every sentence in the answer should be immediately followed by an in-line citation to the search result(s). The cited search result(s) should fully support all the information in the sentence. Search results need to be cited using [index]. When citing several search results, use [1][2][3] format rather than [1, 2, 3]. You should select and cite AT MOST the 5 most relevant and informative sources from the provided search results, and ignore the rest. You can use multiple search results to respond comprehensively while avoiding irrelevant search results.

Question: {query}

Search Results:
{source_text}"""


def format_documents(documents, max_chars_per_doc=3000):
    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "Untitled")
        text = doc.get("text", "")
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc] + "\n...[truncated]"
        parts.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(parts)


def parse_citations(generated_text):
    details = []
    for m in re.finditer(r"\[(\d+)\]", generated_text):
        idx = int(m.group(1)) - 1
        details.append((idx, m.start()))
    cited_indices = {d[0] for d in details if d[0] >= 0}
    return cited_indices, details


def compute_visibility(cited, citation_count, first_position, answer_length):
    if not cited:
        return 0.0
    count_score = min(citation_count / 3.0, 1.0)
    pos_score = 1.0 - (first_position / max(answer_length, 1))
    return round(0.5 * count_score + 0.5 * pos_score, 4)


def select_top_cited(citation_details, max_cited=MAX_CITED):

    stats = {}
    for doc_idx, pos in citation_details:
        if doc_idx not in stats:
            stats[doc_idx] = [0, pos]
        stats[doc_idx][0] += 1
        stats[doc_idx][1] = min(stats[doc_idx][1], pos)

    if len(stats) <= max_cited:
        return set(stats.keys())

    ranked = sorted(stats.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    kept = set(idx for idx, _ in ranked[:max_cited])
    return kept


def clean_answer_citations(answer, dropped_1indexed):
    if not dropped_1indexed:
        return answer

    def replacer(m):
        n = int(m.group(1))
        return "" if n in dropped_1indexed else m.group(0)

    cleaned = re.sub(r"\[(\d+)\]", replacer, answer)

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def annotate(queries_dir, output_jsonl, api_key, api_base, api_model,
             max_docs, max_chars, resume):
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, base_url=api_base)

    done = set()
    if resume and os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                done.add(obj["query_id"])
        print(f"Resuming: {len(done)} queries already done.")

    query_files = sorted(Path(queries_dir).glob("*.json"))
    out_dir = os.path.dirname(output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = len(query_files)
    processed = len(done)

    with open(output_jsonl, "a" if resume else "w") as out_f:
        for qf in query_files:
            with open(qf) as f:
                data = json.load(f)

            qid = data["query_id"]
            if qid in done:
                continue

            query = data["query"]
            docs = data["documents"][:max_docs]
            n_docs = len(docs)

            docs_text = format_documents(docs, max_chars_per_doc=max_chars)
            prompt = GENERATION_PROMPT.format(query=query, source_text=docs_text)

            messages = [{"role": "user", "content": prompt}]

            answer = None
            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=api_model,
                        max_tokens=4096,
                        temperature=0.7,
                        messages=messages,
                    )

                    answer = resp.content[0].text
                    break
                except Exception as e:
                    print(f"  [{qid}] API error (attempt {attempt+1}/3): {e}")
                    if attempt < 2:
                        time.sleep(10)
                    else:
                        print(f"  [{qid}] Failed after 3 attempts, skipping.")

            if answer is None:
                continue

            cited_indices, citation_details = parse_citations(answer)

            n_trunc = 0
            if len(cited_indices) > MAX_CITED:
                kept = select_top_cited(citation_details, MAX_CITED)
                dropped = cited_indices - kept
                n_trunc = len(dropped)
                answer = clean_answer_citations(answer, {d + 1 for d in dropped})

                cited_indices, citation_details = parse_citations(answer)

            record = {
                "query_id": qid,
                "query": query,
                "generated_answer": answer,
                "n_docs": n_docs,
                "documents": [],
            }

            for idx, doc in enumerate(docs):
                is_cited = idx in cited_indices
                cite_count = sum(1 for d_idx, _ in citation_details if d_idx == idx)
                first_pos = min(
                    (pos for d_idx, pos in citation_details if d_idx == idx),
                    default=len(answer),
                )
                vis = compute_visibility(is_cited, cite_count, first_pos, len(answer))

                record["documents"].append({
                    "document_idx": idx,
                    "rank": doc.get("rank"),
                    "url": doc.get("url", ""),
                    "title": doc.get("title", ""),
                    "domain": doc.get("domain", ""),
                    "is_cited": is_cited,
                    "citation_count": cite_count,
                    "first_citation_position": first_pos if is_cited else -1,
                    "visibility_score": vis,
                })

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1
            n_cited = sum(1 for d in record["documents"] if d["is_cited"])
            trunc_note = f" (truncated {n_trunc})" if n_trunc else ""
            print(f"  [{processed}/{total}] {qid}: cited {n_cited}/{n_docs} docs{trunc_note}")

    print(f"Done. Saved to {output_jsonl}")


def main():
    ap = argparse.ArgumentParser(description="Paper 1 - GE 引用标注 (GLM-5)")
    ap.add_argument("--documents-dir", required=True,
                    help="query JSON 文件目录")
    ap.add_argument("--output-jsonl", required=True,
                    help="输出标注文件")
    ap.add_argument("--api-key", default=API_KEY,
                    help="智谱 API Key (默认读 GLM_API_KEY 环境变量)")
    ap.add_argument("--api-base", default=API_BASE,
                    help="API base URL")
    ap.add_argument("--api-model", default=API_MODEL,
                    help="API 模型名称")
    ap.add_argument("--max-docs-per-query", type=int, default=10)
    ap.add_argument("--max-chars-per-doc", type=int, default=15000,
                    help="每篇文档截断字符数")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已标注的 query")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: 需要提供 API Key (--api-key 或设置 GLM_API_KEY 环境变量)")
        sys.exit(1)

    print(f"Paper 1 - Citation Annotation (GLM-5)")
    print(f"  input:  {args.documents_dir}")
    print(f"  output: {args.output_jsonl}")
    print(f"  model:  {args.api_model} @ {args.api_base}")

    annotate(
        queries_dir=args.documents_dir,
        output_jsonl=args.output_jsonl,
        api_key=args.api_key,
        api_base=args.api_base,
        api_model=args.api_model,
        max_docs=args.max_docs_per_query,
        max_chars=args.max_chars_per_doc,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
