import time


# up to 8 attempts: transient API errors back off exponentially, others fail fast.
def call_llm(client, prompt, model="glm-5", max_tokens=4096, temperature=0.4):
    last_err = None
    for attempt in range(8):
        try:
            resp = client.messages.create(model=model, max_tokens=max_tokens,
                                          temperature=temperature,
                                          messages=[{"role": "user", "content": prompt}])
            u = resp.usage
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            if not text:
                text = resp.content[0].text
            return text, (u.input_tokens if u else 0), (u.output_tokens if u else 0)
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = ("429" in msg or "529" in msg or "overloaded" in msg.lower() or "rate" in msg.lower())
            if not transient:
                return f"[ERROR: {last_err}]", 0, 0
            wait = min(10 * (2 ** attempt), 120)
            print(f"    API error (attempt {attempt+1}/8): {msg[:80]}, wait {wait}s", flush=True)
            time.sleep(wait)
    return f"[ERROR: {last_err}]", 0, 0


def rewrite(doc_text, query, specific_diagnosis, client,
           model="glm-5", max_tokens=4096, edit_budget=500):
    prompt = f"""You are a document optimization expert. Revise the document based on the diagnosis below, under strict fact constraints.

Query the document should serve: {query}

Specific diagnosis:
{specific_diagnosis}

Editing rules:
1. FACTS ARE FIXED. Keep every fact, name, number, date, and claim of the original document as given. Do not add facts, statistics, examples, or sources that do not appear in the original.
2. EDIT ONLY DIAGNOSED SPANS. Modify only the parts flagged by the diagnosis; leave every other passage unchanged.
3. FILL GAPS FROM THE DOCUMENT ITSELF. When a diagnosis item asks you to remove a passage, fill the gap by reorganizing or rephrasing material that already exists elsewhere in the document, not by writing new factual content.
4. IMPROVE FORM, NOT CONTENT. You may restructure the diagnosed spans, add headings or transitions, and make them directly responsive to the query; you may not change what the document claims.
5. KEEP LENGTH STABLE. The revised document should be about the same length as the original (~{len(doc_text)} chars); do not expand it into a new article and do not shrink it.

If a diagnosis item cannot be satisfied without inventing new factual content, satisfy it structurally (ordering, phrasing, emphasis) instead.

Output MUST be in English.

Original document:
{doc_text[:6000]}

Output the revised document directly, no explanations."""
    return call_llm(client, prompt, model=model, max_tokens=max_tokens)
