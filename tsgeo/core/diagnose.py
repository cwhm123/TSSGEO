import json

import numpy as np

from .llm import call_llm


# per-dimension deficit vs the prototype, weighted by |w[d]|, mapped to
# dictionary labels as add/reduce feature lists.
def diagnose(doc_act, w, prototype, dim_dict, topk=8):
    dict_dims = [int(k) for k in dim_dict["dimensions"].keys()]
    deficit = []
    for d in dict_dims:
        wd = w[d]; doc_v = doc_act[d]; proto_v = prototype[d]
        if wd > 0 and doc_v < proto_v:
            impact = abs(doc_v - proto_v) * abs(wd)
            deficit.append((d, impact, "+"))
        elif wd < 0 and doc_v > proto_v:
            impact = abs(doc_v - proto_v) * abs(wd)
            deficit.append((d, impact, "-"))
    deficit.sort(key=lambda x: -x[1])
    add_feats, reduce_feats, details = [], [], []
    for d, impact, sign in deficit[:topk]:
        entry = dim_dict["dimensions"].get(str(d), {})
        llm = entry.get("llm", {}) if isinstance(entry.get("llm"), dict) else {}
        label = llm.get("semantic_label", f"dim_{d}")
        feats = llm.get("text_features", [])
        feat_str = ", ".join(feats[:3]) if feats else label
        if sign == "+":
            add_feats.append(feat_str); details.append(f"dim{d}(+): lacking [{feat_str}] -> add (impact={impact:.1f})")
        else:
            reduce_feats.append(feat_str); details.append(f"dim{d}(-): has [{feat_str}] -> remove (impact={impact:.1f})")
    return add_feats, reduce_feats, details


def diagnose_specific(doc_text, query, add_feats, reduce_feats, client,
                     model="glm-5", max_tokens=1024):
    add_str = "; ".join(add_feats) if add_feats else "(none)"
    red_str = "; ".join(reduce_feats) if reduce_feats else "(none)"
    prompt = f"""You are diagnosing a web document that a generative engine cited little or not at all.

Query the document should serve: {query}

A representation-space analysis found this document:
- LACKS these features: {add_str}
- HAS these negative features: {red_str}

Read the document carefully and identify CONCRETELY where these problems occur.

CRITICAL: For every piece of low-quality content that should be REMOVED, you MUST specify what useful content should REPLACE it. The document must not get shorter — every deletion must be compensated.

Output a numbered list of specific findings (3-6 items), each in the form:
- Location: <which part> | Problem: <what's wrong> | Remove: <what to delete> | Replace with: <what useful content to add instead>

Document:
{doc_text[:6000]}

Output ONLY the numbered list."""
    return call_llm(client, prompt, model=model, max_tokens=max_tokens)
