import argparse
import json
import os

import torch

from tsgeo.retrieval_axis.model import ProbingModel
from tsgeo.retrieval_axis.activation_extraction import RepresentationExtractor, mean_pool_role_spans, Span
from tsgeo.retrieval_axis.cli import _build_conversation
from tsgeo.bridge.encoder import RobustSpanEncoder
from tsgeo.config import CONSTRUCTION_SAMPLES

SAMPLES = CONSTRUCTION_SAMPLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", default=SAMPLES)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()

    samples = []
    with open(args.samples) as f:
        for line in f:
            samples.append(json.loads(line))

    done = {}
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        done = torch.load(args.output, map_location="cpu", weights_only=False)
        print(f"[resume] {len(done)} samples already done", flush=True)
    else:

        if os.path.exists(args.output):
            os.remove(args.output)
            print("[resume] removed stale empty all.pt", flush=True)

    todo = [r for r in samples if r["id"] not in done]
    print(f"[plan] total={len(samples)} todo={len(todo)}", flush=True)
    if not todo:
        print("[done] nothing to do")
        return

    pm = ProbingModel(args.model, dtype=torch.bfloat16, device=f"cuda:{args.gpu}", trust_remote_code=True)
    extractor = RepresentationExtractor(pm, encoder=RobustSpanEncoder(pm.tokenizer, pm.model_name))
    n_layers = len(pm.get_layers())
    print(f"[model] {args.model}: {n_layers} layers, hidden={pm.model.config.hidden_size}", flush=True)

    n_done = 0
    for i, r in enumerate(todo):
        conv = _build_conversation(r["query"], r.get("doc"), r.get("answer") or "", with_answer=True)

        acts, _ = extractor.batch_sequences([conv], layers=None, max_length=args.max_length)
        full_ids, spans = extractor.encoder.build_turn_spans(conv)
        # protocol: one (query, document) turn; mean-pool each layer over the
        # user span
        role_spans = [Span(s["start"], s["end"]) for s in spans if s["role"] == "user"]
        if not role_spans:
            print(f"[warn] no user span for {r['id']}, saving zeros", flush=True)
            mean_act = torch.zeros((n_layers, pm.model.config.hidden_size), dtype=torch.bfloat16)
        else:
            seq = acts[:, 0, :, :]
            pooled = []
            for li in range(seq.shape[0]):
                pooled.append(mean_pool_role_spans(seq[li], role_spans))
            mean_act = torch.stack(pooled, dim=0).cpu()
        done[r["id"]] = mean_act
        n_done += 1
        del acts
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % args.checkpoint_every == 0:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(done, args.output)
            print(f"[ckpt] {i+1}/{len(todo)} saved ({n_done} new)", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(done, args.output)
    k = next(iter(done))
    print(f"[done] {len(done)} samples, e.g. {k}: {tuple(done[k].shape)} {done[k].dtype}")


if __name__ == "__main__":
    main()
