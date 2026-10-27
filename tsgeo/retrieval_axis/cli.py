from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from .activation_extraction import RepresentationExtractor
from .discrimination_axis import project_batch_onto_axis
from .data_io import load_samples, load_grouped_activations, load_tensor, save_tensor
from .model import ProbingModel


def _build_conversation(query: str, doc: str | None, answer: str, with_answer: bool = True) -> List[Dict[str, str]]:
    if doc:
        user = (
            f"Write an accurate and concise answer for the given user question, "
            f"using only the provided source document. Every sentence in the answer "
            f"should be supported by the source.\n\n"
            f"Question: {query}\n\nSource document:\n{doc}"
        )
    else:
        user = f"Question: {query}"

    conv = [{"role": "user", "content": user}]
    if with_answer and answer:
        conv.append({"role": "assistant", "content": answer})
    return conv

def _parse_layers(spec: str, n_layers: int) -> List[int]:
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return list(range(n_layers))
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        li = int(p)
        if li < 0 or li >= n_layers:
            raise ValueError(f"layer index out of range: {li} (n_layers={n_layers})")
        out.append(li)
    if not out:
        raise ValueError("layers spec produced empty list")
    return out


def cmd_extract_representations(args: argparse.Namespace) -> int:
    samples = load_samples(Path(args.input), max_samples=args.max_samples)
    pm = ProbingModel(
        args.model,
        dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
        trust_remote_code=bool(args.trust_remote_code),
    )
    extractor = RepresentationExtractor(pm)

    layers = _parse_layers(args.layers, n_layers=len(pm.get_layers()))

    with_answer = bool(getattr(args, "with_answer", True))
    effective_pool_role = args.pool_role if with_answer else "user"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Dict[str, torch.Tensor]] = {}

    existing: Dict[str, Dict[str, torch.Tensor]] = {}
    if args.resume:
        for p in out_dir.glob("*.pt"):
            try:
                existing[p.stem] = load_grouped_activations(p)
            except Exception:
                continue

    def already_done(label: str, ex_id: str) -> bool:
        return label in existing and ex_id in existing[label]

    batch_size = max(1, int(args.batch_size))
    max_len = int(args.max_length)

    buf_convs = []
    buf_meta = []
    for s in samples:
        if args.resume and already_done(s.label, s.id):
            continue
        conv = _build_conversation(s.query, s.doc, s.answer, with_answer=with_answer)
        buf_convs.append(conv)
        buf_meta.append((s.label, s.id))

        if len(buf_convs) >= batch_size:
            acts, _ = extractor.batch_sequences(buf_convs, layers=layers, max_length=max_len)

            for bi in range(acts.shape[1]):
                label, ex_id = buf_meta[bi]
                mean_act = extractor.mean_activations_for_role_from_acts(
                    conversation=buf_convs[bi],
                    acts_layers_seq_hidden=acts[:, bi, :, :],
                    target_role=effective_pool_role,
                ).cpu()
                grouped.setdefault(label, {})[ex_id] = mean_act
            del acts
            torch.cuda.empty_cache()
            buf_convs.clear()
            buf_meta.clear()

    if buf_convs:
        acts, _ = extractor.batch_sequences(buf_convs, layers=layers, max_length=max_len)
        for bi in range(acts.shape[1]):
            label, ex_id = buf_meta[bi]
            mean_act = extractor.mean_activations_for_role_from_acts(
                conversation=buf_convs[bi],
                acts_layers_seq_hidden=acts[:, bi, :, :],
                target_role=effective_pool_role,
            ).cpu()
            grouped.setdefault(label, {})[ex_id] = mean_act
        del acts
        torch.cuda.empty_cache()

    for label, m in existing.items():
        grouped.setdefault(label, {}).update(m)

    for label, acts_dict in grouped.items():
        torch.save(acts_dict, out_dir / f"{label}.pt")

    return 0


def cmd_compute_group_centroids(args: argparse.Namespace) -> int:
    acts_dir = Path(args.acts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(acts_dir.glob("*.pt")):
        acts = load_grouped_activations(p)
        if not acts:
            continue
        stacked = torch.stack([v.float() for v in acts.values()], dim=0)
        mean_vec = stacked.mean(dim=0).to(torch.float32)
        save_tensor(out_dir / f"{p.stem}.pt", mean_vec)
    return 0


def cmd_compute_discrimination_axis(args: argparse.Namespace) -> int:
    a = load_tensor(Path(args.a))
    b = load_tensor(Path(args.b))
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: a={tuple(a.shape)} b={tuple(b.shape)}")
    axis = a - b
    save_tensor(Path(args.output), axis)
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    acts_dir = Path(args.acts_dir)
    axis = load_tensor(Path(args.axis))
    layer = int(args.layer)

    out = {}
    for p in sorted(acts_dir.glob("*.pt")):
        label = p.stem
        acts = load_grouped_activations(p)
        if not acts:
            continue
        ids = list(acts.keys())
        stacked = torch.stack([acts[i].float() for i in ids], dim=0)
        proj = project_batch_onto_axis(stacked, axis, layer=layer, normalize=not args.no_normalize).detach().cpu()
        out[label] = {
            "count": len(ids),
            "mean": float(proj.mean().item()),
            "std": float(proj.std(unbiased=False).item()) if len(ids) > 1 else 0.0,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrieval_axis")
    sp = p.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("extract-representations")
    p1.add_argument("--model", required=True)
    p1.add_argument("--input", required=True)
    p1.add_argument("--output-dir", required=True)
    p1.add_argument("--pool-role", default="assistant", choices=["assistant", "user"])
    p1.add_argument("--max-samples", type=int, default=0)
    p1.add_argument("--batch-size", type=int, default=4)
    p1.add_argument("--max-length", type=int, default=2048)
    p1.add_argument("--layers", type=str, default="all", help='Layers to extract: "all" or comma-separated, e.g. "20" or "10,20,30"')
    p1.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p1.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers")
    p1.add_argument("--resume", action="store_true", help="Skip already-completed samples if output exists")
    p1.add_argument("--with-answer", action="store_true", help="方案B: 对话含 assistant(answer), pool assistant; 不加则方案A: 仅 user, pool user")
    p1.set_defaults(func=cmd_extract_representations)

    p2 = sp.add_parser("compute-group-centroids")
    p2.add_argument("--acts-dir", required=True)
    p2.add_argument("--output-dir", required=True)
    p2.set_defaults(func=cmd_compute_group_centroids)

    p3 = sp.add_parser("compute-discrimination-axis")
    p3.add_argument("--a", required=True, help="A centroid vector (.pt)")
    p3.add_argument("--b", required=True, help="B centroid vector (.pt)")
    p3.add_argument("--output", required=True)
    p3.set_defaults(func=cmd_compute_discrimination_axis)

    p4 = sp.add_parser("project")
    p4.add_argument("--acts-dir", required=True)
    p4.add_argument("--axis", required=True)
    p4.add_argument("--layer", type=int, required=True)
    p4.add_argument("--output", required=True)
    p4.add_argument("--no-normalize", action="store_true")
    p4.set_defaults(func=cmd_project)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
