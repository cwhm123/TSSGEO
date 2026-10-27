from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import torch


@dataclass(frozen=True)
class Sample:
    id: str
    label: str
    query: str
    answer: str
    doc: Optional[str] = None


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_samples(path: Path, max_samples: int = 0) -> list[Sample]:
    out: list[Sample] = []
    for i, ex in enumerate(iter_jsonl(path), start=1):
        if max_samples and len(out) >= max_samples:
            break

        doc = None
        if ex.get("doc") is not None:
            doc = str(ex["doc"])
        elif ex.get("text") is not None:
            doc = str(ex["text"])
        elif ex.get("summary") is not None:
            doc = str(ex["summary"])
        out.append(
            Sample(
                id=str(ex.get("id", f"ex-{i:06d}")),
                label=str(ex.get("label", "unknown")),
                query=str(ex["query"]),
                doc=doc,
                answer=str(ex["answer"]),
            )
        )
    return out


def save_grouped_activations(output_dir: Path, grouped: Dict[str, Dict[str, torch.Tensor]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, acts in grouped.items():
        torch.save(acts, output_dir / f"{label}.pt")


def load_grouped_activations(acts_path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(acts_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {acts_path}, got {type(data)}")
    return data


def save_tensor(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, path)


def load_tensor(path: Path) -> torch.Tensor:
    x = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected tensor in {path}, got {type(x)}")
    return x
