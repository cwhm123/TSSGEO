from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from .model import ProbingModel
from .conversation_encoding import ConversationEncoder


def _to_id_list(ids) -> list:
    if hasattr(ids, "ids"):
        return list(ids.ids)
    if hasattr(ids, "input_ids"):
        return list(ids["input_ids"])
    return list(ids)


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def clamp(self, max_len: int) -> "Span":
        s = max(0, min(self.start, max_len))
        e = max(0, min(self.end, max_len))
        if e < s:
            e = s
        return Span(s, e)


# mean over the given token spans, then averaged across spans
def mean_pool_role_spans(
    activations: torch.Tensor,
    spans: List[Span],
    dim_token: int = -2,
) -> torch.Tensor:
    if activations.ndim < 2:
        raise ValueError("activations must have token and hidden dims.")
    seq_len = activations.shape[dim_token]
    hidden_dim = activations.shape[-1]

    if not spans:
        return torch.zeros((*activations.shape[:-2], hidden_dim), dtype=activations.dtype, device=activations.device)

    pooled = []
    for sp in spans:
        sp = sp.clamp(seq_len)
        if sp.end <= sp.start:
            continue

        if dim_token != -2:
            raise ValueError("mean_pool_role_spans currently expects token dim at -2.")
        sl = activations[sp.start:sp.end, :]
        pooled.append(sl.mean(dim=0))

    if not pooled:
        return torch.zeros((*activations.shape[:-2], hidden_dim), dtype=activations.dtype, device=activations.device)

    return torch.stack(pooled, dim=0).mean(dim=0)


class RepresentationExtractor:

    def __init__(self, probing_model: ProbingModel, encoder: Optional[ConversationEncoder] = None):
        self.pm = probing_model
        self.model = probing_model.model
        self.tokenizer = probing_model.tokenizer
        self.encoder = encoder or ConversationEncoder(self.tokenizer, probing_model.model_name)

    def batch_sequences(
        self,
        conversations: List[List[Dict[str, str]]],
        layers: Optional[List[int]] = None,
        max_length: int = 2048,
        enable_thinking: bool = False,
        **chat_kwargs,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        if layers is None:
            layers = list(range(len(self.pm.get_layers())))

        formatted_ids = []
        for conv in conversations:
            ids = self.tokenizer.apply_chat_template(
                conv,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking if "qwen" in self.pm.model_name.lower() else False,
                **chat_kwargs,
            )
            ids = _to_id_list(ids)
            if len(ids) > max_length:
                ids = ids[:max_length]
            formatted_ids.append(ids)

        batch_size = len(formatted_ids)
        if batch_size == 0:
            raise ValueError("Empty conversations.")

        seq_len = max(len(x) for x in formatted_ids)
        seq_len = min(seq_len, max_length)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=self.pm.device)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.pm.device)
        for i, ids in enumerate(formatted_ids):
            ids = ids[:seq_len]
            input_ids[i, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=self.pm.device)
            attention_mask[i, -len(ids):] = 1

        layer_outputs: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                layer_outputs[layer_idx] = act
            return hook_fn

        model_layers = self.pm.get_layers()
        for li in layers:
            handles.append(model_layers[li].register_forward_hook(make_hook(li)))

        try:
            with torch.inference_mode():
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        if not layer_outputs:
            raise RuntimeError("No layer outputs captured. Check hooks and layer indices.")

        first_li = layers[0]
        if first_li not in layer_outputs:

            first_li = next(iter(layer_outputs.keys()))
        target_device = layer_outputs[first_li].device

        acts = torch.stack(
            [layer_outputs[li].to(target_device) for li in layers],
            dim=0,
        )
        return acts, formatted_ids

    def mean_activations_for_role(
        self,
        conversation: List[Dict[str, str]],
        target_role: str = "assistant",
        layers: Optional[List[int]] = None,
        max_length: int = 2048,
        enable_thinking: bool = False,
        **chat_kwargs,
    ) -> torch.Tensor:
        acts, _ = self.batch_sequences(
            [conversation],
            layers=layers,
            max_length=max_length,
            enable_thinking=enable_thinking,
            **chat_kwargs,
        )

        acts = acts[:, 0, :, :]

        _, spans = self.encoder.build_turn_spans(conversation, enable_thinking=enable_thinking, **chat_kwargs)
        role_spans = [Span(s["start"], s["end"]) for s in spans if s["role"] == target_role]
        if not role_spans:

            return torch.zeros((acts.shape[0], acts.shape[-1]), dtype=acts.dtype)

        pooled = []
        for layer_i in range(acts.shape[0]):
            pooled.append(mean_pool_role_spans(acts[layer_i], role_spans, dim_token=-2))
        return torch.stack(pooled, dim=0)

    def mean_activations_for_role_from_acts(
        self,
        *,
        conversation: List[Dict[str, str]],
        acts_layers_seq_hidden: torch.Tensor,
        target_role: str = "assistant",
        enable_thinking: bool = False,
        **chat_kwargs,
    ) -> torch.Tensor:
        if acts_layers_seq_hidden.ndim != 3:
            raise ValueError("acts_layers_seq_hidden must be (n_layers, seq_len, hidden).")

        _, spans = self.encoder.build_turn_spans(conversation, enable_thinking=enable_thinking, **chat_kwargs)
        role_spans = [Span(s["start"], s["end"]) for s in spans if s["role"] == target_role]
        if not role_spans:
            return torch.zeros((acts_layers_seq_hidden.shape[0], acts_layers_seq_hidden.shape[-1]), dtype=acts_layers_seq_hidden.dtype)

        pooled = []
        for layer_i in range(acts_layers_seq_hidden.shape[0]):
            pooled.append(mean_pool_role_spans(acts_layers_seq_hidden[layer_i], role_spans, dim_token=-2))
        return torch.stack(pooled, dim=0)
