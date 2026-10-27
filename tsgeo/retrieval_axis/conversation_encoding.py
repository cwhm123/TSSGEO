from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union


class ConversationEncoder:

    def __init__(self, tokenizer, model_name: Optional[str] = None):
        self.tokenizer = tokenizer
        name = model_name or getattr(tokenizer, "name_or_path", "")
        self.model_name = str(name).lower()

    def _is_qwen(self) -> bool:
        return "qwen" in self.model_name

    def _apply_template(self, conversation, **chat_kwargs) -> list:
        ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )
        if hasattr(ids, "ids"):
            return list(ids.ids)
        if hasattr(ids, "input_ids"):
            return list(ids["input_ids"])
        return list(ids)

    # token spans of each turn's content, located by the empty-vs-full prefix
    # diff of the chat template (Qwen uses its im_start/im_end markers)
    def build_turn_spans(
        self,
        conversation: List[Dict[str, str]],
        enable_thinking: bool = False,
        **chat_kwargs,
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        chat_kwargs = dict(chat_kwargs)
        if self._is_qwen():
            chat_kwargs["enable_thinking"] = enable_thinking

        full_ids = self._apply_template(conversation, **chat_kwargs)

        if self._is_qwen():
            return full_ids, self._build_turn_spans_qwen(conversation, full_ids, **chat_kwargs)

        spans: List[Dict[str, Any]] = []
        msgs_before: List[Dict[str, str]] = []
        turn_idx = 0

        for msg in conversation:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                msgs_before.append(msg)
                continue

            msgs_empty = msgs_before + [{"role": role, "content": ""}]
            msgs_full = msgs_before + [{"role": role, "content": text}]

            ids_empty = self._apply_template(msgs_empty, **chat_kwargs)
            ids_full_for_this = self._apply_template(msgs_full, **chat_kwargs)

            pref_len = self._longest_common_prefix_len(ids_full_for_this, ids_empty)
            delta = ids_full_for_this[pref_len:]
            delta = self._strip_trailing_special(delta, set(self.tokenizer.all_special_ids))

            content_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            start_in_delta = self._find_subsequence(delta, content_ids)
            if start_in_delta == -1:

                content_ids_sp = self.tokenizer(" " + text, add_special_tokens=False).input_ids
                start_in_delta = self._find_subsequence(delta, content_ids_sp)
                if start_in_delta != -1:
                    content_ids = content_ids_sp

            if start_in_delta == -1:

                start_in_delta = 0
                content_ids = delta

            abs_start = pref_len + start_in_delta
            abs_end = abs_start + len(content_ids)

            spans.append(
                {
                    "turn": turn_idx,
                    "role": role,
                    "start": abs_start,
                    "end": abs_end,
                    "n_tokens": max(0, abs_end - abs_start),
                    "text": text,
                }
            )
            msgs_before.append(msg)
            turn_idx += 1

        return full_ids, spans

    def _build_turn_spans_qwen(self, conversation, full_ids, **chat_kwargs) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []

        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))

        try:
            im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            user_token_id = self.tokenizer.convert_tokens_to_ids("user")
            assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")

            try:
                think_start_id = self.tokenizer.convert_tokens_to_ids("<think>")
                think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")
            except Exception:
                think_start_id = None
                think_end_id = None
        except Exception:

            _, spans_std = self.build_turn_spans(
                conversation, enable_thinking=enable_thinking, **{k: v for k, v in chat_kwargs.items() if k != "enable_thinking"}
            )
            return spans_std

        expected_turns = [(m["role"], m.get("content", "")) for m in conversation if m["role"] != "system"]

        turn_idx = 0
        i = 0
        while i < len(full_ids):
            if i + 1 < len(full_ids) and full_ids[i] == im_start_id:
                role_token = full_ids[i + 1]
                if role_token == user_token_id:
                    role = "user"
                elif role_token == assistant_token_id:
                    role = "assistant"
                else:
                    i += 1
                    continue

                content_start = i + 2
                content_end = None
                for j in range(content_start, len(full_ids)):
                    if full_ids[j] == im_end_id:
                        content_end = j
                        break

                if content_end is None or turn_idx >= len(expected_turns):
                    i += 1
                    continue

                expected_role, expected_text = expected_turns[turn_idx]
                if role != expected_role:
                    i = content_end + 1
                    continue

                raw_indices = list(range(content_start, content_end))

                if (
                    role == "assistant"
                    and not enable_thinking
                    and think_start_id is not None
                    and think_end_id is not None
                ):
                    filtered = []
                    skipping = False
                    for idx in raw_indices:
                        tid = full_ids[idx]
                        if tid == think_start_id:
                            skipping = True
                            continue
                        if tid == think_end_id:
                            skipping = False
                            continue
                        if skipping:
                            continue
                        filtered.append(idx)
                    final_indices = filtered
                else:
                    final_indices = raw_indices

                if final_indices:
                    spans.append(
                        {
                            "turn": turn_idx,
                            "role": role,
                            "start": min(final_indices),
                            "end": max(final_indices) + 1,
                            "n_tokens": len(final_indices),
                            "text": expected_text,
                        }
                    )
                turn_idx += 1
                i = content_end + 1
            else:
                i += 1

        return spans

    @staticmethod
    def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    @staticmethod
    def _strip_trailing_special(ids: List[int], special_ids: set) -> List[int]:
        i = len(ids)
        while i > 0 and ids[i - 1] in special_ids:
            i -= 1
        return ids[:i]

    @staticmethod
    def _find_subsequence(hay: List[int], needle: List[int]) -> int:
        if not needle or len(needle) > len(hay):
            return -1
        for i in range(len(hay) - len(needle) + 1):
            if hay[i : i + len(needle)] == needle:
                return i
        return -1
