from __future__ import annotations

from typing import Optional

from tsgeo.retrieval_axis.conversation_encoding import ConversationEncoder


class RobustSpanEncoder(ConversationEncoder):

    def __init__(self, tokenizer, model_name: Optional[str] = None, header_text: Optional[str] = None):
        super().__init__(tokenizer, model_name)
        if header_text is None:
            if "gemma" in self.model_name:
                header_text = "<start_of_turn>user\n"
            elif "qwen" in self.model_name:
                header_text = "<|im_start|>user\n"
            elif "mistral" in self.model_name:
                header_text = "[INST]"
            else:
                header_text = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.header_text = header_text

    # locate the user span by scanning for the model-specific user turn header
    # instead of relying on a chat template
    def build_turn_spans(self, conversation, enable_thinking: bool = False, **chat_kwargs):
        full_ids = self._apply_template(conversation, **chat_kwargs)
        header_ids = self.tokenizer(self.header_text, add_special_tokens=False).input_ids

        pos = -1
        for i in range(len(full_ids) - len(header_ids) + 1):
            if full_ids[i : i + len(header_ids)] == header_ids:
                pos = i
                break
        if pos == -1:
            return full_ids, []

        start = pos + len(header_ids)
        end = len(full_ids)
        special = set(self.tokenizer.all_special_ids)
        while end > start and full_ids[end - 1] in special:
            end -= 1

        user_text = next((m.get("content", "") for m in conversation if m.get("role") == "user"), "")
        return full_ids, [
            {"turn": 0, "role": "user", "start": start, "end": end,
             "n_tokens": max(0, end - start), "text": user_text}
        ]
