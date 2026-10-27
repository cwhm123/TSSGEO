import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class ProbingModel:

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_memory_per_gpu: Optional[Dict[int, str]] = None,
        chat_model_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.chat_model_name = chat_model_name
        self.dtype = dtype
        self.trust_remote_code = bool(trust_remote_code)

        tokenizer_source = chat_model_name if chat_model_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {"dtype": dtype, "trust_remote_code": self.trust_remote_code}

        if max_memory_per_gpu is not None:
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory_per_gpu
        elif device is None or device == "auto":
            model_kwargs["device_map"] = "auto"
        elif isinstance(device, dict):
            model_kwargs["device_map"] = device
        elif isinstance(device, str) and device.startswith("cuda:"):
            model_kwargs["device_map"] = "auto"
            gpu_id = int(device.split(":")[-1])
            model_kwargs["max_memory"] = {gpu_id: "40GiB"}
        else:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        self._layers: Optional[nn.ModuleList] = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def get_layers(self) -> nn.ModuleList:
        if self._layers is not None:
            return self._layers

        layer_paths = [
            ("model.model.layers", lambda m: m.model.layers),
            ("model.language_model.layers", lambda m: m.language_model.layers),
            ("model.transformer.h", lambda m: m.transformer.h),
            ("model.transformer.layers", lambda m: m.transformer.layers),
            ("model.gpt_neox.layers", lambda m: m.gpt_neox.layers),
        ]

        for _, path_func in layer_paths:
            try:
                layers = path_func(self.model)
                if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                    self._layers = layers
                    return self._layers
            except AttributeError:
                continue

        raise AttributeError("无法定位 transformer layers（请检查模型结构或补充路径）。")
