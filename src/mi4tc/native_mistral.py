"""A small native Tekken adapter; never re-encode decoded Mistral prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NativeMistralTokenizer:
    def __init__(self, path: str | Path):
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        except ImportError as exc:
            raise RuntimeError("Install Mistral dependencies with: pip install -e '.[mistral]'") from exc
        self.native = MistralTokenizer.from_file(str(Path(path) / "tekken.json"))
        self.raw = self.native.instruct_tokenizer.tokenizer
        self.keep = SpecialTokenPolicy.KEEP
        self.ignore = SpecialTokenPolicy.IGNORE
        self.pad_token_id = self.raw.pad_id
        self.eos_token_id = self.raw.eos_id
        self.bos_token_id = self.raw.bos_id
        self.padding_side = "left"
        data = json.loads((Path(path) / "tekken.json").read_text())
        self.special = {r.get("token_str", r.get("token")): r.get("rank", r.get("id")) for r in data["special_tokens"]}

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
        return self.raw.encode(text, bos=add_special_tokens, eos=False)

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **kwargs: Any) -> str:
        return self.raw.decode([int(i) for i in ids], special_token_policy=self.ignore if skip_special_tokens else self.keep)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token in self.special:
            return self.special[token]
        ids = self.encode(token)
        return ids[0] if len(ids) == 1 else None

    def token_offsets(self, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        parts = [self.raw.id_to_byte_piece(int(i), special_token_policy=self.keep) for i in ids]
        data = b"".join(parts)
        offsets = []
        end = 0
        for part in parts:
            start = end
            end += len(part)
            offsets.append((len(data[:start].decode("utf-8", errors="ignore")), len(data[:end].decode("utf-8", errors="ignore"))))
        return data.decode("utf-8", errors="replace"), offsets

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False,
                 return_tensors: str | None = None, **kwargs: Any) -> dict[str, Any]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = self.token_offsets(ids)[1]
        if return_tensors is not None:
            if return_tensors != "pt":
                raise ValueError("Native Mistral tensor output supports return_tensors='pt'")
            import torch
            result = {key: torch.tensor([value], dtype=torch.long) for key, value in result.items()}
        return result

    def apply_chat_template(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                            tokenize: bool = True, return_dict: bool = False, **kwargs: Any) -> Any:
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        request = ChatCompletionRequest.model_validate({"model": "local", "messages": messages, "tools": tools or None})
        ids = self.native.encode_chat_completion(request).tokens
        if not tokenize:
            return self.decode(ids)
        return {"input_ids": ids} if return_dict else ids
