"""Hugging Face GPT-2 BPE 分词器封装，接口与 CharTokenizer 对齐。"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

try:
    from transformers import GPT2Tokenizer as HFGPT2Tokenizer
except ImportError as e:
    HFGPT2Tokenizer = None  # type: ignore[misc, assignment]
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

DEFAULT_MODEL = "gpt2"


class GPT2Tokenizer:
    """
    OpenAI GPT-2 同款 byte-level BPE（vocab≈50257）。
    无独立 BOS 时，prepare 里的 add_bos 使用 <|endoftext|> 作为段首标记（与段尾 eos 相同 id）。
    """

    def __init__(
        self,
        hf_tokenizer: Any,
        tokenizer_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if HFGPT2Tokenizer is None:
            raise ImportError(
                "使用 GPT-2 分词器需要安装 transformers：pip install transformers"
            ) from _IMPORT_ERR
        self._tok = hf_tokenizer
        self.tokenizer_config: Dict[str, Any] = dict(tokenizer_config or {})

        self.pad_token = self._tok.pad_token or "<|endoftext|>"
        self.bos_token = self._tok.bos_token or "<|endoftext|>"
        self.eos_token = self._tok.eos_token or "<|endoftext|>"
        self.unk_token = getattr(self._tok, "unk_token", None) or "<|endoftext|>"

        self.pad_id = int(self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id)
        self.eos_id = int(self._tok.eos_token_id)
        self.bos_id = int(self._tok.bos_token_id if self._tok.bos_token_id is not None else self.eos_id)
        self.unk_id = int(
            self._tok.unk_token_id
            if getattr(self._tok, "unk_token_id", None) is not None
            else self.eos_id
        )

        self._special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        self.token_to_id: Dict[str, int] = dict(self._tok.get_vocab())
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return int(len(self._tok))

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode 需要 str，收到 {type(text).__name__}")
        ids: List[int] = list(self._tok.encode(text, add_special_tokens=False))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self, ids: Iterable[int], skip_special_tokens: bool = True
    ) -> str:
        id_list = [int(i) for i in ids]
        if skip_special_tokens:
            id_list = [i for i in id_list if i not in self._special_ids]
        return self._tok.decode(id_list, skip_special_tokens=False, errors="replace")

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._tok.save_pretrained(output_dir)
        cfg = {
            "tokenizer_type": "gpt2",
            "vocab_size": self.vocab_size,
            "pretrained_name": self.tokenizer_config.get("pretrained_name", DEFAULT_MODEL),
            "special_tokens": {
                "pad": self.pad_token,
                "bos": self.bos_token,
                "eos": self.eos_token,
                "unk": self.unk_token,
            },
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "note": "GPT-2 无独立 BOS；add_bos 时使用 eos_token_id 作段首界标",
        }
        cfg.update(self.tokenizer_config)
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, output_dir: str) -> "GPT2Tokenizer":
        if HFGPT2Tokenizer is None:
            raise ImportError(
                "使用 GPT-2 分词器需要安装 transformers：pip install transformers"
            ) from _IMPORT_ERR
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        extra: Dict[str, Any] = {}
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                extra = json.load(f)
        hf_tok = HFGPT2Tokenizer.from_pretrained(output_dir)
        return cls(hf_tok, tokenizer_config=extra)

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_MODEL) -> "GPT2Tokenizer":
        if HFGPT2Tokenizer is None:
            raise ImportError(
                "使用 GPT-2 分词器需要安装 transformers：pip install transformers"
            ) from _IMPORT_ERR
        hf_tok = HFGPT2Tokenizer.from_pretrained(name)
        return cls(hf_tok, tokenizer_config={"pretrained_name": name})
