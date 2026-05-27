"""统一加载字级或 GPT-2 BPE 分词器。"""

from __future__ import annotations

import json
import os
from typing import Union

from .gpt2_tokenizer import GPT2Tokenizer
from .train_char_tokenizer import CharTokenizer

Tokenizer = Union[CharTokenizer, GPT2Tokenizer]


def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    tok_type = "char"
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            tok_type = json.load(f).get("tokenizer_type", "char")
    if tok_type == "gpt2":
        return GPT2Tokenizer.load(tokenizer_dir)
    return CharTokenizer.load(tokenizer_dir)


def vocab_size_of(tok: Tokenizer) -> int:
    if hasattr(tok, "vocab_size"):
        return int(tok.vocab_size)
    return len(tok.token_to_id)
