#!/usr/bin/env python3
"""
从 train.jsonl 构建字符级词表，并保存 vocab.json / tokenizer_config.json。
仅依赖 Python 标准库与 tqdm。
结果：单个字对应一个id，如 "你" -> 0, "我" -> 1, "他" -> 2, ...
python3 tokenizer/train_char_tokenizer.py \
  --input_path /root/autodl-tmp/data/train.jsonl \
  --output_dir /root/autodl-tmp/char_tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

SPECIAL_TOKENS: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharTokenizer:
    """字符级 tokenizer：encode / decode / save / load。"""

    def __init__(
        self,
        token_to_id: Dict[str, int],
        tokenizer_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.token_to_id: Dict[str, int] = dict(token_to_id)
        self.id_to_token: Dict[int, str] = {
            i: t for t, i in self.token_to_id.items()
        }
        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("token_to_id 中存在重复的 id，无法构建 id_to_token")

        self.tokenizer_config: Dict[str, Any] = dict(tokenizer_config or {})

        self.pad_token = SPECIAL_TOKENS[0]
        self.bos_token = SPECIAL_TOKENS[1]
        self.eos_token = SPECIAL_TOKENS[2]
        self.unk_token = SPECIAL_TOKENS[3]

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]

        self._special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode 需要 str，收到 {type(text).__name__}")
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self, ids: Iterable[int], skip_special_tokens: bool = True
    ) -> str:
        pieces: List[str] = []
        for i in ids:
            if skip_special_tokens and i in self._special_ids:
                continue
            tok = self.id_to_token.get(i)
            if tok is None:
                continue
            pieces.append(tok)
        return "".join(pieces)

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        vocab_path = os.path.join(output_dir, "vocab.json")
        config_path = os.path.join(output_dir, "tokenizer_config.json")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        cfg = {
            "tokenizer_type": "char",
            "vocab_size": len(self.token_to_id),
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
        }
        cfg.update(self.tokenizer_config)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, output_dir: str) -> "CharTokenizer":
        vocab_path = os.path.join(output_dir, "vocab.json")
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"未找到 vocab.json: {vocab_path}")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"未找到 tokenizer_config.json: {config_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id: Dict[str, int] = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            tokenizer_config: Dict[str, Any] = json.load(f)

        return cls(token_to_id, tokenizer_config)


def read_train_jsonl(
    path: str,
) -> tuple[int, int, Counter, int]:
    """
    读取 jsonl，返回：
    raw_records, valid_records, 字符 Counter, total_characters
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    raw_records = 0
    valid_records = 0
    counter: Counter = Counter()
    total_characters = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(
            tqdm(f, desc="Reading train.jsonl"), start=1
        ):
            raw_records += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"第 {line_num} 行 JSON 解析失败: {e.msg} (位置 {e.pos})"
                ) from e

            if not isinstance(obj, dict):
                raise ValueError(
                    f"第 {line_num} 行: 期望 JSON 对象，实际为 {type(obj).__name__}"
                )
            if "text" not in obj:
                raise ValueError(f"第 {line_num} 行: 缺少字段 'text'")

            text = obj["text"]
            if text is None:
                continue
            if not isinstance(text, str):
                raise TypeError(
                    f"第 {line_num} 行: 'text' 应为 str，实际为 {type(text).__name__}"
                )
            if not text:
                continue

            valid_records += 1
            total_characters += len(text)
            counter.update(text)

    return raw_records, valid_records, counter, total_characters


def build_token_to_id(counter: Counter) -> Dict[str, int]:
    """特殊 token 固定在前，其余按频次降序、同频次按字符字典序。"""
    token_to_id: Dict[str, int] = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        token_to_id[tok] = i

    # 按 (-频次, 字符) 排序，等价于频次从高到低，同频次字典序升序
    sorted_chars = sorted(counter.keys(), key=lambda c: (-counter[c], c))
    next_id = len(SPECIAL_TOKENS)
    for ch in sorted_chars:
        if ch in token_to_id:
            continue
        token_to_id[ch] = next_id
        next_id += 1
    return token_to_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 train.jsonl 训练字符级词表并保存 CharTokenizer 产物"
    )
    p.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="训练集 jsonl 路径（仅用于构建词表）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（vocab.json, tokenizer_config.json）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.output_dir)

    raw_records, valid_records, counter, total_characters = read_train_jsonl(
        input_path
    )
    unique_characters = len(counter)
    token_to_id = build_token_to_id(counter)
    vocab_size = len(token_to_id)

    stats_config: Dict[str, Any] = {
        "raw_records": raw_records,
        "valid_records": valid_records,
        "total_characters": total_characters,
        "unique_characters": unique_characters,
        "output_dir": output_dir,
    }

    tokenizer = CharTokenizer(token_to_id, tokenizer_config=stats_config)
    tokenizer.save(output_dir)

    print("raw_records:", raw_records)
    print("valid_records:", valid_records)
    print("total_characters:", total_characters)
    print("unique_characters:", unique_characters)
    print("vocab_size:", vocab_size)
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
