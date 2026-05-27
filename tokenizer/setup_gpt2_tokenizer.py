#!/usr/bin/env python3
"""
保存 OpenAI GPT-2 分词器到本地目录（供 prepare / train 使用）。

默认从仓库内 tokenizer/bundled_gpt2/ 复制（无需访问 huggingface.co）。
若 bundled 缺失，可加 --online 从镜像下载（国内建议先 export HF_ENDPOINT=https://hf-mirror.com）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tokenizer.gpt2_tokenizer import DEFAULT_MODEL, GPT2Tokenizer

BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundled_gpt2")


def _bundled_ready() -> bool:
    return (
        os.path.isfile(os.path.join(BUNDLED_DIR, "vocab.json"))
        and os.path.isfile(os.path.join(BUNDLED_DIR, "merges.txt"))
    )


def setup_from_bundled(output_dir: str) -> GPT2Tokenizer:
    if not _bundled_ready():
        raise FileNotFoundError(
            f"未找到离线词表：{BUNDLED_DIR}\n"
            "请在本机执行（需能访问 hf-mirror）：\n"
            "  export HF_ENDPOINT=https://hf-mirror.com\n"
            "  hf download gpt2 vocab.json merges.txt tokenizer_config.json "
            f"--local-dir {BUNDLED_DIR}\n"
            "或运行：python tokenizer/setup_gpt2_tokenizer.py --online"
        )
    tok = GPT2Tokenizer.load(BUNDLED_DIR)
    tok.save(output_dir)
    return tok


def setup_from_online(output_dir: str, pretrained: str) -> GPT2Tokenizer:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"从 Hub 下载（HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}）...")
    tok = GPT2Tokenizer.from_pretrained(pretrained)
    tok.save(output_dir)
    # 同步更新 bundled，便于下次离线
    if _bundled_ready():
        pass
    else:
        os.makedirs(BUNDLED_DIR, exist_ok=True)
        tok.save(BUNDLED_DIR)
        print(f"已同步到离线目录：{BUNDLED_DIR}")
    return tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="保存 GPT-2 分词器到本地（默认离线）")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/gpt2_tokenizer",
        help="输出目录",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default=DEFAULT_MODEL,
        help="--online 时使用的 HF 模型名",
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="从 Hugging Face（或 HF_ENDPOINT 镜像）下载；默认用 bundled_gpt2 离线复制",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    if args.online:
        tok = setup_from_online(output_dir, args.pretrained)
    else:
        tok = setup_from_bundled(output_dir)
    print(f"已保存 GPT-2 分词器 -> {output_dir}")
    print(f"vocab_size: {tok.vocab_size}")
    print(f"eos_id: {tok.eos_id} ({tok.eos_token})")


if __name__ == "__main__":
    main()
