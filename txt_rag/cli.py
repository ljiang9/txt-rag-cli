"""命令行入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .index import TextIndex
from .retriever import Retriever
from .llm import answer_with_llm


def cmd_build(args) -> int:
    if not Path(args.dir).is_dir():
        print(f"目录不存在: {args.dir}", file=sys.stderr)
        return 2
    idx = TextIndex(args.dir, max_chars=args.chunk)
    n = idx.build()
    idx.save(args.out)
    print(f"已索引 {n} 个文本片段 -> {args.out}")
    return 0


def cmd_ask(args) -> int:
    p = Path(args.index)
    if not p.exists():
        print(f"索引不存在: {p}（请先 build）", file=sys.stderr)
        return 2
    idx = TextIndex.load(p)
    retr = Retriever(idx)
    hits = retr.search(args.question, top_k=args.top_k)
    if not hits:
        print("未检索到相关片段。")
        return 1
    print("【检索结果】")
    for i, h in enumerate(hits):
        print(f"{i+1}. [{h.score:.3f}] ({h.chunk.source}) {h.chunk.text[:120]}...")
    print()
    snippets = [h.chunk.text for h in hits]
    print(answer_with_llm(args.question, snippets))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="txt-rag", description="纯文本目录 RAG CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="对目录建索引")
    pb.add_argument("dir")
    pb.add_argument("--out", default="index.json")
    pb.add_argument("--chunk", type=int, default=400)
    pb.set_defaults(func=cmd_build)

    pa = sub.add_parser("ask", help="提问检索片段")
    pa.add_argument("question")
    pa.add_argument("--index", default="index.json")
    pa.add_argument("--top-k", type=int, default=3)
    pa.add_argument("--llm", action="store_true", help="有 OPENAI_API_KEY 时调用 LLM 作答")
    pa.set_defaults(func=cmd_ask)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
