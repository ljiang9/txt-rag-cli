"""对目录下的 .txt/.md 文件切分、建倒排/词袋索引，持久化为 JSON。"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    tokens: list[str] = field(default_factory=list)


def split_text(text: str, max_chars: int = 400, overlap: int = 60) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars - overlap):
                    chunks.append(p[i:i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


class TextIndex:
    def __init__(self, root, max_chars=400):
        self.root = Path(root).resolve()
        self.max_chars = max_chars
        self.chunks: list[Chunk] = []
        self.postings: dict[str, dict[str, int]] = {}
        self.df: dict[str, int] = {}

    def build(self, exts=(".txt", ".md")) -> int:
        self.chunks.clear()
        self.postings.clear()
        self.df.clear()
        ext_set = {e.lower() for e in exts}
        files = [p for p in self.root.rglob("*")
                 if p.is_file() and p.suffix.lower() in ext_set
                 and ".git" not in p.parts]
        cid = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for piece in split_text(text, self.max_chars):
                toks = tokenize(piece)
                ch = Chunk(id=f"c{cid}", source=str(f.relative_to(self.root)),
                           text=piece, tokens=toks)
                self.chunks.append(ch)
                cid += 1
                tf: dict[str, int] = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                for t, cnt in tf.items():
                    self.postings.setdefault(t, {})[ch.id] = cnt
                for t in tf:
                    self.df[t] = self.df.get(t, 0) + 1
        return len(self.chunks)

    def save(self, path):
        data = {
            "root": str(self.root),
            "max_chars": self.max_chars,
            "chunks": [asdict(c) for c in self.chunks],
            "postings": self.postings,
            "df": self.df,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(data["root"], max_chars=data["max_chars"])
        idx.chunks = [Chunk(**c) for c in data["chunks"]]
        idx.postings = data["postings"]
        idx.df = data["df"]
        return idx

    def chunk_vector(self, ch: Chunk) -> dict[str, float]:
        n = len(self.chunks) or 1
        v: dict[str, float] = {}
        tf: dict[str, int] = {}
        for t in ch.tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, cnt in tf.items():
            idf = math.log((1 + n) / (1 + self.df.get(t, 0))) + 1.0
            v[t] = (1.0 + math.log(cnt)) * idf
        return v

    def query_vector(self, query: str) -> dict[str, float]:
        n = len(self.chunks) or 1
        toks = tokenize(query)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        v: dict[str, float] = {}
        for t, cnt in tf.items():
            idf = math.log((1 + n) / (1 + self.df.get(t, 0))) + 1.0
            v[t] = (1.0 + math.log(cnt)) * idf
        return v
