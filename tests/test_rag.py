import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from txt_rag.index import TextIndex, split_text, tokenize
from txt_rag.retriever import Retriever
from txt_rag.llm import _fallback


DOCS = {
    "python.txt": """Python 是一门通用编程语言。它语法简洁，广泛用于数据科学与人工智能。

Python 的 GIL 是全局解释器锁，影响多线程 CPU 密集型任务。

Python 常用于 Web 后端，例如 FastAPI 与 Django 框架。""",
    "coffee.txt": """咖啡起源于埃塞俄比亚，主要成分是咖啡因。

阿拉比卡豆风味明亮，罗布斯塔豆苦味更重、咖啡因更高。

手冲咖啡需要 92 度左右的水温与均匀研磨。""",
}


class TestTxtRag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name, text in DOCS.items():
            (self.root / name).write_text(text, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokenize_chinese(self):
        toks = tokenize("Python GIL 全局锁")
        self.assertIn("python", toks)
        self.assertIn("gil", toks)
        self.assertIn("全", toks)

    def test_split_text_paragraphs(self):
        text = "段落一内容。\n\n段落二更长一点内容。\n\n段落三也不短。"
        chunks = split_text(text, max_chars=12)
        self.assertGreaterEqual(len(chunks), 2)

    def test_build_and_search(self):
        idx = TextIndex(self.root)
        n = idx.build()
        self.assertGreater(n, 0)
        retr = Retriever(idx)
        hits = retr.search("GIL 全局解释器锁", top_k=2)
        self.assertTrue(hits)
        self.assertIn("python.txt", hits[0].chunk.source)

    def test_search_coffee(self):
        idx = TextIndex(self.root)
        idx.build()
        retr = Retriever(idx)
        hits = retr.search("手冲咖啡水温", top_k=2)
        self.assertTrue(hits)
        self.assertIn("coffee.txt", hits[0].chunk.source)

    def test_persist_roundtrip(self):
        idx = TextIndex(self.root)
        idx.build()
        out = Path(self.tmp.name) / "idx.json"
        idx.save(out)
        idx2 = TextIndex.load(out)
        self.assertEqual(len(idx.chunks), len(idx2.chunks))
        self.assertEqual(idx.postings.keys(), idx2.postings.keys())
        retr = Retriever(idx2)
        hits = retr.search("Django FastAPI", top_k=1)
        self.assertTrue(hits)

    def test_fallback_no_key(self):
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            out = _fallback("问题", ["片段 A", "片段 B"])
            self.assertIn("片段 A", out)
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old


if __name__ == "__main__":
    unittest.main()
