"""可选 LLM 作答：仅用 urllib 走 OpenAI 兼容接口；无 key 时降级为片段拼接。"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Iterable


SYSTEM = (
    "你是一个严谨的助手。请仅根据下面提供的【参考片段】回答用户问题；"
    "若片段中没有答案，请明确说参考资料中未提及。"
)


def answer_with_llm(question: str, snippets: Iterable[str]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    snippets = list(snippets)
    if not api_key:
        return _fallback(question, snippets)
    ctx = "\n---\n".join(f"[片段 {i+1}]\n{s}" for i, s in enumerate(snippets))
    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"参考片段：\n{ctx}\n\n问题：{question}"},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"（LLM 调用失败：{e}，降级为片段拼接）\n" + _fallback(question, snippets)


def _fallback(question: str, snippets: list[str]) -> str:
    if not snippets:
        return "参考资料中未提及。"
    out = [f"问题：{question}", "", "【检索到的参考片段】"]
    for i, s in enumerate(snippets):
        out.append(f"{i+1}. {s}")
    out.append("")
    out.append("提示：设置 OPENAI_API_KEY 后可由 LLM 基于上述片段直接作答。")
    return "\n".join(out)
