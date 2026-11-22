"""llama.cpp chat client for the MiniCut edit contract.

One request per edit: shared system + few-shot messages, greedy decoding,
thinking off, and the GBNF grammar. Loopback calls bypass any HTTP proxy.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from .prompts import edit_messages

GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "minicut.gbnf"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def predict(prompt: str, base_url: str, use_grammar: bool = True, timeout_s: int = 120) -> str:
    """Return the raw assistant content for one edit request."""
    body: dict = {
        "messages": edit_messages(prompt),
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False,
        # MiniCPM5-2B thinks by default, which leaves `content` empty.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if use_grammar:
        body["grammar"] = GRAMMAR_PATH.read_text()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"llama.cpp server HTTP {e.code}: {e.read().decode()[:300]}") from e
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(f"cannot reach llama.cpp server at {base_url}: {reason}") from e
    try:
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected server response: {json.dumps(data)[:300]}") from e
