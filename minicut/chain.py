"""Edit-history chain, stored in file metadata — not in the filename.

Decided 2026-11-20: the chain lives in one ffmpeg metadata tag,
``minicut_chain``, as compact tokens joined by ``;``:

    trim:12-45;speed:2.0;scale:1280x720

The probe reads the tag from the input, the wrapper appends the new
op's token, and build writes the longer chain into the new file with
``-metadata``. The input is never modified. If the tag is missing
(stripped by another tool, or a fresh file), the chain simply starts
from the current op — that is graceful, not an error.

Filenames stay short on purpose: stem + one op token, with a ``_v2``
bump on collision (see wrapper). Filename-as-history was explicitly
rejected, and no sidecar JSON is written.

Format caveats (verified 2026-11-20):
- MP4/M4A need ``-movflags use_metadata_tags`` or the muxer drops
  arbitrary keys silently; MKV/WebM keep them natively, MP3 keeps
  them as ID3 frames.
- WAV drops the tag (its LIST INFO has no place for it), so an
  extract to WAV restarts the chain from that op. Graceful, not fatal.
"""
from __future__ import annotations

from pathlib import Path

CHAIN_KEY = "minicut_chain"


def _num(v) -> str:
    """12.0 -> '12', 2.5 -> '2.5'."""
    f = float(v)
    return str(int(f)) if f == int(f) else str(f)


def token_for(tool_call: dict) -> str:
    """Compact token for one tool call. Raises KeyError/ValueError on bad args."""
    tool = tool_call["tool"]
    if tool == "trim":
        return f"trim:{_num(tool_call['start'])}-{_num(tool_call['end'])}"
    if tool == "cut_out":
        return f"cutout:{_num(tool_call['start'])}-{_num(tool_call['end'])}"
    if tool == "concat":
        return f"concat:{len(tool_call['inputs'])}"
    if tool == "scale":
        return f"scale:{int(tool_call['width'])}x{int(tool_call['height'])}"
    if tool == "speed":
        return f"speed:{_num(tool_call['factor'])}"
    if tool == "convert":
        crf = tool_call.get("crf", 23)
        return f"convert:{tool_call['codec']}-{tool_call['container']}-crf{crf}"
    if tool == "extract_audio":
        return f"extract:{tool_call['format']}"
    if tool == "mute":
        return "mute"
    raise ValueError(f"no chain token for tool: {tool}")


def append_chain(existing: str | None, token: str) -> str:
    """Append one token to an existing chain string (None/'' starts fresh)."""
    parts = [p for p in (existing or "").split(";") if p]
    parts.append(token)
    return ";".join(parts)


def read_chain(path: str | Path) -> str | None:
    """Read the chain tag from a file via ffprobe. None if absent/unreadable."""
    from .probe import probe_file

    try:
        return probe_file(path).chain
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        print(f"{arg}: {read_chain(arg) or '(no chain)'}")
