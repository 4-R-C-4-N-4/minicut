"""MiniCut — local ffmpeg editor driven by MiniCPM5-2B + LoRA.

Model outputs JSON tool calls only (GBNF-constrained). The wrapper in
this package owns probing, validation, ffmpeg argv construction,
execution, and verification. See spec.md and IMPLEMENTATION_GUIDE.md.
"""
__version__ = "0.1.0"
TOOLS_V1 = ["trim", "cut_out", "concat", "scale", "speed", "convert", "extract_audio", "mute", "clarify"]
