# Eval: baseline_v3 — eval_seen

Backend: http · n=50 · verification pass rate: 98.0% · tool accuracy: 92.0% · arg accuracy: 82.0% · mean latency: 22.58s · mean retries: 0.0

## Per-tool verification pass rate

- clarify: 50.0% (2 cases, tool-correct 100.0%)
- concat: 100.0% (4 cases, tool-correct 100.0%)
- convert: 100.0% (6 cases, tool-correct 100.0%)
- cut_out: 100.0% (7 cases, tool-correct 100.0%)
- extract_audio: 100.0% (5 cases, tool-correct 100.0%)
- mute: 100.0% (4 cases, tool-correct 75.0%)
- scale: 100.0% (7 cases, tool-correct 100.0%)
- speed: 100.0% (6 cases, tool-correct 100.0%)
- trim: 100.0% (9 cases, tool-correct 66.7%)

## Failure modes

- wrong_arg: 5
- wrong_tool: 4
