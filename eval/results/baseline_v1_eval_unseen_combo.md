# Eval: baseline_v1 — eval_unseen_combo

Backend: http · n=50 · verification pass rate: 58.0% · tool accuracy: 88.0% · arg accuracy: 52.0% · mean latency: 16.16s · mean retries: 0.0

## Per-tool verification pass rate

- clarify: 50.0% (2 cases, tool-correct 100.0%)
- concat: 25.0% (4 cases, tool-correct 75.0%)
- convert: 33.3% (6 cases, tool-correct 83.3%)
- cut_out: 100.0% (7 cases, tool-correct 100.0%)
- extract_audio: 0.0% (5 cases, tool-correct 80.0%)
- mute: 75.0% (4 cases, tool-correct 75.0%)
- scale: 100.0% (7 cases, tool-correct 100.0%)
- speed: 0.0% (6 cases, tool-correct 100.0%)
- trim: 88.9% (9 cases, tool-correct 77.8%)

## Failure modes

- wrong_arg: 18
- wrong_tool: 5
- parse_fail: 1
