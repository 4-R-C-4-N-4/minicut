# Eval: baseline_v1 — eval_unseen_fixture

Backend: http · n=50 · verification pass rate: 56.0% · tool accuracy: 84.0% · arg accuracy: 50.0% · mean latency: 15.6s · mean retries: 0.0

## Per-tool verification pass rate

- clarify: 50.0% (2 cases, tool-correct 100.0%)
- concat: 25.0% (4 cases, tool-correct 75.0%)
- convert: 0.0% (6 cases, tool-correct 83.3%)
- cut_out: 100.0% (7 cases, tool-correct 100.0%)
- extract_audio: 0.0% (5 cases, tool-correct 40.0%)
- mute: 75.0% (4 cases, tool-correct 75.0%)
- scale: 100.0% (7 cases, tool-correct 100.0%)
- speed: 0.0% (6 cases, tool-correct 100.0%)
- trim: 100.0% (9 cases, tool-correct 77.8%)

## Failure modes

- wrong_arg: 17
- wrong_tool: 7
- parse_fail: 1
