# Eval: baseline_v1 — eval_unseen_paraphrase

Backend: http · n=50 · verification pass rate: 54.0% · tool accuracy: 84.0% · arg accuracy: 42.0% · mean latency: 14.46s · mean retries: 0.0

## Per-tool verification pass rate

- clarify: 50.0% (2 cases, tool-correct 100.0%)
- concat: 0.0% (4 cases, tool-correct 100.0%)
- convert: 16.7% (6 cases, tool-correct 83.3%)
- cut_out: 85.7% (7 cases, tool-correct 100.0%)
- extract_audio: 0.0% (5 cases, tool-correct 40.0%)
- mute: 75.0% (4 cases, tool-correct 75.0%)
- scale: 100.0% (7 cases, tool-correct 100.0%)
- speed: 0.0% (6 cases, tool-correct 100.0%)
- trim: 100.0% (9 cases, tool-correct 66.7%)

## Failure modes

- wrong_arg: 21
- wrong_tool: 8
