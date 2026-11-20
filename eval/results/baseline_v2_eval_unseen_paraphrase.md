# Eval: baseline_v2 — eval_unseen_paraphrase

Backend: http · n=50 · verification pass rate: 92.0% · tool accuracy: 90.0% · arg accuracy: 76.0% · mean latency: 24.64s · mean retries: 0.0

## Per-tool verification pass rate

- clarify: 50.0% (2 cases, tool-correct 100.0%)
- concat: 100.0% (4 cases, tool-correct 100.0%)
- convert: 100.0% (6 cases, tool-correct 100.0%)
- cut_out: 57.1% (7 cases, tool-correct 100.0%)
- extract_audio: 100.0% (5 cases, tool-correct 100.0%)
- mute: 100.0% (4 cases, tool-correct 75.0%)
- scale: 100.0% (7 cases, tool-correct 100.0%)
- speed: 100.0% (6 cases, tool-correct 100.0%)
- trim: 100.0% (9 cases, tool-correct 55.6%)

## Failure modes

- wrong_arg: 7
- wrong_tool: 5
