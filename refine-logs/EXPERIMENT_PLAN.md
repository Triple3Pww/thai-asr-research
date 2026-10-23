# Experiment Plan — ThaiSeek (NECTEC + cross-model CTC splice)

Serial, one GPU job at a time (`adapter/night_bias.sh`, detached). All on the 682 real utts;
hyperparameters never tuned on real (`tau` = −1.5, the published default `D6` used).

| # | run | command | purpose | status |
|---|---|---|---|---|
| E0 | NECTEC through the D4 code path, `none` | `arm_d4_boosting.py --model <nec> --condition none` | harness control | queued |
| E1 | D4 boosting sweep on **synth** split | `--split synth --condition bias@50 --sweep …` | is score-biasing alive on NECTEC at all (`F8b` prior: +8 mentions) | queued |
| E2 | D6 NECTEC `none` | `arm_d6_splice.py --mode rnnt --rnnt-model <nec> --condition none` | splice harness = base? | queued |
| E3 | D6 NECTEC `dis@50` | same, `dis@50` | same-size control (`F8d`) | queued |
| E4 | D6 NECTEC `bias@50` | same, `bias@50` | **C1/C2** | queued |
| E5 | rescore E2–E4 with `eval_canon.py` + paired bootstrap vs tw | — | numbers comparable with every other row | after E4 |
| E6 | latency of NECTEC + CTC pass, batch 1 | timing in `arm_d6_splice` run json | **C3** | after E4 |
| E7 | TVSpeech: tw vs NECTEC vs lot2/lot3 variants | `night_deepseek.sh` tail | **C4** | running |

**Decision after E5:** C1 ∧ C2 pass → write up as `F32`, deploy as an opt-in `:8007` mode
(bias list per request), then E8 = CTC head on NECTEC's frozen encoder to make it one pass.
Fail → record the negative; the real-time name problem then needs a list-reading model made fast
(qwen-0.6b-ctx distillation), which is a new direction, not a tweak.
