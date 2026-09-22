# ForkServe paper (MLSys 2026 format)

Working draft in the MLSys 2026 style (`mlsys2026.sty`).
Body covers the copy-on-write context tree, advanced prefill pruning (hash skip, draft prune, early abort, disaggregated-prefill gate), and the GSM8K GPU forest plus the control-plane prefill comparison.

```bash
cd docs/paper && make
```

Produces `main.pdf`. The bibliography is the checked-in `main.bbl` (numeric citations). Do not run `bibtex` unless `refs.bib` is restored.

## Structure

1. Introduction — branching as the serving object; system architecture (Figure 1)
2. Background — prefill/decode, branch operators, comparison with prior systems
3. Design — CoW tree, LCP commit, advanced prefill pruning (Figure 2), two-class scheduler
4. Evaluation — Qwen3-14B GSM8K forest; control-plane APC / hash prefill / disagg prefill / ForkServe / APP
5. Related work and conclusion

GPU numbers are the A100 forests: Qwen3-14B GSM8K (peak KV, accuracy, fan-out) and Qwen3-8B ForkServe+ (APP knobs plus decode-stop).
Control-plane APP fan-out and transfer numbers are setting L of `docs/exp_report/report_en.tex`.
