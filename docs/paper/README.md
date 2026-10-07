# ForkServe paper (MLSys 2026 format)

Working draft in the MLSys 2026 style (`mlsys2026.sty`).
The body is three coupled mechanisms: a copy-on-write context tree, one cost-ordered action per residual, and the committed spine on the critical path. Evaluation is the GSM8K GPU forest plus the control-plane prefill comparison.

```bash
cd docs/paper && make        # original draft → main.pdf
cd docs/paper && make new    # revised MLSys draft → main_new.pdf
cd docs/paper && make acl-new  # ACL draft, PD-Prune → acl_new/main.pdf
```

`acl_new/` is the ACL-format rewrite of `paper_old/` under the name PD-Prune.
Sections 1--6 are the main text (formulation, the admission algorithm, and the headline results).
Appendices A--D keep the background, the copy-on-write system, the full evaluation, and the extended related work.

Produces `main.pdf`, `main_new.pdf`, or `acl_new/main.pdf`. The MLSys bibliography is the checked-in `*.bbl` (numeric citations). Do not run `bibtex` unless `refs.bib` is restored.

`main_new.tex` is the condensed (≤12pp) revision: the three mechanisms in Figure 1, the per-residual cascade in Figure 2, the work/transfer spans, and the control-plane / GPU / cross-model numbers.

## Structure

1. Introduction — branching as the serving object; system architecture (Figure 1)
2. Background — prefill/decode, branch operators, comparison with prior systems
3. Design — CoW tree, one action per residual, the spine on the critical path
4. Evaluation — Qwen3-14B GSM8K forest; control-plane APC / hash prefill / disagg / cascade
5. Related work and conclusion

GPU numbers are the A100 forests: Qwen3-14B GSM8K (peak KV, accuracy, fan-out) and Qwen3-8B with the prefill threshold in front of ESC, Speculative Rejection, and DPTS. Decode length is a separate answer-stop, not part of the cascade.
Control-plane fan-out and transfer numbers are setting L of `docs/exp_report/report_en.tex`. The in-repo cost model (`experiments/prefill_prune_bench.py`) now drops only the repeated loop, so its percentages are not the blanket-skip rows in that report.
