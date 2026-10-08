"""Control-plane P/D connector table for the paper (tab:connector).

Reproduces the slide / paper numbers:
  10 sessions, k=4, L=256, ell=32
  prefill 12 us/token, transfer 2 us/token

Modes
-----
* APC              : no connector
* P/D              : stock disagg, ships kL + sum ell
* P/D + loop-only  : tau_pre=0.15 on stock prefiller (drop 1/4 residuals)
* APP (alias+gate) : trunk aliased; connector = miss of Keep

No GPU. Run:
  python -m experiments.connector_bench
  python -m experiments.connector_bench --out logs/connector/connector.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREFILL_US = 12.0
XFER_US = 2.0


def account(
    *,
    name: str,
    sessions: int,
    kept: int,
    trunk: int,
    residual: int,
    alias_trunk: bool,
    ship: bool,
) -> dict[str, Any]:
    per = (0 if alias_trunk else trunk) + residual
    work = sessions * kept * per
    xfer = work if ship else 0
    prefill_ms = work * PREFILL_US / 1000.0
    xfer_ms = xfer * XFER_US / 1000.0
    return {
        "method": name,
        "sessions": sessions,
        "kept": kept,
        "work_tokens": work,
        "xfer_tokens": xfer,
        "prefill_ms": round(prefill_ms, 2),
        "xfer_ms": round(xfer_ms, 2),
        "fanout_ms": round(prefill_ms + xfer_ms, 2),
    }


def build_table(
    *,
    sessions: int = 10,
    k: int = 4,
    trunk: int = 256,
    residual: int = 32,
    loops_dropped: int = 1,
) -> list[dict[str, Any]]:
    kept_loop = max(1, k - loops_dropped)
    return [
        account(
            name="APC",
            sessions=sessions,
            kept=k,
            trunk=trunk,
            residual=residual,
            alias_trunk=False,
            ship=False,
        ),
        account(
            name="P/D",
            sessions=sessions,
            kept=k,
            trunk=trunk,
            residual=residual,
            alias_trunk=False,
            ship=True,
        ),
        account(
            name="P/D + loop-only",
            sessions=sessions,
            kept=kept_loop,
            trunk=trunk,
            residual=residual,
            alias_trunk=False,
            ship=True,
        ),
        account(
            name="APP (alias+gate)",
            sessions=sessions,
            kept=kept_loop,
            trunk=trunk,
            residual=residual,
            alias_trunk=True,
            ship=True,
        ),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="P/D connector control-plane table")
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--trunk", type=int, default=256)
    p.add_argument("--residual", type=int, default=32)
    p.add_argument("--loops-dropped", type=int, default=1)
    p.add_argument("--out", default="")
    args = p.parse_args()
    table = build_table(
        sessions=args.sessions,
        k=args.k,
        trunk=args.trunk,
        residual=args.residual,
        loops_dropped=args.loops_dropped,
    )
    print(f"{'method':<18} {'W':>8} {'X':>8} {'xfer_ms':>8} {'fanout_ms':>10}")
    for r in table:
        print(
            f"{r['method']:<18} {r['work_tokens']:8d} {r['xfer_tokens']:8d} "
            f"{r['xfer_ms']:8.1f} {r['fanout_ms']:10.1f}"
        )
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": table}, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
