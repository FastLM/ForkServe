"""CLI for the multi-GPU ForkServe vs vLLM comparison.

    python examples/bench_gpu.py --tp 2,4
    python examples/bench_gpu.py --backend mock
"""

from forkserve.bench import main

if __name__ == "__main__":
    raise SystemExit(main())
