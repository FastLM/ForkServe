"""ToT fan-out: four thought prefixes share one trunk via CoW."""

from forkserve.adapters.templates import ToolWrappers
from forkserve.adapters.tot import ToTAdapter
from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.pages import clone_memory_bytes, cow_memory_bytes


def main() -> None:
    cfg = ForkServeConfig(bytes_per_token=320_000.0)
    eng = Engine(MockBackend(cfg), cfg)
    h = eng.open(" ".join(f"tok{i}" for i in range(400)))  # long trunk
    ad = ToTAdapter(eng, ToolWrappers(), branching=4)
    kids = ad.expand(h.id, h.tip)
    tree = eng.tree(h.id)
    trunk = len(tree.get(h.tip).tokens)
    residuals = [len(tree.get(k).residual) for k in kids]
    print("trunk", trunk, "residuals", residuals)
    print("M_CoW   ", int(cow_memory_bytes(trunk, residuals, cfg.bytes_per_token)))
    print("M_clone ", int(clone_memory_bytes(trunk, residuals, cfg.bytes_per_token, 4)))
    winner = ad.select(h.id, kids, winner=2)
    print("winner", winner, "live_kv_tokens", tree.live_kv_tokens())


if __name__ == "__main__":
    main()
