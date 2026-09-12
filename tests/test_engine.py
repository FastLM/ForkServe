import pytest

from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.engine.mock import MockBackend
from forkserve.types import NodeMode


@pytest.fixture
def eng() -> Engine:
    cfg = ForkServeConfig(page_size=8, bytes_per_token=1.0, max_batched_tokens=2048)
    return Engine(MockBackend(cfg), cfg)


def test_open_fork_speculate_commit_roundtrip(eng: Engine) -> None:
    h = eng.open("system history trunk tokens here")
    tree = eng.tree(h.id)
    wrap = "<tool_response>\n"
    recov = "ERROR bash: "
    happy = eng.fork(h.id, h.tip, "bash", wrap)
    fail = eng.fork(h.id, h.tip, "err", recov)
    eng.speculate(h.id, happy, t_idle_ms=1000)
    eng.speculate(h.id, fail, priority=0.2, t_idle_ms=1000)
    n = eng.drain_slack()
    assert n >= 0
    prompt = wrap + '{"ok": true}'
    cr = eng.commit(h.id, h.tip, prompt, preferred_bid="bash")
    assert cr.known_suffix_hit
    assert tree.get(cr.winner).mode is NodeMode.COMMIT
    assert tree.get(fail).mode is NodeMode.DEAD
    m = eng.metrics[h.id]
    assert m.commits == 1
    assert m.known_suffix_hit_rate == 1.0


def test_promote_keeps_winner_without_join(eng: Engine) -> None:
    h = eng.open("trunk history")
    a = eng.fork(h.id, h.tip, "thought-0", "plan a")
    b = eng.fork(h.id, h.tip, "thought-1", "plan b")
    eng.abort(h.id, b)
    tip = eng.promote(h.id, a)
    tree = eng.tree(h.id)
    assert tip == a
    assert tree.tip == a
    assert tree.get(a).mode is NodeMode.COMMIT
    assert tree.get(b).mode is NodeMode.DEAD
    # No extra join child of the root.
    live = [n for n in tree.live_nodes() if n.parent == h.tip]
    assert [n.id for n in live] == [a]


def test_queue_known_prefill_then_flush(eng: Engine) -> None:
    a = eng.open("trunk alpha", flush=False)
    b = eng.open("trunk beta", flush=False)
    ka = eng.fork(a.id, a.tip, "t0", "plan a")
    kb = eng.fork(b.id, b.tip, "t0", "plan b")
    eng.queue_known_prefill(a.id, ka)
    eng.queue_known_prefill(b.id, kb)
    eng.flush()
    assert eng.tree(a.id).get(ka).residual
    assert eng.tree(b.id).get(kb).residual


def test_generate_many_batches_committed_decode(eng: Engine) -> None:
    a = eng.open("trunk alpha", flush=False)
    b = eng.open("trunk beta", flush=False)
    eng.flush()
    outs = eng.generate_many([a.id, b.id], 2)
    assert len(outs) == 2
    assert all(len(o) >= 1 for o in outs)
    assert outs[0] != outs[1]


def test_generate_refuses_spec_tip(eng: Engine) -> None:
    h = eng.open("trunk")
    spec = eng.fork(h.id, h.tip, "x", "suffix")
    eng.tree(h.id).tip = spec
    with pytest.raises(Exception):
        eng.generate(h.id, 4)


def test_fanout_cow_memory(eng: Engine) -> None:
    h = eng.open(" ".join(f"tok{i}" for i in range(200)))
    parent = h.tip
    kids = [eng.fork(h.id, parent, f"r{i}", f"role-{i} prompt") for i in range(4)]
    tree = eng.tree(h.id)
    trunk = len(tree.get(parent).tokens)
    extra = sum(len(tree.get(k).residual) for k in kids)
    live = tree.live_kv_tokens()
    assert live == trunk + extra
    # clone would be 4*trunk + extra
    assert live < 2 * trunk


def test_prior_and_ngram_observe_on_fork_commit(eng: Engine) -> None:
    h = eng.open("trunk history")
    wrap = "<tool_response>\n"
    happy = eng.fork(h.id, h.tip, "bash", wrap)
    eng.speculate(h.id, happy, t_idle_ms=1000)
    eng.drain_slack()
    eng.commit(h.id, h.tip, wrap + '{"ok": true}', preferred_bid="bash")
    mass = eng.priors.mass("root", ["bash", "err"])
    assert mass["bash"] > mass["err"]
    guessed = eng.ngrams.predict("bash")
    assert guessed  # observed the committed residual


def test_grammar_forks_materialize_nodes(eng: Engine) -> None:
    h = eng.open("trunk")
    chunks = eng.speculate_from_grammar(
        h.id,
        h.tip,
        {"bash": 0.7, "edit": 0.2, "search": 0.1},
        {"bash": "<tool_response>\n", "edit": "<edit>\n", "search": "<search>\n"},
        generic_wrapper="<other>\n",
        t_idle_ms=5000,
    )
    assert chunks
    assert all(c.node_id is not None for c in chunks)
    tree = eng.tree(h.id)
    kids = tree.children_of(h.tip)
    assert len(kids) >= 3


def test_generate_charges_committed_budget(eng: Engine) -> None:
    h = eng.open("trunk")
    happy = eng.fork(h.id, h.tip, "bash", "wrap ")
    eng.speculate(h.id, happy, t_idle_ms=1e9)
    # Saturate the tick with committed decode so spec is not admitted this tick.
    eng.config.max_batched_tokens = 4
    out = eng.generate(h.id, 4)
    assert out
    assert eng.scheduler.pending_spec_tokens() > 0


def test_context_xform_forks_from_root(eng: Engine) -> None:
    h = eng.open("long trunk tokens here")
    xform = eng.fork_context_xform(h.id, "summary of trunk", speculate=False)
    tree = eng.tree(h.id)
    assert tree.get(xform).parent == tree.root
    assert tree.get(h.tip).mode is not NodeMode.DEAD
