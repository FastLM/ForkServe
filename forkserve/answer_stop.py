"""Stop decode only after the answer is on the tape.

Bare markers (``####``, ``</think>``) fire before the number or the reply,
so ForkServe+ scored 0 on DeepSeek-R1 and lost GSM8K points on other
families. These checks wait for a finished answer.
"""

from __future__ import annotations

import re

from forkserve.quality import extract_boxed, game24_correct, iter_game24_exprs

_HASH_ANS = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)(?:\s|$)")
_CODE_BOUND = ("\ndef ", "\nclass ", "\nif __name__")


def stop_mode_for(workload: str, configured: str) -> str:
    """Map a bench workload onto a stop mode. Empty means do not stop."""
    cfg = (configured or "").strip()
    if not cfg:
        return ""
    if cfg != "auto":
        return cfg
    wl = workload or ""
    if wl in ("gsm8k", "svamp", "gsmhard"):
        return "gsm"
    if wl in ("math500", "aime", "amc23"):
        return "math"
    if wl == "game24":
        return "game24"
    if wl == "humaneval":
        return "code"
    return ""


def answer_ready(text: str, mode: str, *, hint: str = "") -> bool:
    """True when ``text`` already contains a gradeable final answer."""
    raw = text or ""
    if not raw or not mode:
        return False
    if mode in ("gsm", "math"):
        body = raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw
        if not body.strip():
            return False
        # Stop only after the model has left a finished answer line
        # (a newline follows it). A ``####`` that is still the last line
        # may be revised, and cutting it dropped Llama and Mistral.
        if re.search(r"(?m)^####\s*-?\d+(?:,\d{3})*(?:\.\d+)?[ \t]*\n", body):
            return True
        if mode == "math" and _boxed_closed(body) and re.search(r"\\boxed\{[^{}]*\}\s*\n", body):
            return True
        return False
    if mode == "game24":
        if hint and game24_correct(raw, hint):
            return True
        # No gold: a finished equation that evaluates to 24.
        if not hint:
            from forkserve.quality import _game24_ok

            for expr in iter_game24_exprs(raw):
                nums = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", expr)]
                if len(nums) == 4 and _game24_ok(expr, nums):
                    return True
        return False
    if mode == "code":
        return any(b in raw for b in _CODE_BOUND)
    return False


def _boxed_closed(text: str) -> bool:
    key = r"\boxed"
    start = text.rfind(key)
    if start < 0:
        return False
    i = start + len(key)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return False
    depth = 0
    for ch in text[i:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


try:
    from vllm.v1.sample.logits_processor.interface import (
        LogitsProcessor as _LogitsProcessor,
    )
except Exception:  # unit tests without the engine
    _LogitsProcessor = object  # type: ignore[misc,assignment]


class AnswerStopLogitsProcessor(_LogitsProcessor):
    """Force EOS once the decoded string contains a finished answer.

    Registered on the vLLM engine. Requests without ``fs_stop`` in
    ``extra_args`` are left alone, so APC and speculative prefills
    run to ``max_tokens``.
    """

    def __init__(self, vllm_config, device, is_pin_memory: bool) -> None:
        del device, is_pin_memory
        self._tok = None
        self._eos = 0
        try:
            from transformers import AutoTokenizer

            path = vllm_config.model_config.tokenizer
            self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            eos = self._tok.eos_token_id
            if isinstance(eos, list):
                eos = eos[0] if eos else 0
            self._eos = int(eos or 0)
        except Exception:
            self._tok = None
        self._rows: dict[int, tuple[list[int], str, str]] = {}

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update) -> None:
        if batch_update is None:
            return
        for index, params, _prompt, output_ids in batch_update.added:
            extra = getattr(params, "extra_args", None) or {}
            mode = str(extra.get("fs_stop") or "")
            hint = str(extra.get("fs_stop_hint") or "")
            if mode and output_ids is not None:
                self._rows[index] = (output_ids, mode, hint)
            else:
                self._rows.pop(index, None)
        for index in batch_update.removed:
            self._rows.pop(index, None)
        for adx, bdx, direct in batch_update.moved:
            row = self._rows.pop(adx, None)
            if row is not None:
                self._rows[bdx] = row
            from vllm.v1.sample.logits_processor.interface import MoveDirectionality

            if direct == MoveDirectionality.SWAP:
                # ``row`` was adx; bdx's previous row was overwritten above.
                # Swap is rare on this decode path; drop rather than alias.
                pass

    def apply(self, logits):
        if not self._rows or self._tok is None or self._eos <= 0:
            return logits
        n = logits.shape[0]
        for index, (ids, mode, hint) in list(self._rows.items()):
            if index >= n or len(ids) < 4:
                continue
            try:
                text = self._tok.decode(ids, skip_special_tokens=True)
            except Exception:
                continue
            if answer_ready(text, mode, hint=hint):
                logits[index].fill_(float("-inf"))
                logits[index, self._eos] = 0
        return logits
