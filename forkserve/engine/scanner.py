"""Streaming JSON/XML tool-call scanner (§8).

Emits a complete tool-call object as soon as it is parsed, so fork of the
observation wrapper overlaps remaining parent decode — charged as committed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re


_XML_TOOL = re.compile(
    r"<tool_call>\s*<name>(?P<name>[^<]+)</name>\s*<arguments>(?P<args>.*?)</arguments>\s*</tool_call>",
    re.DOTALL,
)
_XML_FN = re.compile(
    r"<function=(?P<name>[^>]+)>(?P<args>.*?)</function>",
    re.DOTALL,
)


@dataclass(slots=True)
class ParsedToolCall:
    name: str
    arguments: str
    raw: str
    complete: bool
    start: int
    end: int


class StreamingToolScanner:
    def __init__(self) -> None:
        self.buf = ""

    def feed(self, chunk: str) -> list[ParsedToolCall]:
        self.buf += chunk
        found: list[ParsedToolCall] = []
        found.extend(self._json_objects())
        found.extend(self._xml())
        return found

    def reset(self) -> None:
        self.buf = ""

    def _json_objects(self) -> list[ParsedToolCall]:
        out: list[ParsedToolCall] = []
        s = self.buf
        i = 0
        while True:
            start = s.find("{", i)
            if start < 0:
                break
            depth = 0
            in_str = False
            esc = False
            end = None
            for j in range(start, len(s)):
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end is None:
                break
            blob = s[start:end]
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                i = start + 1
                continue
            name = obj.get("name") or obj.get("tool") or obj.get("function")
            args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
            if isinstance(name, str):
                if isinstance(args, dict):
                    args_s = json.dumps(args, separators=(",", ":"))
                else:
                    args_s = str(args)
                out.append(
                    ParsedToolCall(
                        name=name,
                        arguments=args_s,
                        raw=blob,
                        complete=True,
                        start=start,
                        end=end,
                    )
                )
            i = end
        return out

    def _xml(self) -> list[ParsedToolCall]:
        out: list[ParsedToolCall] = []
        for rx in (_XML_TOOL, _XML_FN):
            for m in rx.finditer(self.buf):
                out.append(
                    ParsedToolCall(
                        name=m.group("name").strip(),
                        arguments=m.group("args").strip(),
                        raw=m.group(0),
                        complete=True,
                        start=m.start(),
                        end=m.end(),
                    )
                )
        return out
