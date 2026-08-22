"""Harness-known suffixes (p=1). Determined by the template, not the tool."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolWrappers:
    style: str = "openai_xml"  # openai_xml | hermes | llama | bfcl

    def chat(self, system: str, user: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def observation(self, tool: str) -> str:
        if self.style == "hermes":
            return f"<tool_response name=\"{tool}\">\n"
        if self.style == "bfcl":
            return f"<|start_header_id|>tool<|end_header_id|>\n\n"
        if self.style == "llama":
            return (
                f"<|eot_id|><|start_header_id|>ipython<|end_header_id|>\n\n"
            )
        return (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<tool_response>\n"
        )

    def recovery(self, tool: str) -> str:
        return (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<tool_response>\nERROR {tool}: "
        )

    def sibling_system(self, role: str, prompt: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are the {role}. {prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
        )

    def join_scaffold(self) -> str:
        return (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<combined_results>\n"
        )

    def thought_prefix(self, index: int) -> str:
        return f"Thought {index + 1}: "

    def close_observation(self) -> str:
        return "\n</tool_response><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
