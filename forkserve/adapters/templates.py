"""Harness-known suffixes (p=1). Determined by the template, not the tool."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolWrappers:
    # Default matches Qwen3-8B (the bench model). Llama3 tokens on Qwen
    # leaked <|eot_id|> and wasted the short decode budget.
    style: str = "qwen"  # qwen | openai_xml | hermes | llama | bfcl

    def chat(self, system: str, user: str) -> str:
        if self.style == "qwen":
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
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
        if self.style == "qwen":
            return f"<|im_end|>\n<|im_start|>user\n<tool_response>\n"
        return (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<tool_response>\n"
        )

    def recovery(self, tool: str) -> str:
        if self.style == "qwen":
            return (
                f"<|im_end|>\n<|im_start|>user\n<tool_response>\nERROR {tool}: "
            )
        return (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<tool_response>\nERROR {tool}: "
        )

    def sibling_system(self, role: str, prompt: str) -> str:
        if self.style == "qwen":
            return (
                f"<|im_start|>system\nYou are the {role}. {prompt}<|im_end|>\n"
                f"<|im_start|>user\n"
            )
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are the {role}. {prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
        )

    def join_scaffold(self) -> str:
        if self.style == "qwen":
            return f"<|im_end|>\n<|im_start|>user\n<combined_results>\n"
        return (
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"<combined_results>\n"
        )

    def thought_prefix(self, index: int) -> str:
        return f"Thought {index + 1}: "

    def close_observation(self) -> str:
        if self.style == "qwen":
            return "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        return "\n</tool_response><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
