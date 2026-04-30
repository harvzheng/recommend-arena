"""Anthropic-backed LLM with optional web_search tool use, capped.

Used by design #13's teacher pipeline. Spec §5: the teacher needs to
research outside the local catalog. We implement a thin loop over the
Messages API: send the prompt, accept text or tool_use stops, let the
server execute server-side tools, and stop when the model emits an
end_turn or the tool-call cap is reached.

Anthropic's web_search tool is server-side, so the SDK resolves
results on the next call without a client-side callback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


@dataclass
class ResearchedResponse:
    text: str
    tool_calls: int
    raw_turns: list[dict] = field(default_factory=list)


class ResearchedLLM:
    def __init__(self, client: Any, model: str = "claude-opus-4-7") -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def research_generate(
        self,
        prompt: str,
        max_tool_calls: int = 3,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> ResearchedResponse:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_calls = 0
        raw_turns: list[dict] = []

        while True:
            tools_for_call = [_WEB_SEARCH_TOOL] if tool_calls < max_tool_calls else []
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if tools_for_call:
                kwargs["tools"] = tools_for_call
            if system:
                kwargs["system"] = system

            resp = self._client.messages.create(**kwargs)
            raw_turns.append({
                "stop_reason": resp.stop_reason,
                "content": list(resp.content),
            })

            if resp.stop_reason == "tool_use" and tools_for_call:
                tool_calls += 1
                messages.append({"role": "assistant", "content": resp.content})
                continue

            text_parts: list[str] = []
            for block in resp.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                else:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(getattr(block, "text", ""))
            return ResearchedResponse(
                text="".join(text_parts),
                tool_calls=tool_calls,
                raw_turns=raw_turns,
            )
