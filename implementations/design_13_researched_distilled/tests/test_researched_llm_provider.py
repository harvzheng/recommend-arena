"""Tests for shared.researched_llm_provider."""
from __future__ import annotations

from shared.researched_llm_provider import ResearchedLLM, ResearchedResponse


class _Resp:
    def __init__(self, payload: dict) -> None:
        self.stop_reason = payload["stop_reason"]
        self.content = payload["content"]


class FakeAnthropic:
    """Fake Anthropic SDK message-create surface used by ResearchedLLM."""

    def __init__(self, scripted_turns: list[dict]) -> None:
        self._turns = list(scripted_turns)
        self.create_calls: list[dict] = []
        self.messages = self  # so .messages.create(**kw) works

    def create(self, **kwargs: object) -> _Resp:
        self.create_calls.append(kwargs)
        return _Resp(self._turns.pop(0))


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_use_block(name: str, tool_id: str, input_obj: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": input_obj}


def test_researched_generate_no_tools() -> None:
    fake = FakeAnthropic([{
        "stop_reason": "end_turn",
        "content": [_text_block(
            '{"score": 0.8, "matched_attributes": {}, '
            '"explanation": "ok", "evidence": []}'
        )],
    }])
    llm = ResearchedLLM(client=fake, model="claude-opus-4-7")
    out = llm.research_generate(prompt="rate this", max_tool_calls=3)
    assert isinstance(out, ResearchedResponse)
    assert out.tool_calls == 0
    assert "0.8" in out.text


def test_researched_generate_caps_tool_calls() -> None:
    final_turn = {
        "stop_reason": "end_turn",
        "content": [_text_block(
            '{"score": 0.5, "matched_attributes": {}, '
            '"explanation": "capped", "evidence": []}'
        )],
    }
    fake = FakeAnthropic([final_turn])
    llm = ResearchedLLM(client=fake, model="claude-opus-4-7")
    out = llm.research_generate(prompt="rate this", max_tool_calls=0)
    assert out.tool_calls == 0
    assert "capped" in out.text
    # When cap is 0, no tools should be passed in the request.
    assert "tools" not in fake.create_calls[0]


def test_researched_generate_loops_through_tool_use() -> None:
    tool_turn = {
        "stop_reason": "tool_use",
        "content": [_tool_use_block("web_search", "tu1", {"query": "ice coast"})],
    }
    final_turn = {
        "stop_reason": "end_turn",
        "content": [_text_block(
            '{"score": 0.7, "matched_attributes": {}, '
            '"explanation": "researched", "evidence": []}'
        )],
    }
    fake = FakeAnthropic([tool_turn, final_turn])
    llm = ResearchedLLM(client=fake, model="claude-opus-4-7")
    out = llm.research_generate(prompt="rate this", max_tool_calls=2)
    assert out.tool_calls == 1
    assert "researched" in out.text
    # Tools should have been included on the first request only.
    assert "tools" in fake.create_calls[0]
