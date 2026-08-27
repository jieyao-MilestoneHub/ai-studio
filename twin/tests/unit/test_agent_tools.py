"""Agent tools. SPEC.md §3.1/C2/C4, §6.3, D29."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from twin.agent.tools.base import Tool, ToolResult
from twin.agent.tools.recall import RecallTool
from twin.agent.tools.reply import ReplyTool
from twin.agent.tools.web_search import WebSearchTool
from twin.core.enums import Modality, SourceClass, Split
from twin.core.fragment import EventTime, Fragment
from twin.ingest.store import write_fragments_jsonl


def _fragment(content: str) -> Fragment:
    return Fragment(
        principal_id="p1",
        source_class=SourceClass.SELF_REPORT,
        modality=Modality.TEXT,
        content=content,
        event_time=EventTime(value="2026-06-01", precision="day", confidence=0.9),
        ingest_time=datetime(2026, 8, 27, tzinfo=UTC),
        split=Split.TRAIN,
    )


def test_recall_reply_and_web_search_all_satisfy_the_tool_protocol() -> None:
    """SPEC.md §3.1/C4: recall shares the exact same interface as any other
    plugin — this is the check that actually confirms that, not just asserts it."""
    tools: list[Tool] = [RecallTool(fragments_uri="file:///nonexistent"), ReplyTool(), WebSearchTool()]
    for tool in tools:
        assert isinstance(tool.name, str) and tool.name
        assert isinstance(tool.schema(), dict)


class TestRecallTool:
    def test_call_returns_matching_fragment_content(self, tmp_path: Path) -> None:
        uri = f"file://{tmp_path}/fragments.jsonl"
        write_fragments_jsonl([_fragment("went hiking with Alice"), _fragment("had coffee")], uri)

        result = RecallTool(fragments_uri=uri).call(query="hiking")

        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert "hiking" in result.content
        assert "coffee" not in result.content

    def test_call_result_digest_is_deterministic(self, tmp_path: Path) -> None:
        uri = f"file://{tmp_path}/fragments.jsonl"
        write_fragments_jsonl([_fragment("went hiking with Alice")], uri)
        tool = RecallTool(fragments_uri=uri)
        assert tool.call(query="hiking").result_digest == tool.call(query="hiking").result_digest


class TestReplyTool:
    def test_call_stages_an_intent_without_sending_anything(self) -> None:
        result = ReplyTool().call(surface="line", content="hey!")
        assert result.ok is True
        assert result.content == "hey!"
        # No send-related attribute/side effect exists on ToolResult or ReplyTool
        # to check "it sent" against — that's the point: there is nothing here
        # that could have sent it.


class TestWebSearchTool:
    def test_call_is_not_yet_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="no backend"):
            WebSearchTool().call(query="anything")
