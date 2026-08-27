"""SPEC.md is the one normative schema (data-contract skill rule 1). This
compares the ```jsonc``` block under SPEC.md §4.4 directly against
`Fragment.model_fields` — not against a hand-maintained schema file, which
would only ever check self-consistency. If SPEC.md and core/fragment.py drift,
this test MUST go red.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from twin.core.fragment import Entities, EventTime, Fragment, ThirdPartySpan

TWIN_ROOT = Path(__file__).resolve().parents[2]
SPEC = TWIN_ROOT / "reference" / "SPEC.md"


def _jsonc_block_under(heading: str) -> str:
    text = SPEC.read_text(encoding="utf-8")
    pattern = re.compile(rf"^###\s+{re.escape(heading)}.*?```jsonc\n(.*?)\n```", re.DOTALL | re.MULTILINE)
    match = pattern.search(text)
    assert match, f"no ```jsonc block found under a '### {heading}' heading in {SPEC}"
    return match.group(1)


def _strip_jsonc_line_comments(text: str) -> str:
    """Strip `// ...` comments, but not `//` that appears inside a JSON string
    (SPEC.md's block has "source_uri": "r2://..." — a naive strip would mangle it)."""
    out: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parsed_fragment_schema() -> dict:
    return json.loads(_strip_jsonc_line_comments(_jsonc_block_under("4.4 碎片 Schema（規範）")))


def test_fragment_top_level_fields_match_spec() -> None:
    spec_fields = set(_parsed_fragment_schema().keys())
    assert spec_fields == set(Fragment.model_fields.keys())


def test_event_time_fields_match_spec() -> None:
    spec_fields = set(_parsed_fragment_schema()["event_time"].keys())
    assert spec_fields == set(EventTime.model_fields.keys())


def test_entities_fields_match_spec() -> None:
    spec_fields = set(_parsed_fragment_schema()["entities"].keys())
    assert spec_fields == set(Entities.model_fields.keys())


def test_third_party_span_fields_match_spec() -> None:
    spec_fields = set(_parsed_fragment_schema()["third_party_spans"][0].keys())
    assert spec_fields == set(ThirdPartySpan.model_fields.keys())


def test_no_fragment_dict_literals_outside_the_constructor() -> None:
    """data-contract skill rule 1: search the repo for `"fragment_id"` string
    literals. Every hit outside core/fragment.py (the constructor) and this
    test file itself is a candidate bug — a hand-built dict that can silently
    drop a field like `split` or `third_party_spans`."""
    allowed = {
        TWIN_ROOT / "src" / "twin" / "core" / "fragment.py",
        TWIN_ROOT / "tests" / "unit" / "test_schema_matches_spec.py",
        TWIN_ROOT / "tests" / "unit" / "test_fragment.py",  # asserts required-field errors by key name
        SPEC,
    }
    result = subprocess.run(
        # The quoted form specifically — a bare `fragment_id` also matches every
        # legitimate `f.fragment_id` attribute access, which is not what this
        # check is for.
        ["grep", "-rl", "--include=*.py", '"fragment_id"', str(TWIN_ROOT / "src"), str(TWIN_ROOT / "tests")],
        capture_output=True,
        text=True,
        check=False,
    )
    hits = {Path(line) for line in result.stdout.splitlines() if line}
    unexpected = hits - allowed
    assert not unexpected, f"unexpected 'fragment_id' string literal(s) outside the constructor: {unexpected}"
