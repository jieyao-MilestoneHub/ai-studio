"""`pipeline.idle`: the grace each kind of job earns a quiet pod."""

from __future__ import annotations

import pytest

from fun_workflow.core.kinds import JobKind
from fun_workflow.pipeline.idle import GRACE_MINUTES_BY_KIND, grace_for


def test_every_kind_has_a_grace_and_chat_is_the_longest() -> None:
    assert set(GRACE_MINUTES_BY_KIND) == set(JobKind)
    assert grace_for(JobKind.CHAT) == max(GRACE_MINUTES_BY_KIND.values())
    assert grace_for(JobKind.IMAGE) < grace_for(JobKind.VIDEO), "Flux reloads faster than H3"
    assert grace_for(JobKind.DRAMA) == grace_for(JobKind.VIDEO), "a drama ends on an H3 clip"


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(KeyError):
        grace_for("nope")  # type: ignore[arg-type]
