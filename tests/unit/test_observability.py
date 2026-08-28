"""`core.observability`: the JSONL trace, the bound context, and the two
guarantees that matter in production -- logging never raises into the
pipeline, and configuring twice does not double the handlers."""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import pytest

from ai_studio.core import observability as obs

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def root_restore():
    """configure_logging installs handlers on the root logger; put it back."""
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_a_record_lands_in_jsonl_with_timestamps_and_the_bound_context(tmp_path, root_restore) -> None:
    obs.configure_logging(service="worker", log_dir=tmp_path, level="INFO", stream=io.StringIO())
    log = logging.getLogger("ai_studio.test")

    with obs.bind(job_id=12, token="aB3dEf9x", kind="video"):
        log.info("job done", extra={"seconds": 287.4, "cost_usd": 0.41, "gpu_tier": "4090"})

    files = list((tmp_path / "worker").glob("*.jsonl"))
    assert len(files) == 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl", files[0].name)
    (rec,) = _lines(files[0])
    assert rec["msg"] == "job done" and rec["level"] == "INFO" and rec["service"] == "worker"
    assert rec["logger"] == "ai_studio.test"
    assert rec["job_id"] == 12 and rec["token"] == "aB3dEf9x" and rec["kind"] == "video"
    assert rec["seconds"] == 287.4 and rec["cost_usd"] == 0.41 and rec["gpu_tier"] == "4090"
    # UTC with offset and milliseconds, plus the Taipei rendering of the same instant
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00", rec["ts"])
    assert rec["local"].endswith("+08:00")


def test_the_human_line_carries_local_time_level_logger_and_context(tmp_path, root_restore) -> None:
    out = io.StringIO()
    obs.configure_logging(service="worker", log_dir=None, level="INFO", stream=out)
    with obs.bind(job_id=7, token="tok"):
        logging.getLogger("ai_studio.worker").warning("requeued")
    line = out.getvalue().strip()
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+08:00 WARNING ai_studio\.worker \[job_id=7 token=tok\] requeued$", line), line


def test_bind_nests_and_restores(root_restore) -> None:
    with obs.bind(job_id=1, kind="video"):
        with obs.bind(kind="image", stage="swap"):
            assert obs.current_context() == {"job_id": 1, "kind": "image", "stage": "swap"}
        assert obs.current_context() == {"job_id": 1, "kind": "video"}
    assert obs.current_context() == {}


def test_a_failing_file_handler_never_raises_into_the_caller(tmp_path, root_restore, monkeypatch) -> None:
    """Disk full on logs/ must not kill the worker: the handler disables
    itself for the day and the stderr line still goes out."""
    err = io.StringIO()
    obs.configure_logging(service="worker", log_dir=tmp_path, level="INFO", stream=err)
    handler = next(h for h in logging.getLogger().handlers if isinstance(h, obs.DailyJsonlHandler))
    log = logging.getLogger("ai_studio.test")
    log.info("first")  # opens the file

    class _Full:
        def write(self, _s: str) -> int:
            raise OSError(28, "No space left on device")

        def close(self) -> None: ...

    handler._stream = _Full()  # type: ignore[assignment]
    monkeypatch.setattr(handler, "handleError", lambda record: None)
    log.info("second")  # must not raise
    log.info("third")
    assert "second" in err.getvalue() and "third" in err.getvalue()
    assert handler._disabled_for == obs.local_day()


def test_configure_twice_does_not_double_the_handlers(tmp_path, root_restore) -> None:
    obs.configure_logging(service="worker", log_dir=tmp_path, level="INFO", stream=io.StringIO())
    obs.configure_logging(service="worker", log_dir=tmp_path, level="INFO", stream=io.StringIO())
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_ai_studio_observability", False)]
    assert len(ours) == 2  # one stream + one jsonl


def test_an_unknown_extra_key_is_dropped_not_fatal(tmp_path, root_restore) -> None:
    obs.configure_logging(service="worker", log_dir=tmp_path, level="INFO", stream=io.StringIO())
    logging.getLogger("x").info("m", extra={"weird": object(), "seconds": 1.0})
    (rec,) = _lines(next((tmp_path / "worker").glob("*.jsonl")))
    assert "weird" not in rec and rec["seconds"] == 1.0


def test_the_pod_server_carries_the_same_human_format() -> None:
    """deploy/inference_server.py cannot import this package, so the format
    string is duplicated there. Same habit as the default-question pin."""
    server = (REPO / "deploy" / "inference_server.py").read_text(encoding="utf-8")
    m = re.search(r'^INFERENCE_LOG_FORMAT = "(.*)"$', server, re.M)
    assert m, "deploy/inference_server.py must define INFERENCE_LOG_FORMAT"
    assert m.group(1) == obs.HUMAN_FORMAT


def test_a_caller_extends_the_allow_list_and_bound_context_always_lands(tmp_path: Path) -> None:
    """The request side has vocabulary this package does not (a delivery
    token, a push outcome). It passes those keys at configure time; what it
    binds is emitted regardless."""
    import json as _json

    obs.configure_logging(
        service="webhook", log_dir=tmp_path, level="INFO", stream=io.StringIO(),
        extra_fields=("outcome",),
    )
    log = logging.getLogger("fun_workflow.test")
    with obs.bind(token="aB3d"):
        log.info("accepted", extra={"outcome": "accepted", "stray": "never"})
    line = next(p for p in (tmp_path / "webhook").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[-1]
    rec = _json.loads(line)
    assert rec["token"] == "aB3d" and rec["outcome"] == "accepted"
    assert "stray" not in rec
