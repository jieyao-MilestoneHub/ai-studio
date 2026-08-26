"""Repo hygiene for the scripts that run on a billing machine.

These are the only files in this project that execute on a host we pay for by
the second, piped in over ssh with no chance to fix a typo. A mistake here is
not caught by mypy, ruff or import-linter, and it is not caught cheaply: the
pod is already running when it fails.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / "deploy").glob("*.sh"))


def _unquoted(line: str) -> str:
    """The parts of a shell line outside single and double quotes."""
    out: list[str] = []
    quote = ""
    for ch in line:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        out.append(ch)
    return "".join(out)


def test_unquoted_ignores_printf_escapes() -> None:
    """The helper above is the whole test's precision, so pin its behaviour."""
    assert chr(92) + "n" not in _unquoted(r"""log() { printf '[setup] %s\n' "$*"; }""")
    assert chr(92) + "n" in _unquoted(r"python3 -c 'x' 2>/dev/null \n  || true")


def test_there_are_deploy_scripts_to_check() -> None:
    """Guard against this whole file silently passing on an empty list."""
    assert SCRIPTS, "expected deploy/*.sh to exist"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_crlf(script: Path) -> None:
    """CRLF is fatal, not cosmetic.

    bash on Linux makes the CR part of each token, so a CRLF script dies on its
    first line with `$'\\r': command not found`. pod_setup.sh was committed this
    way; .gitattributes now pins *.sh to LF, and this is the assertion that the
    pin is working.
    """
    assert b"\r\n" not in script.read_bytes(), (
        f"{script.name} has CRLF endings and will not run on the pod. "
        "Check .gitattributes and re-normalise."
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_parses(script: Path) -> None:
    """`bash -n` is cheap; a syntax error found on the pod is not."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    result = subprocess.run(
        [bash, "-n", str(script)], capture_output=True, text=True, encoding="utf-8"
    )
    assert result.returncode == 0, f"{script.name}: {result.stderr}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_stray_escaped_newline(script: Path) -> None:
    """A literal backslash-n where a line continuation was meant.

    This arrives when a script is written through a heredoc and the escaping is
    off by one. It survived review twice because bash quietly passes the stray
    `n` along as an argument, so the line does the right thing by luck.

    Quote-aware on purpose: `printf '[setup] %s\\n'` is a correct escape and must
    not be flagged. Only an unquoted backslash-n is the bug.
    """
    text = script.read_text(encoding="utf-8")
    offenders = [
        (i, line)
        for i, line in enumerate(text.splitlines(), 1)
        if not line.lstrip().startswith("#") and chr(92) + "n" in _unquoted(line)
    ]
    assert not offenders, f"{script.name}: literal backslash-n on lines " + ", ".join(
        str(i) for i, _ in offenders
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_embedded_python_compiles(script: Path) -> None:
    """Compile the `python -c '...'` blocks these scripts pipe to an interpreter.

    pod_setup.sh verified the H3 nodes with an f-string containing escaped
    quotes -- a SyntaxError. It sat behind `|| die`, so it aborted the run
    *after* the 54.7GB download had been paid for. Nothing else in the toolchain
    looks inside a shell string.
    """
    text = script.read_text(encoding="utf-8")
    blocks, current = [], None
    for line in text.splitlines():
        if current is None:
            if '-c ' + chr(39) in line and line.rstrip().endswith(chr(39)):
                current = []  # opening quote with the body on following lines
            continue
        if line.strip() == chr(39) or line.startswith(chr(39)):
            blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)

    if not blocks:
        pytest.skip(f"no multi-line python -c block in {script.name}")
    for block in blocks:
        compile(block, f"{script.name}:python -c", "exec")


# --------------------------------------------------------------- flux LoRA

# `hf download` keeps the remote filename, so the file lands as
# `lora.safetensors` — a name that says nothing in a directory that also holds
# the H3 turbo LoRA, and that `workflows/flux_dev.json` does not ask for. The
# rename is what makes those two strings the same string, and if it silently
# does not happen ComfyUI reports it only in a log nobody reads at 11:04.

POD_SETUP = REPO / "deploy" / "pod_setup.sh"
FLUX_LORA_NAME = "flux_nsfw_uncensored_v1.safetensors"


def test_the_flux_lora_is_downloaded() -> None:
    body = POD_SETUP.read_text(encoding="utf-8")
    assert "dl Heartsync/Flux-NSFW-uncensored lora.safetensors" in body


def test_the_flux_lora_is_renamed_to_what_the_workflow_loads() -> None:
    body = POD_SETUP.read_text(encoding="utf-8")
    workflow = (REPO / "workflows" / "flux_dev.json").read_text(encoding="utf-8")

    assert FLUX_LORA_NAME in body
    assert FLUX_LORA_NAME in workflow, "the graph asks for a different filename"


def test_a_missing_flux_lora_stops_the_setup_instead_of_being_ignored() -> None:
    """Continuing without it means the pod boots, ComfyUI starts, the first
    image renders, and the adapter was never there — on a billing machine."""
    body = POD_SETUP.read_text(encoding="utf-8")

    rename_block = body[body.index("FLUX_LORA=") :]
    assert "die" in rename_block.split("\n\n")[0], "no failure path after the rename"


def test_the_rename_happens_after_the_downloads_finish() -> None:
    """`dl` backgrounds every download. Renaming before the wait would move a
    file that is still being written, or one that does not exist yet."""
    body = POD_SETUP.read_text(encoding="utf-8")

    assert body.index("weights complete") < body.index("FLUX_LORA=")


def test_the_advertised_download_size_includes_the_lora() -> None:
    """The log line is what an operator watches to know whether a stall is
    normal. It was ~51GB before this LoRA's 0.69GB was added."""
    body = POD_SETUP.read_text(encoding="utf-8")

    assert "starting weight downloads (~52GB)" in body


# ------------------------------------------------- the ComfyUI flag probe

# `deploy/pod_setup.sh` asks ComfyUI which flags it supports rather than
# assuming. It has to: an unrecognised flag is not a warning, it is argparse
# exiting and no ComfyUI at all -- discovered after 52GB of weights have been
# paid for.
#
# The probe is extracted and run rather than reimplemented. `pod_setup.sh` is
# piped into a pod over ssh as a single file, so it cannot be refactored into
# something importable; taking the block out with the same `sed` range the
# comment documents is the closest thing to testing what actually runs.

PROBE_START = "HELP="
PROBE_END = "done"


def _probe_block() -> str:
    """The `HELP=` .. `done` range, minus the line that shells out."""
    lines = POD_SETUP.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(PROBE_START))
    end = next(i for i, line in enumerate(lines[start:], start) if line == PROBE_END)
    body = lines[start : end + 1]
    assert body[0].startswith("HELP="), body[0]
    return "\n".join(body[1:])


def _run_probe(help_text: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = "\n".join(
        [
            "log() { echo \"[setup] $*\"; }",
            f"HELP={help_text!r}".replace("HELP='", "HELP='"),
            _probe_block(),
            'echo "EXTRA=[$EXTRA]"',
        ]
    )
    return subprocess.run(
        [bash, "-c", script], capture_output=True, text=True, encoding="utf-8"
    )


def test_both_flags_are_used_when_comfyui_supports_them() -> None:
    result = _run_probe(
        "usage: main.py [--listen] [--fast-disk] [--use-sage-attention] [--port PORT]"
    )

    assert result.returncode == 0, result.stderr
    assert "EXTRA=[ --fast-disk --use-sage-attention]" in result.stdout


def test_no_flag_is_passed_when_comfyui_supports_none() -> None:
    """The failure this whole block exists to prevent: passing a flag this
    build does not know, and getting no ComfyUI at all."""
    result = _run_probe("usage: main.py [--listen] [--port PORT]")

    assert result.returncode == 0, result.stderr
    assert "EXTRA=[]" in result.stdout
    assert result.stdout.count("skipping") == 2


def test_an_empty_help_output_is_treated_as_supporting_nothing() -> None:
    """`main.py --help` failing entirely must not be read as "all flags fine"."""
    result = _run_probe("")

    assert result.returncode == 0, result.stderr
    assert "EXTRA=[]" in result.stdout
    assert result.stdout.count("skipping") == 2


def test_one_supported_flag_does_not_drag_the_other_along() -> None:
    result = _run_probe("usage: main.py [--listen] [--fast-disk]")

    assert "EXTRA=[ --fast-disk]" in result.stdout
    assert result.stdout.count("skipping") == 1
    assert "--use-sage-attention not supported" in result.stdout


def test_the_flags_are_never_passed_unconditionally() -> None:
    """A structural check on the script itself: every occurrence of a probed
    flag is inside the probe, never on the `main.py` command line."""
    body = POD_SETUP.read_text(encoding="utf-8")
    # The launch is a continued line, so match the whole invocation rather than
    # the first physical line of it -- and skip the pkill that matches the same
    # substring while starting nothing.
    joined = body.replace(chr(92) + "\n", " ")
    launch = [
        chunk
        for chunk in joined.splitlines()
        if "main.py --listen" in chunk and not chunk.lstrip().startswith("pkill")
    ]

    assert launch, "no ComfyUI launch line found"
    for line in launch:
        assert "--fast-disk" not in line, "a probed flag is passed unconditionally"
        assert "--use-sage-attention" not in line
        assert "$EXTRA" in line, "the probe result is not actually used"

