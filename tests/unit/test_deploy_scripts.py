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
    normal. It was ~51GB before this LoRA's 0.69GB was added, then ~52GB
    before the Flux base model's ~17GB was added."""
    body = POD_SETUP.read_text(encoding="utf-8")

    assert "starting weight downloads (~52GB H3 + ~17GB Flux)" in body


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


def _run_probe(help_text: str, *, sageattention_importable: bool = True) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    # $PY is assigned earlier in the real script, outside this extracted
    # range, and the probe now shells out to it to check whether
    # sageattention is actually importable -- not just whether argparse
    # recognises the flag. `true`/`false` stand in for "the import succeeded"
    # / "it didn't", ignoring the `-c ...` argument exactly like the real
    # check only cares about the exit status.
    script = "\n".join(
        [
            "log() { echo \"[setup] $*\"; }",
            f"PY={'true' if sageattention_importable else 'false'}",
            f"HELP={help_text!r}",
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


def test_sage_attention_is_dropped_when_the_package_is_not_installed() -> None:
    """The failure this check exists to prevent: argparse accepting a flag
    whose runtime dependency was never installed, and ComfyUI refusing to
    start at all -- discovered as an empty /object_info response with no
    other clue which flag caused it."""
    result = _run_probe(
        "usage: main.py [--listen] [--fast-disk] [--use-sage-attention] [--port PORT]",
        sageattention_importable=False,
    )

    assert result.returncode == 0, result.stderr
    assert "EXTRA=[ --fast-disk]" in result.stdout
    assert "sageattention is not installed" in result.stdout


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



# ------------------------------------------- the VPS unit set is self-consistent

# `deploy/vps_setup.sh` names its units in one loop and enables them in
# another, and then tells the operator how many to expect in a third place.
# Those three drifted apart once already -- the script generated four timers
# while the next-steps text said "three timers armed" -- so the numbers are now
# derived and compared rather than trusted.

VPS_SETUP = REPO / "deploy" / "vps_setup.sh"

REMOVAL_LOOP = ["open", "drain", "reap", "close", "gc"]
"""Every unit this script has ever installed, which is what it must remove."""


def _phase_loops() -> list[list[str]]:
    """Every `for phase in ...; do` list in the script, in source order.

    Three of them: remove the old set, generate the current set, enable the
    current set.
    """
    import re

    body = VPS_SETUP.read_text(encoding="utf-8")
    return [m.split() for m in re.findall(r"^for phase in (.+?); do$", body, re.M)]


def test_the_script_removes_every_unit_it_has_ever_installed() -> None:
    """The upgrade path, and the reason it is not optional.

    This script is re-run on boxes it already provisioned. Writing only the two
    current timers would leave `ai-studio-open.timer` enabled and firing --
    creating a pod at 03:00 whether or not anyone asked -- and
    `ai-studio-drain.timer` racing the worker for the same queue. An upgrade
    that leaves the thing it replaced still running is worse than no upgrade.
    """
    loops = _phase_loops()
    body = VPS_SETUP.read_text(encoding="utf-8")

    assert loops, "no `for phase in` loop at all"
    assert loops[0] == REMOVAL_LOOP, f"the removal loop is {loops[0]}"
    assert "systemctl disable --now" in body
    assert "rm -f" in body
    assert body.index("rm -f") < body.index("say \"window timers\""), (
        "units are removed after being written, which deletes the new ones"
    )


def test_the_generate_and_enable_loops_use_the_same_unit_list() -> None:
    """A unit generated but never enabled is a file nobody notices is inert."""
    loops = _phase_loops()

    assert len(loops) == 3, f"expected remove/generate/enable, found {len(loops)}"
    generate, enable = loops[1], loops[2]
    assert generate == enable, f"generate={generate} enable={enable}"
    assert generate == ["reap", "close", "gc"]


def test_the_next_steps_text_matches_how_many_timers_are_created() -> None:
    body = VPS_SETUP.read_text(encoding="utf-8")
    count = len(_phase_loops()[1])
    words = {1: "one", 2: "two", 3: "three", 4: "four"}

    assert f"{words[count]} timers armed" in body, (
        f"{count} timers are created; the next-steps text says otherwise"
    )


def test_nothing_on_a_timer_can_open_a_pod() -> None:
    """The whole point of the request-driven worker. A scheduled `session open`
    bills whether or not anybody asked for anything."""
    generate = _phase_loops()[1]
    body = VPS_SETUP.read_text(encoding="utf-8")

    assert "open" not in generate, f"a timer still runs `session open`: {generate}"
    assert "drain" not in generate, "draining is the worker's job now, not a timer's"
    assert "session open" not in body


def test_both_long_running_services_are_created_and_enabled() -> None:
    body = VPS_SETUP.read_text(encoding="utf-8")

    for unit in ("ai-studio.service", "ai-studio-worker.service"):
        assert f"/etc/systemd/system/{unit}" in body, f"{unit} is never written"
        assert f"systemctl enable --now {unit}" in body, f"{unit} is never enabled"


def test_the_worker_is_enabled_after_the_removal_loop() -> None:
    """The removal loop disables `ai-studio-<phase>` units by name. If the
    worker were enabled before it ran, ordering alone would be enough to leave
    the box with nothing that renders."""
    body = VPS_SETUP.read_text(encoding="utf-8")

    assert body.index("for phase in open drain reap close gc") < body.index(
        "systemctl enable --now ai-studio-worker.service"
    )


def test_the_worker_restarts_itself() -> None:
    """It is the only thing that turns a queued request into a pod. If it dies
    at 11:02 and nothing restarts it, the window is silently lost."""
    body = VPS_SETUP.read_text(encoding="utf-8")
    unit = body[body.index("ai-studio-worker.service") : body.index('say "removing')]

    assert "Restart=always" in unit


def test_caddy_reverse_proxies_to_the_loopback_port_the_app_binds() -> None:
    body = VPS_SETUP.read_text(encoding="utf-8")

    assert "reverse_proxy 127.0.0.1:8000" in body
    assert "--host 127.0.0.1 --port 8000" in body, "the app binds a different port"


def test_the_app_port_is_never_opened_to_the_internet() -> None:
    """Only Caddy on loopback should reach 8000; it is what terminates TLS, and
    LINE will not talk to a plain-HTTP webhook. Previously guaranteed by a
    comment alone."""
    body = VPS_SETUP.read_text(encoding="utf-8")

    assert "allow 8000" not in body
    for port in ("22/tcp", "80/tcp", "443/tcp"):
        assert f"ufw allow {port}" in body, f"{port} is no longer opened"


def test_a_missed_close_does_not_fire_late() -> None:
    """`Persistent=true` would run a queued job at boot. Closing is idempotent,
    so this is noise rather than spend -- but it was set deliberately."""
    assert "Persistent=false" in VPS_SETUP.read_text(encoding="utf-8")
