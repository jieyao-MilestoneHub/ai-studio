#!/usr/bin/env python3
"""Generate a `TWIN_ADAPTER_ENCRYPTION_KEY`. SPEC.md §8: adapters are
personal data and MUST be stored encrypted (see `core.encryption`).

Run once per twin (per `principal_id`), then put the printed value in
`.env` (gitignored — see twin/CLAUDE.md's §8 guardrail 2). Losing this key
means losing the ability to ever decrypt that principal's adapter weights —
there is no recovery path, by design (a recoverable key would need to be
stored *somewhere* retrievable, which defeats the point).

    uv run python examples/generate_adapter_encryption_key.py
"""

from __future__ import annotations

from twin.core.encryption import generate_key


def main() -> None:
    key = generate_key().decode("utf-8")
    print(f"TWIN_ADAPTER_ENCRYPTION_KEY={key}")
    print()
    print("Add the line above to .env. Back this key up somewhere safe and")
    print("outside version control — it cannot be regenerated or recovered.")


if __name__ == "__main__":
    main()
