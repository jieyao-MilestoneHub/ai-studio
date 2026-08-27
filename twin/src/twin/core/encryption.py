"""Symmetric encryption for adapter weight artifacts at rest. SPEC.md §8:
"Adapter 為個資。它可反推行為特徵，MUST 加密儲存" — adapters are personal
data (LoRA weights encode learned behavioral characteristics), so they MUST
be encrypted before they leave local storage, not merely relying on a
storage provider's own at-rest encryption (SPEC.md doesn't name that as
sufficient, and this project's own third-party guardrails elsewhere — §8
guardrail 2 — consistently choose a control this codebase enforces itself
over trusting an external default).

Scope, stated explicitly rather than left implicit: this covers the adapter
*weight artifacts* (everything `twin.train.checkpoint` uploads — both
intermediate training checkpoints, which contain adapter weights per SPEC.md
§7.4's own checkpoint contract, and the final adapter). It does NOT cover
`core.adapter.AdapterManifest` (the JSON metadata record) — that holds
`run_id`/hashes/timestamps/a path, none of which by itself reverse-infers
behavioral characteristics the way the actual weights do. SPEC.md §8 says
"Adapter", not "everything about a run"; this module applies that literally
rather than expanding it.

Mechanism: Fernet (AES-128-CBC + HMAC-SHA256, via the `cryptography`
library) — a well-reviewed, misuse-resistant symmetric primitive. SPEC.md
doesn't decide a specific algorithm or key-management scheme beyond "MUST
encrypt"; this module picks Fernet because it's the standard choice for
"one key, authenticated symmetric encryption of a blob" in Python and
doesn't need this project to hand-roll IV/MAC handling.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

# Re-exported so callers (and tests) don't need their own `from
# cryptography.fernet import InvalidToken` — it's the *decryption* half of
# this module's own contract.
__all__ = ["InvalidToken", "decrypt_bytes", "encrypt_bytes", "generate_key"]


def generate_key() -> bytes:
    """One key per twin (per `principal_id`), generated once by the operator
    and stored outside version control (SPEC.md §8 guardrail 2's gitignore/
    pre-commit hook already covers `twin/adapters/`, but the key itself lives
    in `.env`, alongside `TWIN_GEMINI_API_KEY` — see config.settings).
    See `examples/generate_adapter_encryption_key.py`."""
    return Fernet.generate_key()


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    return Fernet(key).encrypt(data)


def decrypt_bytes(data: bytes, key: bytes) -> bytes:
    """Raises `cryptography.fernet.InvalidToken` on a wrong key or corrupted/
    tampered ciphertext — fails loudly (this project's own house style),
    never silently returns garbage bytes."""
    return Fernet(key).decrypt(data)
