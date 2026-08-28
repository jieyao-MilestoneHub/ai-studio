"""LINE webhook signature verification.

The only thing standing between this endpoint and anyone on the internet who
knows the URL. Three details decide whether it actually works, and all three are
easy to get subtly wrong:

1. **The HMAC is over the raw request bytes.** Not the parsed JSON, not a
   re-serialised copy. `await request.json()` followed by `json.dumps()` will
   reorder keys and change whitespace, and the signature will never match. The
   official guidance is explicit: "Don't modify the signature or request body
   string."

2. **The key is the channel *secret*, not the channel access token.** They are
   different strings from different places in the console, and swapping them
   fails in a way that looks like a code bug.

3. **The comparison must be constant-time.** A plain `==` leaks how much of a
   forged signature was correct, one byte at a time.

Reference:
https://developers.line.biz/en/docs/messaging-api/verify-webhook-signature/
"""

from __future__ import annotations

import base64
import hashlib
import hmac

SIGNATURE_HEADER = "x-line-signature"
"""Lowercase on purpose.

LINE warns that header-name casing "may change without notice" and that servers
should treat it case-insensitively. Starlette's header mapping is already
case-insensitive, so looking it up in lowercase is both correct and stable.
"""


def sign(body: bytes, channel_secret: str) -> str:
    """The signature LINE would send for this exact body. Also used in tests."""
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def verify(body: bytes, signature: str | None, channel_secret: str) -> bool:
    """True if `signature` is LINE's signature for `body`.

    Takes `bytes` rather than `str` so a caller physically cannot hand it a
    re-encoded body. Returns False rather than raising on a missing or malformed
    header: an unsigned request is simply not authentic, which is not an
    exceptional condition on a public endpoint.
    """
    if not signature or not channel_secret:
        return False
    return hmac.compare_digest(sign(body, channel_secret), signature)
