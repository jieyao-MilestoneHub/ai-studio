"""LINE Messaging API ceilings, in one place. Every number ai-studio's
tools take as a parameter (`media.poster(max_bytes=)`,
`media.extract_audio(max_bytes=)`, a provider's `max_output_chars`) comes
from here; ai-studio itself knows nothing about LINE."""

MAX_TEXT_CHARS = 5000
"""A text message object's `text` field."""

PREVIEW_IMAGE_MAX_BYTES = 1_000_000
"""`previewImageUrl` on an image/video message: 1 MB, and a poster over it
does not degrade -- the whole message is rejected."""

AUDIO_MESSAGE_MAX_BYTES = 200 * 1024 * 1024
"""An audio message object (`[reported]`, Messaging API reference). An AAC
track at 128 kbps is ~1 MB/min, so a phone clip never comes close; the
check exists so the failure is ours, not LINE's."""

UNDERSTANDING_MAX_OUTPUT_CHARS = 1600
"""Per answer from an understanding or chat model: two answers under
headings for /說影 (📏 ~700 chars each) plus the wrapper text fit
MAX_TEXT_CHARS with room to spare."""
