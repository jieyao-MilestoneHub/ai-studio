"""Assembly of rendered clips into the finished piece.

`timeline.resolve_timeline` is where absolute time gets computed, and
`offsets.json` is the only file it is written to -- see
`docs/editing-grammar.md` section 4.1. The ffmpeg side of assembly is
`ai_studio.media.assemble`, which takes the timeline's numbers and computes
none of its own.
"""
