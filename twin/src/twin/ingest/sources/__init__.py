"""Per-source raw-text parsers. SPEC.md §4.1/§4.2 — each returns plain records,
not Fragments; assembling those into Fragments (source_class, split, event_time
precision) is ingest.fragment's job, kept separate so a new source needs only a
parser here, not a copy of the assembly logic."""
