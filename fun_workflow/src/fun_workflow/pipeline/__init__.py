"""From a queued row to a delivered result: the SQLite queue (`queue`), the
always-on worker loop (`worker`), one render per kind (`drain`), prompt
conversion on the pod (`convert_worker`), the drama stage machine
(`drama`), and the idle grace each kind earns a quiet pod (`idle`).
"""
