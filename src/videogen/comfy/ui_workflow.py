"""Convert a ComfyUI **UI-format** workflow into the API format `/prompt` accepts.

Workflows exported from the ComfyUI canvas — and the example workflows that ship
with custom-node packs — are in the UI format: a node list plus a separate link
array, with widget values stored as a *positional* array per node. The API format
`/prompt` wants `{node_id: {class_type, inputs}}` with inputs keyed by name.

Mapping the positional widget array back to input names needs the node's schema,
so this takes an `/object_info` snapshot rather than guessing. That is also why
this lives here rather than in `editing`: it is a protocol concern.

The subtlety worth knowing: whether an input is a widget or a link is a property
of *this graph*, not of the type. `steps` is an INT widget normally, but in the
stock MiniMax H3 workflow it arrives over a link from a maths node. So the rule
is "wired in this graph → link, otherwise → widget", and only genuinely
link-only types are treated as links unconditionally.
"""

from __future__ import annotations

import copy
from typing import Any

LINK_ONLY_TYPES = frozenset(
    {
        "MODEL", "CLIP", "VAE", "LATENT", "IMAGE", "CONDITIONING", "SAMPLER",
        "SIGMAS", "NOISE", "GUIDER", "AUDIO", "VIDEO", "MASK", "CLIP_VISION",
        "CLIP_VISION_OUTPUT",
    }
)
"""Types that can only arrive over a link.

`INT`, `FLOAT` and `STRING` are deliberately absent. Including them silently
swallowed `noise_seed`, `steps`, `strength` and `fps` the first time this ran —
they are widgets in most graphs, and the ones that *are* wired are already
excluded by the per-graph link check.
"""

NON_EXECUTING_TYPES = frozenset({"MarkdownNote", "Note", "Reroute"})

OUTPUT_TYPES = frozenset({"SaveVideo", "SaveImage", "SaveAudio", "PreviewImage"})


def _widget_names(schema: dict[str, Any], linked: set[str]) -> list[str]:
    """Input names carrying a widget value, in schema order."""
    names: list[str] = []
    for name, spec in list(schema.get("required", [])) + list(schema.get("optional", [])):
        if name in linked:
            continue
        type_ = spec[0] if isinstance(spec, list) and spec else spec
        if isinstance(type_, str) and type_ in LINK_ONLY_TYPES:
            continue
        names.append(name)
    return names


def ui_to_api(ui: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    """Convert a UI workflow to an API graph.

    `object_info` maps class_type -> {"required": [(name, spec), ...],
    "optional": [...]}, as taken from ComfyUI's `/object_info`.
    """
    links = {
        link[0]: (str(link[1]), link[2])
        for link in ui.get("links", [])
        if isinstance(link, list) and len(link) >= 4
    }

    api: dict[str, Any] = {}
    for node in ui.get("nodes", []):
        class_type = node.get("type")
        if not class_type or class_type in NON_EXECUTING_TYPES:
            continue
        schema = object_info.get(class_type)
        if schema is None:
            raise KeyError(
                f"no /object_info schema for node class {class_type!r} — the node pack "
                "that provides it is probably not installed on this ComfyUI"
            )

        inputs: dict[str, Any] = {}
        linked: set[str] = set()
        for inp in node.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is not None and link_id in links:
                source, slot = links[link_id]
                inputs[inp["name"]] = [source, slot]
                linked.add(inp["name"])

        values = node.get("widgets_values") or []
        if isinstance(values, dict):
            inputs.update(values)
        else:
            # Deliberately non-strict: a UI workflow may carry fewer widget
            # values than the schema has widgets, leaving the rest on their
            # defaults. The stock H3 turbo workflow does exactly this — two
            # values for MiniMaxH3TurboLoRA's three widgets.
            for name, value in zip(_widget_names(schema, linked), values, strict=False):
                inputs[name] = value

        api[str(node["id"])] = {"class_type": class_type, "inputs": inputs}

    return api


def prune_unreachable(api: dict[str, Any], roots: set[str] | None = None) -> dict[str, Any]:
    """Drop nodes no output node depends on.

    Needed after replacing wired inputs with literals: the maths and resolution
    helper nodes become orphans, and ComfyUI validates every node in a submitted
    prompt rather than only the reachable ones.
    """
    if roots is None:
        roots = {nid for nid, n in api.items() if n["class_type"] in OUTPUT_TYPES}
    if not roots:
        raise ValueError("no output node found; pass roots explicitly")

    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in api:
            continue
        reachable.add(nid)
        for value in api[nid]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                stack.append(value[0])

    return {nid: node for nid, node in api.items() if nid in reachable}


def set_literal(api: dict[str, Any], class_type: str, key: str, value: Any) -> dict[str, Any]:
    """Replace an input with a literal, returning a copy.

    Use this to cut a helper-node dependency: pointing `length` at a number
    instead of a maths node makes the graph a plain function of the values you
    vary, and `prune_unreachable` then removes the helper.
    """
    api = copy.deepcopy(api)
    for node in api.values():
        if node["class_type"] == class_type:
            node["inputs"][key] = value
            return api
    raise KeyError(f"no {class_type} node in this graph")
