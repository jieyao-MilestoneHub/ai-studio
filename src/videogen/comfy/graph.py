"""Loading ComfyUI API-format workflows and injecting run parameters.

A ComfyUI API graph is a flat map of `node_id -> {class_type, inputs}` with no
stable names, so injecting "the prompt" means knowing which node id holds it.
Guessing by class type breaks the moment a workflow has two text encoders.

Instead a workflow file carries a `_videogen` block declaring where each
parameter lands:

```json
{
  "_videogen": {
    "expect_turbo": true,
    "bindings": {
      "prompt":     ["6", "text"],
      "width":      ["27", "width"],
      "height":     ["27", "height"],
      "length":     ["27", "length"],
      "seed":       ["31", "noise_seed"],
      "steps":      ["31", "steps"],
      "filename":   ["40", "filename_prefix"]
    }
  },
  "6": { "class_type": "CLIPTextEncode", "inputs": { "text": "" } }
}
```

The block is stripped before submission, so ComfyUI never sees it. Declaring
bindings explicitly means a workflow re-export that renumbers nodes fails loudly
at load time instead of silently generating with an empty prompt.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from videogen.comfy.validate import validate_graph
from videogen.core.errors import GraphValidationError, UnknownKeyError

META_KEY = "_videogen"

REQUIRED_BINDINGS = frozenset({"prompt", "width", "height", "length"})
"""Without these a video workflow cannot be driven from a spec at all."""

IMAGE_REQUIRED_BINDINGS = frozenset({"prompt", "width", "height"})
"""A still image has no frame-count binding — there is no `length` to fill."""


class Workflow:
    """A validated, parameterisable ComfyUI workflow."""

    def __init__(
        self,
        graph: dict[str, Any],
        *,
        source: str = "<memory>",
        required_bindings: frozenset[str] = REQUIRED_BINDINGS,
    ) -> None:
        self.source = source
        self._required_bindings = required_bindings
        meta = graph.get(META_KEY, {})
        if not isinstance(meta, dict):
            raise GraphValidationError(f"{source}: {META_KEY} must be an object")

        raw_bindings = meta.get("bindings", {})
        if not isinstance(raw_bindings, dict):
            raise GraphValidationError(f"{source}: {META_KEY}.bindings must be an object")

        self.bindings: dict[str, tuple[str, str]] = {}
        for name, target in raw_bindings.items():
            if not (isinstance(target, list | tuple) and len(target) == 2):
                raise GraphValidationError(
                    f"{source}: binding {name!r} must be [node_id, input_key], got {target!r}"
                )
            self.bindings[str(name)] = (str(target[0]), str(target[1]))

        self.expect_turbo: bool | None = meta.get("expect_turbo")
        self.graph: dict[str, Any] = {k: v for k, v in graph.items() if k != META_KEY}

        self._check_bindings_resolve()
        validate_graph(self.graph, expect_turbo=self.expect_turbo)

    # ------------------------------------------------------------------ load

    @classmethod
    def load(
        cls, path: Path | str, *, required_bindings: frozenset[str] = REQUIRED_BINDINGS
    ) -> Workflow:
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise GraphValidationError(f"workflow not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise GraphValidationError(f"{path}: invalid JSON — {exc}") from None
        if not isinstance(raw, dict):
            raise GraphValidationError(f"{path}: expected a JSON object at the top level")
        return cls(raw, source=str(path), required_bindings=required_bindings)

    # ---------------------------------------------------------------- checks

    def _check_bindings_resolve(self) -> None:
        problems = [
            f"binding {name!r} -> node {node_id!r} does not exist"
            if node_id not in self.graph
            else f"binding {name!r} -> node {node_id!r} has no input {key!r}"
            for name, (node_id, key) in self.bindings.items()
            if node_id not in self.graph
            or key not in self.graph[node_id].get("inputs", {})
        ]
        missing = self._required_bindings - self.bindings.keys()
        if missing:
            problems.append(f"missing required bindings: {sorted(missing)}")
        if problems:
            raise GraphValidationError(
                f"{self.source}: binding problems:\n  - " + "\n  - ".join(problems)
            )

    # --------------------------------------------------------------- inject

    def with_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return a submission-ready copy of the graph with `values` injected.

        Raises on a value with no binding, rather than dropping it. A silently
        ignored `seed` is a run you cannot reproduce and will not notice.
        """
        unknown = values.keys() - self.bindings.keys()
        if unknown:
            raise UnknownKeyError("workflow binding", sorted(unknown), self.bindings.keys())

        graph = copy.deepcopy(self.graph)
        for name, value in values.items():
            node_id, key = self.bindings[name]
            graph[node_id]["inputs"][key] = value
        return graph

    def __repr__(self) -> str:
        return (
            f"Workflow(source={self.source!r}, nodes={len(self.graph)}, "
            f"bindings={sorted(self.bindings)}, expect_turbo={self.expect_turbo})"
        )
