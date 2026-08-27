"""Pure data model. Imports nothing else from this package -- except
`observability`, which uses only stdlib `logging`/`contextvars` and lives
here because `core` is the one package every layer may import."""
