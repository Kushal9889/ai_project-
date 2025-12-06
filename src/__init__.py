"""Compatibility package to mirror top-level modules under the `src` namespace.

This file (and the accompanying shim modules) let code that expects
`src.foo` imports work when modules live at the repository root.
"""

# Intentionally empty; individual shim modules re-export from top-level modules.
