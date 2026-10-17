"""Starter kit for the Catan agent hackathon.

Importing this package applies the determinism fix described in `catan_kit._patch`. Import it
before you create any Game.
"""
from catan_kit import _patch

_patch.apply()

__all__ = ["_patch"]
__version__ = "1.0.0"
