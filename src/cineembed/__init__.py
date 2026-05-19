"""CineEmbed — Multi-modal AE/VAE/DEC for movie metadata clustering.

See docs/superpowers/specs/2026-05-04-modeling-design.md for the full design.
"""

__version__ = "0.1.0"

# Submodules are imported as `from cineembed import data` etc. by tests/notebooks.
# Pyright with `extraPaths: ["src"]` may show "unknown import symbol" until its
# language server re-indexes — IDE-only noise, runtime + pytest are unaffected.
