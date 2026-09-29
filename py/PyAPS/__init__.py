try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development/source installs
    try:
        import importlib.metadata
        __version__ = importlib.metadata.version("PyAPS")
    except Exception:
        # Final fallback - read version.txt directly
        import os
        _root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(_root, "version.txt"), "r") as f:
            __version__ = f.read().strip()


def _load_extensions():
    """Let optional add-on distributions contribute modules to the ``PyAPS`` namespace.

    An add-on registers an entry point in the ``pyaps.extensions`` group whose value is a
    zero-argument callable returning a directory that holds extra ``aps_*.py`` modules.
    That directory is appended to ``PyAPS.__path__``, so ``from PyAPS import aps_xyz``
    works the same whether ``aps_xyz`` ships in this package or in an add-on.

    With no add-on installed (the normal case) this does nothing.
    """
    import os
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="pyaps.extensions") if hasattr(eps, "select") else eps.get("pyaps.extensions", [])
    except Exception:
        return
    for ep in group:
        try:
            path = ep.load()()
        except Exception:
            continue                       # a broken add-on must never break PyAPS itself
        if path and os.path.isdir(path) and path not in __path__:
            __path__.append(path)


_load_extensions()
