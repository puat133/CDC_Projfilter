"""Pickle / un-pickle matplotlib figures for interactive post-editing.

The saved files are plain ``pickle.dump(fig, f)`` of a
``matplotlib.figure.Figure`` object -- no wrapper class, no custom
metadata embedded inside the pickle. That makes them directly loadable
by editors such as
`FigureForge <https://github.com/nogula/FigureForge>`_::

    import pickle
    fig = pickle.load(open('plot.pkl', 'rb'))

so this module is just a thin convenience layer that:

* uses ``pathlib.Path``-style arguments to match the rest of ``utils/``,
* defaults to the ``.pkl`` extension that FigureForge's File→Open dialog
  filters on (``.pickle`` and ``.fig.pickle`` are also accepted on load
  for backward compatibility with older outputs),
* pins LaTeX rendering state per text artist so the LaTeX flag survives
  the round-trip independent of the consumer's ``rcParams``,
* writes an optional ``<path>.rcparams.json`` sidecar that snapshots the
  font / preamble rcParams; :func:`load_figure` re-applies the snapshot
  so a fresh Python session redraws the figure with the same fonts.

matplotlib is **not** imported at module-load time -- the only top-level
imports are ``pickle``, ``pathlib``, ``json``, ``os``, ``sys``,
``logging``, and ``typing``.  matplotlib is pulled in lazily inside the
public functions, so importing this module remains cheap in code paths
that don't use it.

Caveats
-------
1. Figures with lambda callbacks or live event handlers will not pickle.
2. Pickles produced by one major matplotlib release are not guaranteed
   to load cleanly on a different major release.
3. Large data inside the figure (e.g. a 24 M-particle scatter) inflates
   the pickle to roughly the original data size -- pass
   ``rasterized=True`` to the heavy artists before saving if size
   matters.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import pickle
import sys
from typing import TYPE_CHECKING, Any, Iterable, Union

if TYPE_CHECKING:  # pragma: no cover - import only used for type hints
    import matplotlib.figure


__all__ = [
    "save_figure",
    "load_figure",
    "save_figure_with_render",
    "apply_rc_sidecar",
    "DEFAULT_EXT",
    "RC_SIDECAR_SUFFIX",
]


DEFAULT_EXT = ".pkl"
RC_SIDECAR_SUFFIX = ".rcparams.json"

# rcParams keys that affect LaTeX / font rendering. Snapshotted into the
# sidecar so the rendering choice survives a fresh Python session.
_RC_KEYS_TO_SNAPSHOT: tuple[str, ...] = (
    "text.usetex",
    "text.latex.preamble",
    "font.family",
    "font.sans-serif",
    "font.serif",
    "font.size",
    "mathtext.fontset",
    "mathtext.default",
)

_PathLike = Union[str, pathlib.Path]
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_pickle_path(path: _PathLike) -> pathlib.Path:
    """Coerce ``path`` to ``pathlib.Path`` and ensure a recognised suffix.

    Accepts ``.pkl``, ``.pickle`` and ``.fig.pickle`` as-is. Anything else
    (or no suffix at all) gets ``.pkl`` appended (the default).
    """
    p = pathlib.Path(path)
    # Recognise ".pkl", ".pickle", or the legacy compound ".fig.pickle".
    name = p.name.lower()
    if name.endswith(".fig.pickle") or name.endswith(".pickle") or name.endswith(".pkl"):
        return p
    return p.with_name(p.name + DEFAULT_EXT)


def _sidecar_path(pickle_path: pathlib.Path) -> pathlib.Path:
    """Path of the rcParams sidecar JSON next to a pickle file."""
    return pickle_path.with_name(pickle_path.name + RC_SIDECAR_SUFFIX)


# ---------------------------------------------------------------------------
# LaTeX / rcParams helpers (matplotlib imported lazily)
# ---------------------------------------------------------------------------


def _snapshot_rc() -> dict[str, Any]:
    """Snapshot the rcParams keys in :data:`_RC_KEYS_TO_SNAPSHOT` into a
    JSON-serialisable dict using the *current* ``plt.rcParams``."""
    import matplotlib.pyplot as plt  # local import keeps module import cheap

    snap: dict[str, Any] = {}
    for k in _RC_KEYS_TO_SNAPSHOT:
        if k not in plt.rcParams:
            continue
        v = plt.rcParams[k]
        # rcParams sometimes hands back RcParams-internal proxies (e.g.
        # CycleList); convert defensively to JSON-safe primitives.
        if isinstance(v, (list, tuple)):
            v = list(v)
        snap[k] = v
    return snap


def _pin_text_state(fig: "matplotlib.figure.Figure") -> int:
    """Walk every ``Text`` artist in the figure and pin
    ``set_usetex(rcParams['text.usetex'])`` on each. Returns how many
    artists were touched.

    The per-artist ``usetex`` attribute *does* survive pickling, while
    the global ``rcParams['text.usetex']`` does not.  Pinning ensures
    the LaTeX flag travels with the figure into a fresh Python session
    (e.g. FigureForge) even when that session has ``text.usetex=False``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    desired = bool(plt.rcParams.get("text.usetex", False))
    touched = 0
    for artist in fig.findobj(match=Text):
        try:
            artist.set_usetex(desired)
            touched += 1
        except Exception:
            # Some Text subclasses (rare) may not support the flag.
            continue
    return touched


def _atomic_pickle_dump(obj: Any, path: pathlib.Path, *, protocol: int) -> None:
    """Pickle to ``<path>.tmp`` then ``os.replace`` onto ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_figure(
    fig: "matplotlib.figure.Figure",
    path: _PathLike,
    *,
    overwrite: bool = True,
    protocol: int = pickle.HIGHEST_PROTOCOL,
    pin_text_state: bool = True,
    save_rc_sidecar: bool = True,
) -> pathlib.Path:
    """Pickle ``fig`` so it can be re-opened (e.g. in FigureForge).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to pickle.
    path : str or pathlib.Path
        Destination. If the path has no recognised pickle suffix
        (``.pkl``, ``.pickle``, ``.fig.pickle``) then :data:`DEFAULT_EXT`
        (``.pkl``) is appended automatically. Parent directories
        are created as needed.
    overwrite : bool, default True
        Overwrite the destination if it exists. ``False`` raises
        ``FileExistsError`` instead.
    protocol : int, default pickle.HIGHEST_PROTOCOL
        Pickle protocol.
    pin_text_state : bool, default True
        Walk every ``Text`` artist in the figure and call
        ``artist.set_usetex(rcParams['text.usetex'])`` so the LaTeX
        rendering flag survives the pickle independent of the consumer's
        ``rcParams``.
    save_rc_sidecar : bool, default True
        Write a ``<path>.rcparams.json`` sidecar containing a snapshot of
        the rcParams keys in :data:`_RC_KEYS_TO_SNAPSHOT`.
        :func:`load_figure` will re-apply those when it loads the pickle.

    Returns
    -------
    pathlib.Path
        The path the pickle was written to (after suffix normalisation).

    Raises
    ------
    FileExistsError
        ``overwrite=False`` and the target already exists.
    pickle.PicklingError
        The figure contains unpicklable objects (lambda callbacks,
        ``threading.Lock``, ...).
    """
    out = _resolve_pickle_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(out)

    if pin_text_state:
        n_pinned = _pin_text_state(fig)
        _logger.debug("pinned text.usetex on %d artist(s) in %s", n_pinned, out)

    _atomic_pickle_dump(fig, out, protocol=protocol)

    if save_rc_sidecar:
        sidecar = _sidecar_path(out)
        with open(sidecar, "w") as f:
            json.dump(_snapshot_rc(), f, indent=2, sort_keys=True)
        _logger.debug("wrote rcParams sidecar %s", sidecar)

    return out


def load_figure(
    path: _PathLike,
    *,
    apply_rc_sidecar: bool = True,
) -> "matplotlib.figure.Figure":
    """Load a figure produced by :func:`save_figure`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.pkl`` (or ``.pickle`` / legacy ``.fig.pickle``) file.
    apply_rc_sidecar : bool, default True
        If a ``<path>.rcparams.json`` sidecar exists, apply its keys to
        ``plt.rcParams`` after loading. The applied keys are logged to
        stderr at INFO level.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    The returned figure is detached from pyplot's figure manager.  Call
    ``fig.show()`` to display it inline, or feed it back into FigureForge.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    import matplotlib  # noqa: F401  -- ensure mpl is importable
    with open(p, "rb") as f:
        fig = pickle.load(f)

    if apply_rc_sidecar:
        sidecar = _sidecar_path(p)
        if sidecar.exists():
            applied = _apply_rc_sidecar_inner(sidecar)
            if applied:
                _logger.info(
                    "applied rcParams sidecar %s (%d keys: %s)",
                    sidecar.name, len(applied), ", ".join(sorted(applied)),
                )
                # Also surface to stderr for users who haven't configured logging.
                print(
                    f"[fig_pickle] applied rcParams sidecar from {sidecar.name}: "
                    f"{sorted(applied)}",
                    file=sys.stderr,
                )

    return fig


def save_figure_with_render(
    fig: "matplotlib.figure.Figure",
    path: _PathLike,
    *,
    render_fmt: str = "pdf",
    pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
    overwrite: bool = True,
    pin_text_state: bool = True,
    save_rc_sidecar: bool = True,
    **savefig_kwargs: Any,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write both a rendered image (PDF/PNG/SVG/...) AND a pickle.

    The pickle gets the same treatment as :func:`save_figure`
    (per-artist ``usetex`` pinning + optional rcParams sidecar).
    The render uses the *current* rcParams.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or pathlib.Path
        Stem; this routine produces ``<path>.<render_fmt>`` and
        ``<path>.pkl``. Any pickle-recognised suffix on ``path`` is
        stripped before deriving the two final paths so callers can
        pass either ``"plot"`` or ``"plot.pkl"``.
    render_fmt : str, default "pdf"
        Forwarded to ``fig.savefig(..., format=render_fmt)``.
    pickle_protocol : int, default pickle.HIGHEST_PROTOCOL
    overwrite : bool, default True
    pin_text_state : bool, default True
    save_rc_sidecar : bool, default True
    **savefig_kwargs
        Forwarded to ``fig.savefig`` (e.g. ``bbox_inches="tight"``,
        ``transparent=True``, ``dpi=300``).

    Returns
    -------
    (pathlib.Path, pathlib.Path)
        (rendered_path, pickle_path).
    """
    stem_path = pathlib.Path(path)
    # Strip any pickle suffix(es) the caller may have included.
    while stem_path.suffix.lower() in {".pickle", ".pkl", ".fig"}:
        stem_path = stem_path.with_suffix("")

    rendered_path = stem_path.with_name(stem_path.name + f".{render_fmt}")
    pickle_path = stem_path.with_name(stem_path.name + DEFAULT_EXT)

    rendered_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for q in (rendered_path, pickle_path):
            if q.exists():
                raise FileExistsError(q)

    fig.savefig(rendered_path, format=render_fmt, **savefig_kwargs)

    # save_figure handles pinning + sidecar + atomic write.
    save_figure(
        fig,
        pickle_path,
        overwrite=True,  # we already gated on overwrite above
        protocol=pickle_protocol,
        pin_text_state=pin_text_state,
        save_rc_sidecar=save_rc_sidecar,
    )

    return rendered_path, pickle_path


def apply_rc_sidecar(path: _PathLike) -> dict[str, Any]:
    """Read ``<path>.rcparams.json`` (or treat ``path`` itself as the
    sidecar) and apply it to ``plt.rcParams``.

    Useful when launching FigureForge from a small Python wrapper::

        from utils.fig_pickle import apply_rc_sidecar
        apply_rc_sidecar('plots/cross_entropy_comparison.pkl')
        from FigureForge.main import main; main()

    Returns the dict that was applied.
    """
    p = pathlib.Path(path)
    if p.suffix == ".json" and p.name.endswith(RC_SIDECAR_SUFFIX):
        sidecar = p
    else:
        sidecar = _sidecar_path(p)
    if not sidecar.exists():
        raise FileNotFoundError(sidecar)
    return _apply_rc_sidecar_inner(sidecar)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _apply_rc_sidecar_inner(sidecar: pathlib.Path) -> dict[str, Any]:
    """Apply a sidecar JSON to ``plt.rcParams`` and return what was set."""
    import matplotlib.pyplot as plt

    with open(sidecar, "r") as f:
        data: dict[str, Any] = json.load(f)
    applied: dict[str, Any] = {}
    for k, v in data.items():
        try:
            plt.rcParams[k] = v
            applied[k] = v
        except (KeyError, ValueError) as exc:
            _logger.warning("skipping rcParams[%r] = %r (%s)", k, v, exc)
    return applied
