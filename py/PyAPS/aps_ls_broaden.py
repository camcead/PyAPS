"""Broadening of a spectrum with a Gaussian whose width changes from pixel to pixel.

The line-strength modules (IFUExGalLS, MOSExGalLS) broaden every spectrum to the
resolution of the Lick/LIS system. The width of the extra Gaussian depends on the
wavelength (instrument LSF, velocity dispersion), so a plain
``scipy.ndimage.gaussian_filter1d`` (one sigma for the whole array) cannot be used.

Until 2026-11 the modules called ``ppxf.ppxf_util.gaussian_filter1d(spec, sigma,
truncate=..., mode='nearest')``. That function (deprecated in pPXF, signature
``(spec, sig)``) takes neither ``truncate`` nor ``mode``; the call raised
``TypeError: gaussian_filter1d() got an unexpected keyword argument 'truncate'``,
the surrounding ``try/except`` printed a WARNING and kept the spectrum **unbroadened**
(FWHM_FLAG = 3). This module has no dependency on the pPXF or SciPy version:
NumPy only.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Largest temporary window array (number of float64 values) allocated at once.
_MAX_WINDOW_ELEMENTS = 8_000_000


def broaden_variable_sigma(spec, sigma, truncate=4.0, mode='nearest'):
    """Convolve ``spec`` with a Gaussian of dispersion ``sigma[j]`` (pixels) at pixel ``j``.

    Parameters
    ----------
    spec : array_like, shape (n,)
        The spectrum (also usable for an error spectrum).
    sigma : array_like, shape (n,) or scalar
        Gaussian dispersion in pixels for every output pixel. Values at or below
        1e-3 pixel leave the pixel unchanged.
    truncate : float
        Kernel half-width in units of the local ``sigma`` (like scipy's ``truncate``).
        The truncated kernel is renormalised to unit sum. Default 4.
    mode : {'nearest', 'constant', 'reflect'}
        How the spectrum is continued beyond its ends. ``'nearest'`` repeats the
        edge pixel (the behaviour the callers asked for), ``'constant'`` uses zeros
        (what ``ppxf_util.gaussian_filter1d`` does), ``'reflect'`` mirrors.

    Returns
    -------
    ndarray, shape (n,), float64.
    NaN in the input only spreads to the pixels whose kernel reaches the NaN pixel.
    """
    spec = np.asarray(spec, dtype=float)
    if spec.ndim != 1:
        raise ValueError("spec must be one-dimensional")
    n = spec.size
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma must be finite")
    if truncate <= 0:
        raise ValueError("truncate must be positive")
    pad_mode = {'nearest': 'edge', 'constant': 'constant', 'reflect': 'reflect'}.get(mode)
    if pad_mode is None:
        raise ValueError("unknown mode %r" % (mode,))

    sig = np.where(sigma > 1e-3, sigma, 0.0)
    if n == 0 or not np.any(sig > 0):
        return spec.copy()

    p = int(np.ceil(truncate * sig.max()))
    m = 2 * p + 1
    padded = np.pad(spec, p, mode=pad_mode)
    windows = sliding_window_view(padded, m)                 # (n, m) view, no copy
    offsets = np.arange(-p, p + 1, dtype=float)

    out = spec.copy()                                         # sigma == 0 pixels stay as they are
    chunk = max(1, _MAX_WINDOW_ELEMENTS // m)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        s = sig[lo:hi]
        use = s > 0
        if not np.any(use):
            continue
        s = s[use]
        w = np.exp(-0.5 * (offsets[None, :] / s[:, None]) ** 2)
        w[np.abs(offsets)[None, :] > truncate * s[:, None]] = 0.0
        w /= w.sum(axis=1, keepdims=True)
        win = windows[lo:hi][use]
        # a zero-weight pixel must not let a NaN in: only the kernel support counts
        out[lo:hi][use] = np.einsum('ij,ij->i', w, np.where(w > 0, win, 0.0))
    return out


def fwhm_to_sigma_pixels(fwhm_angstrom, wave):
    """Gaussian dispersion in PIXELS of ``wave`` for an extra broadening of ``fwhm_angstrom`` (FWHM, A).

    The line-strength modules broaden the spectrum after it has been rebinned from log to LINEAR
    wavelength (``log_unbinning``), so the pixel is a constant number of Angstrom wide. The old
    expression ``FWHM / wave * c / 2.355 / velscale`` is the width in pixels of the LOG grid and is
    wrong on a linear grid by the factor ``wave / wave_ref`` (``wave_ref`` = dlambda * c / velscale,
    about 4100 A for the WEAVE LIFU range): 18 % too small at 5000 A and 31 % too small at 6000 A.
    Using the local pixel step ``np.gradient(wave)`` is right for a linear grid and for a log grid.
    """
    wave = np.asarray(wave, dtype=float)
    step = np.abs(np.gradient(wave))
    return np.asarray(fwhm_angstrom, dtype=float) / 2.355 / step
