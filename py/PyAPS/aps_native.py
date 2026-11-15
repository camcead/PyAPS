"""Native byte order for arrays that come out of FITS files.

FITS stores numbers big-endian, and astropy hands them over that way (dtype ``>f8``). numpy computes with them
correctly, but not in the same way as with native ``float64``: a reduction (``sum``, ``mean``, ``std``, ...) over
byte-swapped data runs through numpy's buffered casting in chunks of 8192 values, a native contiguous array is
reduced in one pairwise pass, so the additions happen in another order and the last digits differ. A smooth fit
does not notice; an ill-conditioned one (a low S/N Voronoi bin, a bound hit) amplifies it to 1e-6 .. 1e-4.

Python's ``multiprocessing`` queues pickle every job, and a pickled non-contiguous slice comes out native. So a pool
worker computes on native data while the serial loop of the same code computes on the big-endian view of the same
numbers: the same input, two results (IFU_ExGal, `--mp_ExGal 1` against 2 or more workers, 2026-11-15; the stellar
re-fit of the last bin of OB 26140 patch 3: sum of the pixel values differs in 8 of 21 bins, the standard deviation
that sets the regularisation in one, and that bin's emission lines change by up to 1e-4).

Rule: an array read from a FITS table that is going to be fitted, summed or compared between runs is made native
once, where it is read, with :func:`native_array`. Then a serial loop, a pool worker and a lane see the same bytes.
"""
import numpy as np


def is_native(a):
    """True when ``a`` is a numpy array whose dtype is in native byte order."""
    return isinstance(a, np.ndarray) and a.dtype.isnative


def native_array(a):
    """Copy of ``a`` in native byte order: same values, shape and memory order, nothing else changed.

    Always a copy, like ``np.array(a)``, so it can replace that call on data that is memory-mapped from a FITS
    file. A native input stays native (same dtype, same bits); a big-endian one is converted value for value.
    """
    a = np.asarray(a)
    return np.array(a, dtype=a.dtype.newbyteorder('='))
