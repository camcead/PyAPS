"""IFUExGalLS: the pixel index range built from np.where() results must work with NumPy 2 (one-element arrays are no longer
converted to scalars)."""
import numpy as np

from PyAPS.IFUExGalLS import _pixel_index_range


def test_range_from_where_results():
    loglam = np.linspace(3.5, 3.9, 11)
    first = np.where(loglam[2] == loglam)[0]       # array([2])
    last = np.where(loglam[8] == loglam)[0]        # array([8])
    assert list(_pixel_index_range(first, last)) == list(range(2, 9))


def test_range_from_plain_integers():
    assert list(_pixel_index_range(3, 5)) == [3, 4, 5]
