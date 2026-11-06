"""MOSExGalLS.run_ls: a bin that covers only part of the indices must keep the ones it could measure.

``data`` / ``index_names`` hold the SSP-conversion indices (spp == 1) only. When none of them is inside the
spectrum (UV-only fibres of OB 26133) the old code returned NaN for EVERY index of the bin, also for the
UV indices that had been measured. IFUExGalLS never did that.
"""
import numpy as np
import pytest

MOS = pytest.importorskip("PyAPS.MOSExGalLS")


def _setup(tmp_path):
    wave = np.arange(1700.0, 3900.0, 0.2)                 # UV only
    spec = 10.0 + 0.0 * wave
    spec -= 3.0 * np.exp(-0.5 * ((wave - 2800.0) / 6.0) ** 2)
    lick = tmp_path / "bands.config"
    lick.write_text(
        "# b1 b2 b3 b4 b5 b6 b7 names spp origin\n"
        "2750.0 2780.0 2785.0 2815.0 2820.0 2850.0 1 UVline 0 test\n"           # inside, not an SSP index
        "5142.625 5161.375 5160.125 5192.625 5191.375 5206.375 1 Mgb 1 test\n")  # SSP index, outside
    return wave, spec, str(lick)


def test_indices_outside_the_ssp_set_survive(tmp_path):
    wave, spec, lick = _setup(tmp_path)
    cfg = {"MC_LS": 0}
    names = np.array(["UVline", "Mgb"])
    out = MOS.run_ls(wave, spec.copy(), np.full(wave.size, 0.05), [0.0, 0.0], cfg, lick, names,
                     ["Mgb"], "dummy", "dummy", "dummy", "dummy", str(tmp_path), 1, 0, False)
    indices = np.asarray(out[0], dtype=float)
    assert indices.shape == (2,)
    assert np.isfinite(indices[0]) and indices[0] > 1.0      # the UV absorption line is measured
    assert np.isnan(indices[1])                              # Mgb is outside the spectrum: legitimately NaN
