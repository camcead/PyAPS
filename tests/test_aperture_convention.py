"""
Aperture-boundary tests for the A_world / B_world convention.

One definition everywhere: A and B are FULL axis lengths (diameters), the
``width`` / ``height`` of ``regions.EllipseSkyRegion``. An aperture with A=10"
and B=6" therefore contains points up to 5" from the centre along the major
axis and 3" along the minor axis, not 10" and 6".

    cd <PYAPS_DIR> && pytest tests/test_aperture_convention.py -v
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from PyAPS.aps_utils import aperture_sky_region

RA0, DEC0 = 150.0, 2.0
PIX = 0.05 / 3600.0  # deg per pixel


@pytest.fixture(scope="module")
def wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [2000, 2000]
    w.wcs.cdelt = [-PIX, PIX]
    return w


def _inside(region, wcs, d_arcsec, pa_deg):
    """Is the point d_arcsec from the centre, at position angle pa_deg (from
    +Dec towards +RA, i.e. north through east), inside the region?"""
    d = d_arcsec / 3600.0
    dec = DEC0 + d * np.cos(np.radians(pa_deg))
    ra = RA0 + d * np.sin(np.radians(pa_deg)) / np.cos(np.radians(DEC0))
    return bool(region.contains(SkyCoord(ra, dec, unit="deg"), wcs))


def _max_extent(region, wcs, pa_deg):
    offs = np.arange(0.05, 20.0, 0.05)
    ok = [o for o in offs if _inside(region, wcs, o, pa_deg)]
    return max(ok)


def test_circle_diameter_is_full_width(wcs):
    reg = aperture_sky_region(RA0, DEC0, 10.0, 10.0, 0.0)
    assert _inside(reg, wcs, 4.8, 0.0) and _inside(reg, wcs, 4.8, 90.0)
    assert not _inside(reg, wcs, 5.2, 0.0) and not _inside(reg, wcs, 5.2, 90.0)


def test_ellipse_semi_axes_are_half_the_entries(wcs):
    # width (A) lies along the angle direction; with angle=0 the region's
    # width axis is the sky x (RA) axis in the pixel frame.
    reg = aperture_sky_region(RA0, DEC0, 10.0, 6.0, 0.0)
    ext_ra = _max_extent(reg, wcs, 90.0)
    ext_dec = _max_extent(reg, wcs, 0.0)
    assert ext_ra == pytest.approx(5.0, abs=0.1)
    assert ext_dec == pytest.approx(3.0, abs=0.1)


def test_rotation_swaps_the_axes(wcs):
    reg = aperture_sky_region(RA0, DEC0, 10.0, 6.0, 90.0)
    assert _max_extent(reg, wcs, 90.0) == pytest.approx(3.0, abs=0.1)
    assert _max_extent(reg, wcs, 0.0) == pytest.approx(5.0, abs=0.1)


def test_make_patch_array_stores_full_axes_in_degrees():
    from PyAPS.aps_ifu_ExGal import make_patch_array as mp_ex
    from PyAPS.aps_ifu_Gal import make_patch_array as mp_gal

    for mp in (mp_ex, mp_gal):
        row = mp(RA0, DEC0, 10.0, 6.0, 0.1, 1e-4, "GALAXY")
        assert row["A_world"] * 3600.0 == pytest.approx(10.0)
        assert row["B_world"] * 3600.0 == pytest.approx(6.0)


def test_make_patch_array_row_gives_5_and_3_arcsec_extraction(wcs):
    """patch row -> area list exactly as the ExGal loop builds it -> region."""
    from PyAPS.aps_ifu_ExGal import make_patch_array

    row = make_patch_array(RA0, DEC0, 10.0, 6.0, 0.1, 1e-4, "GALAXY")
    area = [row["RA_icrs"], row["DEC_icrs"], row["A_world"] * 3600.0,
            row["B_world"] * 3600.0, row["angle"]]
    reg = aperture_sky_region(*area)
    assert _max_extent(reg, wcs, 90.0) == pytest.approx(5.0, abs=0.1)
    assert _max_extent(reg, wcs, 0.0) == pytest.approx(3.0, abs=0.1)


def test_gaia_mask_row_radius_matches_extraction_radius(wcs):
    """aps_ifu_prepare stores mask_rad_arcsec*2 for a mask star: the region
    built from it must reach exactly mask_rad_arcsec from the star."""
    mask_rad = 3.0
    reg = aperture_sky_region(RA0, DEC0, mask_rad * 2.0, mask_rad * 2.0, 0.0)
    assert _inside(reg, wcs, mask_rad - 0.2, 0.0)
    assert not _inside(reg, wcs, mask_rad + 0.2, 0.0)


class TestGalApertureFactor:
    def setup_method(self):
        from PyAPS.aps_ifu_Gal import resolve_gal_aperture_factor
        self.f = resolve_gal_aperture_factor

    def test_defaults_unchanged(self):
        assert self.f(None, patch_array_mode=False) == 0.5
        assert self.f(None, patch_array_mode=True) == 1.0

    def test_explicit_value_applies_to_both_routes(self):
        assert self.f(0.75, patch_array_mode=False) == 0.75
        assert self.f(0.75, patch_array_mode=True) == 0.75

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite(self, bad):
        with pytest.raises(ValueError):
            self.f(bad, patch_array_mode=False)

    def test_same_aperture_same_selection_on_both_routes(self):
        a = 10.0 / 3600.0
        patch_file_area = a * 3600.0 * self.f(1.0, False)
        patch_array_area = a * 3600.0 * self.f(1.0, True)
        assert patch_file_area == patch_array_area


def test_add_mask_region_typed_radius_is_the_masked_radius(wcs):
    from astropy.table import Table
    from PyAPS.aps_ifu_ExGal import make_patch_array
    from PyAPS.aps_ifu_utils import add_mask_region, ensure_table

    tbl = ensure_table(make_patch_array(RA0, DEC0, 10.0, 6.0, 0.1, 1e-4, "GALAXY"))
    tbl = add_mask_region(tbl, RA0, DEC0, 3.0)
    row = tbl[tbl["type"] == "M"][0]
    reg = aperture_sky_region(RA0, DEC0, float(row["A_world"]) * 3600.0,
                              float(row["B_world"]) * 3600.0, float(row["angle"]))
    assert _inside(reg, wcs, 2.8, 0.0) and not _inside(reg, wcs, 3.2, 0.0)
