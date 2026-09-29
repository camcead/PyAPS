"""Shared pytest configuration.

Several tests exercise real WEAVE data (L1 exposures, calibration files, L2 products).
Point them at your copy with ``PYAPS_TEST_DATA=<PYAPS_DATA>``. When that variable is not
set, any test that fails only because the placeholder data root does not exist is reported
as *skipped* instead of failed, so the suite is green on a machine without WEAVE data while
still failing on real regressions.
"""
import os

import pytest

_PLACEHOLDER = "<PYAPS_DATA>"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when != "call" or call.excinfo is None:
        return
    if os.environ.get("PYAPS_TEST_DATA"):
        return
    if _PLACEHOLDER in str(call.excinfo.value):
        report.outcome = "skipped"
        report.wasxfail = None
        report.longrepr = (str(item.fspath), item.location[1] or 0,
                           "Skipped: real WEAVE data not available (set PYAPS_TEST_DATA)")
