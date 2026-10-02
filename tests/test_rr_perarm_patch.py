"""Synthetic tests of the multi-arm Redrock chi2 modification in aps_rr.py.

No WEAVE data needed. These pin down WHAT the modification does (see
doc/aps_rr.md, "Per-arm chi2 modification"), not that it improves
classification: per-arm vs joint behaviour under weight rescaling, calibration
offsets, missing/degenerate arms, priors and per-camera (arm-specific) columns.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))
from PyAPS import aps_rr  # noqa: E402  (installs the patch at import)

HUGE = 9999999.0
joint = aps_rr._original_calc_zchi2_batch
perarm = aps_rr._patched_calc_zchi2_batch


class Arm:
    def __init__(self, h, n):
        self.wavehash = h
        self.Rcsr = sp.identity(n, format="csr")


def basis(n, shift=0.0):
    x = np.linspace(0, 1, n)
    return np.stack([np.ones(n), x, np.sin(6 * x + shift), np.cos(11 * x + shift)], 1)


def run(solver, f, w, T, nbasis, prior=None, method="PCA", solver_args=None):
    """f, w, T: lists with one entry per arm."""
    arms = [Arm(str(i), len(x)) for i, x in enumerate(f)]
    td = {a.wavehash: t[None] for a, t in zip(arms, T)}
    W, F = np.concatenate(w), np.concatenate(f)
    return solver(arms, td, W, F, W * F, 1, nbasis, solve_matrices_algorithm=method,
                  prior=prior, solver_args=solver_args)


def chi2_at(c, f, w, T):
    return sum(np.dot((fi - Ti @ c) ** 2, wi) for fi, wi, Ti in zip(f, w, T))


@pytest.fixture
def shared():
    rng = np.random.default_rng(1)
    n = (300, 400)
    T = [basis(n[0]), basis(n[1], 0.3)]
    sig = (0.2, 0.1)
    c = np.array([5, 2, 1.0, -0.5])
    f = [Ti @ c + rng.normal(0, s, len(Ti)) for Ti, s in zip(T, sig)]
    w = [np.full(len(Ti), 1 / s**2) for Ti, s in zip(T, sig)]
    return f, w, T, c


def test_joint_switch_is_upstream(shared, monkeypatch):
    f, w, T, _ = shared
    monkeypatch.setattr(aps_rr, "_RR_SOLVER", "joint")
    a = run(perarm, f, w, T, 4)
    b = run(joint, f, w, T, 4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_shared_model_agrees_with_joint(shared):
    f, w, T, c = shared
    a, b = run(joint, f, w, T, 4), run(perarm, f, w, T, 4)
    assert b[0][0] <= a[0][0] + 1e-9            # extra freedom cannot raise chi2
    assert abs(a[0][0] - b[0][0]) / a[0][0] < 0.05
    assert np.allclose(b[1][0], c, atol=0.15)


def test_reported_chi2_is_not_chi2_of_returned_coeff(shared):
    f, w, T, _ = shared
    chi, coeff = run(perarm, f, w, T, 4)
    assert chi2_at(coeff[0], f, w, T) >= chi[0] - 1e-9


@pytest.mark.parametrize("k", [0.01, 0.1, 10, 100])
def test_arm_ivar_rescale_coeff_invariant_chi2_not(shared, k):
    f, w, T, _ = shared
    base = run(perarm, f, w, T, 4)
    scaled = run(perarm, f, [w[0], w[1] * k], T, 4)
    assert np.allclose(base[1], scaled[1], atol=1e-8)       # coefficients invariant
    assert not np.isclose(base[0][0], scaled[0][0], rtol=0.05)  # score is not
    # the joint solver's coefficients DO depend on the weights
    ja, jb = run(joint, f, w, T, 4), run(joint, f, [w[0], w[1] * k], T, 4)
    assert not np.allclose(ja[1], jb[1], atol=1e-8)


def test_calibration_offset_absorbed_by_perarm(shared):
    f, w, T, _ = shared
    g = [f[0], f[1] * 1.15]
    assert run(perarm, g, w, T, 4)[0][0] < 0.5 * run(joint, g, w, T, 4)[0][0]


def test_missing_arm_equals_single_arm(shared):
    f, w, T, _ = shared
    w0 = [w[0], np.zeros_like(w[1])]
    a = run(perarm, f, w0, T, 4)
    b = run(joint, [f[0]], [w[0]], [T[0]], 4)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_all_weights_zero_is_huge(shared):
    f, w, T, _ = shared
    chi, coeff = run(perarm, f, [np.zeros_like(x) for x in w], T, 4)
    assert chi[0] == HUGE and not coeff.any()


def test_uncovered_arm_does_not_kill_fit(shared):
    """Template with no support in one arm (all-zero rows): upstream joint solve
    handles it; the per-arm solve must not return HUGE_CHI2 for the whole target."""
    f, w, T, _ = shared
    T2 = [T[0], np.zeros_like(T[1])]
    chi, coeff = run(perarm, f, w, T2, 4)
    assert chi[0] < HUGE
    ref = run(joint, f, w, T2, 4)[0][0]
    assert chi[0] == pytest.approx(ref, rel=1e-6)


def _percamera(rng, n=(300, 400), sig=(0.2, 0.1)):
    nb = 5

    def arch(m, which):
        x = np.linspace(-1, 1, m)
        A = np.zeros((m, nb))
        A[:, 0] = basis(m)[:, 2] + 2
        A[:, 1 + 2 * which] = 1
        A[:, 2 + 2 * which] = x
        return A

    T = [arch(n[0], 0), arch(n[1], 1)]
    ct = np.array([3.0, 0.4, 0.2, -0.3, 0.1])
    f = [Ti @ ct + rng.normal(0, s, len(Ti)) for Ti, s in zip(T, sig)]
    w = [np.full(len(Ti), 1 / s**2) for Ti, s in zip(T, sig)]
    prior = np.zeros((nb, nb))
    prior[1:, 1:] = np.eye(4) * 1e2
    return f, w, T, ct, prior


def test_percamera_columns_not_diluted():
    f, w, T, ct, prior = _percamera(np.random.default_rng(2))
    chi, coeff = run(perarm, f, w, T, 5, prior=prior)
    assert np.allclose(coeff[0], ct, atol=0.1)               # was ~half before the fix
    assert chi[0] == pytest.approx(chi2_at(coeff[0], f, w, T), rel=0.05)


def test_percamera_bvls_bounds_subset():
    f, w, T, ct, prior = _percamera(np.random.default_rng(3))
    bounds = np.zeros((2, 5))
    bounds[0][1:] = -np.inf
    bounds[1] = np.inf
    chi, coeff = run(perarm, f, w, T, 5, prior=prior, method="BVLS",
                     solver_args={"bounds": bounds})
    assert chi[0] < HUGE and np.allclose(coeff[0], ct, atol=0.1)


def test_diag_hook(shared, monkeypatch):
    f, w, T, _ = shared
    sink = []
    monkeypatch.setattr(aps_rr, "_RR_DIAG", sink)
    chi, coeff = run(perarm, f, w, T, 4)
    assert len(sink) == 1
    assert sink[0]["zchi2_at_mean_coeff"][0] == pytest.approx(chi2_at(coeff[0], f, w, T))
    assert sink[0]["zchi2"][0] == chi[0]
