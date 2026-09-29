"""
=====================

redrock wrapper tools for WEAVE DATA
"""

from __future__ import absolute_import, division, print_function

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import datetime
import json
import re
import sys
import traceback
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Column, Table, hstack, join, vstack
from scipy import sparse

from redrock._version import __version__
from redrock.archetypes import All_archetypes
from redrock.results import write_zscan
from redrock.targets import DistTargetsCopy, Spectrum, Target
from redrock.templates import load_dist_templates
from redrock.utils import distribute_work, elapsed, get_mp, getGPUCountMPI
from redrock.zfind import zfind
from redrock.zwarning import ZWarningMask as _RR_ZWarningMask

import PyAPS
from PyAPS import aps_constants
from PyAPS.aps_utils import (
    APSOB,
    add_extra_columns,
    check_and_fix_overlap,
    gen_targlist,
    l1_fileinfo,
    makeR,
    none_or_str,
    print_args,
    str2bool,
)
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS.apsPlot.redrock import MAX_RANKS, make_rrplot

warnings.filterwarnings('ignore')
APSVERS = PyAPS.__version__


#################################################
"""
Python wrapper to run REDROCK code on WEAVE DATA and generate output tables

versions:
 1.0 By A. Molaeinezhad (IAC, July 2019) - First version
 1.2 By A. Molaeinezhad (IAC, August 2019)
 2.0 By A. Molaeinezhad (IAC, November 2019)
 3.0 By A. Molaeinezhad (IAC, October 2020)
 3.1 By A. Molaeinezhad (CASU, May 2021)
 3.2 By A. Molaeinezhad (CASU, Oct 2025)

NOTES:


TODO list:


########### BAISC APS PARAM ###################################################################################################
|    param                |    APS default  |  RRWEAVE EQUIVALENT          |  RRWEAVE DEFAULT  |   availble options/notes      |
infiles (Required)        |        -        |  --infiles (Required)        |        -          |                               |
aps_ids (Optional)        |     None        |  --aps_ids (Optional)        |        None       |                               |
targsrvy (Optional)       |     None        |  --targsrvy (Optional)       |        None       |                               |
targclass (Optional)      |     None        |  --targclass (Optional)      |        None       |                               |
mask_aps_ids (Optional)   |     None        |  --mask_aps_ids (Optional)   |        None       |                               |
area (Optional)           |     None        |  --area (Optional)           |        None       |                               |
mask_areas (Optional)     |     None        |  --mask_areas (Optional)     |        None       |                               |
wlranges (Optional)       |     None        |  --wlranges (Optional)       |        None       |                               |
sens_corr (Optional)      |     True        |  --sens_corr (Optional)      |        True       |                               |
mask_gaps (Optional)      |     True        |  --mask_gaps (Optional)      |        True       |                               |
safe_mask_gaps (Optional) |     False       |  --safe_mask_gaps (Optional) |        True       |                               |
vacuum (Optional)         |     False       |  --vacuum (Optional)         |        True       |                               |
tellurics (Optional)      |     False       |  --tellurics (Optional)      |        False      |                               |
fill_gap (Optional)       |     False       |  --fill_gap (Optional)       |        False      |                               |
arms_ratio (Optional)     |     None        |  --arms_ratio (Optional)     |        None       | for R band in OPR3B  is 0.83  |
join_arms (Optional)      |     False       |  --join_arms (Optional)      |        False      |                               |
funit (Optional)          |     1.0e18      |  USE DEFAULT APS value       |        AS APS     |                               |
offset_gap_pix (Optional) |     10          |  USE DEFAULT APS value       |        AS APS     |                               |
                          |                 |                              |                   |                               |
######### RRWEAVE INITAIL PARAM ################################################################################################
cache_Rcsr  (Optional)    |     -           |  --cache_Rcsr  (Optional)    |         True      |                               |
                          |                 |                              |                   |                               |
######### DEDICATED RRWEAVE PARAM ##############################################################################################
                          |                 |  --outpath   (Required)      |          -        |                               |
                          |                 |  --srvyconf  (Required)      |          -        |                               |
                          |                 |  --templates (Required)      |          -        |                               |
                          |                 |  --headname  (Optional)      |     'headname'    |                               |
                          |                 |  --fig       (Optional)      |        False      |                               |
                          |                 |  --overwrite (Optional)      |        False      |                               |
                          |                 |  --mp        (Optional)      |         2         |                               |
                          |                 |  --archetypes(Optional)      |        None       |                               |
                          |                 |  --zall      (Optional)      |        False      |                               |
                          |                 |  --priors    (Optional)      |        None       |                               |
                          |                 |  --chi2_scan (Optional)      |        None       |                               |
                          |                 |  --debug     (Optional)      |        False      |                               |
                          |                 |  --gpu       (Optional)      |        False      |                               |
                          |                 |  --max_gpuprocs (Optional)   |        2          |                               |
                          |                 |  --ntop  (Optional)          |        1       |                               |
################################################################################################################################



Example:

Short mode:
python3 <PYAPS_DIR>/py/PyAPS/aps_rr.py --infiles <PYAPS_DATA>/star_test/stacked_1002046.fit <PYAPS_DATA>/star_test/stacked_1002045.fit --outpath <PYAPS_DIR>/PyAPS_results/20160903/3294/ --headname stacked_1002046__stacked_1002045 --wlranges 4200.0,6000.0 6000.0,8000.0 --aps_ids 1007,1006 --targsrvy None --targclass None --mask_aps_ids None --area None --mask_areas None --templates <PYAPS_DIR>/PyAPS_templates/templates_RR/ --srvyconf <PYAPS_DIR>/PyAPS/configs/weave_cls.json --join_arms False --mp 2 --archetypes None --zall True --priors None --chi2_scan None --nminima 3 --cache_Rcsr False --debug False --overwrite True --fig True --ntop 1 --sens_corr True --mask_gaps True --safe_mask_gaps True --tellurics False --vacuum True --fill_gap False --arms_ratio 1.0,0.83 --gpu False --max_gpuprocs 2

Full mode:
python3 <PYAPS_DIR>/py/PyAPS/aps_rr.py --infiles <PYAPS_DATA>/star_test/stacked_1002046.fit <PYAPS_DATA>/star_test/stacked_1002045.fit --outpath <PYAPS_DIR>/PyAPS_results/20160903/3294/ --headname stacked_1002046__stacked_1002045 --wlranges 4200.0,6000.0 6000.0,8000.0 --aps_ids 1007,1006 --targsrvy None --targclass None --mask_aps_ids None --area None --mask_areas None --templates <PYAPS_DIR>/PyAPS_templates/templates_RR/ --srvyconf <PYAPS_DIR>/PyAPS/configs/weave_cls.json --join_arms False --mp 2 --archetypes None --zall True --priors None --chi2_scan None --nminima 3 --cache_Rcsr False --debug False --overwrite True --fig True --ntop 1  --sens_corr True --mask_gaps True --safe_mask_gaps True --tellurics False --vacuum True --fill_gap False --arms_ratio 1.0,0.83 --gpu False --max_gpuprocs 2



Note:
How to Solve the issue with the error [too many open file]:
add the following lines to the /etc/security/limits.conf [root access required]

*                soft   nproc            100000
*                hard   nproc            100000
*                soft   nofile           100000
*                hard   nofile           100000

Then, close all open sessions and reset your systemd using:
sysctl -p


History:
10 Jan 2020: Add an example to run the code through command line (By A. Molaeinezhad)
23 Jan 2020: Remove res param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
23 Jan 2020: Remove mode param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
23 Jan 2020: Remove setups param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
23 Jan 2020: ADD targsrvy param. to specific the list of surveys you want to proceed with (By A. Molaeinezhad)
23 Jan 2020: ADD targclass param. to specific the type of targets (coming from targclass) to proceed with (By A. Molaeinezhad)
23 Jan 2020: ADD mask_aps_ids param. to specific the list of aps_ids that you want to be masked from the process (By A. Molaeinezhad)
23 Jan 2020: update examples and readme/header content (By A. Molaeinezhad)
23 Jan 2020: handle exception in case an error occurred in plotting routine
31 Jan 2020: Add a new feature using l1_fileinfo to make sure order or inputs (infiles, wlranges) are OK, as well as double-checking the
             the merge_arms parameters. This checks done at two level 1. at script level (this code itself) and 2. and APSOB level
06 Feb 2020: Add area and mask_areas into the criteria to select fibres/spaxels (By A. Molaeinezhad)
06 Feb 2020: Update working examples
10 Feb 2020: Break the main rrweave function into 4 semi-independent function. Now we have rrweave_worder which is the master function
             and 3 other function mainly deal with generating/preparing the outputs.
             Using this, now user could use this function as a independent too
             to measure redshift. However, still the module to read IFU data is missed.
10 Feb 2020: Update the output structure and add an option 'return_outputs' into rrweave_worker function. So if user set it to True
             then, the rrweave_worker function return scandata, zbest, zspec, zfitall
11 Feb 2020: Add a new param to rrweave_worker named binmode with two possible values [BIN or FIB] to support IFU data
             However, the code, at the moment, is functional just for the FIB mode.
12 Feb 2020: Important: I heavily revised the code to make sure it is compatible with archetypes.
             Before that, we never used archetypes and all measurements were only based on templates
             We have to modify the code to generate model spectra, based on the info [e.g. coefficients] coming from archetypes
             I noticed, as for archetypes the code use a c-like function structure, the code return a bad warning, including
             NumbaDeprecationWarning or NumbaPerformanceWarning. all because the code fails to compile the c-like function inside python
             using @numba.jit inside the rebin.py (a redrock function)
             IMPORTANT: AT the moment, just to suppress it, I added the following line to this code:
             import warnings
             warnings.filterwarnings('ignore')
             which simply ignore all warnings. However, we should check this on LINUX and make sure if it's a good way to proceed.
             However, aniother solution is just to ignore archetypes at the moment (set it to None) and do as before.
             Probably, baecuase of the errors/warnings raising while using archtypes, the code gets slower than usual.
25 Feb 2020: Replace 'FIBREID' with 'APS_ID' (and all other relevant params) to be compatible with all possible L1 structures
25 Feb 2020: Replace NSPEC (int) with TARGID (str), as NSPEC is not available in all L1 structures
25 Feb 2020: Add ylablel to plot. ADD/update units to output tables/spectra.
25 feb 2020: replace bad values in output tables and plots by nan value
25 Feb 2020: Update header of all outputs (tables, spectra) to include units, reference files, sampling ,etc.
25 Feb 2020: Update Versioning of all sub-modules for STAR modules.
04 March 2020: replace the parameter RB_ratio (float) with a more general arms_ratio (list of floats).
04 March 2020: A new feature added to support more than 2 arms/bands (to handle possible R+G+B setup in HR)
04 March 2020: Add two reserved columns for P(Z) and its error into the zftillall and zbest
04 April 2020: Add new check to make sure we exclude all fibres with fib_status other than A:Active from analysis
              This is done at 3 levels 1- pack_2_redrock function in the aps_utils.py
              2- before running rrweave_worker, where we check the input parameters
              3- inside the rrweave_worker function
27 May 2020: Remove the 3rd check on the fib-status from the gen_zbest function, to avoid redundancy
27 May 2020: As requested by Dan Smith from WEAVE-LOFAR, we added CHI2(Z) for all available templates
            on the first search run (COARSE GRIDS). To do this, we added a new function that create
            a Table from the scandata dict and then, join this table with the zbest
28 July 2020: Remove binmode param from the rrweave_worker
28 July 2020: Add a new option to rrweave_worker called collapse.
            If true, it will collapse all spectra into one spectrum and then run redrock on that.
            In this mode, APS_ID for this new single spectrum is -999
            Adding this feature required a few changes in the aps_util (APSOB class) and also pack_2_redrock
            to make sure, it is consistent with this new feature
28 July 2020: changed srvyconf from mandatory to optional parameter (given that for some cases we do not need it)
29 July 2020: Added collapse_fname option to the rrweave_worker and make_rrplot.
            In this way, if it's a collapsed spectrum, we have this option to use an specific name for the outputs.
20 Oct 2020: bugs fixed in output path
20 Oct 2020: in-line documentation updated
28 Dec 2020: Example and demos updated
28 Dec 2020: Fixed a bug related to targsrvy in the aps_utils (affected this routine)
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
5 July 2021: To create and check figdir, we now use Path library instead of old os.path
17 May 2022: Minor changes: zall_fname removed from gen_zbest function
26 Aug 2022: The function to write final fits file separated from the table generators.
30 Aug 2022: 'add_extra_columns' added to be able to add extra columns to the final fits outputs
30 Aug 2022: Now deployed the updated templates for redrock, has two low redshift and high redshift subclass for quasars.
12 Oct 2023: GPU support added to the code. gpu and max_gpuprocs keywords added.  Note, dama does not support GPU at the moment. So we set it to off.
12 Oct 2023: Now the code is able to return top n best redshifts, instead of the top first. user can control it through the ntop= a number between 2 to 5 keyword
12 Oct 2023: CD/CI note: code still fully support the previous format of zbest and redrock output (top best guess only)
15 Oct 2023: single_mode keyword chnaged to ntop
06 Aug 2024: in case we are dealing with collapsed spectra, it is recommended to use cosmic ray cleaning code, as stacking too many spectra together, increase the chance of dealing with cosmic rays
14 Aug 2025: caldir, catdir and configdir added to inputs. So it can now generate LSF from the L1 calibrations insated of default resolution
05 Oct 2025: Fixed SRVY_CLASS handling for ntop>1 mode - proper 2D array creation and assignment
06 Oct 2025: Improved survey classification with robust exact and substring matching
21 Jan 2026: Improved use of TARGCLASS, TARGPROG for decidng about SRVY_CLASS
21 Jan 2026: Tellurics correction added to the default mode in the script params
23 Jan 2026: pack_2_redrock now can be run in parallel
05 Jun 2026: This is a major change in the code, as now we are using archetypes instead of PCA templates only.
09 Jun 2026: We added a fallback to handle cases where chi2 from archtypes are signgificantly worse than the PCA templates, which can happen for some targets. In this case, we use the PCA templates to get the best redshift and then, we use the archetypes to get the best coefficients for that redshift. This way, we can have the best of both worlds, and we can still use the archetypes to get the best coefficients, even if the chi2 is not good enough.
09 Jun 2026: zbest for multiple mode now caple of handling targ_class and make sure at least one of the top ntop answers is related to the target class (e.g. if targ_class is STAR, then at least one of the top ntop answers should be related to STAR). This is done by a new function called _ensure_srvy_class_represented, which is called after the initial zbest is generated. This function checks if the top ntop answers are related to the target class, and if not, it injects a new answer that is related to the target class, using the information from the zfitall table. This way, we can ensure that the top ntop answers are always related to the target class, which can help with the classification and redshift determination.
02 Aug 2026: New PyAPS.apsPlot platform introduced. make_rrplot moved to PyAPS.apsPlot.redrock (Plotly-based), so the same figure code serves the pipeline's static PNG export, a future live web viewer, and the WEAVE Operational Hub - aps_rr just imports it. Fixed a real bug caught in review: gen_zspec previously called with a single fixed rank=0, so MODEL_RR_<arm> only ever held the rank-0 model and every rank panel in the diagnostic plot showed the SAME model curve overlaid on different titles/colours. gen_zspec now computes and stores one model per available rank (MODEL_RR_<arm> is now float[NRANK,NW], was float[NW] - see doc/weave_datamodel_v8.md), so each rank panel shows its own genuine best-fit model. Verified by re-running MOS OB 5011 APS_ID=327 before/after (kept on disk as PyAPS_data/L2/20230514/5011_V2 pre-fix and 5011_V3 post-fix, vs the original 5011/ for comparison).
"""


# ================================================================
# FORCE FORK MODE — ensures monkey patches are inherited by
# multiprocessing worker processes. Must be set before any
# Process or Pool is created.
# On Linux this is already the default but we set it explicitly
# to be safe.
# ================================================================
import multiprocessing as _mp

try:
    _mp.set_start_method('fork')
except RuntimeError:
    pass  # Already set elsewhere — fine on Linux


import redrock.zscan as _rr_zscan

_original_calc_zchi2_batch = _rr_zscan.calc_zchi2_batch



def _patched_calc_zchi2_batch(spectra, tdata, weights, flux, wflux, nz,
                               nbasis, solve_matrices_algorithm=None,
                               solver_args=None, use_gpu=False,
                               fullprecision=True, prior=None):
    """
    Patched version of calc_zchi2_batch that solves each arm
    independently and sums chi2 for multi-arm targets.

    Key design decisions:
    - chi2 is summed across arms (per-arm independent solve)
    - zcoeff is a simple EQUAL-WEIGHT mean of per-arm coefficients
      regardless of IVAR normalisation mode or arm pixel count
      This ensures fitz always starts from a neutral point and
      the result is completely independent of IVAR scaling
    """

    # Single arm or GPU: use original implementation unchanged
    if len(spectra) <= 1 or use_gpu:
        if use_gpu:
            print("  [APS PATCH] WARNING: per-arm chi2 patch not applied "
                  "in GPU mode — results may be suboptimal")
        return _original_calc_zchi2_batch(
            spectra, tdata, weights, flux, wflux, nz, nbasis,
            solve_matrices_algorithm=solve_matrices_algorithm,
            solver_args=solver_args, use_gpu=use_gpu,
            fullprecision=fullprecision, prior=prior)

    import numpy as np
    from redrock.zscan import HUGE_CHI2, solve_matrices

    zchi2  = np.zeros(nz, dtype=np.float64)
    zcoeff = np.zeros((nz, nbasis), dtype=np.float64)

    # Pre-compute per-arm pixel offsets into the concatenated arrays
    arm_offsets = []
    offset = 0
    for s in spectra:
        n = tdata[s.wavehash].shape[1]
        arm_offsets.append((s, offset, offset + n))
        offset += n

    # Pre-compute total weight for prior scaling
    total_weight = weights.sum()

    for i in range(nz):
        chi2_total     = 0.0
        coeff_sum      = np.zeros(nbasis, dtype=np.float64)
        n_arms_valid   = 0
        valid          = True

        for arm_idx, (s, i_start, i_end) in enumerate(arm_offsets):
            key    = s.wavehash
            Tb_arm = s.Rcsr.dot(tdata[key][i, :, :])  # (n_pix, nbasis)
            w_arm  = weights[i_start:i_end]
            wf_arm = wflux[i_start:i_end]
            f_arm  = flux[i_start:i_end]

            if w_arm.sum() == 0:
                continue

            M_arm = Tb_arm.T.dot(np.multiply(w_arm[:, None], Tb_arm))
            y_arm = Tb_arm.T.dot(wf_arm)

            # Scale prior by this arm's fractional weight so the
            # total prior effect across all arms equals the original.
            # e.g. 2 equal arms → each gets prior × 0.5
            if prior is not None:
                arm_frac = w_arm.sum() / total_weight \
                           if total_weight > 0 else 1.0 / len(arm_offsets)
                M_arm = M_arm + prior * arm_frac
                # if arm_idx == 0 and i == 0:
                #     print(f"  [APS PATCH] prior scaled by arm fraction "
                #           f"{arm_frac:.4f} for arm {arm_idx}")

            try:
                c_arm = solve_matrices(
                    M_arm, y_arm,
                    solve_algorithm=solve_matrices_algorithm
                    if solve_matrices_algorithm else 'PCA',
                    solver_args=solver_args,
                    use_gpu=False)
            except (np.linalg.LinAlgError, NotImplementedError):
                chi2_total = HUGE_CHI2
                valid = False
                break

            model_arm = Tb_arm.dot(c_arm)
            chi2_arm  = np.dot((f_arm - model_arm) ** 2, w_arm)

            chi2_total   += chi2_arm

            # --------------------------------------------------------
            # EQUAL-WEIGHT coefficient mean (not IVAR-weighted).
            #
            # Using arm_weight = w_arm.sum() here would bias zcoeff
            # toward whichever arm has higher scaled IVAR, making
            # the fitz starting point sensitive to IVAR normalisation
            # mode (balanced vs hybrid vs pixels).
            #
            # Equal weighting (count arms, not pixels) ensures zcoeff
            # is completely independent of IVAR scaling and gives fitz
            # a neutral starting point regardless of normalisation.
            # --------------------------------------------------------
            coeff_sum    += c_arm
            n_arms_valid += 1

        if valid and n_arms_valid > 0:
            zchi2[i]  = chi2_total
            zcoeff[i] = coeff_sum / n_arms_valid
        else:
            zchi2[i]  = HUGE_CHI2
            zcoeff[i] = 0.0

    return zchi2, zcoeff



# Apply the patch
_rr_zscan.calc_zchi2_batch = _patched_calc_zchi2_batch

# Also patch the reference used inside zscan itself
# (needed because calc_zchi2 calls calc_zchi2_batch by local name)
import redrock.zscan

redrock.zscan.calc_zchi2_batch = _patched_calc_zchi2_batch

print("  [APS] Redrock coarse scan patched: per-arm independent chi2 solve")
print("        This fixes joint PCA coefficient bias for WEAVE blue+red arms")





##########################################################
def classify_target_comprehensive(targclass, targprog, targsrvy,
                                  survey_config=None,
                                  default_class=None,
                                  verbose=False):
    """
    Classify a target using hierarchical priority system

    Priority order:
    1. TARGCLASS - Direct classification (STAR, QSO, GALAXY keywords)
    2. TARGPROG - Program-based classification (STAR, QSO, GALAXY keywords)
    3. TARGSRVY - Survey code lookup via config file
    4. default_class - Fallback (usually None to keep original CLASS)

    Parameters:
    -----------
    targclass : str or array
        Target classification from catalog (e.g., 'WD', 'STAR', 'ELG')
    targprog : str or array
        Target program (e.g., 'STELLAR_POPULATIONS', 'QSO_SURVEY')
    targsrvy : str or array
        Target survey code (e.g., 'WS2024A2-007')
    survey_config : dict
        Configuration dictionary from load_survey_classification
    default_class : str or None
        Fallback if no match found (None keeps original CLASS)
    verbose : bool
        Print matching information

    Returns:
    --------
    str or None : The survey classification, or None if no override

    Notes:
    ------
    - NEBULA types are classified as GALAXY (processed by extragalactic modules)
    - Keyword matching is case-insensitive and uses substrings
    - Empty/None/NULL values are skipped
    """

    # Classification keywords (case-insensitive, substring matching)
    STAR_KEYWORDS = ['STAR', 'WD', 'CV', 'MS', 'STELLAR', 'OB', 'GIANTS', 'DWARFS']
    QSO_KEYWORDS = ['QSO', 'QUASAR', 'AGN', 'BLAZAR', 'SEYFERT']
    GALAXY_KEYWORDS = ['GALAXY', 'GALAXIES', 'ELG', 'LRG', 'BCG', 'NEBULA', 'NEBULAE',
                       'HII', 'EMISSION', 'EXTRAGAL']

    def normalize_value(val):
        """Convert input to clean string"""
        if isinstance(val, (list, np.ndarray)):
            if len(val) > 0:
                val = val[0]
            else:
                return None

        if val is None:
            return None

        val_str = str(val).strip().upper()

        # Check for empty/invalid values
        if not val_str or val_str in ['NONE', 'NULL', '', 'NAN']:
            return None

        return val_str

    def contains_keywords(text, keywords):
        """Check if text contains any keyword"""
        if text is None:
            return False
        text_upper = text.upper()
        for keyword in keywords:
            if keyword.upper() in text_upper:
                return keyword
        return None

    # Normalize all inputs
    targclass_norm = normalize_value(targclass)
    targprog_norm = normalize_value(targprog)
    targsrvy_norm = normalize_value(targsrvy)

    if verbose:
        print(f"  Input: TARGCLASS='{targclass_norm}', TARGPROG='{targprog_norm}', TARGSRVY='{targsrvy_norm}'")

    # PRIORITY 1: Check TARGCLASS
    if targclass_norm:
        # Check for STAR
        star_match = contains_keywords(targclass_norm, STAR_KEYWORDS)
        if star_match:
            if verbose:
                print(f"  → SRVY_CLASS='STAR' (from TARGCLASS, matched '{star_match}')")
            return 'STAR'

        # Check for QSO
        qso_match = contains_keywords(targclass_norm, QSO_KEYWORDS)
        if qso_match:
            if verbose:
                print(f"  → SRVY_CLASS='QSO' (from TARGCLASS, matched '{qso_match}')")
            return 'QSO'

        # Check for GALAXY (includes NEBULA)
        galaxy_match = contains_keywords(targclass_norm, GALAXY_KEYWORDS)
        if galaxy_match:
            if verbose:
                print(f"  → SRVY_CLASS='GALAXY' (from TARGCLASS, matched '{galaxy_match}')")
            return 'GALAXY'

    # PRIORITY 2: Check TARGPROG
    if targprog_norm:
        # Check for STAR
        star_match = contains_keywords(targprog_norm, STAR_KEYWORDS)
        if star_match:
            if verbose:
                print(f"  → SRVY_CLASS='STAR' (from TARGPROG, matched '{star_match}')")
            return 'STAR'

        # Check for QSO
        qso_match = contains_keywords(targprog_norm, QSO_KEYWORDS)
        if qso_match:
            if verbose:
                print(f"  → SRVY_CLASS='QSO' (from TARGPROG, matched '{qso_match}')")
            return 'QSO'

        # Check for GALAXY
        galaxy_match = contains_keywords(targprog_norm, GALAXY_KEYWORDS)
        if galaxy_match:
            if verbose:
                print(f"  → SRVY_CLASS='GALAXY' (from TARGPROG, matched '{galaxy_match}')")
            return 'GALAXY'

    # PRIORITY 3: Check TARGSRVY via config
    if targsrvy_norm and survey_config is not None:
        # Use the existing classify_target_by_survey function
        srvy_cls = classify_target_by_survey(targsrvy_norm, survey_config, default_class=None)
        if srvy_cls:
            if verbose:
                print(f"  → SRVY_CLASS='{srvy_cls}' (from TARGSRVY via config)")
            return srvy_cls

    # No match found
    if verbose:
        print(f"  → SRVY_CLASS=None (no match, keeping original CLASS)")
    return default_class



##########################################################
def load_survey_classification(srvyconf):
    """Load and validate survey classification config

    Parameters:
    -----------
    srvyconf : str
        Path to JSON configuration file

    Returns:
    --------
    dict : Survey configuration dictionary
    """
    if not os.path.exists(srvyconf):
        raise FileNotFoundError(f'Survey Config file not found: {srvyconf}')

    with open(srvyconf, 'r') as f:
        config = json.load(f)

    return config


##########################################################
def classify_target_by_survey(targsrvy_value, survey_config, default_class=None):
    """
    Classify a target based on exact match or substring matching in TARGSRVY value

    Priority:
    1. Exact match of full TARGSRVY string (case-insensitive)
    2. Substring match for survey codes (case-insensitive)

    Parameters:
    -----------
    targsrvy_value : str or array
        The TARGSRVY value from the target
    survey_config : dict
        Configuration dictionary from load_survey_classification
    default_class : str or None
        Class to return if no match found (None keeps original CLASS)

    Returns:
    --------
    str or None : The survey classification, or None if no override
    """
    # Handle array values
    if isinstance(targsrvy_value, (list, np.ndarray)):
        if len(targsrvy_value) > 0:
            targsrvy_value = targsrvy_value[0]
        else:
            return default_class

    # Normalize the survey value
    targsrv_str = str(targsrvy_value).strip()
    targsrv_upper = targsrv_str.upper()

    # Empty or invalid values
    if not targsrv_str or targsrv_upper in ['NONE', 'NULL', '']:
        return default_class

    # First: Try exact match (case-insensitive)
    for classification, survey_codes in survey_config.items():
        for survey_code in survey_codes:
            if targsrv_str == survey_code or targsrv_upper == survey_code.upper():
                return classification

    # Second: Try substring match (case-insensitive)
    for classification, survey_codes in survey_config.items():
        for survey_code in survey_codes:
            if survey_code.upper() in targsrv_upper:
                return classification

    # No match found
    return default_class


##########################################################
# make_rrplot now lives in PyAPS.apsPlot.redrock, built on the shared
# spectrum_overlay_figure() (Plotly) used across the new PyAPS.apsPlot
# platform, so the same figure code path serves the pipeline's static
# PNG export, the live web viewer, and the WEAVE Operational Hub.
# Imported at module top as `from PyAPS.apsPlot.redrock import make_rrplot`.

##########################################################

def gen_scandata_table(scandata):

    # recover aps_ids from the scandata dictionary
    scan_data_id=list(scandata.keys())
    # recover the templates available and create a new dict, where keys are template names and values
    # are the corrected templated names, to be used to generate new column names
    tmpl_dict={t: t.replace(':::','_') for t in list(scandata[scan_data_id[0]].keys())}


    ### GENERATE OUTPUT TABLE STRUTURE
    columns = ['APS_ID']

    # loop over all available templates
    for tmpl_dict_vals in list(tmpl_dict.values()):
        columns.append('CZZ_%s' % tmpl_dict_vals)
        columns.append('CZZ_CHI2_%s' % tmpl_dict_vals)

    scan_dict = OrderedDict()
    for c in columns:
            scan_dict[c]=[]

    ### FILL THE scan_dict with scandata
    for scn_id in scan_data_id:
        scan_dict['APS_ID'].append(scn_id)

        for tmpl_val in tmpl_dict.keys():
            scan_dict[ 'CZZ_%s' % tmpl_dict[tmpl_val] ].append(scandata[scn_id][tmpl_val]['redshifts'])
            scan_dict[ 'CZZ_CHI2_%s' % tmpl_dict[tmpl_val] ].append(scandata[scn_id][tmpl_val]['zchi2'])

    scan_table= Table(scan_dict)
    scan_table.sort(['APS_ID'])

    return scan_table
##########################################################

def rr_write_fits(outfile, ztable, dtemplates, darchetypes=None, match_table=None):
    """Write output Table to outfile

    Args:
        outfile (str): output file.
        ztable (Table): input table (astropy).
    """
    header = fits.Header()
    header['RR_V'] = (__version__, 'Redrock version')
    header['APSVERS'] = (APSVERS,'APS version')
    header['APS_RR_V'] = (aps_constants.__aps_rr_version__,'PyAPS Redrock wrapper version')



    ## add any extra columns if necessary from the match_table (if given)
    if match_table is not None:
        ztable = add_extra_columns(ztable, match_table=match_table)



    ztable.meta['RR_V'] = (__version__, 'Redrock version')
    ztable.meta['APS_V'] = (APSVERS,'PyAPS version')
    ztable.meta['APS_RR_V'] = (aps_constants.__aps_rr_version__,'PyAPS Redrock wrapper version')

    # header['MIN_WAVE'] = lmin
    # header.comments['MIN_WAVE']= 'Min wavelength (Ang), used by APS_REDROCK'
    # header['MAX_WAVE'] = lmax
    # header.comments['MAX_WAVE']= 'MAX wavelength (Ang), used by APS_REDROCK'



    template_version = {t._template.full_type:t._template._version for t in dtemplates}
    archetype_version = None
    if not darchetypes is None:
        archetype_version = {name:arch._version for name, arch in darchetypes.items()}


    for i, fulltype in enumerate(template_version.keys()):
        header['TEMNAM'+str(i).zfill(2)] = fulltype
        header['TEMVER'+str(i).zfill(2)] = template_version[fulltype]

        ztable.meta['TEMNAM'+str(i).zfill(2)] = fulltype
        ztable.meta['TEMVER'+str(i).zfill(2)] = template_version[fulltype]

    if not archetype_version is None:
        for i, fulltype in enumerate(archetype_version.keys()):
            header['ARCNAM'+str(i).zfill(2)] = fulltype
            header['ARCVER'+str(i).zfill(2)] = archetype_version[fulltype]

            ztable.meta['ARCNAM'+str(i).zfill(2)] = fulltype
            ztable.meta['ARCVER'+str(i).zfill(2)] = archetype_version[fulltype]

    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU(header=header))
    hx.append(fits.convenience.table_to_hdu(ztable))
    hx.writeto(os.path.expandvars(outfile), overwrite=True)
    return

##########################################################


def read_spectra(infiles, aps_ids=None, targsrvy= None, targclass = None, mask_aps_ids = None , area=None, mask_areas=None, wlranges=None,
  cache_Rcsr=None, sens_corr=None, mask_gaps=None,safe_mask_gaps=None, vacuum=None,
   tellurics=None, fill_gap=None, arms_ratio=None, join_arms=None, collapse=False, catdir=None, caldir=None, configdir=None, ncpus=1, skysub_mask_residuals=True,
   spaxel_weighted_lsf=False, extinction_corr=False, extinction_ebv_scale=1.0, extinction_mapdir=None):

    """Read targets from a list of spectra files
    Args:
         infiles (list): input files
    Returns:
        tuple: (APStargets,APSmeta , setups) where targets is a list of Target objects and
    This setups is the output setups, after checking for availability of data in the requested arms.
    """

    # in case we are dealing with collapsed spectra, it is recommended to use cosmic ray cleaning code, as stacking too many spectra together, increase the chance of dealing with cosmic rays
    if collapse:
        crr = True
    else:
        crr = False

    # fix any overlap in the wavelength ranges, if given, to avoid any issue with redrock when dealing with overlapping spectra:
    if wlranges is not None and len(wlranges) >= 2:
        wlranges = check_and_fix_overlap(wlranges)
        print(f"  Wavelength ranges after overlap fix: {wlranges}")



    # READ DATA and put them in the APSOBJ OBJECT
    _APSOB = APSOB(infiles, aps_ids=aps_ids, targsrvy= targsrvy, targclass = targclass, mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps , vacuum=vacuum, tellurics=tellurics,
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, collapse=collapse, crr=crr, catdir=catdir, caldir=caldir, configdir=configdir,
        # was hardcoded `skysub_mask_residuals=True` -- silently ignored
        # this function's own received parameter, so callers requesting
        # False (e.g. to disable it for a diagnostic run) had no effect.
        # Harmless in practice since the default is also True, but a real
        # dead/misleading parameter -- found 28 Aug 2026 reviewing the
        # full aps_rr/aps_utils preprocessing chain end to end.
        skysub_mask_residuals=skysub_mask_residuals,
        spaxel_weighted_lsf=spaxel_weighted_lsf, extinction_corr=extinction_corr,
        extinction_ebv_scale=extinction_ebv_scale, extinction_mapdir=extinction_mapdir)


    # spec = _APSOB.data()[_APSOB.apstoid(778)].spectra[1]

    # Determine resolution mode based on data type. FOR LIFU related modes we use global average LSF
    if 'IFU' in str(_APSOB.fmode()).upper():
        resolution_mode = 'global_average'
        print(f"INFO: IFU data detected. Using resolution_mode='{resolution_mode}' for LSF handling.")
    else:
        resolution_mode = 'fiber_specific'
        print(f"INFO: Using resolution_mode='{resolution_mode}' for LSF handling.")

    # return _APSOB.pack_2_redrock(cache_Rcsr=cache_Rcsr)
    return _APSOB.pack_2_redrock(cache_Rcsr=cache_Rcsr, use_interpolated_fwhm=True, fwhm_key='fwhm', resolution_mode=resolution_mode, ncpus=ncpus)
##########################################################

def gen_zfitall(zfit, apsmeta,comm = None):

    for colname in zfit.colnames:
        if colname.islower():
            zfit.rename_column(colname, colname.upper())

    zfit.rename_column('SPECTYPE', 'CLASS')
    zfit.rename_column('SUBTYPE', 'SUBCLASS')
    zfit.sort(['APS_ID', 'ZNUM'])

    ## Add this level, we first add TARGID, APS_ID and CNAME to the new table
    ## and then, add other columns
    ## This is just to avoid reshuffling columns, we observed in the new version of astropy and numpy
    zfitall=Table()
    zfitall.add_columns([zfit['APS_ID'],zfit['TARGID'], zfit['CNAME']])
    zfitall.add_columns([zfit['Z'], zfit['ZERR'], zfit['ZWARN'], zfit['ZNUM'], zfit['CLASS'],
    zfit['SUBCLASS'], zfit['TARGSRVY'],zfit['TARGCLASS'],zfit['TARGPROG'],zfit['FIB_STATUS'], zfit['SNR'], zfit['CHI2'],
    zfit['DELTACHI2'], zfit['NCOEFF'],zfit['COEFF'],zfit['NPIXELS'],zfit['ZZ'],zfit['ZZCHI2']])


    ## Create two columns, reserved for P(z) and its error. Later it fill be filled with right values
    ## However, how it works, needs to be clarified with Dan Smith from WEAVE_LOFAR
    ## Please note, we just copied a column structure of a column in zfitall and then assigned a new new to that.
    ## We just used zfitall['z'] as reference, as it has dtype =float
    # zfitall.add_column(zfitall['Z'], name='PZ')
    # zfitall['PZ'][:] = np.nan
    # zfitall.add_column(zfitall['Z'], name='PZERR')
    # zfitall['PZERR'][:] = np.nan



    ## ADD META to ZBEST table
    zfitall.meta['EXTNAME'] = 'ZFITALL'
    ## ADD Original filenames (infiles) as meta to the fits file (As provinces)
    for n_province, province in enumerate(apsmeta['files']):
        zfitall.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    zfitall.meta['CSB_RR'] = (apsmeta['stitched'], 'Combines Spectral Bands Status for REDROCK')



    return zfitall

#########################################################################################

# STAR/QSO/GALAXY keyword lists — kept identical to the inline mapping in
# _ensure_srvy_class_represented (that function's own copy is left as-is to
# avoid touching its already-validated behaviour); this module-level copy
# is for _apply_archetype_fallback's targeting-corroboration check, added
# after the GA-LRDISC investigation below.
_TARGETING_STAR_KW   = ['STAR', 'WD', 'CV', 'MS', 'STELLAR', 'OB', 'GIANTS', 'DWARFS']
_TARGETING_QSO_KW    = ['QSO', 'QUASAR', 'AGN', 'BLAZAR', 'SEYFERT']
_TARGETING_GALAXY_KW = ['GALAXY', 'GALAXIES', 'ELG', 'LRG', 'BCG', 'NEBULA',
                         'NEBULAE', 'HII', 'EMISSION', 'EXTRAGAL']


def _targeting_class_hint(targclass_val=None, targprog_val=None, targsrvy_val=None):
    """STAR/QSO/GALAXY (or None) from targeting info, checked in priority
    order TARGCLASS > TARGPROG > TARGSRVY -- same convention as
    _ensure_srvy_class_represented. Used by _apply_archetype_fallback to
    tell "archetype spuriously flipped a genuine extragalactic target"
    apart from "archetype correctly flipped a genuine star that the PCA
    coarse scan misjudged" -- see that function's docstring."""
    def _match(val):
        if val is None:
            return None
        s = str(val).strip().upper()
        if not s or s in ('NONE', 'NULL', ''):
            return None
        for kw in _TARGETING_STAR_KW:
            if kw in s:
                return 'STAR'
        for kw in _TARGETING_QSO_KW:
            if kw in s:
                return 'QSO'
        for kw in _TARGETING_GALAXY_KW:
            if kw in s:
                return 'GALAXY'
        return None

    for val in (targclass_val, targprog_val, targsrvy_val):
        hint = _match(val)
        if hint is not None:
            return hint
    return None

#########################################################################################
def _ensure_srvy_class_represented(zbest_table, zfitall_cp,
                                    ntop=3):
    """
    Ensure the survey classification (SRVY_CLASS, derived from
    TARGCLASS/TARGPROG/TARGSRVY) is represented in at least one
    of the ntop ranks in zbest_table.

    If none of the top ntop ranks match SRVY_CLASS, replace the
    last rank (ntop-1) with the best matching rank from zfitall.

    This guarantees that if targeting says STAR, at least one STAR
    answer appears in the output — even if Redrock ranks all top
    answers as GALAXY or QSO.

    Parameters
    ----------
    zbest_table : Table — stacked ntop table (after ZWARN reordering)
    zfitall_cp  : Table — full zfitall (without ZZ/ZZCHI2, with ZNUM)
    ntop        : int

    Returns
    -------
    zbest_table : Table — modified in place
    n_injected  : int — number of targets where injection was applied
    """
    n_injected = 0

    # SRVY_CLASS not yet set at this point — derive it directly
    # from TARGCLASS using the same keyword matching as
    # classify_target_comprehensive, but inline and lightweight.
    # We only need a coarse match: STAR/QSO/GALAXY.

    STAR_KW   = ['STAR', 'WD', 'CV', 'MS', 'STELLAR',
                 'OB', 'GIANTS', 'DWARFS']
    QSO_KW    = ['QSO', 'QUASAR', 'AGN', 'BLAZAR', 'SEYFERT']
    GALAXY_KW = ['GALAXY', 'GALAXIES', 'ELG', 'LRG', 'BCG',
                 'NEBULA', 'NEBULAE', 'HII', 'EMISSION',
                 'EXTRAGAL']

    def _targclass_to_srvy(targclass_val):
        if targclass_val is None:
            return None
        s = str(targclass_val).strip().upper()
        if not s or s in ['NONE', 'NULL', '']:
            return None
        for kw in STAR_KW:
            if kw in s:
                return 'STAR'
        for kw in QSO_KW:
            if kw in s:
                return 'QSO'
        for kw in GALAXY_KW:
            if kw in s:
                return 'GALAXY'
        return None

    has_targclass = 'TARGCLASS' in zbest_table.colnames
    has_targprog  = 'TARGPROG'  in zbest_table.colnames
    has_targsrvy  = 'TARGSRVY'  in zbest_table.colnames

    id_columns = ['APS_ID', 'TARGID', 'CNAME']

    for idx in range(len(zbest_table)):
        aps_id = zbest_table['APS_ID'][idx]

        # ----------------------------------------------------------------
        # Derive SRVY_CLASS for this target
        # ----------------------------------------------------------------
        srvy_cls = None

        # Priority: TARGCLASS > TARGPROG > TARGSRVY
        for col, has_col in [('TARGCLASS', has_targclass),
                              ('TARGPROG',  has_targprog),
                              ('TARGSRVY',  has_targsrvy)]:
            if not has_col:
                continue
            val = zbest_table[col][idx]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                val = val[0]
            srvy_cls = _targclass_to_srvy(val)
            if srvy_cls is not None:
                break

        if srvy_cls is None:
            # No targeting classification — nothing to enforce
            continue

        # ----------------------------------------------------------------
        # Check if SRVY_CLASS already present in any of the ntop ranks
        # ----------------------------------------------------------------
        current_classes = np.array(zbest_table['CLASS'][idx]).ravel()
        already_present = any(
            str(c).strip().upper() == srvy_cls
            for c in current_classes[:ntop])

        if already_present:
            continue

        # ----------------------------------------------------------------
        # SRVY_CLASS not represented — find best matching rank in zfitall
        # ----------------------------------------------------------------
        targ_zfitall = zfitall_cp[zfitall_cp['APS_ID'] == aps_id]

        if len(targ_zfitall) == 0:
            continue

        # Find best matching rank: SRVY_CLASS match,
        # ZWARN=0 preferred, lowest chi2 among matches
        best_idx  = None
        best_chi2 = np.inf

        for ri in range(len(targ_zfitall)):
            row_class = str(targ_zfitall['CLASS'][ri]).strip().upper()
            row_zwarn = int(targ_zfitall['ZWARN'][ri])
            row_chi2  = float(targ_zfitall['CHI2'][ri])
            row_znum  = int(targ_zfitall['ZNUM'][ri])

            if row_class != srvy_cls:
                continue

            # Skip ranks already in top ntop
            # (they clearly don't match so this is a safety check)
            if row_znum < ntop:
                continue

            # ZWARN=0 strongly preferred
            chi2_weighted = row_chi2 if row_zwarn == 0 \
                            else row_chi2 * 1.5

            if chi2_weighted < best_chi2:
                best_chi2 = chi2_weighted
                best_idx  = ri

        if best_idx is None:
            # No matching rank found beyond current ntop
            # (may all be within ntop already — recheck)
            for ri in range(len(targ_zfitall)):
                row_class = str(targ_zfitall['CLASS'][ri]).strip().upper()
                row_zwarn = int(targ_zfitall['ZWARN'][ri])
                row_chi2  = float(targ_zfitall['CHI2'][ri])
                if row_class != srvy_cls:
                    continue
                chi2_weighted = row_chi2 if row_zwarn == 0 \
                                else row_chi2 * 1.5
                if chi2_weighted < best_chi2:
                    best_chi2 = chi2_weighted
                    best_idx  = ri

        if best_idx is None:
            # Redrock found no SRVY_CLASS match at all — skip
            print(f"  SRVY_CLASS injection: APS_ID={aps_id} "
                  f"no {srvy_cls} rank found in zfitall — skipped")
            continue

        # ----------------------------------------------------------------
        # Replace rank ntop-1 (last slot) with the matching rank
        # Rank 0 and 1 are preserved — only the last slot is replaced
        # ----------------------------------------------------------------
        inject_row   = targ_zfitall[best_idx]
        last_slot    = ntop - 1

        cols_to_replace = ['Z', 'ZERR', 'ZWARN', 'CLASS',
                            'SUBCLASS', 'CHI2', 'DELTACHI2',
                            'COEFF', 'NCOEFF', 'NPIXELS']

        for col in cols_to_replace:
            if col not in zbest_table.colnames:
                continue
            if col not in zfitall_cp.colnames:
                continue
            try:
                zbest_table[col][idx][last_slot] = inject_row[col]
            except Exception:
                pass

        n_injected += 1
        print(f"  SRVY_CLASS injection: APS_ID={aps_id} "
              f"SRVY_CLASS={srvy_cls} not in top {ntop} "
              f"[{', '.join(str(c).strip() for c in current_classes[:ntop])}] "
              f"→ injected {srvy_cls} z={float(inject_row['Z']):.5f} "
              f"chi2={float(inject_row['CHI2']):.1f} "
              f"into rank {last_slot}")

    print(f"\n  SRVY_CLASS injection summary: "
          f"{n_injected}/{len(zbest_table)} targets injected")

    return zbest_table, n_injected

##########################################################
def _apply_archetype_fallback(zfitall, scandata, apsmeta,
                               degradation_threshold=1.5,
                               targeting_ratio_relax=3.0,
                               comm=None):
    """
    Detect targets where archetype fine fitting gave a different or
    degraded result compared to the coarse PCA scan, and fall back
    to the PCA coarse scan class/redshift.

    Three trigger conditions (any is sufficient):

    1. CLASS MISMATCH — archetype gave a different class than the
       coarse PCA scan winner. GALAXY/QSO mismatches at the same z
       are suppressed (AGN host). Any other mismatch triggers --
       *unless* targeting independently corroborates the archetype's
       class over the coarse scan's (see "Targeting corroboration"
       below), which suppresses this specific trigger only.

    2. RATIO TRIGGER — archetype chi2 is significantly worse than
       the coarse PCA chi2 for the same class (threshold widened by
       targeting_ratio_relax under targeting corroboration -- see below).

    3. Z SHIFT TRIGGER — archetype found a different chi2 minimum
       than the coarse PCA scan within the same class. Triggered by
       dv(arch vs coarse_best_z) > DV_SHIFT_THRESHOLD km/s.
       Uses velocity-based threshold to handle all redshifts.

    Targeting corroboration
    ------------------------
    Originally (any mismatch involving STAR always triggers) this was a
    blanket, direction-blind rule added to stop archetype fitting from
    spuriously flipping genuine galaxies to STAR with catastrophic
    redshifts. But it is symmetric where the real failure isn't: it also
    silently discards cases where the archetype is *right* and the coarse
    PCA scan is wrong -- confirmed via a real GA-LRDISC OB (20260122,
    stack_3134457__stack_3134456) where genuine, TARGCLASS='STAR' targets
    at low Galactic latitude (b~1 deg, so significantly reddened) lose the
    PCA coarse scan to GALAXY outright (independent of archetypes: the
    10-coefficient GALAXY PCA basis's extra shape freedom out-fits the
    5-coefficient STAR basis's fixed, unreddened continuum shape). An
    archetype STAR fit -- built from real stellar spectra, not a small PCA
    basis -- is exactly the kind of answer this fallback should let
    through in that situation, not revert.

    This uses TARGCLASS/TARGPROG/TARGSRVY (already columns on `zfitall`,
    via `_targeting_class_hint`, same priority order as
    `_ensure_srvy_class_represented`) as an independent corroborating
    signal: when the archetype's class agrees with targeting AND the
    coarse PCA scan's class does NOT, the class-mismatch trigger is
    suppressed and the ratio-trigger threshold is widened by
    `targeting_ratio_relax` (default 3x) rather than fully disabled --
    a genuinely bad archetype fit (large degradation) still reverts. The
    z-shift trigger is untouched (it only fires within one shared class,
    orthogonal to this). Targets with no resolvable targeting hint, or
    where targeting does NOT support the archetype's class, keep the
    original direction-blind behaviour exactly as before.

    Verified on the real catastrophic-redshift regression set this
    trigger was built against (WC OB 20250630/16287,
    stack_3095664__stack_3095663, catastrophic_redshifts_apsmod.txt /
    PyAPS_local/<investigation_dir>): the two genuine galaxy-flipped-to-STAR cases in
    that set (FIBREID/APS_ID 791, 938) are UNCHANGED by this addition
    (TARGCLASS=GALAXY there, archetype says STAR, so targeting_override
    never applies) -- no regression. Those two are not actually fixed by
    this trigger even in its original form, though: both already hit
    "no valid same-class rank found" in the fallback search before this
    change, i.e. coarse PCA and archetype independently agree on the
    wrong class there, so there is no disagreement for this function to
    catch either way -- a PCA-coarse-level degeneracy needing a different
    fix (same category as GA-LRDISC APS_ID 613/6 above).

    Archetype z improvement (same-class redshift, not class)
    -----------------------------------------------------------
    A second, related asymmetry found while validating the above: within
    a single agreed-upon class, if the archetype refit lands on a very
    different redshift (large dv vs the coarse PCA scan), the *unmodified*
    z-shift trigger always treats this as suspected aliasing and reverts
    toward whatever zfitall row sits closest to the coarse PCA's OWN z --
    even when the archetype's chi2 is decisively BETTER than the coarse
    scan's own best. Confirmed on the same WC OB, APS_ID 348 and 618: the
    archetype found z matching literature almost exactly (z=0.0432 vs
    Z_LIT=0.04323; z=0.0425 vs Z_LIT=0.04247) with lower chi2 than the
    coarse scan's own best, but the fallback discarded it anyway and
    landed on a third, unrelated, still-wrong z~0.40 by searching near
    the coarse scan's own (worse) reference z.

    Fix: when `arch_chi2 <= coarse_best_chi2` at the same class
    (`archetype_improved_same_class`), the z-shift trigger is suppressed
    and `z_reference` is set to the archetype's own z rather than the
    coarse/PCA-refined z -- a genuine redshift refinement is trusted
    instead of being penalised for having moved far from a worse answer.
    New ZWARN bit 25 (APS_ARCHETYPE_Z_IMPROVED) marks every target where
    this applied, for audit.

    Fallback strategy:
    ------------------
    Find the zfitall rank whose redshift is closest to z_reference,
    among rows matching the coarse scan class, ZWARN<=2 only (tier 1).
    If no such row exists, retry allowing rows whose only extra Redrock
    flag is BAD_MINFIT (tier 2, bit 26 APS_MINFIT_RELAXED -- see below).

    z_reference selection:
      z_pca (parabola fit) if it confirms the coarse grid point
                           (dv(coarse-PCA) < DV_PCA_COARSE_MAX)
      coarse_best_z        otherwise

    ZWARN encoding:
    ---------------
    Bits 0-15  : Redrock native + reserved (never modified here)
    Bits 16-23 : APS-specific fallback history flags
      Bit 16 (  65536): APS_FALLBACK_CLASS_MISMATCH
      Bit 17 ( 131072): APS_FALLBACK_RATIO
      Bit 18 ( 262144): APS_FALLBACK_Z_SHIFT
      Bit 19 ( 524288): APS_FALLBACK_SWAP_APPLIED
      Bit 20 (1048576): APS_FALLBACK_NO_VALID_RANK
      Bit 21 (2097152): APS_PCA_UNRELIABLE
      Bit 22 (4194304): APS_PCA_CONFIRMS_COARSE
      Bit 23 (8388608): APS_Z_REFERENCE_IS_PCA
      Bit 24 (16777216): APS_TARGETING_OVERRIDE (class-mismatch trigger
                          suppressed / ratio threshold widened because
                          targeting corroborated the archetype's class
                          over the coarse PCA scan's -- see "Targeting
                          corroboration" above)
      Bit 25 (33554432): APS_ARCHETYPE_Z_IMPROVED (z-shift trigger
                          suppressed and z_reference set to the
                          archetype's own z, because its chi2 was at
                          least as good as the coarse PCA scan's own
                          best at the same class -- a genuine redshift
                          refinement, not aliasing)
      Bit 26 (67108864): APS_MINFIT_RELAXED (the fallback rank-
                          substitution search found no same-class
                          candidate passing the strict ZWARN<=2 filter,
                          but found one whose only extra Redrock flag is
                          BAD_MINFIT (bit 10 -- uncertain parabola fit to
                          the chi2 minimum, not a wrong-redshift flag) --
                          used as a second-tier candidate rather than
                          silently keeping a worse-chi2 wrong-class
                          answer. See "Second-tier BAD_MINFIT relaxation"
                          below.)

    ZWARN is extended in place — Redrock bits 0-15 are preserved.

    Parameters
    ----------
    zfitall              : astropy Table
    scandata             : dict from zfind
    apsmeta              : dict
    degradation_threshold: float (default 1.5)

    Returns
    -------
    zfitall    : astropy Table — modified in place
    n_fallback : int
    """
    n_fallback   = 0
    n_checked    = 0
    fallback_log = []

    # ---- constants ----
    C_LIGHT            = 299792.458   # km/s
    ZWARN_MAX_ALLOWED  = 2            # ZWARN Redrock bits 0-1 only
    PLAUSIBLE_PAIRS    = {('GALAXY', 'QSO'), ('QSO', 'GALAXY')}
    Z_AGREE_THRESHOLD  = 0.01
    DV_SHIFT_THRESHOLD = 3000.0       # km/s
    DV_PCA_COARSE_MAX  = 1000.0       # km/s

    # ---- APS ZWARN bits (16-23) ----
    APS_ZWARN_CLASS_MISMATCH  = 65536     # bit 16
    APS_ZWARN_RATIO           = 131072    # bit 17
    APS_ZWARN_Z_SHIFT         = 262144    # bit 18
    APS_ZWARN_SWAP_APPLIED    = 524288    # bit 19
    APS_ZWARN_NO_VALID_RANK   = 1048576   # bit 20
    APS_ZWARN_PCA_UNRELIABLE  = 2097152   # bit 21
    APS_ZWARN_PCA_CONFIRMS    = 4194304   # bit 22
    APS_ZWARN_REF_IS_PCA      = 8388608   # bit 23
    APS_ZWARN_TARGETING_OVERRIDE = 16777216   # bit 24
    APS_ZWARN_ARCHETYPE_Z_IMPROVED = 33554432 # bit 25
    APS_ZWARN_MINFIT_RELAXED = 67108864       # bit 26

    aps_ids = np.unique(zfitall['APS_ID'])

    for aps_id in aps_ids:
        aps_id_int = int(aps_id)

        if aps_id_int not in scandata:
            continue

        n_checked += 1

        # ----------------------------------------------------------------
        # Get archetype best result (ZNUM=0)
        # ----------------------------------------------------------------
        targ_rows = zfitall[zfitall['APS_ID'] == aps_id]
        targ_rows = targ_rows[np.argsort(targ_rows['ZNUM'])]

        if len(targ_rows) == 0:
            continue

        arch_chi2  = float(targ_rows['CHI2'][0])
        arch_z     = float(targ_rows['Z'][0])
        arch_class = str(targ_rows['CLASS'][0]).strip()
        arch_dchi2 = float(targ_rows['DELTACHI2'][0])

        # ----------------------------------------------------------------
        # Targeting hint (TARGCLASS > TARGPROG > TARGSRVY), for the
        # targeting-corroboration carve-out below — see "Targeting
        # corroboration" in the docstring.
        # ----------------------------------------------------------------
        def _first(col):
            if col not in targ_rows.colnames:
                return None
            v = targ_rows[col][0]
            return v[0] if isinstance(v, (list, np.ndarray)) and len(v) else v

        targeting_hint = _targeting_class_hint(
            _first('TARGCLASS'), _first('TARGPROG'), _first('TARGSRVY'))

        # ----------------------------------------------------------------
        # Get coarse PCA best chi2 across all templates
        # ----------------------------------------------------------------
        coarse_best_chi2  = np.inf
        coarse_best_z     = np.nan
        coarse_best_class = ''

        for ft in scandata[aps_id_int].keys():
            zz   = scandata[aps_id_int][ft]['redshifts']
            chi  = scandata[aps_id_int][ft]['zchi2']
            imin = chi.argmin()
            if chi[imin] < coarse_best_chi2:
                coarse_best_chi2  = chi[imin]
                coarse_best_z     = zz[imin]
                coarse_best_class = ft.split(':::')[0]

        if np.isinf(coarse_best_chi2) or coarse_best_chi2 <= 0:
            continue

        # ----------------------------------------------------------------
        # PCA fine fit reliability check
        # ----------------------------------------------------------------
        pca_check = _pca_reliability_check(
            scandata, aps_id_int,
            coarse_best_class, coarse_best_z)

        z_pca         = pca_check['z_pca']
        pca_reliable  = pca_check['reliable']
        dv_pca_coarse = pca_check['dv_coarse'] \
                        if np.isfinite(pca_check['dv_coarse']) \
                        else np.inf

        pca_confirms_coarse = (pca_reliable and
                               dv_pca_coarse < DV_PCA_COARSE_MAX)

        z_reference = z_pca if pca_confirms_coarse else coarse_best_z

        # ----------------------------------------------------------------
        # mask for ZNUM=0 of this target — used for ZWARN updates
        # ----------------------------------------------------------------
        mask_znum0 = ((zfitall['APS_ID'] == aps_id) &
                      (zfitall['ZNUM']   == 0))

        # ----------------------------------------------------------------
        # Check trigger conditions
        # ----------------------------------------------------------------
        degradation = arch_chi2 / coarse_best_chi2

        # Same-class case where the archetype's own fit is at least as
        # good as the coarse PCA scan's own best chi2 at its own
        # (possibly very different) redshift. A large z jump alongside a
        # genuine chi2 improvement is the archetype doing exactly its
        # job -- refining to a better redshift -- not aliasing to a
        # spurious minimum, so it should not be penalised the same way a
        # z jump alongside a *worse* fit would be. Confirmed on real data
        # (WC OB 20250630/16287, APS_ID 348/618): archetype chi2 was
        # decisively lower than the coarse scan's own best, at a redshift
        # matching literature almost exactly, but the unconditional
        # z-shift trigger discarded it in favour of a third, unrelated,
        # still-wrong z found by searching for whatever zfitall row
        # happened to sit closest to the (worse) coarse-PCA reference z.
        archetype_improved_same_class = (
            arch_class == coarse_best_class and arch_chi2 <= coarse_best_chi2)

        if archetype_improved_same_class:
            z_reference = arch_z

        # Condition 1: class mismatch (pair-aware)
        pair = (coarse_best_class, arch_class)
        dz   = abs(arch_z - coarse_best_z)

        # Targeting corroboration: archetype's class agrees with
        # independent targeting info, coarse PCA's does not. This is the
        # "archetype is plausibly correcting the coarse scan" case (see
        # docstring) — suppress the class-mismatch trigger for it, and
        # give the ratio trigger a wider berth, rather than the original
        # blanket revert-on-any-STAR-mismatch rule.
        targeting_override = (
            targeting_hint is not None and
            arch_class == targeting_hint and
            coarse_best_class != targeting_hint)

        if arch_class == coarse_best_class:
            class_mismatch = False

        elif pair in PLAUSIBLE_PAIRS:
            z_agree        = dz < Z_AGREE_THRESHOLD
            class_mismatch = not z_agree

        elif targeting_override:
            class_mismatch = False

        else:
            class_mismatch = True

        # Condition 2: ratio trigger (threshold widened under targeting
        # corroboration — a genuinely much worse archetype fit still
        # reverts, this only gives a plausible-but-not-dominant archetype
        # answer room to stand when targeting backs it)
        effective_ratio_threshold = (
            degradation_threshold * targeting_ratio_relax
            if targeting_override else degradation_threshold)
        ratio_trigger = (degradation > effective_ratio_threshold)

        # Condition 3: z shift within same class
        dv_arch_coarse = abs(arch_z - coarse_best_z) / \
                         (1.0 + coarse_best_z) * C_LIGHT \
                         if np.isfinite(coarse_best_z) else 0.0

        z_shift_trigger = (arch_class == coarse_best_class) and \
                          (dv_arch_coarse > DV_SHIFT_THRESHOLD) and \
                          not archetype_improved_same_class

        # ----------------------------------------------------------------
        # Always set PCA reliability bits regardless of trigger
        # ----------------------------------------------------------------
        if mask_znum0.sum() > 0:
            if not pca_reliable:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_PCA_UNRELIABLE)
            if pca_confirms_coarse:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_PCA_CONFIRMS)
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_REF_IS_PCA)
            # Set regardless of whether a trigger ends up firing below --
            # this marks every target where targeting corroboration was
            # actually considered, for QC/audit of how often this new
            # carve-out is exercised and on what (survey, SNR, etc.).
            if targeting_override:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_TARGETING_OVERRIDE)
            if archetype_improved_same_class:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_ARCHETYPE_Z_IMPROVED)

        # ----------------------------------------------------------------
        # Skip if no trigger
        # ----------------------------------------------------------------
        if not (class_mismatch or ratio_trigger or z_shift_trigger):
            if targeting_override:
                print(f"  Targeting override: APS_ID={aps_id_int} "
                      f"targeting={targeting_hint} arch={arch_class} "
                      f"(coarse PCA said {coarse_best_class}) — "
                      f"kept archetype answer, degradation={degradation:.2f}x "
                      f"(effective threshold {effective_ratio_threshold:.2f}x)")
            if archetype_improved_same_class and dv_arch_coarse > DV_SHIFT_THRESHOLD:
                print(f"  Archetype z improvement: APS_ID={aps_id_int} "
                      f"class={arch_class} coarse_z={coarse_best_z:.5f} "
                      f"(chi2={coarse_best_chi2:.1f}) -> arch_z={arch_z:.5f} "
                      f"(chi2={arch_chi2:.1f}) dv={dv_arch_coarse:.0f} km/s "
                      f"— kept archetype's better redshift instead of "
                      f"reverting to the coarse PCA reference")
            continue

        # ----------------------------------------------------------------
        # Set trigger bits
        # ----------------------------------------------------------------
        if mask_znum0.sum() > 0:
            if class_mismatch:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_CLASS_MISMATCH)
            if ratio_trigger:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_RATIO)
            if z_shift_trigger:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_Z_SHIFT)

        # Build trigger reason string for log
        trigger_parts = []
        if class_mismatch:
            trigger_parts.append(
                f"class mismatch "
                f"(coarse={coarse_best_class} "
                f"arch={arch_class} "
                f"dz={dz:.4f})")
        if ratio_trigger:
            trigger_parts.append(
                f"ratio={degradation:.2f}x")
        if z_shift_trigger:
            trigger_parts.append(
                f"z shift within {arch_class} "
                f"(coarse z={coarse_best_z:.4f} "
                f"arch z={arch_z:.4f} "
                f"dv={dv_arch_coarse:.0f} km/s "
                f"ref={'pca' if pca_confirms_coarse else 'coarse'}"
                f"={z_reference:.4f})")
        trigger_str = ' + '.join(trigger_parts)

        # ----------------------------------------------------------------
        # Fallback rank selection
        #
        # Two tiers. Tier 1 (strict, as before): same-class candidate with
        # Redrock ZWARN <= ZWARN_MAX_ALLOWED (bits 0-1 only). Tier 2 only
        # runs if tier 1 finds nothing: same-class candidate whose ONLY
        # native Redrock flag beyond bits 0-1 is BAD_MINFIT (bit 10 --
        # "bad parabola fit to the chi2 minimum", i.e. Redrock is unsure
        # of the *formal error bar* on this z, not that the z is wrong).
        # Real case this fixes: WC APS_ID 948 (TARGCLASS=GALAXY, coarse
        # scan already correctly prefers GALAXY chi2=18097 over the
        # archetype's STAR chi2=18551, but every GALAXY zfitall row is
        # BAD_MINFIT-flagged, so the old strict-only search found nothing
        # and silently kept the wrong STAR answer despite a demonstrably
        # better same-class candidate sitting right there). See
        # doc/aps_rr.md.
        # ----------------------------------------------------------------
        BAD_MINFIT_BIT = _RR_ZWarningMask.BAD_MINFIT   # 1024, redrock native bit 10

        # Tier 2 is gated to the genuine class-mismatch case only (coarse
        # class disagrees with the archetype's class, and that disagreement
        # is NOT already explained away by targeting corroboration). Without
        # this gate, tier 2 also fires for a ratio/z-shift trigger that only
        # exists because targeting_override *widened* (not eliminated) the
        # ratio threshold -- there, coarse_best_class is the class targeting
        # actively argues AGAINST, so relaxing the ZWARN filter to find a
        # coarse-class substitute is actively harmful, not helpful. Caught
        # by real end-to-end testing: an earlier, ungated version of this
        # tier flipped 20 genuinely-correct STAR classifications in the
        # GA-LRDISC population batch to wrong GALAXY (APS_ID 38, 182, 199,
        # ... — every one of them a targeting_override case where ratio
        # marginally exceeded even the widened 4.5x threshold). See
        # doc/aps_rr.md Version History 3.11.
        allow_tier2 = class_mismatch and not targeting_override

        fallback_idx     = None
        fallback_dz      = np.inf
        fallback_relaxed = False

        for relax_pass in (False, True) if allow_tier2 else (False,):
            for rank_idx in range(len(targ_rows)):
                row_class = str(targ_rows['CLASS'][rank_idx]).strip()
                row_zwarn = int(targ_rows['ZWARN'][rank_idx])
                row_z     = float(targ_rows['Z'][rank_idx])

                if row_class != coarse_best_class:
                    continue

                # Only consider Redrock bits (0-15) for ZWARN check
                redrock_zwarn = row_zwarn & 0xFFFF
                if not relax_pass:
                    if redrock_zwarn > ZWARN_MAX_ALLOWED:
                        continue
                else:
                    # Already covered by tier 1 -- skip so tier 2 only
                    # ever adds BAD_MINFIT-only candidates.
                    if redrock_zwarn <= ZWARN_MAX_ALLOWED:
                        continue
                    if (redrock_zwarn & ~BAD_MINFIT_BIT) > ZWARN_MAX_ALLOWED:
                        continue

                dz_i        = abs(row_z - z_reference)
                dz_weighted = dz_i if (row_zwarn & 0xFFFF) == 0 \
                              else dz_i + 0.5

                if dz_weighted < fallback_dz:
                    fallback_dz      = dz_weighted
                    fallback_idx     = rank_idx
                    fallback_relaxed = relax_pass

            if fallback_idx is not None:
                break

        # ----------------------------------------------------------------
        # Apply fallback
        # ----------------------------------------------------------------
        if fallback_idx is not None and fallback_idx > 0:

            fallback_row   = targ_rows[fallback_idx]
            fallback_z     = float(fallback_row['Z'])
            fallback_class = str(fallback_row['CLASS']).strip()
            fallback_zwarn = int(fallback_row['ZWARN'])
            fallback_znum  = int(fallback_row['ZNUM'])

            mask_fallback = ((zfitall['APS_ID'] == aps_id) &
                             (zfitall['ZNUM']   == fallback_znum))

            if mask_znum0.sum() > 0 and mask_fallback.sum() > 0:
                cols_to_swap = ['Z', 'ZERR', 'ZWARN', 'CLASS',
                                 'SUBCLASS', 'CHI2', 'DELTACHI2',
                                 'COEFF', 'NCOEFF', 'NPIXELS']

                znum0_vals    = {}
                fallback_vals = {}

                for col in cols_to_swap:
                    if col in zfitall.colnames:
                        znum0_vals[col]    = \
                            zfitall[col][mask_znum0][0]
                        fallback_vals[col] = \
                            zfitall[col][mask_fallback][0]

                for col in cols_to_swap:
                    if col in zfitall.colnames:
                        zfitall[col][mask_znum0]    = fallback_vals[col]
                        zfitall[col][mask_fallback] = znum0_vals[col]

                # Re-apply APS bits to new rank 0 after swap
                # (swap overwrote ZWARN with fallback rank's ZWARN)
                # Preserve Redrock bits from fallback rank,
                # re-add APS bits that were set before the swap
                aps_bits_to_restore = (
                    APS_ZWARN_SWAP_APPLIED |
                    (APS_ZWARN_CLASS_MISMATCH if class_mismatch else 0) |
                    (APS_ZWARN_RATIO          if ratio_trigger   else 0) |
                    (APS_ZWARN_Z_SHIFT        if z_shift_trigger else 0) |
                    (APS_ZWARN_PCA_UNRELIABLE if not pca_reliable else 0)|
                    (APS_ZWARN_PCA_CONFIRMS   if pca_confirms_coarse
                                              else 0) |
                    (APS_ZWARN_REF_IS_PCA     if pca_confirms_coarse
                                              else 0) |
                    (APS_ZWARN_MINFIT_RELAXED if fallback_relaxed
                                              else 0))

                zfitall['ZWARN'][mask_znum0] = (
                    (zfitall['ZWARN'][mask_znum0] & 0xFFFF) |
                    aps_bits_to_restore)

                n_fallback += 1
                fallback_log.append(
                    f"  APS_ID={aps_id_int:6d}: "
                    f"[{trigger_str}]  "
                    f"coarse={coarse_best_class} "
                    f"z={coarse_best_z:.4f} "
                    f"(ref={z_reference:.4f} "
                    f"{'pca' if pca_confirms_coarse else 'coarse'}) "
                    f"chi2={coarse_best_chi2:.0f}  "
                    f"arch={arch_class} z={arch_z:.4f} "
                    f"chi2={arch_chi2:.0f}  "
                    f"→ {fallback_class} z={fallback_z:.4f} "
                    f"chi2={float(targ_rows['CHI2'][fallback_idx]):.0f} "
                    f"dz={fallback_dz:.4f} "
                    f"ZWARN={fallback_zwarn & 0xFFFF}"
                    f"{' (BAD_MINFIT-relaxed)' if fallback_relaxed else ''} "
                    f"APS_ZWARN={hex(aps_bits_to_restore)}")

        else:
            # No valid rank found or rank 0 already best
            reason = (
                "no valid same-class rank found "
                "(all Redrock ZWARN>2 or class absent)") \
                if fallback_idx is None \
                else "rank 0 is already best match"

            if mask_znum0.sum() > 0:
                zfitall['ZWARN'][mask_znum0] = (
                    zfitall['ZWARN'][mask_znum0] |
                    APS_ZWARN_NO_VALID_RANK)

            fallback_log.append(
                f"  APS_ID={aps_id_int:6d}: "
                f"[{trigger_str}]  "
                f"coarse={coarse_best_class} "
                f"z={coarse_best_z:.4f} "
                f"(ref={z_reference:.4f} "
                f"{'pca' if pca_confirms_coarse else 'coarse'}) "
                f"chi2={coarse_best_chi2:.0f}  "
                f"arch={arch_class} z={arch_z:.4f} "
                f"chi2={arch_chi2:.0f}  "
                f"— {reason}, keeping rank 0")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n  Archetype fallback summary:")
    print(f"    Targets checked   : {n_checked}")
    print(f"    Fallback applied  : {n_fallback} "
          f"({100*n_fallback/max(n_checked,1):.1f}%)")
    print(f"    Threshold ratio   : {degradation_threshold}×")
    print(f"    ZWARN max allowed : {ZWARN_MAX_ALLOWED} "
          f"(Redrock bits 0-1 only)")
    print(f"    DV shift thresh   : {DV_SHIFT_THRESHOLD:.0f} km/s")
    print(f"    DV PCA-coarse max : {DV_PCA_COARSE_MAX:.0f} km/s")
    print(f"\n  APS ZWARN bits used:")
    print(f"    Bit 16 ({APS_ZWARN_CLASS_MISMATCH:>8}): "
          f"APS_FALLBACK_CLASS_MISMATCH")
    print(f"    Bit 17 ({APS_ZWARN_RATIO:>8}): "
          f"APS_FALLBACK_RATIO")
    print(f"    Bit 18 ({APS_ZWARN_Z_SHIFT:>8}): "
          f"APS_FALLBACK_Z_SHIFT")
    print(f"    Bit 19 ({APS_ZWARN_SWAP_APPLIED:>8}): "
          f"APS_FALLBACK_SWAP_APPLIED")
    print(f"    Bit 20 ({APS_ZWARN_NO_VALID_RANK:>8}): "
          f"APS_FALLBACK_NO_VALID_RANK")
    print(f"    Bit 21 ({APS_ZWARN_PCA_UNRELIABLE:>8}): "
          f"APS_PCA_UNRELIABLE")
    print(f"    Bit 22 ({APS_ZWARN_PCA_CONFIRMS:>8}): "
          f"APS_PCA_CONFIRMS_COARSE")
    print(f"    Bit 23 ({APS_ZWARN_REF_IS_PCA:>8}): "
          f"APS_Z_REFERENCE_IS_PCA")
    print(f"    Bit 24 ({APS_ZWARN_TARGETING_OVERRIDE:>8}): "
          f"APS_TARGETING_OVERRIDE")
    print(f"    Bit 25 ({APS_ZWARN_ARCHETYPE_Z_IMPROVED:>8}): "
          f"APS_ARCHETYPE_Z_IMPROVED")
    print(f"    Bit 26 ({APS_ZWARN_MINFIT_RELAXED:>8}): "
          f"APS_MINFIT_RELAXED")
    if fallback_log:
        print(f"    Details:")
        for entry in fallback_log:
            print(entry)

    return zfitall, n_fallback


##########################################################
def gen_zbest_single(zfitall, scandata, apsmeta,
                     srvyconf=None, comm=None):

    start = elapsed(None, "", comm=comm)

    # ----------------------------------------------------------------
    # STEP 1: Archetype fallback
    # ----------------------------------------------------------------
    print("\n  [Step 1] Archetype fallback check...")
    zfitall, n_fallback = _apply_archetype_fallback(
        zfitall, scandata, apsmeta,
        degradation_threshold=1.5)

    # ----------------------------------------------------------------
    # STEP 2: Select best rank (ZNUM=0 after fallback)
    # ZWARN-aware promotion DISABLED — causes incorrect swaps.
    # Archetype fallback in Step 1 handles genuine failures.
    # Use Redrock bits only (mask 0xFFFF) for rank selection.
    # ----------------------------------------------------------------
    print("\n  [Step 2] Rank selection (no ZWARN promotion)...")

    zfitall_grp = zfitall.group_by(['APS_ID'])
    zbest_rows  = []
    for grp in zfitall_grp.groups:
        grp = grp[np.argsort(grp['ZNUM'])]
        zbest_rows.append(grp[0:1])

    from astropy.table import vstack
    zbest = vstack(zbest_rows)
    zbest.remove_columns(['ZZ', 'ZZCHI2', 'ZNUM'])

    # ----------------------------------------------------------------
    # STEP 3: Survey classification (SRVY_CLASS)
    # ----------------------------------------------------------------
    print("\n  [Step 3] Survey classification...")

    zbest.add_column(zbest['CLASS'].copy(), name='SRVY_CLASS')

    has_targclass = 'TARGCLASS' in zbest.colnames
    has_targprog  = 'TARGPROG'  in zbest.colnames
    has_targsrvy  = 'TARGSRVY'  in zbest.colnames

    print(f"  TARGCLASS available: {has_targclass}")
    print(f"  TARGPROG available : {has_targprog}")
    print(f"  TARGSRVY available : {has_targsrvy}")

    survey_config = None
    if srvyconf is not None:
        try:
            survey_config = load_survey_classification(srvyconf)
            print(f"  Survey config loaded: {srvyconf}")
        except Exception as e:
            print(f"  WARNING: Failed to load survey config: {e}")

    n_overridden = 0
    n_unmatched  = 0
    classification_counts = {'STAR': 0, 'QSO': 0,
                              'GALAXY': 0, 'OTHER': 0}
    source_counts = {'TARGCLASS': 0, 'TARGPROG': 0,
                     'TARGSRVY': 0, 'NONE': 0}

    for idx in range(len(zbest)):
        targclass = zbest['TARGCLASS'][idx] if has_targclass else None
        targprog  = zbest['TARGPROG'][idx]  if has_targprog  else None
        targsrvy  = zbest['TARGSRVY'][idx]  if has_targsrvy  else None

        verbose  = (idx < 5)
        srvy_cls = classify_target_comprehensive(
            targclass, targprog, targsrvy,
            survey_config=survey_config,
            default_class=None,
            verbose=verbose)

        if srvy_cls is not None:
            zbest['SRVY_CLASS'][idx] = srvy_cls
            n_overridden += 1
            if srvy_cls in classification_counts:
                classification_counts[srvy_cls] += 1
            else:
                classification_counts['OTHER'] += 1

            if has_targclass and targclass:
                tc_norm = str(targclass).strip().upper()
                if tc_norm and tc_norm not in ['NONE', 'NULL', '']:
                    if any(kw in tc_norm for kw in
                           ['STAR', 'WD', 'QSO', 'GALAXY', 'NEBULA']):
                        source_counts['TARGCLASS'] += 1
                        continue
            if has_targprog and targprog:
                tp_norm = str(targprog).strip().upper()
                if tp_norm and tp_norm not in ['NONE', 'NULL', '']:
                    if any(kw in tp_norm for kw in
                           ['STAR', 'QSO', 'GALAXY']):
                        source_counts['TARGPROG'] += 1
                        continue
            if has_targsrvy and targsrvy:
                source_counts['TARGSRVY'] += 1
        else:
            n_unmatched += 1
            source_counts['NONE'] += 1

    print(f"\n  Classification summary:")
    print(f"    Overridden : {n_overridden}/{len(zbest)} "
          f"({100*n_overridden/len(zbest):.1f}%)")
    print(f"    Unmatched  : {n_unmatched}/{len(zbest)} "
          f"({100*n_unmatched/len(zbest):.1f}%)")
    for cls, count in classification_counts.items():
        if count > 0:
            print(f"    {cls}: {count}")

    # ----------------------------------------------------------------
    # STEP 4: Finalise
    # ----------------------------------------------------------------
    zbest.remove_columns(['FIB_STATUS'])

    zbest.meta['EXTNAME'] = 'ZBEST'
    for n_province, province in enumerate(apsmeta['files']):
        zbest.meta['APSREF_%d' % n_province] = (
            os.path.basename(province), 'L1 reference file')
    zbest.meta['CSB_RR'] = (
        apsmeta['stitched'], 'Combines Spectral Bands Status for REDROCK')

    try:
        scan_table = gen_scandata_table(scandata)
        zbest = join(zbest, scan_table, join_type='left', keys='APS_ID')
    except Exception:
        print('Failed to add chi2(Z) scan data')

    print(zbest)
    stop = elapsed(start, "Preparing ZBEST table took", comm=comm)
    return zbest


##########################################################
def gen_zbest_multiple(zfitall, scandata, apsmeta,
                        srvyconf=None, ntop=3,
                        add_extra=True, comm=None):

    start = elapsed(None, "", comm=comm)

    # ----------------------------------------------------------------
    # STEP 1: Archetype fallback
    # ----------------------------------------------------------------
    print("\n  [Step 1] Archetype fallback check...")
    zfitall, n_fallback = _apply_archetype_fallback(
        zfitall, scandata, apsmeta,
        degradation_threshold=1.5)

    # ----------------------------------------------------------------
    # STEP 2: ntop stacking
    # ZWARN-aware promotion DISABLED.
    # Use Redrock bits only (mask 0xFFFF) for rank ordering.
    # ----------------------------------------------------------------
    print(f"\n  [Step 2] ntop={ntop} stacking "
          f"(no ZWARN promotion)...")

    zfitall_cp = deepcopy(zfitall)
    zfitall_cp.remove_columns(['ZZ', 'ZZCHI2'])

    id_columns  = ['APS_ID', 'TARGID', 'CNAME']
    zfitall_grp = zfitall_cp.group_by(id_columns)
    zbest_table = Table()

    for grp in zfitall_grp.groups:
        assert len(grp) > ntop, \
            f"Need at least {ntop} redshift guesses, found {len(grp)}"

        grp = grp[np.argsort(grp['ZNUM'])]

        new_dummy_table = Table()
        for clnme in grp.colnames:
            if clnme in id_columns:
                new_dummy_table.add_column(
                    Column([grp[clnme][0]], name=clnme))
            else:
                new_dummy_table.add_column(
                    Column([np.array(grp[clnme])[0:ntop]], name=clnme))

        zbest_table = vstack([zbest_table, new_dummy_table])

    zbest_table.remove_columns(['ZNUM', 'FIB_STATUS'])

    # ----------------------------------------------------------------
    # STEP 2b: Ensure SRVY_CLASS represented in top ntop ranks
    # ----------------------------------------------------------------
    print(f"\n  [Step 2b] SRVY_CLASS representation check...")
    zbest_table, n_injected = _ensure_srvy_class_represented(
        zbest_table, zfitall_cp, ntop=ntop)

    # ----------------------------------------------------------------
    # STEP 3: Survey classification (SRVY_CLASS)
    # ----------------------------------------------------------------
    print(f"\n  [Step 3] Survey classification (ntop={ntop})...")

    import copy
    srvy_class_data = copy.deepcopy(zbest_table['CLASS'].data)
    zbest_table.add_column(Column(srvy_class_data, name='SRVY_CLASS'))

    has_targclass = 'TARGCLASS' in zbest_table.colnames
    has_targprog  = 'TARGPROG'  in zbest_table.colnames
    has_targsrvy  = 'TARGSRVY'  in zbest_table.colnames

    survey_config = None
    if srvyconf is not None:
        try:
            survey_config = load_survey_classification(srvyconf)
            print(f"  Survey config loaded: {srvyconf}")
        except Exception as e:
            print(f"  WARNING: Failed to load survey config: {e}")

    n_overridden = 0
    n_unmatched  = 0
    classification_counts = {'STAR': 0, 'QSO': 0,
                              'GALAXY': 0, 'OTHER': 0}
    source_counts = {'TARGCLASS': 0, 'TARGPROG': 0,
                     'TARGSRVY': 0, 'NONE': 0}

    for idx in range(len(zbest_table)):
        targclass = zbest_table['TARGCLASS'][idx] \
                    if has_targclass else None
        targprog  = zbest_table['TARGPROG'][idx]  \
                    if has_targprog  else None
        targsrvy  = zbest_table['TARGSRVY'][idx]  \
                    if has_targsrvy  else None

        if isinstance(targclass, (list, np.ndarray)) \
                and len(targclass) > 0:
            targclass = targclass[0]
        if isinstance(targprog, (list, np.ndarray)) \
                and len(targprog) > 0:
            targprog = targprog[0]
        if isinstance(targsrvy, (list, np.ndarray)) \
                and len(targsrvy) > 0:
            targsrvy = targsrvy[0]

        verbose  = (idx < 5)
        srvy_cls = classify_target_comprehensive(
            targclass, targprog, targsrvy,
            survey_config=survey_config,
            default_class=None,
            verbose=verbose)

        if srvy_cls is not None:
            for rank_idx in range(zbest_table['SRVY_CLASS'].shape[1]):
                zbest_table['SRVY_CLASS'][idx][rank_idx] = srvy_cls
            n_overridden += 1
            if srvy_cls in classification_counts:
                classification_counts[srvy_cls] += 1
            else:
                classification_counts['OTHER'] += 1

            if has_targclass and targclass:
                tc_norm = str(targclass).strip().upper()
                if tc_norm and tc_norm not in ['NONE', 'NULL', '']:
                    if any(kw in tc_norm for kw in
                           ['STAR', 'WD', 'QSO', 'GALAXY', 'NEBULA']):
                        source_counts['TARGCLASS'] += 1
                        continue
            if has_targprog and targprog:
                tp_norm = str(targprog).strip().upper()
                if tp_norm and tp_norm not in ['NONE', 'NULL', '']:
                    if any(kw in tp_norm for kw in
                           ['STAR', 'QSO', 'GALAXY']):
                        source_counts['TARGPROG'] += 1
                        continue
            if has_targsrvy and targsrvy:
                source_counts['TARGSRVY'] += 1
        else:
            n_unmatched += 1
            source_counts['NONE'] += 1

    print(f"\n  Classification summary:")
    print(f"    Overridden : {n_overridden}/{len(zbest_table)} "
          f"({100*n_overridden/len(zbest_table):.1f}%)")
    print(f"    Unmatched  : {n_unmatched}/{len(zbest_table)} "
          f"({100*n_unmatched/len(zbest_table):.1f}%)")
    for cls, count in classification_counts.items():
        if count > 0:
            print(f"    {cls}: {count}")

    # ----------------------------------------------------------------
    # STEP 4: Finalise
    # ----------------------------------------------------------------
    zbest_table.meta['EXTNAME'] = 'ZBEST'
    for n_province, province in enumerate(apsmeta['files']):
        zbest_table.meta['APSREF_%d' % n_province] = (
            os.path.basename(province), 'L1 reference file')
    zbest_table.meta['CSB_RR'] = (
        apsmeta['stitched'],
        'Combines Spectral Bands Status for REDROCK')

    if add_extra:
        try:
            scan_table = gen_scandata_table(scandata)
            zbest_table = join(zbest_table, scan_table,
                               join_type='left', keys='APS_ID')
        except Exception:
            print('Failed to add chi2(Z) scan data')

    print(zbest_table)
    stop = elapsed(start, "Preparing ZBEST table took", comm=comm)
    return zbest_table
###########################################################

def gen_zspec(zbest, targets, apsmeta, dtemplates, rank=0,
              darchetypes=None, sens_corr=True, comm=None):
    """
    Build the per-target CLASS_SPECTRA table (observed spectrum + model)
    used for Redrock diagnostic plots and output.

    MODEL_RR_<arm> stores one best-fit model spectrum per available rank
    (shape [n_ranks, n_wave] per row, n_ranks capped at
    PyAPS.apsPlot.redrock.MAX_RANKS), not just the rank-0 model — each rank
    shown in the diagnostic plot is now backed by its own model curve.
    `rank` is kept only for call-signature compatibility and no longer
    restricts which rank is computed.
    """

    start_zspec = elapsed(None, "", comm=comm)

    template_version = {t._template.full_type: t._template._version
                        for t in dtemplates}

    archetype_version = None
    if darchetypes is not None:
        archetype_version = {name: arch._version
                             for name, arch in darchetypes.items()}

    # Build templates dict keyed by full_type
    templates = dict()
    for dt in dtemplates:
        templates[dt.template.full_type] = dt.template

    # ----------------------------------------------------------------
    # Output table structure
    # ----------------------------------------------------------------
    columns = ['APS_ID', 'TARGID', 'CNAME']
    for s in apsmeta['setups']:
        columns.append('LAMBDA_RR_%s' % s[0])
        columns.append('FLUX_RR_%s'   % s[0])
        columns.append('IVAR_RR_%s'   % s[0])
        columns.append('MODEL_RR_%s'  % s[0])

    outdict = OrderedDict()
    for c in columns:
        outdict[c] = []

    # ----------------------------------------------------------------
    # Loop over targets
    # ----------------------------------------------------------------
    for i in range(len(targets)):
        targets[i].sharedmem_unpack()
        dtg      = targets[i]
        id_zbest = np.where(zbest['APS_ID'] == dtg.id)[0]

        if id_zbest.size == 0:
            continue

        zz = zbest[id_zbest]

        # ----------------------------------------------------------------
        # Determine how many ranks are available for this target, and
        # extract (z, class, subclass, coeff) for each of them.
        # ----------------------------------------------------------------
        try:
            if zz['Z'].ndim == 1:
                n_ranks = 1
            elif zz['Z'].ndim == 2:
                n_ranks = min(MAX_RANKS, zz['Z'].shape[1])
            else:
                print(f"Warning: unsupported Z ndim={zz['Z'].ndim} "
                      f"for APS_ID={dtg.id}")
                continue

            rank_fits = []
            for rk in range(n_ranks):
                if zz['Z'].ndim == 1:
                    z_val     = float(zz['Z'][0])
                    class_val = str(zz['CLASS'][0]).strip()
                    sub_val   = str(zz['SUBCLASS'][0]).strip()
                    aps_id    = zz['APS_ID'][0]
                    targid    = zz['TARGID'][0]
                    cname     = zz['CNAME'][0]

                    coeff_raw = np.array(zz['COEFF']).squeeze()
                else:
                    z_val     = float(zz['Z'].value.data[0][rk])
                    class_val = str(zz['CLASS'].value.data[0][rk]).strip()
                    sub_val   = str(zz['SUBCLASS'].value.data[0][rk]).strip()
                    aps_id    = zz['APS_ID'].value.data[0]
                    targid    = zz['TARGID'].value.data[0]
                    cname     = zz['CNAME'].value.data[0]

                    coeff_raw = np.array(zz['COEFF'].value.data[0]).squeeze()

                if coeff_raw.ndim == 2:
                    coeff_val = coeff_raw[rk].ravel()
                elif coeff_raw.ndim == 1:
                    coeff_val = coeff_raw.ravel()
                else:
                    coeff_val = np.array([float(coeff_raw)])

                # Always ensure 1D float64
                coeff_val = np.asarray(coeff_val, dtype=np.float64).ravel()

                rank_fits.append((z_val, class_val, sub_val, coeff_val))

        except Exception as e:
            print(f"Warning: failed to extract fit params "
                  f"for APS_ID={dtg.id}: {e}")
            continue

        # ----------------------------------------------------------------
        # Append ID columns
        # ----------------------------------------------------------------
        outdict['APS_ID'].append(aps_id)
        outdict['TARGID'].append(targid)
        outdict['CNAME'].append(cname)

        # ----------------------------------------------------------------
        # Generate one model per rank, per arm
        # ----------------------------------------------------------------
        for dwc, s in enumerate(apsmeta['setups']):
            try:
                wave = dtg.spectra[dwc].wave.copy()
                flux = dtg.spectra[dwc].flux.copy()
                ivar = dtg.spectra[dwc].ivar.copy()
            except Exception as e:
                print(f"Warning: failed to read spectrum for "
                      f"APS_ID={dtg.id} arm={dwc}: {e}")
                import traceback
                traceback.print_exc()
                wave = np.zeros(1)
                flux = np.zeros(1)
                ivar = np.zeros(1)

            rank_models = []
            for z_val, class_val, sub_val, coeff_val in rank_fits:
                try:
                    if darchetypes is not None:
                        # ------------------------------------------------
                        # ARCHETYPE MODE
                        #
                        # sub_val may contain multiple archetypes separated
                        # by semicolons when n_nearest > 1:
                        #   e.g. 'LRG_1;LRG_3;BGS_0'
                        #
                        # coeff_val contains one amplitude per archetype.
                        # The model is the weighted sum of all archetypes.
                        #
                        # After building the archetype model, we re-fit
                        # per-arm Legendre continuum correction terms.
                        # These were used during fitz (per_camera=True) but
                        # are not stored in the COEFF output column.
                        # Re-fitting them here recovers the correct continuum
                        # level and fixes the flux offset in the plot.
                        # ------------------------------------------------
                        tp = darchetypes[class_val]

                        # Parse archetype subtypes
                        subtypes = [st.strip()
                                    for st in sub_val.split(';')
                                    if st.strip()]
                        n_arch   = len(subtypes)
                        archcoeff = np.asarray(
                            coeff_val[:n_arch], dtype=np.float64)

                        try:
                            # Build model as weighted sum of all archetypes
                            model = np.zeros(len(wave), dtype=np.float64)
                            for subtype_i, amp_i in zip(subtypes, archcoeff):
                                arch_model_i = tp.eval(
                                    subtype_i,
                                    np.array([amp_i], dtype=np.float64),
                                    wave,
                                    z_val,
                                    R=dtg.spectra[dwc].R
                                ) * (1 + z_val)
                                model += arch_model_i

                            # --------------------------------------------
                            # RE-FIT PER-ARM LEGENDRE CONTINUUM CORRECTION
                            #
                            # Recovers the per_camera Legendre terms that
                            # fitz computed with deg_legendre=2 but did not
                            # store in the COEFF output column.
                            #
                            # Method: at the known best-fit redshift, solve
                            # for Legendre coefficients that minimise:
                            #   chi2 = sum( ivar × (flux - model - L@c)² )
                            # → c = (L^T W L)^{-1} L^T W (flux - model)
                            # --------------------------------------------
                            from scipy.special import legendre as _legendre

                            deg_leg = 2   # must match deg_legendre in zfind
                            valid   = ivar > 0

                            if valid.sum() > deg_leg + 1:
                                # Reduced wavelength in [-1, 1]
                                w_min = float(wave[valid].min())
                                w_max = float(wave[valid].max())
                                if w_max > w_min:
                                    w_red = (2.0 * (wave - w_min)
                                             / (w_max - w_min) - 1.0)

                                    # Legendre basis (n_pix × deg_leg)
                                    L = np.array([
                                        _legendre(k)(w_red)
                                        for k in range(deg_leg)
                                    ]).T

                                    # Weighted residual
                                    residual = flux - model
                                    L_v  = L[valid]
                                    r_v  = residual[valid]
                                    iv_v = ivar[valid]

                                    LtWL = L_v.T @ (iv_v[:, None] * L_v)
                                    LtWr = L_v.T @ (iv_v * r_v)

                                    try:
                                        leg_c = np.linalg.solve(LtWL, LtWr)
                                        model = model + L @ leg_c
                                    except np.linalg.LinAlgError:
                                        print(f"  Warning: Legendre solve "
                                              f"failed APS_ID={dtg.id} "
                                              f"arm={dwc}")

                        except Exception as e_arch:
                            print(f"  Warning: archetype eval failed "
                                  f"APS_ID={dtg.id} arm={dwc}: {e_arch}")
                            import traceback
                            traceback.print_exc()
                            model = np.zeros(len(wave))

                    else:
                        # ------------------------------------------------
                        # PCA TEMPLATE MODE
                        # ------------------------------------------------
                        fulltype = class_val
                        if sub_val != '':
                            fulltype = fulltype + ':::' + sub_val

                        if fulltype not in templates:
                            print(f"  Warning: template '{fulltype}' not "
                                  f"found for APS_ID={dtg.id}")
                            model = np.zeros(len(wave))
                        else:
                            tp    = templates[fulltype]
                            model = tp.eval(
                                coeff_val[0:tp.nbasis],
                                wave,
                                z_val
                            ) * (1 + z_val)
                            model = dtg.spectra[dwc].R.dot(model)

                except Exception as e:
                    print(f"Warning: model generation failed "
                          f"APS_ID={dtg.id} arm={dwc}: {e}")
                    import traceback
                    traceback.print_exc()
                    model = np.zeros(len(wave))

                rank_models.append(model)

            model_stack = np.array(rank_models)  # shape (n_ranks, n_wave)

            # Mask bad pixels
            isbad              = (ivar == 0)
            flux[isbad]        = np.nan
            model_stack[:, isbad] = np.nan

            outdict['LAMBDA_RR_%s' % s[0]].append(wave)
            outdict['FLUX_RR_%s'   % s[0]].append(flux)
            outdict['IVAR_RR_%s'   % s[0]].append(ivar)
            outdict['MODEL_RR_%s'  % s[0]].append(model_stack)

    # ----------------------------------------------------------------
    # Build output table
    # ----------------------------------------------------------------
    zspec = Table(outdict)
    zspec.sort(['APS_ID'])
    zspec.meta['EXTNAME'] = 'CLASS_SPECTRA'

    for n_province, province in enumerate(apsmeta['files']):
        zspec.meta['APSREF_%d' % n_province] = (
            os.path.basename(province), 'L1 reference file')

    # Units
    if sens_corr:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' % apsmeta['funits']
        ivar_unit_str = '%2e cm**4 Angstrom**2 s**2 / (erg**2)' % \
                        (apsmeta['funits'] ** -2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'

    for s in apsmeta['setups']:
        zspec['LAMBDA_RR_%s' % s[0]].unit = wave_unit_str
        zspec['FLUX_RR_%s'   % s[0]].unit = flux_unit_str
        zspec['IVAR_RR_%s'   % s[0]].unit = ivar_unit_str
        zspec['MODEL_RR_%s'  % s[0]].unit = flux_unit_str

    zspec.meta['VACUUM']   = (True,
                               'Wavelengths are in vacuum')
    zspec.meta['SAMPLING'] = (0,
                               'Sampling mode (0: linear, 1: logarithmic)')
    zspec.meta['CSB_RR']   = (apsmeta['stitched'],
                               'Combines Spectral Bands Status for REDROCK')

    stop = elapsed(start_zspec, "Writing ZSPEC fits files took", comm=comm)

    return zspec


##########################################################

def _pca_reliability_check(scandata, aps_id,
                            coarse_best_class,
                            coarse_best_z,
                            dz_window=0.05):
    """
    Fit a parabola around the coarse PCA minimum and extract
    reliability indicators.

    Returns
    -------
    result : dict with keys:
      z_pca       : float  — refined z from parabola minimum
      chi2_pca    : float  — chi2 at parabola minimum
      zerr_pca    : float  — z uncertainty from parabola width
      dv_coarse   : float  — velocity offset coarse vs PCA (km/s)
      curvature   : float  — parabola curvature (larger = sharper)
      reliable    : bool   — True if minimum is well-defined
      reason      : str    — explanation of reliability assessment
    """
    C_LIGHT = 299792.458

    # Find the matching template key in scandata
    best_key  = None
    best_chi2 = np.inf

    for ft in scandata[aps_id].keys():
        if ft.split(':::')[0] != coarse_best_class:
            continue
        zz  = scandata[aps_id][ft]['redshifts']
        chi = scandata[aps_id][ft]['zchi2']
        if chi.min() < best_chi2:
            best_chi2 = chi.min()
            best_key  = ft

    if best_key is None:
        return {'z_pca': coarse_best_z, 'chi2_pca': np.inf,
                'zerr_pca': np.nan, 'dv_coarse': np.nan,
                'curvature': 0.0, 'reliable': False,
                'reason': 'no matching template'}

    zz  = scandata[aps_id][best_key]['redshifts']
    chi = scandata[aps_id][best_key]['zchi2']

    # Select window around coarse minimum
    window = (zz >= coarse_best_z - dz_window) & \
             (zz <= coarse_best_z + dz_window)

    if window.sum() < 5:
        return {'z_pca': coarse_best_z, 'chi2_pca': best_chi2,
                'zerr_pca': np.nan, 'dv_coarse': 0.0,
                'curvature': 0.0, 'reliable': False,
                'reason': f'too few points in window ({window.sum()})'}

    zz_win  = zz[window]
    chi_win = chi[window]

    # Fit parabola
    try:
        coeffs   = np.polyfit(zz_win, chi_win, 2)
        a, b, c  = coeffs
        z_pca    = -b / (2.0 * a)
        chi2_pca = np.polyval(coeffs, z_pca)

        # z uncertainty: delta_chi2=1 → sigma_z = 1/sqrt(a)
        zerr_pca = 1.0 / np.sqrt(abs(a)) if a > 0 else np.nan

        # Velocity offset between coarse grid and parabola minimum
        dv_coarse = abs(z_pca - coarse_best_z) / \
                    (1.0 + coarse_best_z) * C_LIGHT

        # ----------------------------------------------------------------
        # Reliability assessment
        # ----------------------------------------------------------------
        reliable = True
        reasons  = []

        # Check 1: parabola opens upward (genuine minimum)
        if a <= 0:
            reliable = False
            reasons.append('parabola opens downward — no minimum')

        # Check 2: parabola minimum is within the window
        # (not extrapolating outside the data range)
        if z_pca < zz_win.min() or z_pca > zz_win.max():
            reliable = False
            reasons.append(
                f'minimum z={z_pca:.4f} outside window '
                f'[{zz_win.min():.4f},{zz_win.max():.4f}]')

        # Check 3: chi2 at parabola minimum is close to coarse chi2
        # Large difference means the parabola is a poor fit
        # (flat or multi-modal landscape)
        chi2_relative_diff = abs(chi2_pca - best_chi2) / \
                             max(best_chi2, 1.0)
        if chi2_relative_diff > 0.05:
            reliable = False
            reasons.append(
                f'chi2 mismatch: parabola={chi2_pca:.0f} '
                f'vs coarse={best_chi2:.0f} '
                f'({100*chi2_relative_diff:.1f}% diff)')

        # Check 4: curvature — how sharp is the minimum?
        # Low curvature → flat landscape → unreliable z
        # Normalise by chi2 to make it scale-independent
        curvature_norm = a / max(best_chi2, 1.0)
        if curvature_norm < 1e-5:
            reliable = False
            reasons.append(
                f'low curvature ({curvature_norm:.2e}) '
                f'— flat chi2 landscape')

        # Check 5: coarse grid point vs parabola minimum
        # If they disagree by > 1000 km/s the coarse grid
        # may have found the wrong local minimum
        if dv_coarse > 1000.0:
            # This is informational, not necessarily unreliable
            # (could be the parabola finding a better minimum)
            reasons.append(
                f'coarse-PCA offset = {dv_coarse:.0f} km/s '
                f'(coarse grid may be imprecise here)')

        reason = '; '.join(reasons) if reasons \
                 else 'well-defined minimum'

        return {
            'z_pca'      : float(z_pca),
            'chi2_pca'   : float(chi2_pca),
            'zerr_pca'   : float(zerr_pca) if np.isfinite(zerr_pca)
                           else np.nan,
            'dv_coarse'  : float(dv_coarse),
            'curvature'  : float(a),
            'curvature_norm': float(curvature_norm),
            'reliable'   : reliable,
            'reason'     : reason,
        }

    except Exception as ex:
        return {
            'z_pca': coarse_best_z, 'chi2_pca': best_chi2,
            'zerr_pca': np.nan, 'dv_coarse': np.nan,
            'curvature': 0.0, 'curvature_norm': 0.0,
            'reliable': False,
            'reason': f'parabola fit failed: {ex}'}


##########################################################

def rrweave_worker(infiles, templates, srvyconf=None, zbest_fname= None, zall_fname = None ,zspec_fname=None, aps_ids=None, ntop=1,
    targsrvy= None, targclass = None, mask_aps_ids=None , area=None, mask_areas=None, wlranges=None, sens_corr=True,
    mask_gaps=True, safe_mask_gaps=True, vacuum=True, tellurics=False, fill_gap=False, arms_ratio=None, join_arms=False, ncpus= 1, comm=None,
    comm_rank=0, comm_size=1, nminima=3, archetypes=None, cache_Rcsr= False, priors=None, chi2_scan=None, figdir=None,
    debug=False, return_outputs=False, collapse = False, collapse_fname=None, match_table=None, gpu=False, max_gpuprocs = None, catdir=None, caldir=None, configdir=None, prior_sigma=0.01, n_nearest=3,skysub_mask_residuals=True,
    spaxel_weighted_lsf=False, extinction_corr=False, extinction_ebv_scale=1.0, extinction_mapdir=None,
    star_z_prior_sigma=None):

    """
    Function description:

    """

    C_LIGHT = 299792.458   # km/s
    # ----------------------------------------------------------------
    # TEMPORARY: Auto-derive archetypes path from templates path
    # if archetypes is not provided.
    # This is to handle the currenly available scripts
    # TODO: Replace this with explicit archetypes path in all configs
    # ----------------------------------------------------------------
    if archetypes is None and templates is not None:
        templates_str = str(templates).strip()
        if 'templates_RR' in templates_str:
            archetypes = templates_str.replace('templates_RR', 'templates_ARC_RR')
            print(f"\n{'='*70}")
            print(f"  *** TEMPORARY WARNING ***")
            print(f"  archetypes not set — auto-derived from templates path:")
            print(f"  templates : {templates_str}")
            print(f"  archetypes: {archetypes}")
            print(f"  Please update your config/script to set archetypes")
            print(f"  explicitly. This auto-derivation will be removed.")
            print(f"{'='*70}\n")
            # Verify the derived path actually exists
            from pathlib import Path
            if not Path(archetypes).exists():
                print(f"  *** WARNING: derived archetypes path does not exist:")
                print(f"  {archetypes}")
                print(f"  Continuing without archetypes — results may be wrong!")
                archetypes = None
        else:
            print(f"\n  WARNING: archetypes not set and cannot be auto-derived")
            print(f"  (templates path does not contain 'templates_RR')")
            print(f"  templates path: {templates_str}")
            print(f"  Continuing without archetypes — results may be wrong!\n")





    global_start = elapsed(None, "", comm=comm)
    # Multiprocessing processes to use if MPI is disabled.
    mpprocs = 0
    if comm is None:
        mpprocs = get_mp(ncpus)
        print("Running with {} processes".format(mpprocs))
        if "OMP_NUM_THREADS" in os.environ:
            nthread = int(os.environ["OMP_NUM_THREADS"])
            if nthread != 1:
                print("WARNING:  {} multi-processes running, each with "
                    "{} threads ({} total)".format(mpprocs, nthread,
                    mpprocs*nthread))
                print("WARNING:  Please ensure this is <= the number of "
                    "physical cores on the system")
        else:
            print("WARNING:  using multiprocessing, but the OMP_NUM_THREADS")
            print("WARNING:  environment variable is not set- your system may")
            print("WARNING:  be oversubscribed.")
        sys.stdout.flush()
    elif comm_rank == 0:
        print("Running with {} processes".format(comm_size))
        sys.stdout.flush()


    # GPU configuration
    if gpu:
        # Determine which processes will use a GPU
        if max_gpuprocs is not None:
            max_gpuprocs = max_gpuprocs
        else:
            #Check actual number of GPUs available
            if (comm is not None):
                #Use custom method that checks PCI ids for MPI
                max_gpuprocs = getGPUCountMPI(comm)
            else:
                #cupy getDeviceCount works for non MPI
                import cupy
                max_gpuprocs = cupy.cuda.runtime.getDeviceCount()
        use_gpu = comm_rank < max_gpuprocs

        # Determine cpu/gpu process capacities for target distribution
        if comm is not None:
            gpu_proc_flags = comm.allgather(use_gpu)
        else:
            gpu_proc_flags = [use_gpu, ]
            if (mpprocs > 1):
                #Force mpprocs == 1 for multiprocessing mode with GPU
                print("WARNING:  using GPU mode without MPI requires --mp 1")
                print("WARNING:  Overriding {} multiprocesses to force this.".format(mpprocs))
                print("WARNING:  Running with 1 process.")
                mpprocs = 1
        ngpu_procs = sum(gpu_proc_flags)
        ncpu_procs = comm_size - ngpu_procs
        if ngpu_procs > 0 and ncpu_procs > 0:
            # On Perlmutter, 1:15 seems like a good ratio
            #capacities = [1 if is_gpu_proc else 1.0/15 for is_gpu_proc in gpu_proc_flags]
            #With new GPU implementation of zscan, use 1:10000 so that only GPU-enabled
            #procs get allocated targets
            capacities = [1 if is_gpu_proc else 1.0/10000 for is_gpu_proc in gpu_proc_flags]
        else:
            capacities = None

        # Redistribute templates after rebinning when using GPUs
        redistribute_templates = True
    else:
        use_gpu = False
        capacities = None
        redistribute_templates = False

    try:
        # Load and distribute the targets
        if comm_rank == 0:
            print("Loading targets (according to their APS_IDs)...")
            sys.stdout.flush()

        # start = elapsed(None, "", comm=comm)
        # Read the spectra on the root process.  Currently the "meta" Table
        # returned here is not propagated to the output zbest file.  However,
        # that could be changed to work like the DESI write_zbest() function.
        # Each target contains metadata which is propagated to the output zbest
        # table though.


        targets, apsmeta = read_spectra(infiles, aps_ids= aps_ids, targsrvy= targsrvy, targclass = targclass,
            mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, wlranges=wlranges, cache_Rcsr=cache_Rcsr,
            sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap,
            arms_ratio=arms_ratio, join_arms=join_arms, collapse=collapse, catdir=catdir, caldir=caldir, configdir=configdir,ncpus= ncpus, skysub_mask_residuals=skysub_mask_residuals,
            spaxel_weighted_lsf=spaxel_weighted_lsf, extinction_corr=extinction_corr,
            extinction_ebv_scale=extinction_ebv_scale, extinction_mapdir=extinction_mapdir)

        # stop = elapsed(start, "Read of {} targets".format(len(targets)), comm=comm)

        # ------------------------------------------------------------------
        # Opt-in TARGCLASS-based redshift prior (star_z_prior_sigma, default
        # None = disabled). Unlike the archetype-fallback fixes above (which
        # only arbitrate AFTER the coarse+archetype fit has already run),
        # this acts directly on the coarse scan itself: redrock's own
        # `priors` mechanism (see redrock.priors.Priors) adds a chi2 penalty
        # as a function of (TARGETID, z), identically across every template
        # class for that target, evaluated immediately after the coarse
        # scan and before minima are selected. For a target independently
        # known (via TARGCLASS/TARGPROG/TARGSRVY) to be a STAR, a tight
        # Gaussian prior centred at z=0 penalises any high-z GALAXY/QSO
        # solution directly at the source, rather than needing a
        # post-hoc archetype/targeting arbitration to catch it afterward.
        #
        # Verified on real GA-LRDISC data (20260122, stack_3134457__
        # stack_3134456, sigma=0.0067 ~ 2000 km/s): fixed APS_ID 6, a
        # coarse=GALAXY/archetype=QSO case the targeting-corroboration fix
        # above cannot touch at all (it only fires when the archetype's own
        # class already agrees with targeting). Does not help APS_ID 613,
        # where the coarse GALAXY answer is *already* at z~0 (a genuine
        # continuum-shape degeneracy, not a high-z excursion) -- expected
        # and a useful sanity check that this prior does exactly what it
        # claims to and nothing more.
        #
        # Only used when the caller hasn't already supplied their own
        # `--priors` file (that always wins) -- see doc/aps_rr.md.
        # ------------------------------------------------------------------
        if star_z_prior_sigma is not None and priors is None:
            star_ids = []
            for tg in targets:
                hint = _targeting_class_hint(
                    tg.meta.get('TARGCLASS'), tg.meta.get('TARGPROG'),
                    tg.meta.get('TARGSRVY'))
                if hint == 'STAR':
                    star_ids.append(int(tg.id))
            if len(star_ids) > 0:
                import tempfile
                from astropy.table import Table as _PriorTable
                pt = _PriorTable()
                pt['TARGETID'] = np.array(star_ids, dtype=np.int64)
                pt['Z'] = np.zeros(len(star_ids))
                pt['SIGMA'] = np.full(len(star_ids), float(star_z_prior_sigma))
                pt['FUNCTION'] = np.array(['gaussian'] * len(star_ids))
                prior_hdu = fits.BinTableHDU(pt)
                prior_hdu.header['EXTNAME'] = 'PRIORS'
                prior_dir = (os.path.dirname(zbest_fname) if zbest_fname
                             else (figdir or '.')) or '.'
                prior_fh = tempfile.NamedTemporaryFile(
                    dir=prior_dir, prefix='star_z_prior_', suffix='.fits', delete=False)
                prior_fh.close()
                prior_hdu.writeto(prior_fh.name, overwrite=True)
                priors = prior_fh.name
                print(f"  star_z_prior_sigma={star_z_prior_sigma}: generated z=0 "
                      f"prior for {len(star_ids)}/{len(targets)} targets "
                      f"(TARGCLASS/TARGPROG/TARGSRVY resolved to STAR) -> {priors}")
            else:
                print(f"  star_z_prior_sigma={star_z_prior_sigma} requested but no "
                      f"targets resolved to STAR via targeting info -- no prior generated")

        t = targets[0]
        print(f"N spectra: {len(t.spectra)}")
        for i, sp in enumerate(t.spectra):
            v = sp.ivar > 0
            print(f"Arm {i}: wave {sp.wave[0]:.1f}-{sp.wave[-1]:.1f}  "
                f"n={len(sp.wave)}  "
                f"median flux={np.median(sp.flux[v]):.3f}  "
                f"median ivar={np.median(sp.ivar[v]):.6f}  "
                f"sum ivar={np.sum(sp.ivar[v]):.4e}")


        from redrock.zscan import spectral_data
        weights, flux, wflux = spectral_data(t.spectra)
        n0 = len(t.spectra[0].flux)
        w0 = weights[:n0]
        w1 = weights[n0:]
        f0 = flux[:n0]
        f1 = flux[n0:]
        print(f"Blue: sum_ivar={w0.sum():.3e}  ({100*w0.sum()/weights.sum():.1f}%)")
        print(f"Red:  sum_ivar={w1.sum():.3e}  ({100*w1.sum()/weights.sum():.1f}%)")
        print(f"Blue: median SNR/pix={np.median(f0[w0>0]*np.sqrt(w0[w0>0])):.3f}")
        print(f"Red:  median SNR/pix={np.median(f1[w1>0]*np.sqrt(w1[w1>0])):.3f}")


        # Distribute the targets.
        start = elapsed(None, "", comm=comm)
        dtargets = DistTargetsCopy(targets, comm=comm, root=0)

        # Get the dictionary of wavelength grids
        dwave = dtargets.wavegrids()
        stop = elapsed(start, "Distribution of {} targets"\
            .format(len(dtargets.all_target_ids)), comm=comm)

        # Read the template data
        dtemplates = load_dist_templates(dwave, templates=templates,
            comm=comm, mp_procs=mpprocs,redistribute=redistribute_templates, use_gpu=use_gpu, gpu_mode=gpu)

        # if gpu mode is set, we generate a copy of dtemplates without any parallel processing.
        # It will be used to generate model spectra and plotting purpose
        if use_gpu:
            print('Reading templates in non-distributed mode for generating model spectra')
            dtemplates_no_gpu = load_dist_templates(dwave, templates=templates,
                comm=comm, mp_procs=mpprocs,redistribute=False, use_gpu=False, gpu_mode=False)
        else:
            dtemplates_no_gpu = dtemplates



        # Read archetypes data if available
        if not archetypes is None:
            start_arctype = elapsed(None, "", comm=comm)
            darchetypes = All_archetypes(archetypes_dir=archetypes).archetypes
            stop = elapsed(start_arctype, "Preparing archetypes data took", comm=comm)
        else:
            darchetypes = None


        # Compute the redshifts, including both the coarse scan and the
        # refinement.  This function only returns data on the rank 0 process.
        start_redshift = elapsed(None, "", comm=comm)
        # Determine ncamera dynamically from the actual number of spectra arms
        ncamera = len(targets[0].spectra) if len(targets) > 0 else 1

        scandata, zfit = zfind(dtargets, dtemplates, mp_procs=mpprocs, nminima=nminima, archetypes = darchetypes, priors=priors, chi2_scan=chi2_scan, use_gpu=use_gpu, deg_legendre=2, per_camera=True,ncamera=ncamera, prior_sigma=prior_sigma, n_nearest=n_nearest, zminfit_npoints=25)


        # # ================================================================
        # # ARCHETYPE FALLBACK DIAGNOSTIC
        # # Run on specific APS_IDs from the catastrophic redshift list.
        # # ================================================================

        # # DEBUG_IDS = [335]
        # DEBUG_IDS = list(scandata.keys())

        # PLAUSIBLE_PAIRS    = {('GALAXY', 'QSO'), ('QSO', 'GALAXY')}
        # Z_AGREE_THRESHOLD  = 0.01
        # DEGRADATION_THRESH = 1.5
        # Z_LIT_APPROX       = 0.04846   # A2142 cluster redshift

        # for tid in DEBUG_IDS:
        #     if tid not in scandata:
        #         continue

        #     targ_zfit = zfit[zfit['targetid'] == tid]
        #     if len(targ_zfit) == 0:
        #         continue

        #     print(f"\n{'='*70}")
        #     print(f"ARCHETYPE FALLBACK DIAGNOSTIC — APS_ID={tid}")
        #     print(f"{'='*70}")

        #     # ----------------------------------------------------------------
        #     # SECTION 1: Coarse scan summary
        #     # ----------------------------------------------------------------
        #     print(f"\n--- Coarse scan ---")
        #     print(f"{'Template':<22} {'chi2 @ z_lit':>14} "
        #         f"{'best chi2':>12} {'best z':>10}")
        #     print(f"{'-'*22} {'-'*14} {'-'*12} {'-'*10}")

        #     coarse_best_chi2  = np.inf
        #     coarse_best_z     = np.nan
        #     coarse_best_class = ''

        #     for ft in sorted(scandata[tid].keys()):
        #         zz    = scandata[tid][ft]['redshifts']
        #         chi   = scandata[tid][ft]['zchi2']
        #         imin  = chi.argmin()
        #         i_lit = np.argmin(np.abs(zz - Z_LIT_APPROX))
        #         print(f"{ft:<22} {chi[i_lit]:>14.1f} "
        #             f"{chi[imin]:>12.1f} {zz[imin]:>10.4f}")
        #         if chi[imin] < coarse_best_chi2:
        #             coarse_best_chi2  = chi[imin]
        #             coarse_best_z     = zz[imin]
        #             coarse_best_class = ft.split(':::')[0]

        #     print(f"\nCoarse scan winner: "
        #         f"class={coarse_best_class}  "
        #         f"z={coarse_best_z:.5f}  "
        #         f"chi2={coarse_best_chi2:.1f}")

        #     # ----------------------------------------------------------------
        #     # SECTION 2: zfit results (all ranks)
        #     # ----------------------------------------------------------------
        #     print(f"\n--- zfit results ---")
        #     print(f"{'znum':<6} {'z':>10} {'class':<12} "
        #         f"{'chi2':>12} {'dchi2':>10} {'zwarn':>6}")
        #     print(f"{'-'*6} {'-'*10} {'-'*12} "
        #         f"{'-'*12} {'-'*10} {'-'*6}")
        #     for row in targ_zfit:
        #         print(f"{int(row['znum']):<6} "
        #             f"{float(row['z']):>10.5f} "
        #             f"{str(row['spectype']).strip():<12} "
        #             f"{float(row['chi2']):>12.1f} "
        #             f"{float(row['deltachi2']):>10.1f} "
        #             f"{int(row['zwarn']):>6}")

        #     # ----------------------------------------------------------------
        #     # SECTION 3: Best archetype (rank 0)
        #     # ----------------------------------------------------------------
        #     row0 = targ_zfit[targ_zfit['znum'] == 0]
        #     if len(row0) == 0:
        #         print(f"\n  WARNING: no ZNUM=0 row found")
        #         continue

        #     arch_chi2  = float(row0['chi2'][0])
        #     arch_z     = float(row0['z'][0])
        #     arch_class = str(row0['spectype'][0]).strip()
        #     arch_zwarn = int(row0['zwarn'][0])
        #     arch_sub   = str(row0['subtype'][0]).strip() \
        #                 if 'subtype' in targ_zfit.colnames else 'N/A'

        #     print(f"\n--- Best archetype (rank 0) ---")
        #     print(f"  class   = {arch_class}")
        #     print(f"  subtype = {arch_sub}")
        #     print(f"  z       = {arch_z:.5f}")
        #     print(f"  chi2    = {arch_chi2:.1f}")
        #     print(f"  zwarn   = {arch_zwarn}")

        #     # ----------------------------------------------------------------
        #     # SECTION 4: Degradation ratio
        #     # ----------------------------------------------------------------
        #     degradation = arch_chi2 / coarse_best_chi2 \
        #                 if coarse_best_chi2 > 0 else 0.0

        #     print(f"\n--- Degradation ---")
        #     print(f"  arch chi2    = {arch_chi2:.1f}")
        #     print(f"  coarse chi2  = {coarse_best_chi2:.1f}")
        #     print(f"  ratio        = {degradation:.3f}x  "
        #         f"({'TRIGGERS' if degradation > DEGRADATION_THRESH else 'silent'} "
        #         f"at threshold={DEGRADATION_THRESH})")


        #     # ----------------------------------------------------------------
        #     # SECTION 4b: PCA fine fit reliability check
        #     # ----------------------------------------------------------------
        #     pca_check = _pca_reliability_check(
        #         scandata, tid,
        #         coarse_best_class, coarse_best_z)

        #     z_pca        = pca_check['z_pca']
        #     pca_reliable = pca_check['reliable']
        #     z_reference  = z_pca if pca_reliable else coarse_best_z

        #     print(f"\n--- PCA fine fit reliability ---")
        #     print(f"  coarse z         = {coarse_best_z:.5f}")
        #     print(f"  z_pca (parabola) = {z_pca:.5f}")
        #     if np.isfinite(pca_check['zerr_pca']):
        #         print(f"  zerr_pca         = {pca_check['zerr_pca']:.5f}")
        #     else:
        #         print(f"  zerr_pca         = N/A")
        #     if np.isfinite(pca_check['dv_coarse']):
        #         print(f"  dv(coarse-PCA)   = {pca_check['dv_coarse']:.1f} km/s")
        #     else:
        #         print(f"  dv(coarse-PCA)   = N/A")
        #     print(f"  curvature_norm   = {pca_check['curvature_norm']:.2e}")
        #     print(f"  reliable         = {pca_reliable}")
        #     print(f"  reason           = {pca_check['reason']}")
        #     print(f"  z_reference used = {z_reference:.5f} "
        #         f"({'pca' if pca_reliable else 'coarse — PCA unreliable'})")

        #     # dv between arch and PCA reference
        #     dv_arch_pca = abs(arch_z - z_reference) / \
        #                 (1.0 + z_reference) * C_LIGHT \
        #                 if np.isfinite(z_reference) else np.nan
        #     if np.isfinite(dv_arch_pca):
        #         print(f"  dv(arch-PCA ref) = {dv_arch_pca:.1f} km/s  "
        #             f"({'> 3000 → z_shift trigger' if dv_arch_pca > 3000 else '< 3000 → no z_shift trigger'})")

        #     # ----------------------------------------------------------------
        #     # SECTION 5: Class mismatch evaluation (pair-aware)
        #     # ----------------------------------------------------------------
        #     pair = (coarse_best_class, arch_class)
        #     dz   = abs(arch_z - coarse_best_z)

        #     print(f"\n--- Class mismatch evaluation ---")
        #     print(f"  coarse class = {coarse_best_class}")
        #     print(f"  arch class   = {arch_class}")
        #     print(f"  pair         = {pair}")

        #     if arch_class == coarse_best_class:
        #         class_mismatch  = False
        #         mismatch_reason = 'same class → no mismatch'

        #     elif pair in PLAUSIBLE_PAIRS:
        #         z_agree         = dz < Z_AGREE_THRESHOLD
        #         class_mismatch  = not z_agree
        #         mismatch_reason = (
        #             f"plausible pair (AGN/host)  "
        #             f"dz={dz:.4f}  "
        #             f"z_agree={z_agree} "
        #             f"(threshold={Z_AGREE_THRESHOLD})  "
        #             f"→ mismatch={class_mismatch}")

        #     else:
        #         class_mismatch  = True
        #         mismatch_reason = (
        #             f"STAR involved — always genuine mismatch  "
        #             f"(dz={dz:.4f} irrelevant)")

        #     print(f"  result       = {mismatch_reason}")


        #     # ----------------------------------------------------------------
        #     # SECTION 6: Trigger decision — update to include z_shift
        #     # ----------------------------------------------------------------
        #     ratio_trigger = (degradation > DEGRADATION_THRESH)

        #     z_shift_trigger = (arch_class == coarse_best_class) and \
        #                     pca_reliable                        and \
        #                     np.isfinite(dv_arch_pca)            and \
        #                     (dv_arch_pca > 3000.0)

        #     trigger_parts = []
        #     if class_mismatch:
        #         trigger_parts.append(
        #             f"CLASS MISMATCH ({mismatch_reason})")
        #     if ratio_trigger:
        #         trigger_parts.append(
        #             f"RATIO={degradation:.2f}x > {DEGRADATION_THRESH}")
        #     if z_shift_trigger:
        #         trigger_parts.append(
        #             f"Z SHIFT within {arch_class} "
        #             f"dv={dv_arch_pca:.0f} km/s > 3000")
        #     if not trigger_parts:
        #         trigger_parts.append(
        #             f"SILENT  "
        #             f"(ratio={degradation:.2f}x  "
        #             f"class_mismatch={class_mismatch}  "
        #             f"z_shift={z_shift_trigger}  "
        #             f"pca_reliable={pca_reliable})")

        #     print(f"\n--- Trigger decision ---")
        #     if any('SILENT' not in p for p in trigger_parts):
        #         print(f"  FIRES: {' + '.join(trigger_parts)}")
        #     else:
        #         print(f"  {trigger_parts[0]}")

        #     # ----------------------------------------------------------------
        #     # SECTION 6: Final trigger decision
        #     # ----------------------------------------------------------------
        #     ratio_trigger = (degradation > DEGRADATION_THRESH)

        #     trigger_parts = []
        #     if class_mismatch:
        #         trigger_parts.append(
        #             f"CLASS MISMATCH "
        #             f"(coarse={coarse_best_class} "
        #             f"arch={arch_class} "
        #             f"dz={dz:.4f})")
        #     if ratio_trigger:
        #         trigger_parts.append(
        #             f"RATIO={degradation:.2f}x > {DEGRADATION_THRESH}")

        #     print(f"\n--- Trigger decision ---")
        #     if trigger_parts:
        #         print(f"  FIRES: {' + '.join(trigger_parts)}")
        #     else:
        #         print(f"  SILENT  "
        #             f"(ratio={degradation:.2f}x  "
        #             f"class_mismatch={class_mismatch})")

        #     # ----------------------------------------------------------------
        #     # SECTION 7: Fallback rank search (only if trigger fired)
        #     # ----------------------------------------------------------------
        #     if class_mismatch or ratio_trigger or z_shift_trigger:
        #         print(f"\n--- Fallback rank search ---")
        #         print(f"  Reference z: {z_reference:.5f} "
        #             f"({'PCA fine fit' if pca_reliable else 'coarse grid'})")
        #         print(f"  Target: class={coarse_best_class}")
        #         print(f"\n--- Fallback rank search ---")
        #         print(f"  Target: class={coarse_best_class}  "
        #             f"z≈{coarse_best_z:.5f}")
        #         print(f"\n  {'znum':<6} {'z':>10} {'class':<12} "
        #             f"{'chi2':>12} {'dz':>10} {'dz_w':>10} "
        #             f"{'zwarn':>6}  note")
        #         print(f"  {'-'*6} {'-'*10} {'-'*12} "
        #             f"{'-'*12} {'-'*10} {'-'*10} {'-'*6}  {'-'*12}")

        #         best_idx = None
        #         best_dz  = np.inf

        #         for ri in range(len(targ_zfit)):
        #             row_class = str(targ_zfit['spectype'][ri]).strip()
        #             row_zwarn = int(targ_zfit['zwarn'][ri])
        #             row_z     = float(targ_zfit['z'][ri])
        #             row_chi2  = float(targ_zfit['chi2'][ri])
        #             row_znum  = int(targ_zfit['znum'][ri])
        #             dz_i      = abs(row_z - coarse_best_z)
        #             dz_w      = dz_i if row_zwarn == 0 else dz_i + 0.5
        #             matches   = (row_class == coarse_best_class)

        #             note = ''
        #             if matches and dz_w < best_dz:
        #                 best_dz  = dz_w
        #                 best_idx = ri
        #                 note     = '← best'
        #             elif matches:
        #                 note = '(class match)'

        #             print(f"  {row_znum:<6} {row_z:>10.5f} "
        #                 f"{row_class:<12} {row_chi2:>12.1f} "
        #                 f"{dz_i:>10.4f} {dz_w:>10.4f} "
        #                 f"{row_zwarn:>6}  {note}")

        #         print(f"\n  Fallback outcome:")
        #         if best_idx is None:
        #             print(f"  → No {coarse_best_class} rank found "
        #                 f"— keeping rank 0 unchanged")
        #         elif best_idx == 0:
        #             print(f"  → Best match is already rank 0 "
        #                 f"— no swap needed")
        #         else:
        #             sel = targ_zfit[best_idx]
        #             print(f"  → Would swap rank 0 with ZNUM={int(sel['znum'])}: "
        #                 f"class={str(sel['spectype']).strip()}  "
        #                 f"z={float(sel['z']):.5f}  "
        #                 f"chi2={float(sel['chi2']):.1f}  "
        #                 f"ZWARN={int(sel['zwarn'])}")
        #             # Cross-check against Z_LIT
        #             dv_fallback = abs(float(sel['z']) - Z_LIT_APPROX) \
        #                         / (1 + Z_LIT_APPROX) * 3e5
        #             dv_arch     = abs(arch_z - Z_LIT_APPROX) \
        #                         / (1 + Z_LIT_APPROX) * 3e5
        #             print(f"  → dV(fallback vs Z_LIT) = {dv_fallback:.1f} km/s")
        #             print(f"  → dV(arch     vs Z_LIT) = {dv_arch:.1f} km/s")
        #             if dv_fallback < dv_arch:
        #                 print(f"  → IMPROVEMENT ✓")
        #             else:
        #                 print(f"  → NO IMPROVEMENT ✗  "
        #                     f"(coarse scan z may also be wrong)")

        #     # ----------------------------------------------------------------
        #     # SECTION 8: IVAR balance and masking profile
        #     # ----------------------------------------------------------------
        #     t_obj = [tg for tg in targets if tg.id == tid]
        #     if t_obj:
        #         t_obj = t_obj[0]
        #         print(f"\n--- IVAR balance ---")
        #         for arm_idx, arm_name in enumerate(['Blue', 'Red ']):
        #             iv    = t_obj.spectra[arm_idx].ivar
        #             fl    = t_obj.spectra[arm_idx].flux
        #             wv    = t_obj.spectra[arm_idx].wave
        #             valid = iv > 0
        #             if valid.sum() > 0:
        #                 snr = float(np.median(
        #                     fl[valid] * np.sqrt(iv[valid])))
        #                 print(f"  {arm_name}: "
        #                     f"sum_ivar={iv[valid].sum():.3e}  "
        #                     f"valid={valid.sum()}px  "
        #                     f"SNR={snr:.2f}  "
        #                     f"pct_valid={100*valid.mean():.1f}%")

        #         # Red arm masking profile by wavelength bin
        #         if len(t_obj.spectra) > 1:
        #             iv_red = t_obj.spectra[1].ivar
        #             wv_red = t_obj.spectra[1].wave
        #             valid_red = iv_red > 0
        #             print(f"\n  Red arm masking by region:")
        #             bins = [(5925, 6500), (6500, 7000),
        #                     (7000, 7600), (7600, 7700),
        #                     (7700, 8200), (8200, 8700),
        #                     (8700, 9380)]
        #             for wmin, wmax in bins:
        #                 mask = (wv_red >= wmin) & (wv_red < wmax)
        #                 if mask.sum() > 0:
        #                     pct_masked = 100 * (~valid_red[mask]).mean()
        #                     n_valid    = valid_red[mask].sum()
        #                     bar_len    = int(pct_masked / 5)
        #                     bar        = '█' * bar_len
        #                     print(f"    {wmin}-{wmax}Å: "
        #                         f"{mask.sum():5d}px  "
        #                         f"{n_valid:5d} valid  "
        #                         f"{pct_masked:5.1f}% masked  "
        #                         f"|{bar:<20}|")

        #     print(f"\n{'='*70}\n")

        # breakpoint()


        stop = elapsed(start_redshift, "Computing redshifts took", comm=comm)

        # print(scandata)

        start_tables = elapsed(None, "", comm=comm)
        if comm_rank == 0:

            zfitall = gen_zfitall(zfit, apsmeta)
            if zall_fname is not None:
                rr_write_fits(zall_fname, zfitall, dtemplates, darchetypes=darchetypes, match_table=match_table)

            if ntop == 1:
                zbest = gen_zbest_single(zfitall, scandata, apsmeta, srvyconf=srvyconf)
                if zbest_fname is not None:
                    rr_write_fits(zbest_fname, zbest, dtemplates, darchetypes=darchetypes, match_table=match_table)
            else:
                zbest = gen_zbest_multiple(zfitall, scandata, apsmeta, ntop=ntop, add_extra=True, srvyconf=srvyconf)
                if zbest_fname is not None:
                    rr_write_fits(zbest_fname, zbest, dtemplates, darchetypes=darchetypes, match_table=match_table)

            ## As zbest has been modified (change class of targets with fib_stat !='A' to nan), for the following step
            ## we use the original zbest table (zbest_original)

            # gen_zspec now computes a model per available rank (up to
            # PyAPS.apsPlot.redrock.MAX_RANKS), not just rank 0.
            zspec = gen_zspec(zbest, dtargets.local(), apsmeta, dtemplates_no_gpu, rank=0, darchetypes=darchetypes, sens_corr = sens_corr)
            if zspec_fname is not None:
                rr_write_fits(zspec_fname, zspec, dtemplates, darchetypes=darchetypes, match_table=match_table)

        stop = elapsed(start_tables, "Preparing ALL output tables/fits files took", comm=comm)

        ## START generating the figures, demonstrating the raw spectra and the best fits
        if (figdir is not None) and (comm_rank == 0):
            start_fig = elapsed(None, "", comm=comm)
            try:
                make_rrplot(zbest, zspec, apsmeta['setups'],  figdir, collapse_fname = collapse_fname)
            except Exception as _rrplot_exc:
                import traceback as _tb
                print('Failed to generate plots. 1- Check your X11 configs. 2- Check spectra in output fits file.')
                print(f'  Exception type : {type(_rrplot_exc).__name__}')
                print(f'  Exception value: {_rrplot_exc}')
                print('  Traceback:')
                _tb.print_exc()


            stop = elapsed(start_fig, "Generating figures took", comm=comm)

        global_stop = elapsed(global_start, "Total run time", comm=comm)

        if debug:
            import IPython
            IPython.embed()

        if return_outputs:
            return scandata, zbest, zspec, zfitall
        else:
            return


    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "Proc {}: {}".format(comm_rank, x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()
        if comm is not None:
            comm.Abort()

        global_stop = elapsed(global_start, "Total run time", comm=comm)

        if debug:
            import IPython
            IPython.embed()

        if return_outputs:
            return None, None, None, None
        else:
            return

##########################################################
def rrweave(options=None, comm=None):
    """Estimate redshifts for WEAVE targets.

    This loads targets serially and copies them into a DistTargets class.
    It then runs redshift fitting and writes the output to a catalogue.

    Args:
        options (list): optional list of command line options to parse.
        comm (mpi4py.Comm): MPI communicator to use.

    """

    # Shared registry for the ~20 flags essentially every aps_*.py
    # processing script re-declares (--infiles, --aps_ids, --targsrvy,
    # --sens_corr, --join_arms, --caldir, --outpath, ...) -- see
    # aps_common_args.py's own module docstring for the full rationale
    # (explicit request: "I hate this repea[t]ation in all codes...").
    # `overrides` below reproduces this script's own current --help
    # wording verbatim wherever it genuinely differs from the shared
    # default (including the pre-existing "ellipse" vs "circle" wording
    # for --area -- aps_rr.py is the one script using the ellipse form
    # description, [RA_CENT, DEC_CENT, A, B, ANGLE], matching how `area`
    # is actually used everywhere else in this codebase; every other
    # script's own --area help text describing a circle looks like a
    # stale copy-paste, left as-is here since fixing it isn't this
    # migration's job).
    parser = build_common_parser(
        description="Estimate redshifts from WEAVE target spectra.",
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "caldirs", "output"],
        overrides={
            "area": {"help": "The area in [RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec], "
                              "ANGLE [CCW in deg]] to be considered in analysis"},
            "outpath": {"help": "Directory to keep WEAVE_REDROCK outputs"},
        },
        extra_args=[
            (("--ntop",), dict(type=int, default=1, required=False,
                                help="number of top z/class to be returned. ntop=1 means single mode")),
            (("-t", "--templates"), dict(type=none_or_str, default=None, required=False,
                                          help="template file or directory")),
            (("--srvyconf",), dict(type=none_or_str, default=None, required=False,
                                    help="json config file (Pre-defined class type, based on TARGSRVY)")),
            (("--archetypes",), dict(type=none_or_str, default=None, required=False,
                                      help="archetype file or directory for final redshift comparisons")),
            (("--zall",), dict(type=str2bool, default=False, required=False,
                                help="if True, it creates a fits file contains all WEAVE_REDROCK outputs. [FITS file]")),
            (("--priors",), dict(type=none_or_str, default=None, required=False,
                                  help="optional redshift prior file")),
            (("--chi2_scan",), dict(type=none_or_str, default=None, required=False,
                                     help="Load the file containing already computed chi2 scan")),
            (("--nminima",), dict(type=int, default=3, required=False,
                                   help="the number of redshift minima to search")),
            (("--fig",), dict(type=str2bool, default=False, required=False,
                               help="if True, the code also produces plots of the best fitted model")),
            (("--cache_Rcsr",), dict(type=str2bool, default=True, required=False, help="Cache Rcsr")),
            (("--debug",), dict(type=str2bool, default=False, required=False,
                                 help="debug with ipython (only if communicator has a single process)")),
            (("--mp",), dict(type=int, default=1, required=False,
                              help="if not using MPI, the number of multiprocessing processes to use "
                                   "(defaults to half of the hardware threads)")),
            (("--gpu",), dict(type=str2bool, default=False, required=False, help="Use GPU if available")),
            (("--max_gpuprocs",), dict(type=int, default=2, required=False,
                                        help="limit number of processes using GPUs")),
            (("--skysub_mask_residuals",), dict(type=str2bool, default=True, required=False,
                                                 help="Mask sky-subtraction residuals by putting ivar=0")),
            (("--extinction_corr",), dict(type=str2bool, default=False, required=False,
                                           help="opt-in Galactic (SFD98+Fitzpatrick99) dust extinction "
                                                "correction, applied per-target before REDROCK fitting. "
                                                "See APSOB's extinction_corr docstring re: extinction_ebv_scale "
                                                "-- full correction (ebv_scale=1.0) is validated for GALAXY/QSO "
                                                "but empirically hurts STAR classification for reddened, "
                                                "low-Galactic-latitude targets.")),
            (("--extinction_ebv_scale",), dict(type=float, default=1.0, required=False,
                                                help="fraction of the full SFD E(B-V) to apply when "
                                                     "--extinction_corr is set (default 1.0 = full)")),
            (("--extinction_mapdir",), dict(type=none_or_str, default=None, required=False,
                                             help="directory containing SFD_dust_4096_{ngp,sgp}.fits "
                                                  "(default: PyAPS_data/DUST or $DUST_DIR)")),
            (("--star_z_prior_sigma",), dict(type=float, default=None, required=False,
                                              help="opt-in: if set, auto-generate a z=0 Gaussian "
                                                   "redshift prior (this sigma, in dz) for every "
                                                   "target whose TARGCLASS/TARGPROG/TARGSRVY "
                                                   "resolves to STAR, applied at the coarse-scan "
                                                   "stage (redrock's own priors mechanism) rather "
                                                   "than post-hoc. Ignored if --priors is also set. "
                                                   "~0.0067 corresponds to ~2000 km/s.")),
        ],
    )

    ## Check if any command-line argument has been passed to the module. It counts the number of system arguments to check this.
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')

        args = parser.parse_args(options)



    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.size
        comm_rank = comm.rank

    # Check arguments- all processes have this, so just check on the first
    # process

    if comm_rank == 0:
        if args.debug and comm_size != 1:
            print("--debug can only be used if the communicator has one "
                " process")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

    if args.gpu:
        try:
            import cupy
            gpu_ok = cupy.is_available()
        except ImportError:
            gpu_ok = False
        if not gpu_ok:
            print("ERROR: cupy or GPU not available")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()
            else:
                sys.exit(1)


    ### Now we have both infiles and wlranges array. We use resolve_common_args
    ### (aps_common_args.py) to update infiles/wlranges/arms_ratio/join_arms
    ### and convert the remaining comma-separated CLI strings into real
    ### Python types, exactly matching this script's own previous logic
    ### (see that function's own docstring). However, we had similar test
    ### done by APSOB.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio
    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
    area, mask_areas = resolved.area, resolved.mask_areas

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ', '')


    figdir=None
    if args.fig:

        figdir = Path(args.outpath).joinpath('figs')
        if not Path(figdir).exists():
            Path(figdir).mkdir(parents=True, exist_ok=True)
            print("FIGDIR: %s Created!" %(figdir))





    # print args and assigned/default values on the screen
    print_args(args,module='RR', version= aps_constants.__aps_rr_version__, path=outpath, headname=args.headname)

    zbest_fname=os.path.join(args.outpath,'zbest_'+str(args.headname)+'.fits')

    zall_fname=None
    if args.zall:
        zall_fname=os.path.join(args.outpath,'zall_'+str(args.headname)+'.fits')
        zall_fname=zall_fname.replace(' ', '')


    zspec_fname=os.path.join(args.outpath,'zspec_'+str(args.headname)+'.fits')
    zspec_fname=zspec_fname.replace(' ', '')

    if (not args.overwrite) and os.path.exists(zbest_fname):
        print('skipping, products already exist. If this is a test run, change the --headname')
        sys.exit()


    ### Before running the main worker, we first make sure everything is OK
    # try:
    infiles_check = l1_fileinfo(args.infiles)
    fcheck_targs, fcheck_idt, fcheck_info, fcheck_la, fcheck_wcs = gen_targlist(infiles_check['infiles'][0], infiles_check['mode'], aps_ids = aps_ids, targsrvy= targsrvy, targclass = targclass,
        mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, la_out=False)

    # ## Make sure at least one valid aps_id exist after applying all filters
    assert len(fcheck_targs) > 0, 'No valid APS_ID(s) found'


    status_fibs = [fcheck_info['FIB_STATUS'][fcheck_idt[fchid]] for fchid in fcheck_targs]


    ## Count number of targets with ACTIVE fibres
    ## Note, we also check if fib_status of fibres are OK through 1- the main rrweave_worker where we read data and
    ## 2- where we pack data into the redrock in the pack_2_redrock function
    n_active_fibs = status_fibs.count('A')

    ## Make sure at least one valid aps_id exist (with fib_status = A: ACTIVE) after applying all filters
    assert n_active_fibs > 0, 'FIB_STATUS CHECK: No valid APS_ID(s) found'


    ## Update args.mp if it was greater than number of targets (with valid fib-type)
    if n_active_fibs < args.mp:
        args.mp = n_active_fibs
        print('NOTE: --mp param is updated to %d'%(args.mp))
    # except:
    #     print('ERROR : %s' %(sys.exc_info()[1]))
    #     return


    rrweave_worker(args.infiles, args.templates, srvyconf= args.srvyconf,
        zbest_fname=zbest_fname, zall_fname=zall_fname, zspec_fname=zspec_fname, aps_ids=aps_ids, ntop = args.ntop,
        targsrvy= targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=args.sens_corr, mask_gaps=args.mask_gaps, safe_mask_gaps=args.safe_mask_gaps, vacuum=args.vacuum, tellurics=args.tellurics,
        fill_gap=args.fill_gap, arms_ratio=arms_ratio, join_arms=args.join_arms, ncpus= args.mp, comm=comm,
        comm_rank=comm_rank, comm_size=comm_size,nminima=args.nminima, archetypes=args.archetypes,cache_Rcsr=args.cache_Rcsr,
        priors=args.priors, chi2_scan=args.chi2_scan, figdir=figdir, debug=args.debug, return_outputs=False, collapse=False,
        collapse_fname=None, match_table=None,gpu=args.gpu, max_gpuprocs = args.max_gpuprocs, catdir=args.catdir, caldir=args.caldir, configdir=args.configdir, skysub_mask_residuals=args.skysub_mask_residuals,
        extinction_corr=args.extinction_corr, extinction_ebv_scale=args.extinction_ebv_scale, extinction_mapdir=args.extinction_mapdir,
        star_z_prior_sigma=args.star_z_prior_sigma)

    ## Please note, if you set return_outputs=True, then the rrweave_worker function will return
    ## scandata, zbest, zspec, zfitall
    ## which is a useful option to use this function (rrweave_worker) as a standalone function




##########################################################
if __name__ == '__main__':



    ##  TEST 1
    debug_demo= ['--infiles', '<PYAPS_DATA>_dev/L1/star_test/stacked_1002046.fit', '<PYAPS_DATA>_dev/L1/star_test/stacked_1002045.fit',
    '--aps_ids', '1006,1007', # or 'None' to run for all available fibreids
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area', 'None',
    '--mask_areas', 'None',
    '--wlranges', '4200.0,6000', '6000.0,8000',
    '--sens_corr', 'True',
    '--safe_mask_gaps', 'True',
    '--mask_gaps', 'True',
    '--tellurics', 'False',
    '--vacuum', 'True',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0, 0.83',
    '--join_arms', 'False',
    '--templates', '<PYAPS_DIR>/PyAPS_templates/templates_RR/',
    '--srvyconf', '<PYAPS_DIR>/configs/weave_cls.json',
    '--archetypes', '<PYAPS_DIR>/PyAPS_templates/templates_ARC_RR/',
    '--outpath', '<PYAPS_DATA>_dev/L2/20160903/3294/',
    '--headname', 'stacked_1002046__stacked_1002045',
    '--zall', 'True',
    '--priors', 'None',
    '--chi2_scan', 'None',
    '--nminima' , '3',
    '--fig', 'True',
    '--cache_Rcsr', 'False',
    '--debug', 'False',
    '--overwrite', 'True',
    '--mp' ,'2',
    '--ntop' ,'1',
    '--gpu', 'False',
    '--max_gpuprocs', '2',
    '--caldir', '<PYAPS_DATA>/CAL',
    '--catdir', '<PYAPS_DATA>/CAT',
    '--configdir', '<PYAPS_DIR>/configs/ExGal_configs',
    '--skysub_mask_residuals', 'True'
    ]

    ## TEST 2
    debug_demo= ['--infiles', '<PYAPS_DATA>/L1/20240308/single_3048995.fit', '<PYAPS_DATA>/L1/20240308/single_3048994.fit',
    '--aps_ids', 'None',
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area', 'None',
    '--mask_areas', 'None',
    '--wlranges', '4200.0,6000', '6000.0,8000',
    '--sens_corr', 'True',
    '--mask_gaps', 'True',
    '--safe_mask_gaps', 'True',
    '--tellurics', 'False',
    '--vacuum', 'True',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0, 0.83',
    '--join_arms', 'False',
    '--templates', '<PYAPS_DIR>/PyAPS_templates/templates_RR/',
    '--srvyconf', '<PYAPS_DIR>/configs/weave_cls.json',
    # '--archetypes', '<PYAPS_DIR>/PyAPS_templates/templates_ARC_RR/',
    '--archetypes', 'None',
    '--outpath', '<PYAPS_DATA>/L2/20240308_test/12052/',
    '--headname', 'single_3048995__single_3048994_test',
    '--zall', 'True',
    '--priors', 'None',
    '--chi2_scan', 'None',
    '--nminima' , '3',
    '--fig', 'True',
    '--cache_Rcsr', 'False',
    '--debug', 'False',
    '--overwrite', 'True',
    '--mp' ,'8',
    '--ntop' ,'3',
    '--gpu', 'False',
    '--max_gpuprocs', '2',
    '--caldir', '<PYAPS_DATA>/CAL',
    '--catdir', '<PYAPS_DATA>/CAT',
    '--configdir', '<PYAPS_DIR>/configs/ExGal_configs',
    '--skysub_mask_residuals', 'True'
    ]


    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    rrweave(options=debug_demo, comm=None)
##########################################################
