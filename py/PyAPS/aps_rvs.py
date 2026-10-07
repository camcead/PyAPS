import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Real bug caught by a real end-to-end SLURM test run (1 Sep 2026, all 3
# cluster nodes): rvspecfit's neural-net interpolator (nn/RVSInterpolator.py)
# reads its torch device from metadata baked into the interpolator .h5
# file at template-build time (rvspecfit's spec_inter.getInterpolator ->
# fd['device'] from the .h5, not from rvs_config.yaml) -- the current
# templates_RVS/*.h5 files (regenerated ~19-21 Aug 2026) were apparently
# built on a machine with a GPU, so that baked-in value is 'cuda'. This
# Batch RVS SLURM jobs normally never request a GPU allocation
# (aps_runner.py's RVS sbatch step has no --gres=gpu), so
# torch.cuda.is_available() is False inside the job regardless of which
# node it lands on -- including nodes that do have a physical
# GPU, because SLURM's own GPU cgroup isolation hides it from a job that didn't ask for it. Result:
# every RVS run crashed with "Attempting to deserialize object on a CUDA
# device but torch.cuda.is_available() is False" -- 100% failure rate,
# previously masked because it had apparently never been exercised
# end-to-end against these specific (~2-week-old) templates before.
#
# rvspecfit's own nn/RVSInterpolator.py already supports exactly this
# override, checked at interpolator-load time: RVS_NN_DEVICE in the
# environment wins over the .h5's baked-in device. resolve_rvs_device()
# is the single source of truth for the policy (also driven by the CLI's
# own --device flag below, resolved again -- and forced, not just
# setdefault -- right after argparse, still before proc_rvs() opens its
# multiprocessing pool, so forked workers inherit whatever was decided).
# This module-load-time call is only the safety-net default for anything
# that imports proc_rvs()/this module directly without going through the
# CLI (tests, other scripts) -- 'cpu', matching --device's own default,
# since no SLURM node on this cluster currently gets a real --gres=gpu
# allocation regardless of physical hardware (see above).
def resolve_rvs_device(requested="cpu"):
    """
    Decide (and set RVS_NN_DEVICE to) the torch device rvspecfit's neural-net
    interpolator should use, and return what was chosen ('cpu' or 'cuda').

    requested: 'cpu' (always CPU), 'gpu' (CUDA if actually available and
    visible to this process, else falls back to CPU with a warning -- never
    hard-fails just because the GPU isn't there), or 'auto' (same
    availability probe as 'gpu', but silently -- no warning, since not
    getting a GPU is the expected/default case, not a fallback from an
    explicit request).
    """
    requested = (requested or "cpu").strip().lower()

    def _cuda_available():
        try:
            import torch as _torch_cuda_probe
            avail = bool(_torch_cuda_probe.cuda.is_available())
            del _torch_cuda_probe
            return avail
        except ImportError:
            return False

    if requested == "cpu":
        device = "cpu"
    elif requested in ("gpu", "auto"):
        if _cuda_available():
            device = "cuda"
        else:
            device = "cpu"
            if requested == "gpu":
                print("WARNING: --device gpu requested but no CUDA device is "
                      "visible to this process (no --gres=gpu allocation, or "
                      "no physical GPU on this node) -- falling back to CPU.")
    else:
        print(f"WARNING: unrecognised --device '{requested}', falling back to 'cpu'.")
        device = "cpu"

    os.environ["RVS_NN_DEVICE"] = device
    return device


resolve_rvs_device("cpu")

import glob
import sys
import argparse
import time
import itertools
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
import warnings
import pandas
from astropy.io import fits
import astropy.wcs as pywcs
import numpy as np
from astropy.table import  Table, Column, vstack, join
import datetime
import yaml
from rvspecfit import fitter_ccf, vel_fit, spec_fit, utils, frozendict
import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist, add_extra_columns, makeR_from_fwhm_array
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import aps_constants
from PyAPS.apsPlot.rvs import make_rvs_plot
APSVERS = PyAPS.__version__
C_KMS = 299792.458
#################################################
"""
aps_rvs
Python wrapper to run RVSPECFIT code (BY SERGEY KOPOSOV) on WEAVE DATA and generate output tables

versions:
 1.0 By A. Molaeinezhad (IAC, July 2019) - First version
 1.2 By A. Molaeinezhad (IAC, August 2019)
 2.0 By A. Molaeinezhad (IAC, November 2019)
 3.0 By A. Molaeinezhad (IAC, October  2020)
 3.1 By A. Molaeinezhad (CASU, May 2021)
 3.2 By A. Molaeinezhad (CASU, Oct 2021)
 3.3 By A. Molaeinezhad (CASU, Dec 2025)


TODO list:

########### BAISC APS PARAM ###################################################################################################
|    param                |    APS default  |  rvsweave EQUIVALENT          |  rvsweave DEFAULT  |   available options/notes   |
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
vacuum (Optional)         |     False       |  --vacuum (Optional)         |        False      |                               |
tellurics (Optional)      |     False       |  --tellurics (Optional)      |        True       |                               |
fill_gap (Optional)       |     False       |  --fill_gap (Optional)       |        True       |                               |
arms_ratio (Optional)     |     None        |  --arms_ratio (Optional)     |        None       | for R band in OPR3B  is 0.83  |
join_arms (Optional)      |     False       |  --join_arms (Optional)      |        False      |                               |
funit (Optional)          |     1.0e18      |  USE DEFAULT APS value       |        AS APS     |                               |
offset_gap_pix (Optional) |     10          |  USE DEFAULT APS value       |        AS APS     |                               |
                          |                 |                              |                   |                               |
######### rvsweave INITAIL PARAM ################################################################################################
                          |                 |                              |                   |                               |
######### DEDICATED rvsweave PARAM ##############################################################################################
                          |                 |  --outpath   (Required)      |          -        |                               |
                          |                 |  --config    (Required)      |          -        |                               |
                          |                 |  --classfile (Required)      |          -        |                               |
                          |                 |  --templates (Optional)      |        None       |                               |
                          |                 |  --headname  (Optional)      |     'headname'    |                               |
                          |                 |  --fig       (Optional)      |        False      |                               |
                          |                 |  --overwrite (Optional)      |        False      |                               |
                          |                 |  --mp        (Optional)      |         1         |                               |
                          |                 |  --outspec   (Optional)      |        False      |                               |
################################################################################################################################

Example:
python3 <PYAPS_DIR>/py/PyAPS/aps_rvs.py --infiles <PYAPS_DATA>/star_test/stacked_1002046.fit <PYAPS_DATA>/star_test/stacked_1002045.fit --outpath <PYAPS_DIR>/PyAPS_results/20160903/3294/ --headname stacked_1002046__stacked_1002045 --wlranges 4200.0,6000.0 6000.0,8000.0 --aps_ids 1007,1006 --targsrvy None --targclass None --mask_aps_ids None --area None --mask_areas None --classfile <PYAPS_DIR>/PyAPS_results/20160903/3294/zbest_stacked_1002046__stacked_1002045.fits --config <PYAPS_DIR>/configs/rvs_config.yaml --join_arms False --mp 2 --outspec True --overwrite True --fig True --sens_corr True --mask_gaps True --safe_mask_gaps True --tellurics False --vacuum False --fill_gap True --arms_ratio 1.0,0.83


History:
10 Jan 2020: Fixed a bug, occurred when number of CPUs was larger than targets classified as 'STAR' (By A. Molaeinezhad)
10 Jan 2020: Add an example to run the code through command line (By A. Molaeinezhad)
23 Jan 2020: Remove "res" param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
23 Jan 2020: Remove mode param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
23 Jan 2020: Remove setups param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
23 Jan 2020: ADD targsrvy param. to specific the list of surveys you want to proceed with (By A. Molaeinezhad)
23 Jan 2020: ADD targclass param. to specific the type of targets (coming from targclass) to proceed with (By A. Molaeinezhad)
23 Jan 2020: ADD mask_aps_ids param. to specific the list of aps_ids that you want to be masked from the process (By A. Molaeinezhad)
23 Jan 2020: update examples and readme/header content (By A. Molaeinezhad)
23 Jan 2020: handle exception in case an error occurred in plotting routine (By A. Molaeinezhad)
23 Jan 2020: updated templates names, to consider CAMERA, mode and xbin, ybin (By A. Molaeinezhad)
31 Jan 2020: (Important update) put the common part of parallel and serial modes of code (generating output tables/spec)
             into a function and test it.
31 Jan 2020: Add a new feature using l1_fileinfo to make sure order or inputs (infiles, wlranges) are OK, as well as double-checking the
             the merge_arms parameters. This checks done at two level 1. at script level (this code itself) and 2. and APSOB level
06 Feb 2020: Add area and mask_areas into the criteria to select aps_ids/spaxels (By A. Molaeinezhad)
06 Feb 2020: Update working examples
25 Feb 2020: Replace 'FIBREID' with 'APS_ID' (and all other relevant params) to be compatible with all possible L1 structures
25 Feb 2020: Replace NSPEC (INT) with TARGID (STR), as NSPEC is not available in all L1 structures
25 Feb 2020: Update the whole plotting routine for this module. e.g. Add ylablel to plot. ADD/update units to output tables/spectra.
25 feb 2020: replace bad values in output tables and plots by np.nan value
25 Feb 2020: Update header of all outputs (tables, spectra) to include units, reference files, etc.
25 Feb 2020: Update Versioning of all sub-modules for STAR modules.
04 March 2020: replace the parameter RB_ratio (float) with a more general arms_ratio (list of floats).
04 March 2020: A new feature added to support more than 2 arms/bands (to handle possible R+G+B setup in HR)
04 March 2020: changing the way code save fits tables. Now the code use more low-level HDU instead of fits.write function
09 September 2020: replaced read_config from rvs.utils with read_config_APS_RVS
             in which if the parameter be a path (e.g. $HOME) it will be automatically expanded.
             Useful to have portable version of config file
07 October 2020: Update some functions and definitions like chisq_cont_array or SKEWNESS --> VEL_SKEWNESS or Kurtosis to VEL_...
            to be matched with the latest version of RVSPECFIT (Version 0.0.4.200429)
20 Oct 2020: in-line documentation updated
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
26 Aug 2022: the main RVS code updated from the Sergey git
26 Aug 2022: IFU support added to this code (user must set classfile  param to 'IFU')
26 Aug 2022: new method to measure paramDict0['vsini'], based on the updated from Sergey for DESSI project.
26 Aug 2022: green templates regenerated to solve the inconsistency between wavelength ranges of templates and data
26 Aug 2022: The function to write final fits file separated from the table generators.
30 Aug 2022: 'add_extra_columns' added to be able to add extra columns to the final fits outputs
15 Oct 2023: Support for multiple z/class enteries added. index_in_class function added to the aps_utils to extract z/calss info based on the targeted_class
14 Aug 2025: caldir, catdir and configdir added to inputs. So it can now generate LSF from the L1 calibrations insated of default resolution
06 Oct 2025: IFU mode retracted as it is now a seperate aps module (aps_ifu_rvs)
08 Dec 2025: first attemp of using FWHM interpolation to generate resolution matrix for each target/setup
08 Dec 2025: vrad_ccd and quality flags added to output tables/files
21 Jan 2026: Tellurics correction added to the default mode in the script params
23 Jan 2026: Resolution matrix generation updated to use empirical sigma_0 values based on LSF templates from January 2026
23 Jan 2026: Full implementation of the Resolution matrix in  spec_fit (ResolMatrix)
02 Feb 2026: Use fmode_unit to generate templates names (like MOS, MIFU, LIFU) instead of mode
02 Aug 2026: make_rvs_plot moved to PyAPS.apsPlot.rvs (Plotly), built on the same spectrum_overlay_figure shared with aps_rr - part of the new cross-module PyAPS.apsPlot platform. Same output convention and layout (2 arms x 2 halves), verified against the legacy matplotlib output on OB 5011 APS_ID=319.

"""
#################################################
bitmasks = {
    'CHISQ_WARN': 1,      # Chi-square vs continuum too small
    'RV_WARN': 2,         # RV too close to edge
    'RVERR_WARN': 4,      # RV error too large
    'PARAM_WARN': 8,      # Parameters too close to edge
    'VSINI_WARN': 16,     # vsini too large
    'BAD_SPECTRUM': 32,   # Issue with spectrum
    'BAD_HESSIAN': 64     # Issue with Hessian
}
#################################################

def get_rvs_warn(res1, chisq_cont_total, config):
    """Evaluate fit quality and return warning bitmask"""
    rvs_warn = 0

    # 1. Chi-square warning (fit not better than continuum?)
    chisq_total = sum(res1['chisq_array'])
    delta_chisq = chisq_cont_total - chisq_total
    if delta_chisq < 50:
        rvs_warn |= bitmasks['CHISQ_WARN']

    # 2. RV edge warning
    vrad = res1['vel']
    if (vrad < config['min_vel'] + 5) or (vrad > config['max_vel'] - 5):
        rvs_warn |= bitmasks['RV_WARN']

    # 3. RV error warning
    if res1['vel_err'] > 100:  # km/s
        rvs_warn |= bitmasks['RVERR_WARN']

    # 4. Parameter edge warnings
    param_checks = [
        ('teff', [2300, 15000], 10),
        ('feh', [-4, 1], 0.01),
        ('logg', [-0.5, 6.5], 0.01)
    ]
    for pname, (pmin, pmax), thresh in param_checks:
        pval = res1['param'][pname]
        if (pval < pmin + thresh) or (pval > pmax - thresh):
            rvs_warn |= bitmasks['PARAM_WARN']
            break

    # 5. vsini warning
    if res1.get('vsini', 0) > 20:
        rvs_warn |= bitmasks['VSINI_WARN']

    # 6. Bad Hessian
    if res1.get('bad_hessian', False):
        rvs_warn |= bitmasks['BAD_HESSIAN']

    return rvs_warn, delta_chisq

#################################################

def read_config_APS_RVS(fname=None):
    """
    APS customised version of the read_config from rvspecfit.utils

    Read the configuration file and return the frozendict with it

    Parameters:
    -----------
    fname: string, optional
        The path to the configuration file. If not given config.yaml in the
        current directory is used

    Returns:
    --------
    config: frozendict
        The dictionary with the configuration from a file
    """
    if fname is None:
        fname = 'config.yaml'
    with open(fname) as fp:
        return freezeDict_APS_RVS(yaml.safe_load(fp))

#################################################

def freezeDict_APS_RVS(d):
    """
    APS customised version of the freezeDict from rvspecfit.utils

    Take the input object and if it is a dictionary, freeze it (i.e. return frozendict)
    If not, do nothing

    Parameters:
    d: dict
        Input dictionary

    Returns:
    --------
    d: frozendict
        Frozen input dictionary
    """
    if isinstance(d, dict):
        d1 = {}
        for k, v in d.items():
            if isinstance(v, str):
                v = os.path.expanduser(os.path.expandvars(v))
            d1[k] = freezeDict_APS_RVS(v)
        return frozendict.frozendict(d1)
    else:
        return d

#################################################################################################
def interp_masker(lam, spec, badmask):
    """ Mask as spectrum by linearly interpolating across a badmask
    """
    spec1 = spec*1
    xbad= np.nonzero(badmask)[0]
    xgood= np.nonzero(~badmask)[0]
    xpos = np.searchsorted(xgood,xbad)
    leftedge= xpos == 0
    rightedge= xpos == len(xgood)
    mid = (~leftedge)&(~rightedge)
    l1,l2 = lam[xgood[xpos[mid]-1]],lam[xgood[xpos[mid]]]
    s1,s2= spec[xgood[xpos[mid]-1]],spec[xgood[xpos[mid]]]
    l0 = lam[xbad[mid]]
    spec1[xbad[leftedge]] = spec[xgood[0]]
    spec1[xbad[rightedge]] = spec[xgood[-1]]
    spec1[xbad[mid]] = (-(l1-l0)*s2+(l2-l0)*s1)/(l2-l1)
    return spec1
#################################################################################################



#################################################################################################
# make_rvs_plot now lives in PyAPS.apsPlot.rvs, built on the shared
# spectrum_overlay_figure() (Plotly) used across the PyAPS.apsPlot platform.
# Imported at module top as `from PyAPS.apsPlot.rvs import make_rvs_plot`.
#################################################################################################
def estimate_sigma0_for_deconvolution(fpmode, resmode):
    """
    Estimate sigma_0 parameter for LSF deconvolution based on instrument configuration.

    The sigma_0 parameter represents the intrinsic instrumental broadening (in sigma, not FWHM)
    that needs to be deconvolved from the measured FWHM to obtain the true spectral resolution.

    Parameters
    ----------
    fpmode : str
        Focal plane mode (case-insensitive):
        - 'MOS-A/B': Multi-Object Spectroscopy
        - 'LIFU': Large Integral Field Unit
        - 'MIFU': Mini Integral Field Unit

    resmode : str
        Resolution mode (case-insensitive):
        - 'HR': High Resolution
        - 'LR': Low Resolution

    Returns
    -------
    sigma_0 : float
        Instrumental broadening parameter in Angstroms (Gaussian sigma, not FWHM)

    Notes
    -----
    Based on LSF analysis templates generated in January 2026.

    Template generation uses resolution functions of form R = λ/C where:
    - For a given C value, the Gaussian sigma is: σ = C/2.355
    - The relationship: FWHM = 2.355 × σ

    Empirically determined template resol_func values (and derived sigma):
    - MOS-A/B/mIFU + HR: resol_func='x/0.5'  → σ₀ = 0.212 Å
    - MOS-A/B/mIFU + LR: resol_func='x/2.4'  → σ₀ = 1.019 Å
    - LIFU + HR:         resol_func='x/0.7'  → σ₀ = 0.297 Å
    - LIFU + LR:         resol_func='x/3.5'  → σ₀ = 1.486 Å

    If no match is found, returns default value of 1.0 Å with a warning.

    Examples
    --------
    >>> estimate_sigma0_for_deconvolution('MOS-A', 'HR')
    0.212  # σ from resol_func='x/0.5'

    >>> estimate_sigma0_for_deconvolution('LIFU', 'LR')
    1.486  # σ from resol_func='x/3.5'

    >>> estimate_sigma0_for_deconvolution('MIFU', 'HR')
    0.212  # σ from resol_func='x/0.5'
    """

    # Normalize to uppercase
    fpmode_upper = str(fpmode).strip().upper()
    resmode_upper = str(resmode).strip().upper()

    # Conversion factor: FWHM = 2.355 × σ, so σ = FWHM/2.355
    FWHM_TO_SIGMA = 1.0 / 2.355

    # Lookup table: template resol_func coefficient C (where resol_func = 'x/C')
    # For R = λ/C, the constant Gaussian sigma is: σ = C/2.355
    resol_coeff_lookup = {
        ('MOS-A', 'HR'): 0.2,
        ('MOS-A', 'LR'): 1.0,
        ('MOS-B', 'HR'): 0.2,
        ('MOS-B', 'LR'): 1.0,
        ('MIFU', 'HR'): 0.2,
        ('MIFU', 'LR'): 1.0,
        ('LIFU', 'HR'): 0.3,
        ('LIFU', 'LR'): 1.5,
        # extra entries for generality(fallback)
        ('MOS', 'HR'): 0.2,
        ('MOS', 'LR'): 1.0,
        ('IFU', 'HR'): 0.3,
        ('IFU', 'LR'): 1.5,
    }

    # Get coefficient from lookup table
    key = (fpmode_upper, resmode_upper)
    resol_coeff = resol_coeff_lookup.get(key)

    if resol_coeff is None:
        print(f"WARNING: Unknown configuration fpmode='{fpmode_upper}', resmode='{resmode_upper}' - using default sigma_0=1.0 Å")
        print(f"         Valid fpmode: MOS-A, MOS-B, LIFU, MIFU")
        print(f"         Valid resmode: HR, LR")
        return 1.0

    # Convert coefficient to sigma: σ = C / 2.355
    sigma_0 = resol_coeff * FWHM_TO_SIGMA

    return sigma_0

#################################################################################################
def proc_weave(infiles, classfile, aps_ids, targsrvy, targclass, mask_aps_ids, area, mask_areas, wlranges, figdir, config, threadid,
 nthreads, sens_corr, mask_gaps, safe_mask_gaps, tellurics, vacuum, fill_gap, arms_ratio, join_arms, catdir, caldir, configdir,
 linemask=None, uselinemasks=False, maskbalmer=False):
    """
    Process on WEAVE spectra

    Parameters:
    -----------
    infiles (list): input files
    classfile (str) : Class file, contains classifications and redshift estimations.
    aps_ids: int (list) The list of WEAVE APS_IDs ro be analysed
    setups:  a list [length=length(infiles)], indicating the configuration (arm/resolution/binning)
        (e.g. "BLUE_REDL", "BLUE_REDH", "BLUEL", "REDH", "GREENH" )
    wlranges: a list [length=length(infiles)], wavelength range, for each infname
    figdir: (str) The output fig directory where the figures will be stored
    sens_corr (bool): if True, then it apply the sensitivity function to the flux and ivar and
        brings them into the 1e-18 erg/(s cm2 Angstroms) units
    """

    large_error = aps_constants.large_error

    ## For MOS mode, we need the APSID of the interested sources, coming from the classfile.
    ## get the list of aps_ids for each threadid (for MOS, this second loop has been implemented inside the aps_ids_class function)

    # this return the aps_ids of the targets, satisfied the class condition.
    # It also accept the nthreads and threadid for parallel processing
    aps_ids_in_class = aps_ids_class(classfile, ['STAR', 'WD'], aps_ids=aps_ids, nthreads=nthreads, threadid=threadid, rank=3, ncchar=2)

    ## Check if ids_in_class is OK or not. If not, it will force the code to exit (back to one level up)
    if aps_ids_in_class is None or not aps_ids_in_class.size:
        return


    # READ DATA and put them in the APSOBJ OBJECT
    APSOBJ = APSOB(infiles, targsrvy= targsrvy, targclass = targclass, mask_aps_ids = mask_aps_ids ,
     area=area, mask_areas=mask_areas, wlranges=wlranges,
     aps_ids=aps_ids_in_class, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics,
     fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, catdir= catdir, caldir=caldir, configdir=configdir,
     normalize_ivar=False)

    targs=APSOBJ.data()
    targs_infiles = APSOBJ.infiles()
    targs_id=APSOBJ.id()
    targs_idfx=APSOBJ.idfx()
    targs_idxf=APSOBJ.idxf()
    targs_nbands = APSOBJ.nbands()
    targs_funits = APSOBJ.funits()
    targs_mode = APSOBJ.mode()
    # Return the normalized spectrograph unit name like MOS, MIFU or LIFU
    targs_fmode_unit = APSOBJ.fmode_unit()
    targs_join_arms = APSOBJ.join_arms()
    # new: get fmode to be used for sigma0_angstrom in resolution matrix generation
    targs_fmode = APSOBJ.fmode()
    targs_resmode = list(set(APSOBJ.res_mode()))[0]
    # now evaluate the sigma0_angstrom based on fmode and resmode (look at the scripts to genearte the RVS templates to see the set values)
    targs_sigma0_angstrom = estimate_sigma0_for_deconvolution(targs_fmode, targs_resmode)

    ## This is the setups param, generated by APSOBJ. For stitched arms it will be something like 'combined'
    ## AS RVSPECFIT use setups param to search for the appropriate templates, we also generate a list of
    ## setups called orig_setups, including the arm name, resolution (L,H) and binning ('', 2, 4, etc)
    targs_setups = APSOBJ.setups()
    setups_original = APSOBJ.setups_original()


    # just to make sure stitching arms is done:
    if targs_join_arms and (targs_nbands ==1) and (len(setups_original) > 1):
        orig_setups = ['_'.join(setups_original)]
    else:
        orig_setups = targs_setups

    is_hr_ob = any(setup.startswith(("GREENH", "BLUEH", "REDH")) for setup in setups_original)
    if uselinemasks and not is_hr_ob:
        warnings.warn("Linemasks are only applicable to HR observations. Full spectrum fit will be performed.")
        uselinemasks = False
    if uselinemasks and (linemask is None or not Path(linemask).exists()):
        warnings.warn("No linemask csv has been detected. Full spectrum fit will be performed.")
        uselinemasks = False
    if uselinemasks:
        mask_windows = np.genfromtxt(linemask, delimiter=',', names=True, dtype=None, encoding=None)
        mask_shrink_angstrom = 0.2
        mask_list = []
        for start, end in zip(mask_windows['start'], mask_windows['end']):
            left = min(start, end) + mask_shrink_angstrom
            right = max(start, end) - mask_shrink_angstrom
            if left < right:
                mask_list.append([left, right])

    options = {'npoly': 7 if (uselinemasks) else 19}

    ## Please note, in case we had join arms, both targs_setups and orig_setups has lenght=1
    ## so you can safely use index (e.g. ns) to use this params, without worring about it



    ### GENERATE OUTPUT TABLE STRUTURE
    columns = ['APS_ID','TARGID','CNAME',
                'VRAD','VRAD_ERR','VRAD_CCF',
                'SKEWNESS_RVS', 'KURTOSIS_RVS',
                'LOGG_RVS','TEFF_RVS','VSINI_RVS','FEH_RVS','ALPHA_RVS',
                'LOGG_ERR_RVS','TEFF_ERR_RVS','FEH_ERR_RVS','ALPHA_ERR_RVS',
                'SNR_RVS','CHISQ_TOT_RVS','RVS_WARN', 'DELTA_CHISQ']


    ## here we use original targs_setups as it is used to create the columns in the output fits/tables
    ## and it is independent of the templates used by RVSPECFIT
    for s in targs_setups:
        columns.append('LAMBDA_RVS_%s' % s[0])
        columns.append('FLUX_RVS_%s' % s[0])
        columns.append('ERROR_RVS_%s' % s[0])
        columns.append('MODEL_RVS_%s' % s[0])
        columns.append('CHISQ_C_%s' % s[0])

    outdict = OrderedDict()
    for c in columns:
        outdict[c]=[]


    ## Here we define units, as we will use it later to generate output plots
    if sens_corr:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' %(targs_funits)
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' %(targs_funits**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'
    ## Create a dict to keep all this units
    units_str = {'flux':flux_unit_str, 'ivar':ivar_unit_str,'wave':wave_unit_str}


    #### LOOP OVER DIFFERENT TARGETS, BASED ON THE ID COMMING FROM CLASSTABLE

    ## It loop over aps_ids coming form APSOBJ module.
    ## This might be different to the input/initial aps_ids

    for tgs in targs_id:

        # Convert aps_id into the index in the APSOBJ
        # targs_idfx is the dictionary linking each aps_id to the index in the output object
        tgs_indx = targs_idfx[tgs]


        tgsTARGID=targs[tgs_indx].targid
        tgsAPS_ID=targs[tgs_indx].aps_id
        tgsCNAME=targs[tgs_indx].cname

        print( '--> RUNNING RVS on the spectrum:: ' + 'APS_ID: %s TARGID:%s CNAME: %s'
            % (tgsAPS_ID, tgsTARGID , tgsCNAME))

        if figdir is not None:
            fig_fname = figdir + 'RVS_%s_%s_%s.png' % (tgsTARGID, tgsCNAME, tgsAPS_ID)

        specdata = []

        sns = {}

        ## loop over data at each possible bands
        ## here we use  orig_setups as setups indicator as it directly deals with RVSPECFIT templates
        ## IF join_arms is True, then  orig_setups is the original setups[0]+"_"+setups[1]
        ## otherwise, it is the targs_setups, generated by APSOBJ

        badmask_all_setups={}
        for ns, s in enumerate(orig_setups):
            spec = targs[tgs_indx].spectra[ns].flux
            ivar = targs[tgs_indx].spectra[ns].ivar
            wave_s = targs[tgs_indx].spectra[ns].wave

            badmask = (ivar <= 0)
            badmask_all_setups[ns] = badmask

            ivar[badmask] = 1.0/(large_error**2)
            espec = 1. / (ivar**.5)
            sns[s] = np.nanmedian(spec / espec)

            # =====================================================================
            # NEW: Create resolution matrix from FWHM interpolation
            # =====================================================================
            try:
                # Get FWHM function for this target and setup
                fwhm_funcs = APSOBJ.get_fwhm(aps_id=tgsAPS_ID, fwhm_key='fwhm')
                fwhm_func = fwhm_funcs[ns]

                # Evaluate FWHM at wavelengths
                fwhm_values = fwhm_func(wave_s)

                # Note how we pass sigma0_angstrom for deconvolution
                # it is comming from the --resol_func (e.g. 'x/0.5') for each fmmode (MOS,LIFU,MIFU) and resmode (HR,LR)
                # as defined in the scripts we used to generate the RVS templates
                # For example After generating templates at fixed sigma. e.g. for # For FWHM = 0.5 Å:
                # --resol_func 'x/0.5'

                # Create resolution matrix using SHARED function
                R, _ = makeR_from_fwhm_array(
                    wave_s, fwhm_values, cache_Rcsr=False,
                    use_deconvolution=True,  # Enable deconvolution
                    sigma0_angstrom=targs_sigma0_angstrom      # Template resolution (Phoenix/Kurucz)
                )

                # Wrap for rvspecfit
                resolution_matrix = spec_fit.ResolMatrix(mat=R)

                print(f"  APS_ID {tgsAPS_ID} setup {s}: Using FWHM-based resolution matrix")

            except Exception as e:
                # Fallback to fixed resolution if FWHM interpolation fails
                print(f"  APS_ID {tgsAPS_ID} setup {s}: FWHM failed ({e}), using fixed resolution")
                R, _ = makeR(wave_s, APSOBJ.resolution()[ns], cache_Rcsr=False)
                resolution_matrix = spec_fit.ResolMatrix(mat=R)
            specdata.append(spec_fit.SpecData('%s' % targs_fmode_unit+'_'+s, wave_s, spec, espec, resolution=resolution_matrix, badmask=badmask))


        # t1 = time.time()
        # res = fitter_ccf.fit(specdata, config)

        # t2 = time.time()
        # paramDict0 = res['best_par']
        # fixParam = []
        t1 = time.time()

        # ADD ccf_init parameter (from config or function argument)
        ccf_init = config.get('ccf_init', True)  # Default True for backward compatibility

        if ccf_init:
            res = fitter_ccf.fit(specdata, config)
            paramDict0 = res['best_par']
            vrad_ccf = res['best_vel']
        else:
            # Fallback to brute-force grid search
            res = vel_fit.firstguess(specdata, config=config, options=options)
            res['best_vsini'] = res.get('vsini')
            paramDict0 = res
            vrad_ccf = None  # Not available in firstguess mode

        # First pass is unmasked; use its RV to apply line windows in the rest frame.
        if uselinemasks or maskbalmer:
            rv_pass1 = res['best_vel'] if ccf_init else res['vel']
            for i, spec_item in enumerate(specdata):
                rest_wave = spec_item.lam / (1.0 + rv_pass1 / C_KMS)
                line_badmask = np.zeros_like(rest_wave, dtype=bool)
                if uselinemasks:
                    for mask_start, mask_end in mask_list:
                        line_badmask |= (rest_wave >= mask_start) & (rest_wave <= mask_end)
                if maskbalmer:
                    for mask_start, mask_end in ((4334, 4349), (4853, 4868), (6555, 6570)):
                        line_badmask |= (rest_wave >= mask_start) & (rest_wave <= mask_end)
                badmask_all_setups[i] |= line_badmask
                spec_item.espec[line_badmask] = large_error
                spec_item.badmask[line_badmask] = True

            # Re-run the existing fit as pass 2 with the rest-frame windows excluded.
            if ccf_init:
                res = fitter_ccf.fit(specdata, config)
                paramDict0 = res['best_par']
                vrad_ccf = res['best_vel']
            else:
                res = vel_fit.firstguess(specdata, config=config, options=options)
                res['best_vsini'] = res.get('vsini')
                paramDict0 = res
                vrad_ccf = None

        t2 = time.time()
        fixParam = []
        if res['best_vsini'] is not None:
            # paramDict0['vsini'] = res['best_vsini']
            paramDict0['vsini'] = min(max(res['best_vsini'], config['min_vsini']), config['max_vsini'])

        res1 = vel_fit.process(specdata, paramDict0, fixParam=fixParam,
                            config=config, options=options)

        t3 = time.time()
        chisq_cont_array = spec_fit.get_chisq_continuum(specdata, options=options)['chisq_array']
        chisq_cont_total = sum(chisq_cont_array)

        rvs_warn, delta_chisq = get_rvs_warn(res1, chisq_cont_total, config)


        t4 = time.time()

        outdict['APS_ID'].append(tgsAPS_ID)
        outdict['TARGID'].append(tgsTARGID)
        outdict['CNAME'].append(tgsCNAME)
        outdict['VRAD'].append(res1['vel'])
        outdict['VRAD_ERR'].append(res1['vel_err'])
        outdict['SKEWNESS_RVS'].append(res1['vel_skewness'])
        outdict['KURTOSIS_RVS'].append(res1['vel_kurtosis'])
        outdict['LOGG_RVS'].append(res1['param']['logg'])
        outdict['TEFF_RVS'].append(res1['param']['teff'])
        outdict['ALPHA_RVS'].append(res1['param']['alpha'])
        outdict['FEH_RVS'].append(res1['param']['feh'])
        outdict['LOGG_ERR_RVS'].append(res1['param_err']['logg'])
        outdict['TEFF_ERR_RVS'].append(res1['param_err']['teff'])
        outdict['ALPHA_ERR_RVS'].append(res1['param_err']['alpha'])
        outdict['FEH_ERR_RVS'].append(res1['param_err']['feh'])
        outdict['VSINI_RVS'].append(res1['vsini'])
        outdict['SNR_RVS'].append(np.mean(list(sns.values())))
        outdict['CHISQ_TOT_RVS'].append(sum(res1['chisq_array']))
        outdict['RVS_WARN'].append(rvs_warn)
        outdict['DELTA_CHISQ'].append(delta_chisq)

        # Add CCF velocity if available
        if vrad_ccf is not None:
            outdict['VRAD_CCF'].append(vrad_ccf)
        else:
            outdict['VRAD_CCF'].append(np.nan)


        # As we are dealing with columns in the output fits file, we use the targs_setups
        # generated by APSOBJ
        ## At this level , we also Mask bad values in spec (flux and error) and model ( by setting to nan)

        for i, s in enumerate(targs_setups):
            outdict['LAMBDA_RVS_%s' % s[0]].append(specdata[i].lam)
            specdata[i].spec[badmask_all_setups[i]]=np.nan
            outdict['FLUX_RVS_%s' % s[0]].append(specdata[i].spec)
            specdata[i].espec[badmask_all_setups[i]]=np.nan
            outdict['ERROR_RVS_%s' % s[0]].append(specdata[i].espec)
            res1['yfit'][i][badmask_all_setups[i]]=np.nan
            outdict['MODEL_RVS_%s' % s[0]].append(res1['yfit'][i])
            outdict['CHISQ_C_%s' % s[0]].append(float(chisq_cont_array[i]))

        title = 'FEH=%.3f  TEFF=%.1f  LOGG=%.2f  ALPHA=%.2f  Vrad=%.2f+/-%.2f' % (res1['param']['feh'],
                                                                                      res1['param']['teff'],
                                                                                      res1['param']['logg'],
                                                                                      res1['param']['alpha'],
                                                                                      res1['vel'],
                                                                                      res1['vel_err'])


        out_msg= 'APS_ID: %d RESULTS [based on %d arm(s) data]:' % (tgsAPS_ID, len(specdata))
        print(out_msg + title)


        if figdir is not None:
            try:
                make_rvs_plot(specdata, res1, title, fig_fname, units_str= units_str)
            except:
                print('Failed to generate plots. 1- Check your X11 configs. 2- Check spectra in output fits file.')
                pass


    outtab = Table(outdict)
    outtab['VRAD'].unit='km/s'
    outtab['VRAD_ERR'].unit='km/s'
    outtab['VRAD_CCF'].unit='km/s'
    outtab['SKEWNESS_RVS'].unit='km/s'
    outtab['KURTOSIS_RVS'].unit='km/s'
    outtab['VSINI_RVS'].unit='km/s'
    outtab['TEFF_RVS'].unit='K'
    outtab['TEFF_ERR_RVS'].unit='K'
    outtab['LOGG_RVS'].unit='dex'
    outtab['FEH_RVS'].unit='dex'
    outtab['ALPHA_RVS'].unit='dex'
    outtab['LOGG_ERR_RVS'].unit='dex'
    outtab['FEH_ERR_RVS'].unit='dex'
    outtab['ALPHA_ERR_RVS'].unit='dex'



    ## Add unit to the spectra, to be reflect in the spec output
    for s in targs_setups:
        outtab['LAMBDA_RVS_%s' % s[0]].unit=wave_unit_str
        outtab['FLUX_RVS_%s' % s[0]].unit= flux_unit_str
        outtab['ERROR_RVS_%s' % s[0]].unit=ivar_unit_str
        outtab['MODEL_RVS_%s' % s[0]].unit=flux_unit_str


    return outtab
#################################################################################################

def proc_weave_wrapper(*args, **kwargs):
    try:
        ret = proc_weave(*args, **kwargs)
        return ret
    except:
        print('failed with these arguments', args, kwargs)
        raise


proc_weave_wrapper.__doc__ = proc_weave.__doc__

#################################################################################################

def rvs_write_fits(tabs, infiles, param_fits, spec_fits=None, join_arms=False,
                   match_table=None, linemask=None, uselinemasks=False):
    """
    Generates outputs for rvspecfit
    """

    tabs.sort('APS_ID')
    RVS_TABLE=Table(tabs['APS_ID','TARGID', 'CNAME', 'VRAD', 'VRAD_ERR','VRAD_CCF','SKEWNESS_RVS', 'KURTOSIS_RVS',
        'LOGG_RVS','TEFF_RVS','VSINI_RVS','FEH_RVS','ALPHA_RVS','LOGG_ERR_RVS','TEFF_ERR_RVS','FEH_ERR_RVS',
        'ALPHA_ERR_RVS','SNR_RVS','CHISQ_TOT_RVS','RVS_WARN', 'DELTA_CHISQ'])


    ## add any extra columns if necessary from the match_table (if given)
    if match_table is not None:
        RVS_TABLE = add_extra_columns(RVS_TABLE, match_table=match_table)



    RVS_TABLE.meta['EXTNAME'] = 'RVS_TABLE'
    RVS_TABLE.meta['APSVERS'] = (APSVERS,'APS version')
    RVS_TABLE.meta['APS_RV_V'] = (aps_constants.__aps_rvs_version__,'PyAPS RVSPECFIT wrapper version')
    linemask_used = int(bool(uselinemasks)) and linemask is not None
    linemask_name = Path(linemask).name if linemask_used else "None"
    RVS_TABLE.meta['LMASKUSE'] = (linemask_used, '1 if linemask used, 0 otherwise')
    RVS_TABLE.meta['LMASK'] = (linemask_name, 'Path to RVS linemask file')


    #keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(infiles):
        RVS_TABLE.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    RVS_TABLE.meta['CSB_RVS'] = (join_arms, 'Combines Spectral Bands Status for RVSPECFIT')

    RVS_TABLE.sort('APS_ID')


    # header = fits.Header()
    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU())
    hx.append(fits.convenience.table_to_hdu(RVS_TABLE))
    hx.writeto(os.path.expandvars(param_fits), overwrite=True)



    if spec_fits is not None:
        tabs.remove_columns(['VRAD', 'VRAD_ERR', 'VRAD_CCF','SKEWNESS_RVS','KURTOSIS_RVS','LOGG_RVS','TEFF_RVS','VSINI_RVS',
            'FEH_RVS','ALPHA_RVS','LOGG_ERR_RVS','TEFF_ERR_RVS','FEH_ERR_RVS','ALPHA_ERR_RVS',
            'SNR_RVS','CHISQ_TOT_RVS','RVS_WARN', 'DELTA_CHISQ'])

        ## add any extra columns if necessary from the match_table (if given)
        if match_table is not None:
            tabs = add_extra_columns(tabs, match_table=match_table)

        ## remove columns from tabs, contains CHISQ_C_*
        CHISQ_C_cols = [tabscols for tabscols in tabs.colnames if "CHISQ_C_" in tabscols]
        if len(CHISQ_C_cols) > 0: tabs.remove_columns(CHISQ_C_cols)

        tabs.meta['EXTNAME'] = 'RVS_SPEC'
        tabs.meta['APSVERS'] = (APSVERS,'APS version')
        tabs.meta['APS_RV_V'] = (aps_constants.__aps_rvs_version__,'PyAPS RVSPECFIT wrapper version')
        tabs.meta['VACUUM'] = (False, 'Wavelengths are in VACUUM')
        tabs.meta['SAMPLING'] = (0,'Sampling mode (0: linear, 1: logarithmic')
        tabs.meta['LMASKUSE'] = (linemask_used, '1 if linemask used, 0 otherwise')
        tabs.meta['LMASK'] = (linemask_name, 'Path to RVS linemask file')


        # header = fits.Header()
        hx = fits.HDUList()
        hx.append(fits.PrimaryHDU())
        hx.append(fits.convenience.table_to_hdu(tabs))
        hx.writeto(os.path.expandvars(spec_fits), overwrite=True)
#################################################################################################


def proc_rvs(infiles, classfile , param_fits, figdir=None, spec_fits=None, aps_ids=None,
    targsrvy= None, targclass = None, mask_aps_ids = None, area=None, mask_areas=None,
    wlranges=None , config= None, nthreads=None, overwrite=None, sens_corr=None,
    mask_gaps = True,safe_mask_gaps=True, tellurics= True, vacuum=False, fill_gap=True,
    arms_ratio=None, join_arms=False, match_table=None, catdir=None, caldir=None, configdir=None,
    linemask=None, uselinemasks=False, maskbalmer=False):

    """
    Process many spectral infiles
    """

    ###########################################################################################
    ## Important note by APS:
    ## In this module, we assume all inputs, especially infiles, wlranges, setups and join_arms
    ## have been already checked/tested/corrected during the previous steps
    ## If you want to make it as a stand-alone function, make sure you check all these params
    ## for example by using l1_fileinfo function.
    ###########################################################################################


    # this is just to make sure number of aps_ids dedicated to the specific type of targets (here "STAR")
    # is not larger than nthreads.

    aps_ids_satis_class = aps_ids_class(classfile, ['STAR', 'WD'], aps_ids=aps_ids, nthreads=1, threadid=0, rank=3, ncchar=2)

    ## Check if aps_ids_satis_class is OK or not. If not, it will force the code to exit (back to one level up)
    if aps_ids_satis_class is None:
        return

    ## After the check above (if aps_ids_satis_class is None: return) we made sure aps_ids_satis_class is a list with > 0 elements
    ## Now we just want to make sure that nthreads and number of targets are consistent.
    ## And we do not assign an empty job to a thread
    if len(aps_ids_satis_class) < nthreads:
        nthreads = len(aps_ids_satis_class)
        print('Update nthreads param to %d'%(nthreads))

    # config = utils.read_config(config)
    config = read_config_APS_RVS(config)


    if nthreads > 1:
        parallel = True
    else:
        parallel = False

    if parallel:
        pool = mp.Pool(nthreads)

    # Updated by ALireza on 30-Jan-2020
    res = []

    if (not overwrite) and os.path.exists(param_fits):
        print('skipping, products already exist')
        sys.exit()

    if (not overwrite) and os.path.exists(spec_fits):
        print('skipping, products already exist')
        sys.exit()


    if parallel:
        for i in range(nthreads):
            res.append(pool.apply_async(
            proc_weave_wrapper, (infiles, classfile, aps_ids, targsrvy, targclass, mask_aps_ids, area, mask_areas,  wlranges , figdir,
             config, i, nthreads, sens_corr, mask_gaps,safe_mask_gaps, tellurics, vacuum, fill_gap, arms_ratio, join_arms, catdir, caldir, configdir,
             linemask, uselinemasks, maskbalmer)))

        tabs = []
        for r in res:
            tabs.append(r.get())

        tabs = ([_ for _ in tabs if _ is not None] )
        if len(tabs)==0:
            sys.exit('Empty tab: Cannot generate the RVS output table')
        tabs = vstack(tabs)
        ## GENERARE RVS OUTPUT TABLES
        rvs_write_fits(tabs, infiles, param_fits, spec_fits=spec_fits, join_arms=join_arms,
                   match_table=match_table, linemask=linemask, uselinemasks=uselinemasks)

    else:
        tabs=proc_weave_wrapper(infiles, classfile, aps_ids,targsrvy, targclass, mask_aps_ids,area, mask_areas, wlranges, figdir,
         config, 0, 1, sens_corr, mask_gaps,safe_mask_gaps, tellurics, vacuum, fill_gap, arms_ratio, join_arms, catdir, caldir, configdir,
         linemask, uselinemasks, maskbalmer)

        if tabs is not None:
            ## GENERARE RVS OUTPUT TABLES
            rvs_write_fits(tabs, infiles, param_fits, spec_fits=spec_fits, join_arms=join_arms,
                           match_table=match_table, linemask=linemask, uselinemasks=uselinemasks)


    if parallel:
        pool.close()
        pool.join()


################################################################################

def rvsweave(options=None):

    # Shared registry -- see aps_common_args.py's own module docstring.
    # `overrides` reproduces this script's own current --help wording
    # verbatim wherever it genuinely differs from the shared default
    # (including the "APS_IDS" capitalization on --mask_aps_ids, a
    # pre-existing typo left as-is here).
    parser = build_common_parser(
        description=None,
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "caldirs", "output"],
        overrides={
            "mask_aps_ids": {"help": "comma-separated list of APS_IDS to be masked"},
            "vacuum": {"default": False},
            "fill_gap": {"default": True},
            "outpath": {"help": "Directory to keep WEAVE_RVS outputs"},
            "overwrite": {"help": "If enabled the code will overwrite the existing products, otherwise it will skip them"},
        },
        extra_args=[
            (("--config",), dict(type=none_or_str, default=None,
                                  help="The filename of the configuration file")),
            (("--classfile",), dict(type=none_or_str, default=none_or_str, required=True,
                                     help="The input fits file, contains the classification table")),
            (("--outspec",), dict(type=str2bool, default=False, required=False,
                                   help="if True, the code return the spectra and best fitted model [FITS file]")),
            (("--fig",), dict(type=str2bool, default=False, required=False,
                               help="if True, the code also produces plots of the best fitted model")),
            (("--mp",), dict(type=int, default=1, help="Number of threads for the fits")),
            (("--device",), dict(type=str, default="cpu", choices=["cpu", "gpu", "auto"],
                                  help="Torch device for the neural-net interpolator: 'cpu' "
                                       "(default -- forced CPU, safe on this cluster since no "
                                       "SLURM job currently gets a --gres=gpu allocation), "
                                       "'gpu' (use CUDA if available, else fall back to CPU "
                                       "with a warning), or 'auto' (same availability check, "
                                       "no warning on fallback). See resolve_rvs_device().")),
              (('--linemask',), dict(type=none_or_str, default=None,
                                help="Path to RVS linemask CSV file")),
              (('--uselinemasks',), dict(type=str2bool, default=False,
                                    help="Use rest-frame linemasks in a two-pass RVS fit")),
              (('--maskbalmer',), dict(type=str2bool, default=False,
                                    help="Exclude H-alpha, H-beta, and H-gamma regions from the RVS fit")),
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



    ### Now we have both infiles and wlranges array. We use resolve_common_args
    ### (aps_common_args.py) to update infiles/wlranges/arms_ratio/join_arms
    ### and put them in the right order, if needed -- see that function's
    ### own docstring. However, we had similar test done by APSOB.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio

    # Resolve the actual --device choice now (forced, not setdefault --
    # overrides the module-load-time 'cpu' default above) and still well
    # before proc_rvs() opens its multiprocessing pool, so forked workers
    # inherit the decision.
    device_chosen = resolve_rvs_device(getattr(args, "device", "cpu"))
    print(f"RVS neural-net interpolator device: {device_chosen} (requested: {getattr(args, 'device', 'cpu')})")

    if args.classfile == None:
        raise Exception('You need to specify the CLASS/Z file to proceed')
    classfile = args.classfile


    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ','')



    param_fits=os.path.join(args.outpath,'rvs_'+str(args.headname)+'.fits')
    param_fits=param_fits.replace(' ', '')

    spec_fits=None
    if args.outspec:
        spec_fits=os.path.join(args.outpath,'rvsspec_'+str(args.headname)+'.fits')
        spec_fits.replace(' ', '')

    figdir=None
    if args.fig:
        figdir=args.outpath + '/figs/'
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            print("FIGDIR: %s Created!" %(figdir))
        figdir=figdir+'/'#+str(args.headname)
        figdir=figdir.replace(' ', '')

    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
    area, mask_areas = resolved.area, resolved.mask_areas

    # print args and assigned/default values on the screen
    print_args(args,module='RVS', version= aps_constants.__aps_rvs_version__, path=outpath, headname=args.headname)

    proc_rvs(args.infiles, classfile , param_fits, figdir=figdir, targsrvy= targsrvy, targclass = targclass,
        mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, spec_fits=spec_fits, aps_ids=aps_ids,
        wlranges=wlranges , nthreads=args.mp, overwrite=args.overwrite, config=args.config,
        sens_corr=args.sens_corr, mask_gaps = args.mask_gaps, safe_mask_gaps = args.safe_mask_gaps, tellurics=args.tellurics, vacuum=args.vacuum,
        fill_gap=args.fill_gap, arms_ratio=arms_ratio, join_arms=args.join_arms, match_table=None,
        catdir=args.catdir, caldir=args.caldir, configdir=args.configdir,
        linemask=args.linemask, uselinemasks=args.uselinemasks,
        maskbalmer=args.maskbalmer)

#################################################################################################


#################################################################################################
if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    MOS_demo= ['--infiles', '<PYAPS_DATA>/star_test/stacked_<runid>.fit', '<PYAPS_DATA>/star_test/stacked_<runid>.fit',
    '--classfile' , '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/zbest_stacked_<runid>__stacked_<runid>.fits',
    '--aps_ids', '1006,1007,1005,1004', # or 'None' to run for all available fibreids
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area', 'None',
    '--mask_areas', 'None',
    '--wlranges', '4200.0,6000', '6000.0,8000',
    '--sens_corr', 'True',
    '--safe_mask_gaps', 'True',
    '--mask_gaps', 'True',
    '--tellurics', 'True',
    '--vacuum', 'False',
    '--fill_gap', 'True',
    '--arms_ratio', '1.0, 0.83',
    '--join_arms', 'False',
    '--config', '<PYAPS_DIR>/configs/rvs_config.yaml',
    '--outpath', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/',
    '--headname', 'stacked_<runid>__stacked_<runid>',
    '--outspec', 'True',
    '--fig', 'True',
    '--overwrite', 'True',
    '--mp' ,'4' ]

    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    rvsweave(options=MOS_demo)
