
"""
=====================

APS tools for WEAVE MOS ExGAL sources analysis
"""

import numpy as np

import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pickle
import logging
import glob
import json
import argparse
import datetime
from   scipy.interpolate import interp1d
from multiprocessing import Queue, Process

from astropy.io import fits, ascii
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord, Galactic, ICRS
from astropy.wcs import WCS
from astropy import units

import PyAPS
from PyAPS import MOSExGalPrepare    as MOSExGalPrepare
from PyAPS import ExGalPrepare    as ExGalPrepare
from PyAPS import MOSExGalPPXF
from PyAPS import MOSExGalLS
from PyAPS import MOSExGalEMIPPXF
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class, l1_fileinfo, gen_targlist, index_in_class, apply_redshift_to_fwhm_corrected
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import aps_constants


#################################################
"""
APS_mosExGal
Python code for analysing APS MOS ExGal sources



versions:
 3.2 By A. Molaeinezhad (IAC, July 2020)
 3.3 By A. Molaeinezhad (CASU, May 2023)
 3.5 By A. Molaeinezhad (CASU, Oct 2023)
 3.6 By A. Molaeinezhad (CASU, Apr 2024)
 3.7 By A. Molaeinezhad (CASU, June 2024)
 3.8 By A. Molaeinezhad (CASU, August 2025)
NOTES:


TODO list:
1- Discuss with Adrian and Jesus about velscale_ratio issue (see the history below)

########### BAISC APS PARAM #####################################################################################################
|    param                |    APS default  |  RRWEAVE EQUIVALENT          |APS_mosExGal DEFAULT|   available options/notes     |
infiles (Required)        |        -        |  --infiles (Required)        |        -           |                               |
aps_ids (Optional)        |     None        |  --aps_ids (Optional)        |        None        |                               |
targsrvy (Optional)       |     None        |  --targsrvy (Optional)       |        None        |                               |
targclass (Optional)      |     None        |  --targclass (Optional)      |        None        |                               |
mask_aps_ids (Optional)   |     None        |  --mask_aps_ids (Optional)   |        None        |                               |
area (Optional)           |     None        |  --area (Optional)           |        None        |                               |
mask_areas (Optional)     |     None        |  --mask_areas (Optional)     |        None        |                               |
wlranges (Optional)       |     None        |  --wlranges (Optional)       |        None        |                               |
sens_corr (Optional)      |     True        |  --sens_corr (Optional)      |        True        |                               |
mask_gaps (Optional)      |     True        |  --mask_gaps (Optional)      |        True        |                               |
safe_mask_gaps (Optional) |     False       |  --safe_mask_gaps (Optional) |        True        |                               |
vacuum (Optional)         |     False       |  --vacuum (Optional)         |        False       |                               |
tellurics (Optional)      |     False       |  --tellurics (Optional)      |        False       |                               |
fill_gap (Optional)       |     False       |  --fill_gap (Optional)       |        False       |                               |
arms_ratio (Optional)     |     None        |  --arms_ratio (Optional)     |        None        | for R band in OPR3B  is 0.83  |
join_arms (Optional)      |     False       |  --join_arms (Optional)      |        True        |                               |
funit (Optional)          |     1.0e18      |  USE DEFAULT APS value       |        AS APS      |                               |
offset_gap_pix (Optional) |     10          |  USE DEFAULT APS value       |        AS APS      |                               |
                          |                 |                              |                    |                               |
######### mosExGal INITAIL PARAM ##################################################################################################

######### DEDICATED mosExGal PARAM ################################################################################################
                          |                 |  --classfile      (Required) |          -         |                               |
                          |                 |  --config_dir     (Required) |          -         |                               |
                          |                 |  --templates_dir  (Required) |          -         |                               |
                          |                 |  --outpath   (Required)      |          -         |                               |
                          |                 |  --params    (Required)      |          -         |                               |
                          |                 |  --outpath   (Required)      |          -         |                               |
                          |                 |  --outpath   (Required)      |          -         |                               |
                          |                 |  --headname  (Optional)      |     'headname'     |                               |
                          |                 |  --fig       (Optional)      |        False       |                               |
                          |                 |  --PPXF      (Optional)      |        False       |                               |
                          |                 |  --EMIPPXF   (Optional)      |        False       |                               |
                          |                 |  --EMIPPXF_LEVEL (Optional)  |        'BIN'       |  'BIN', 'SPAXEL'              |
                          |                 |  --LS        (Optional)      |        False       |                               |
                          |                 |  --LS_MODE   (Optional)      |        1           |   1 [no MCMC] , 2[with MCMC]  |
                          |                 |  --LS_RES    (Optional)      |     'ADAPTED'     |   'ORIGINAL', 'ADAPTED'       |
                          |                 |  --mp        (Optional)      |         1          |                               |
##################################################################################################################################


Example:
python3 <PYAPS_DIR>/py/PyAPS/aps_mosExGal.py --infiles <PYAPS_DATA>/gal_test/superstack_100001.fits <PYAPS_DATA>/gal_test/superstack_100000.fits --outpath <PYAPS_DIR>/PyAPS_results/20170223/3800/ --headname superstack_100001__superstack_100000 --wlranges 4200.0,6000.0 6000.0,8000.0 --aps_ids 1007,1006 --targsrvy None --targclass None --mask_aps_ids None --area None --mask_areas None --classfile <PYAPS_DIR>/PyAPS_results/20170223/3800/zbest_superstack_100001__superstack_100000.fits --config_dir <PYAPS_DIR>/configs/Exgal_configs/ --templates_dir <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ --params <PYAPS_DIR>/configs/ExGal_configs/MOSLR11.json --join_arms True --mp 2 --PPXF True --EMIPPXF True --LS True --EMIPPXF_LEVEL BIN --LS_MODE 1 --LS_RES ADAPTED --fig True --sens_corr True --mask_gaps True --safe_mask_gaps True --tellurics False --vacuum False --fill_gap False --arms_ratio 1.0,0.83

History:
10 Jan 2020: Add an example to run the code through command line (By A. Molaeinezhad)
10 Jan 2020: Update the skylines to mask (By A. Molaeinezhad)
27 Jan 2020: Remove res param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
27 Jan 2020: Remove mode param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
27 Jan 2020: Remove setups param. Now it automatically extracted from aps_util (By A. Molaeinezhad)
27 Jan 2020: ADD targsrvy param. to specific the list of surveys you want to proceed with (By A. Molaeinezhad)
27 Jan 2020: ADD targclass param. to specific the type of targets (coming from targclass) to proceed with (By A. Molaeinezhad)
27 Jan 2020: ADD mask_aps_ids param. to specific the list of aps_ids that you want to be masked from the process (By A. Molaeinezhad)
27 Jan 2020: update examples and readme/header content (By A. Molaeinezhad)
27 Jan 2020: handle exception in case an error occurred in plotting routine (By A. Molaeinezhad)
27 Jan 2020: updated templates names, to consider CAMERA, mode and xbin, ybin (By A. Molaeinezhad)
31 Jan 2020: Add a new feature using l1_fileinfo to make sure order or inputs (infiles, wlranges) are OK, as well as double-checking the
             the merge_arms parameters. This checks done at two level 1. at script level (this code itself) and 2. and APSOB level
06 Feb 2020: Add area and mask_areas into the criteria to select aps_ids/spaxels (By A. Molaeinezhad)
06 Feb 2020: Update working examples
11 Feb 2020: Update velscale_ratio for both Gandalf and PPXF to 1 (in MOSExGalPPXF and MOSExGalGANALF)
As we noticed, sometimes, depends on the length of templates we are using, Gandalf fails to convolve
the templates (function convolve_templates_new in the gandalf_util.py)
with verlscale_ratio = 2, simply because the length of template array is odd and not compatible with
the even velscale_ratio that you have set by default. I just noticed since Jan 2019, Michele has updated
the PPXF code (which uses the same method/approach to deal with the templates as GANDALF) to avoid
this and has changed the default velscale_ratio  to 1.0.  That is why PPXF, can handle this with no problem.
If you check the convolve_gauss_hermite function in the gist_util.py of the latest version of
PPXF, you can see that he mentioned this issue.
To fix this, (inspiring from the latest version of PPXF), in gandalf_utils.py we changed:
npad = fftpack.next_fast_len(templates.shape[0])
to:
npad = 2**int(np.ceil(np.log2(templates.shape[0])))
However, I'm still not sure if it fix all problems mentioned above.

11 Feb 2020: Update the path of all gist related modules and put them all in the PyAPS directory
11 Feb 2020: update definition of error limit to mask CCD gaps in MOSExGalPPXf and MOSExGalGANDALF. See below for the logic behind it.
11 Feb 2020: Please note, as error array is affected by Voronoi Binning
             and the error after Voronio binning is av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
             we use 0.95 * np.sqrt(error_limit) as our error limit to mask CCD gaps
             where 0.95 is 2 sigma around this value
             In MOS mode, in opposite to IFU, we do not run VORONOI
             So, we do not expect the error to be affected.
             However, as we "correct" the resampled log_error array to account for variance conservation (see update 25 Oct 2020)
             and that process strongly affect ivar (and consequently the errors) we apply the same limit for errors
25 Feb 2020: Replace 'FIBREID' with 'APS_ID' (and all other relevant params) to be compatible with all possible L1 structures
25 Feb 2020: Replace NSPEC (int) with TARGID (str), as NSPEC is not available in all L1 structures
25 Feb 2020: Add ylablel to plot. ADD/update units to output tables/spectra.
25 Feb 2020: Update header of all outputs (tables, spectra) to include units, sampling, templ, etc.
25 Feb 2020: Update Versioning of all sub-modules for galaxy module.
25 Feb 2020: For PPXF and GANDALF, we replace V_ppxf by Z_Corr:
            As the galaxy is at significant redshift z and as the wavelength has been
            de-redshifted, the best-fitting redshift is now given by the following
            formula (equation 2 of Cappellari et al. 2009, ApJ, 704, L34):

            Z_CORR =  (z_input + 1)*(1 + PPXF_V/Clight) - 1
            and error in Z_corr is coming from
            ZERR_CORR = (z_input_err)^2 + (1/Clight ^2) * [(z_input_err)^2)/(z_input)^2) + (PPXF_V_ERR)^2)/(PPXF_V)^2) + (PPXF_V_ERR)^2)]
            where Z_CORR is the corrected redshift. Z_input is the input redshift and PPXF_V is the velocity coming directly from PPXF
            *** However, still, I'm not sure about the accuracy of the formula for ZERR_CORR (evaluated by myself) ASK JESUS about this.
            *** STILL I'm not sure if it's necessary for V_LINES coming from GANDALF or it has been already corrected by the GANDALF code
            *** ASK JESUS to make sure!
25 Feb 2020: We also replace V_lines in gandalf by Z_lines, describing above.
25 Feb 2020: Update the fits filenames to be consistent with STAR module's products.
04 March 2020: replace the parameter RB_ratio (float) with a more general arms_ratio (list of floats).
04 March 2020: A new feature added to support more than 2 arms/bands (to handle possible R+G+B setup in HR)
05 March 2020: Add all necessary config params into the header of relevant products.
05 March 2020: Add all GANDALF EMISSION line info into the header of the GANDALF products.
05 March 2020: developed A function (emifile )to read and regenerate Emission-line table from the headers.
07 June 2020: Implementing the new version of Gandalf (Python 3 supported).
            For that, it is necessary to install BVLS
            https://github.com/treverhines/
            and install it (python3 setup.py install)
            I have put the original directory in PyAPS_modules, renamed it to BVLS and installed it.
            The full path of this new package in my mac is:
            /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/bvls
            I have slightly edited the header of the gandalf_util.py to address the apsGITS version instead of the original
            version developed by Adrian Binter
20 July 2020: Add lmin and lmax range (both in rest frame) to the PPXF and GNADALF (MOS modes). It has been done by updating the goodpixels
            through the masking routines. More checks are needed to make sure, such large masked area, does not affect the accuracy of the results.
20 July 2020: Also solve an issue with shading the masked area in plotting routines, if the masked area by setting the first and last pixels of goodpixels to 0
22 July 2020: TODO: Find a way to bypass the redundant voronoi binning step.
31 August 2020: Since PPXF (VERSION > 7.0), ppxf does not include bvls_solve function. Therefore, to make sure the code is compatible with the latest version
            of PPXF, we added this function to the gandalf_util.py
31 August 2020: Added support to PPXF version> 7.0
31 August 2020: Now PyAPS is independent of the GITS pipeline.
16 October 2020: fixed a bug in setting the wlranges (rest frame) for PPXF (lmin_ppxf, lmax_ppxf)
16 October 2020: Now, velscale is a free parameter to be set by user (through the configuration file). If empty/None/Null, if will be
                estimated by the code itself (Suggested by Stefano).
20 Oct 2020: in-line documentation updated
25 Oct 2020: A new param called 'pixel_scale' added to the exgal_targ dict.
25 Oct 2020: Following the comment made S. Zibetti" The IVAR spectrum cannot be re-binned as is, but must be inverted (VAR=1/IVAR) and
             rebinned using "flux" conservation So we "correct" the resampled log_error array to account for variance conservation.
             To do so, we updated the run_logrebinning function in both IFUExGalPrepare and MOSExGalPrepare and also modified the log_rebinning
             function accordingly. Now it accept pixel_scale as input. and has a new optional boolean key (ivar_cor).
28 March 2021: New error handling added to MOSExGalPrepare and MOSExGalGANDALF and MOSExGalPPXF to handle the case where
             null spectrum remain after applying LMIN/LMAX_PPXF or LMIN/LMAX_GANDALF limits on the redshift-corrected spectrum. In this case goodpixel
             will be returned as an empty list
28 March 2021: defined two new functions called gandalf_wlimit and MaskGaps (applied on both Gandalf and PPXF) in the MOSExGalPrepare
              to define working wlranges for PPXF and GANDALF and also to mask CCD GAPS
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
** [NEW VERSION] **
09 Dec 2021: prepareSpectralTemplateLibrary module has been fully upgraded:
            1- It accepts WEAVE FORMAT (npy) templates
            2- SSP parameters can be extracted
            3- PCA compressed templates added. Now you can use PCA compressed grids for kinematics
            4- The way the code handle wlranges of templates have been significantly improved
09 Dec 2021: A keyword called PCA added to the config file, to trigger PCA mode
09 Dec 2021: Again we have separated ExGalPrepare module for IFU and MOS. From now on, we have two separate IFUExGalPrepare and MOSExGalPrepare
09 Dec 2021: All Voronoi binning related stuff have been removed from MOS mode. Two new functions prepare_table and prepare_spec_file now added to manage things in similar way we do to
            prepare IFU data (creating table and spec). These two functions have been added to ExGalPrepare (earlier called: gistVoronoi)
09 Dec 2021: A new definition for velscale has been added. Now we have an array of velscale instead of a single value. The way we measure it, has been extracted from Capellari's codes.
27 Jan 2022: Configs files for all possible binning modes added
27 Jan 2022: configs file directory updated to configs/ExGal_configs (which contains all necessary configs for EXGAL modules)
30 Jan 2022: Spotted an error due to exception handling of prepareSpectralTemplateLibrary in the IFUExGalGANDALF. Now the except only catches the very specific errors. Not all warnings and errors.
            The same has been added to MOSExGalGandalf (Not fully tested yet).
31 Jan 2022: in MOSExGalPrepare, in run_logrebinning function, velscale[i] updated to velscale[0] (using a single velscale value for all spectra) to avoid issue with unequal length of output from log_rebin function
02 Feb 2022: Retracted the change into velscale (31 Jan 20202)
02 Feb 2022: Given that applying velscale (array structure) may change the size of the output logLam and SSpNEW in the log_rebinning process,in run_logrebinning [MOSExGalPrepare] we first run a dummy run to find the min length for the logLam among all fibres
            And then, create the empty arrays to be filled up by log_wave_i and log_bin_data_i for each fibre. No modification is required for IFU (as we have single velscale and identical WL range)
16 March 2022: A bug fixed in MOSExGalGANDALF and IFUExGalGANDAL. Now, EBMV and ERR_EBMV only measured once, and not for each index.
16 March 2022: For_error set to 1 in all config files.
2 May 2023: Following the discussion with Scott, LS mode for LS module updated to ADAPTED to broadening the spectra to LIS resolution
15 Oct 2023: Support for multiple z/class enteries added. index_in_class function added to the aps_utils to extract z/calss info based on the targeted_class
19 Feb 2024: Following WEAVEQAG-34 ticket reported on 16 April 2024 by Scott about wrong SNR calculation for IFU. we made the following changes
            given that aps_util already masked bad ivars (zero and negatives) and changed them to 1.0/(large_error**2), we consider those pixels
             with values < 90% of this 1.0/(large_error**2 as our mask (mask_tgs) we only use the nomask targets for SNR calculation
5 Nov 2024: Gandalf is now supporting no ppxf output scenario.
1 Aug 2025: Major upgrade to the code. Now it is fully adopted PPXF for the emi emission line analysis, instead of GANDALF.
            This is a major change, as it is now using the same PPXF code for both
14 Aug 2025: caldir, catdir and configdir added to inputs. So it can now generate LSF from the L1 calibrations insated of default resolution
14 Aug 2025: New LSF implementation based on L1 calibrations added
"""

########################################
# PHYSICAL CONSTANTS and global parameters
Clight = 299792.458  # km/s

## set default value for error of badpixels/badspaxels
large_error = aps_constants.large_error

##################################################################################################################

def proc_mosExGaL(infiles, headname , params, classfile, outpath, config_dir, templates_dir, aps_ids=None, wlranges=None,
    targsrvy= None, targclass = None, mask_aps_ids = None, area=None, mask_areas=None,
    PPXF=None, EMIPPXF= None, LS= None, EMIPPXF_LEVEL = "BIN", LS_MODE = None, LS_RES = None,
    sens_corr=None, mask_gaps = None, safe_mask_gaps = None, tellurics=None, vacuum=None, fill_gap=None, arms_ratio=None, join_arms=None, nthreads=None, figdir=None, catdir=None, caldir=None, configdir=None,
    extinction_corr=None, extinction_ebv_fixed=None):

    # read APSExGal config file
    configs = json.load(open(params))
    ## We add the name of config file into the configs dic to be passed into the headers
    configs['CONFIG_FILE'] = os.path.basename(params)

    # Galactic extinction correction (SFD98+Fitzpatrick99, shared with
    # REDROCK -- see doc/aps_rr.md) -- resolved from the config's EBmV
    # key unless explicitly overridden here. See
    # ExGalPrepare.resolve_ebmv_extinction's own docstring for the full
    # "None" (automatic per-target) vs a-number (fixed) semantics.
    extinction_kwargs = ExGalPrepare.resolve_ebmv_extinction(
        configs, extinction_corr=extinction_corr,
        extinction_ebv_fixed=extinction_ebv_fixed)
    ExGalPrepare.note_reddening_unused(configs)
    # Record what was actually decided for honest header provenance
    # downstream (see MOSExGalEMIPPXF.py's EBmV header stamp) -- distinct
    # from the raw, unresolved 'EBmV' config string. Finalised below,
    # after APSOBJ exists, with the real measured E(B-V) rather than a
    # mode label -- explicit request: "add ... the one you measure before
    # running ppxf (what we have from the Schlegel model) to the header
    # so people can see what absorption correction is already applied."
    if not extinction_kwargs['extinction_corr']:
        configs['EBmV_APPLIED'] = 'OFF'
    elif extinction_kwargs['extinction_ebv_fixed'] is not None:
        configs['EBmV_APPLIED'] = extinction_kwargs['extinction_ebv_fixed']
    else:
        configs['EBmV_APPLIED'] = 'AUTO (per-target SFD98)'  # placeholder,
        # replaced below with the real measured median once APSOBJ exists

    # here we set the list of classes we are going to proceed with
    working_classlist = ['GALAXY','QSO']

    # read classifier output [updated/commented: 15/10/2023]
    # class_tab = fits.getdata(classfile, 1)

    # this return the aps_ids of the targets, satisfied the class condition.
    # It also accept the nthreads and threadid for parallel processing
    aps_ids_in_class = aps_ids_class(classfile, working_classlist, aps_ids=aps_ids, nthreads=1, threadid=0)

    ## Check if aps_ids_in_class is OK or not. If not, it will force the code to exit (back to one level up)
    if aps_ids_in_class is None or not aps_ids_in_class.size:
        return

    # also get the class info for each target incliding the index in z array, dict of class types, flags, etc
    class_index_dict = index_in_class(classfile, working_classlist, aps_ids=list(aps_ids_in_class), ncchar=2)

    APSOBJ = APSOB(infiles, targsrvy= targsrvy, targclass = targclass, mask_aps_ids = mask_aps_ids , area=area, mask_areas=mask_areas,
     wlranges=wlranges, aps_ids=aps_ids_in_class, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics,
     fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, catdir=catdir, caldir=caldir, configdir=configdir,
     **extinction_kwargs)

    # Replace the 'AUTO' placeholder above with the real measured E(B-V)
    # -- median SFD98 value actually looked up for this batch's targets
    # (each target got its own per-target lookup inside APSOB; MOS
    # targets in one field are close enough on sky that a median is a
    # fair single-number summary for header provenance). Recomputed here
    # rather than threaded out of APSOB's internals -- a second SFD map
    # query is cheap, and this keeps the extinction-correction code path
    # itself unchanged.
    if extinction_kwargs['extinction_corr'] and extinction_kwargs['extinction_ebv_fixed'] is None:
        try:
            from PyAPS.aps_utils import get_sfd_ebv as _get_sfd_ebv
            _ra = np.array([t.targra for t in APSOBJ.data()])
            _dec = np.array([t.targdec for t in APSOBJ.data()])
            _ebv_arr = np.atleast_1d(_get_sfd_ebv(_ra, _dec))
            configs['EBmV_APPLIED'] = float(np.nanmedian(_ebv_arr))
            configs['EBmV_APPLIED_MIN'] = float(np.nanmin(_ebv_arr))
            configs['EBmV_APPLIED_MAX'] = float(np.nanmax(_ebv_arr))
        except Exception as e:
            print(f"WARNING: could not recompute median E(B-V) for header "
                  f"provenance ({e}) -- EBmV_APPLIED stays 'AUTO'")

    targs=APSOBJ.data()
    targs_infiles = APSOBJ.infiles()
    targs_id=APSOBJ.id()
    targs_idfx=APSOBJ.idfx()
    targs_idxf=APSOBJ.idxf()
    targs_nbands = APSOBJ.nbands()
    targs_funits = APSOBJ.funits()
    targs_wavelist= APSOBJ.wavelist()
    targs_mode = APSOBJ.mode()
    targs_join_arms = APSOBJ.join_arms()





    # targs_origin contains the RA and DEC origins in degs.
    targs_origin = APSOBJ.origin()

    ## This is the setups param, generated by APSOBJ. For stitched arms it will be something like 'combined'
    ## As this module use setups param to search for the appropriate templates, we also generate a list of
    ## setups called orig_setups, including the arm name, resolution (L,H) and binning ('', 2, 4, etc.)
    targs_setups = APSOBJ.setups()
    setups_original = APSOBJ.setups_original()

    # just to make sure stitching arms is done:
    if join_arms and (targs_nbands ==1) and (len(setups_original) > 1):
        orig_setups = ['_'.join(setups_original)]
    else:
        orig_setups = targs_setups

    original_wave = targs[0].spectra[0].wave
    nwave = len(original_wave)

    # Storing everything into a structure
    targ_len = len(targs_id)
    exgal_targ = {'aps_id':np.zeros((targ_len), dtype= np.int32), 'targid' : np.empty((targ_len),dtype='U40'),
     'cname' : np.empty((targ_len),dtype='U40'), 'x':np.zeros((targ_len)), 'y':np.zeros((targ_len)),
     'z':np.zeros((targ_len)),'zerr':np.zeros((targ_len)),'healpix':np.zeros((targ_len),dtype=np.int64),
     'x_0':np.zeros((targ_len)), 'y_0':np.zeros((targ_len)), 'wave': np.zeros((nwave, targ_len)),
     'spec': np.zeros((nwave, targ_len)),'error' : np.zeros((nwave, targ_len)), 'snr':np.zeros((targ_len)),
     'signal':np.zeros((targ_len)), 'noise':np.zeros((targ_len)), 'velscale':np.zeros((targ_len)), 'pixelsize':0.0,
     'lsf':np.zeros((targ_len), dtype=object), 'glsf':np.zeros((targ_len), dtype=object)}


    ## Update configs dict to include high level APSOB info
    configs['sens_corr'] = sens_corr
    configs['mask_gaps'] = mask_gaps
    configs['safe_mask_gaps'] = safe_mask_gaps
    configs['vacuum']    = vacuum
    configs['tellurics'] = tellurics
    configs['fill_gap']  = fill_gap
    configs['arms_ratio'] = arms_ratio
    configs['funits']    = targs_funits
    configs['stitched']  = targs_join_arms
    configs['infiles']   = targs_infiles

    ## Call HEALPix
    hp = HEALPix(nside=1024, order='nested', frame=ICRS())


    for ntgs, tgs in enumerate(targs_id):

        # Convert aps_id into the index in the APSOBJ
        # targs_idfx is the dictionary linking each aps_id to the index in the output object
        tgs_indx = targs_idfx[tgs]

        assert targs[targs_idfx[tgs]].id == tgs, 'It should never happen. Something wrong in APSOB'

        # [updated/commented: 15/10/2023]
        # id_in_class= np.ravel(np.where(class_tab['APS_ID'] == tgs))[0]

        exgal_targ['targid'][ntgs] = targs[tgs_indx].targid
        exgal_targ['aps_id'][ntgs] = targs[tgs_indx].aps_id
        exgal_targ['cname'][ntgs] = targs[tgs_indx].cname

        # store the lsf and global lsf for this target
        exgal_targ['lsf'][ntgs] = APSOBJ.get_fwhm(aps_id=targs[tgs_indx].aps_id, fwhm_key='fwhm')[0]
        exgal_targ['glsf'][ntgs] = APSOBJ.get_fwhm(aps_id=targs[tgs_indx].aps_id, fwhm_key='gfwhm')[0]

        exgal_targ['spec'][:,ntgs] = targs[tgs_indx].spectra[0].flux

        ivar_tgs = targs[tgs_indx].spectra[0].ivar
        # given that aps_util already masked bad ivars (zero and negatives) and changed them
        # to 1.0/(large_error**2), we consider those pixels with values < 90% of this 1.0/(large_error**2
        # as our mask (mask_tgs)
        # we only use the nomask targets for SNR calculation

        ivar_mask_value = 1.0/(large_error**2)

        mask_tgs = ( ivar_tgs <= 10 * ivar_mask_value)
        nomask_tgs  = ( ivar_tgs > 10 * ivar_mask_value)

        ivar_tgs[mask_tgs] = ivar_mask_value
        espec_tgs = 1. / (ivar_tgs**.5)
        exgal_targ['error'][:,ntgs] = espec_tgs

        # here we only consider nonmask pixels for calculationg signal and noise and snr
        exgal_targ['signal'][ntgs]  = np.nanmean(targs[tgs_indx].spectra[0].flux[nomask_tgs],axis=0)
        exgal_targ['noise'][ntgs]  = np.abs(np.nanmean(np.sqrt(espec_tgs[nomask_tgs]),axis=0))

        if exgal_targ['noise'][ntgs] > 0.0:
            exgal_targ['snr'][ntgs] = exgal_targ['signal'][ntgs] / exgal_targ['noise'][ntgs]
        else:
            exgal_targ['snr'][ntgs] = 0.0


        exgal_targ['x'][ntgs] = (targs[tgs_indx].targra - targs_origin[0])* 3600.0
        exgal_targ['y'][ntgs] = (targs[tgs_indx].targdec - targs_origin[1])* 3600.0

        # [updated/commented: 15/10/2023]
        # exgal_targ['z'][ntgs] = class_tab['Z'][id_in_class]
        # exgal_targ['zerr'][ntgs] = class_tab['ZERR'][id_in_class]

        # update on 15/10/2023 to support multiple/single z/class enteries
        exgal_targ['z'][ntgs] = class_index_dict[tgs]['Z']
        exgal_targ['zerr'][ntgs] = class_index_dict[tgs]['ZERR']

        coord = SkyCoord('%fd %fd' %(targs[tgs_indx].targra,targs[tgs_indx].targdec))
        exgal_targ['healpix'][ntgs] = hp.skycoord_to_healpix(coord)

        exgal_targ['x_0'][ntgs] = targs_origin[0]
        exgal_targ['y_0'][ntgs] = targs_origin[1]


        # De-redshift spectra
        exgal_targ['wave'][:,ntgs] = targs[tgs_indx].spectra[0].wave / (1 + exgal_targ['z'][ntgs])


        ## If velscale is not set by user in the config file, here we evaluate it
        if str(configs['VELSCALE']).replace(" ",'').lower() in ['none', 'null' , '']:

            s_sampling = 1.0
            lam_range = [exgal_targ['wave'][0,ntgs],exgal_targ['wave'][-1,ntgs]]
            s_lam = len(exgal_targ['wave'][:,ntgs])
            dlam = (lam_range[1]-lam_range[0])/(s_lam-1.) # Assumes constant lambda steps ...
            lim = lam_range/dlam + [-0.5,0.5] # All in units of dlam
            loglim = np.log(lim)
            exgal_targ['velscale'][ntgs] = (np.diff(loglim)/(s_sampling*s_lam)*Clight)[0]
        else:
            exgal_targ['velscale'][ntgs] = configs['VELSCALE']

    # exgal_targ['pixelsize'] = configs['PIXELSIZE']

    ##################################################################################################################



    # Save LSF data
    lsf_filepath = MOSExGalPrepare.save_lsf_data(exgal_targ, headname, outpath)

    # Update configs if needed
    configs['lsfdir'] = lsf_filepath

    # # TODO: DO we need a code to reject defunct spaxels and apply SNR threshold??

    ## create a table structure from the exgal_targ
    binNum = ExGalPrepare.prepare_table(exgal_targ['aps_id'],exgal_targ['targid'],exgal_targ['cname'] , exgal_targ['x'], exgal_targ['y'], exgal_targ['z'],exgal_targ['zerr'], \
        exgal_targ['healpix'], exgal_targ['x_0'], exgal_targ['y_0'], exgal_targ['signal'], exgal_targ['noise'], exgal_targ['pixelsize'], exgal_targ['snr'], headname, outpath, configs)


    ## create a SPEC file from the exgal_targ [LINEAR MODE]

    ExGalPrepare.prepare_spec_file(binNum, exgal_targ['spec'], exgal_targ['error'], headname, outpath, exgal_targ['wave'], 'lin', verbose = False)

    ### Log-rebin spectra and apply bins, or read them from file
    log_spec, log_error, logLam = MOSExGalPrepare.log_rebinning(exgal_targ, configs, headname, outpath)

    ## create a SPEC file from the exgal_targ [LOG MODE]
    ExGalPrepare.prepare_spec_file(binNum, log_spec,log_error, headname,outpath, logLam, 'log', verbose = False)


    ### Read LSF of templates and construct an interpolation function
    LSF_Temp           = np.genfromtxt(config_dir+'LSF-Config_'+configs['SSP_LIB'], comments='#')
    LSF_Templates = interp1d(LSF_Temp[:,0], LSF_Temp[:,1], 'linear', fill_value = 'extrapolate')

    ###MAIN runners

    if PPXF:
        MOSExGalPPXF.runModule_PPXF(nthreads, configs, exgal_targ['velscale'], LSF_Templates, outpath, config_dir, templates_dir, figdir, headname, large_error)
    if EMIPPXF:
        MOSExGalEMIPPXF.runModule_EMIPPXF(nthreads, configs, exgal_targ['velscale'], LSF_Templates, outpath, config_dir, templates_dir, figdir, headname, large_error, tie_mode='optimised', debug=False, diag_plots=False, LEVEL=EMIPPXF_LEVEL)
    if LS:
        MOSExGalLS.runModule_LINESTRENGTH(LS_MODE, LS_RES, nthreads, configs, exgal_targ['velscale'], outpath, config_dir, templates_dir, headname)

    return
#################################################################################################

def mosExGal_weave(options=None):

    # Shared registry -- see aps_common_args.py's own module docstring.
    # `overrides` reproduces this script's own current --help wording
    # verbatim wherever it genuinely differs from the shared default
    # (including the "wavelenght"/"whill" typos, left as-is here).
    # `exclude=["overwrite"]` because this script genuinely has no such
    # flag today (commented out in the source this was migrated from).
    parser = build_common_parser(
        description="RUN APS_MOS_GAL for WEAVE target spectra [MOS Mode].",
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "caldirs", "output"],
        overrides={
            "wlranges": {"help": "wavelenght range array for each elements of the setup"},
            "vacuum": {"default": False, "help": "transform wavelenght from air to vacuum"},
            "join_arms": {"default": True},
            "headname": {"help": "Output headname. The output filenames whill be generated based on this headname"},
        },
        exclude=["overwrite"],
        extra_args=[
            (("--classfile",), dict(type=none_or_str, default=None, required=True,
                                     help="The input fits file, contains the classification table")),
            (("--config_dir",), dict(type=none_or_str, default=None, required=True,
                                      help="Directory contains APSExGal configuration files")),
            (("--templates_dir",), dict(type=none_or_str, default=None, required=True,
                                         help="Parent directory contains working templates")),
            (("--params",), dict(type=none_or_str, default=None, required=True,
                                  help="parameter (.json) file (full path)")),
            (("--PPXF",), dict(type=str2bool, default=False, required=False, help="Run PPXF?")),
            (("--EMIPPXF",), dict(type=str2bool, default=False, required=False, help="RUN EMIPPXF?")),
            (("--LS",), dict(type=str2bool, default=False, required=False, help="RUN LS?")),
            (("--EMIPPXF_LEVEL",), dict(type=none_or_str, default='BIN', required=False,
                                         help="Running EMIPPXF at BIN or SPAXEL level. Availble options are [BIN and SPAXEL]")),
            (("--LS_MODE",), dict(type=int, default=1, required=False,
                                   help="Line Strength [1] or Line Strength+SSP [2]")),
            (("--LS_RES",), dict(type=none_or_str, default='ADAPTED', required=False,
                                  help="Running LS on original or the resolution adapted to the SSP libraries. "
                                       "Availble options are [ORIGINAL and ADAPTED]")),
            (("--mp",), dict(type=int, default=1, required=False,
                              help="The number of threads to run original FERRE CODE in Fortran")),
            (("--fig",), dict(type=str2bool, default=False, required=False,
                               help="if True, the code also produces plots of the best fitted model")),
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

    if args.classfile is None:
        raise Exception('You need to specify the CLASS/Z file to proceed')
    classfile = args.classfile

    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
    area, mask_areas = resolved.area, resolved.mask_areas

    if not os.path.exists(args.params):
        sys.exit('No parameter file found! in the %s' %(args.params))

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ', '')


    if not os.path.exists(args.templates_dir):
        sys.exit('TEMPLATES directory: %s does not exist'%(args.templates_dir))

    templates_dir = args.templates_dir

    if not os.path.exists(args.config_dir):
        sys.exit('COnfigurations directory: %s does not exist'%(args.config_dir))

    config_dir = args.config_dir

    if (len(args.infiles) > 1) and (args.join_arms is False):
        sys.exit('No way to proceed with more than one input file without joining arms')


    if (args.PPXF is False) and ((args.EMIPPXF is True) or (args.LS is True)):
        print('Warning: Running EMIPPXF or LS without ruuning PPXF in the same run')
        print('may cause inconsistency among results and order of results in the outputs')
        print('Make sure you are using the same aps_ids list as you used for PPXF, earlier')


    figdir=None
    if args.fig:
        figdir=args.outpath + '/figs/'
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            print("FIGDIR: %s Created!" %(figdir))
        figdir=figdir.replace(' ', '')

    # print args and assigned/default values on the screen
    print_args(args,module='mosExGaL', version= aps_constants.__aps_mosExGal_version__ , path=outpath, headname=args.headname)


    proc_mosExGaL(args.infiles, args.headname , args.params, classfile, outpath, config_dir, templates_dir, aps_ids=aps_ids, wlranges=wlranges,
            targsrvy= targsrvy, targclass = targclass, mask_aps_ids = mask_aps_ids, area= area, mask_areas=mask_areas, PPXF=args.PPXF,
            EMIPPXF= args.EMIPPXF, LS= args.LS, EMIPPXF_LEVEL = args.EMIPPXF_LEVEL, LS_MODE = args.LS_MODE, LS_RES = args.LS_RES, sens_corr=args.sens_corr,
            mask_gaps = args.mask_gaps, safe_mask_gaps = args.safe_mask_gaps,tellurics=args.tellurics, vacuum=args.vacuum, fill_gap=args.fill_gap, arms_ratio=arms_ratio,
            join_arms=args.join_arms, nthreads=args.mp, figdir=figdir, catdir=args.catdir, caldir=args.caldir, configdir=args.configdir)


#################################################################################################
if __name__ == '__main__':

    debug_demo= ['--infiles', '<PYAPS_DATA>/gal_test/superstack_100001.fits', '<PYAPS_DATA>/gal_test/superstack_100000.fits',
        '--classfile' , '<PYAPS_DIR>/PyAPS_results/20170223/3800/zbest_superstack_100001__superstack_100000.fits',
        '--config_dir','<PYAPS_DIR>/configs/Exgal_configs/',
        '--templates_dir','<PYAPS_DIR>/PyAPS_templates/templates_ExGal/',
        '--outpath', '<PYAPS_DIR>/PyAPS_results/20170223/3800/',
        '--aps_ids', '1006,1007', # or 'None' to run for all availble fibreids
        '--wlranges', 'None',
        '--targsrvy', 'None',
        '--targclass', 'None',
        '--mask_aps_ids', 'None',
        '--area', 'None',
        '--mask_areas', 'None',
        '--headname', 'superstack_100001__superstack_100000',
        '--params', '<PYAPS_DIR>/configs/Exgal_configs/MOSLR11.json',
        '--PPXF', 'False',
        '--EMIPPXF', 'True',
        '--EMIPPXF_LEVEL', 'BIN',
        '--LS', 'True',
        '--LS_MODE' ,'1',
        '--LS_RES', 'ADAPTED',
        '--sens_corr' , 'True',
        '--mp' ,'2',
        '--safe_mask_gaps', 'True',
        '--mask_gaps', 'True',
        '--tellurics', 'False',
        '--vacuum', 'False',
        '--fill_gap', 'False',
        '--arms_ratio', '1.0, 0.83',
        '--join_arms', 'True',
        '--fig', 'True',
        '--caldir', '<PYAPS_DATA>/CAL',
        '--catdir', '<PYAPS_DATA>/CAT',
        '--configdir', '<PYAPS_DIR>/configs/ExGal_configs'
        ]


    # ## MOSLIFU mode
    # debug_demo= ['--infiles', '<PYAPS_DATA>/opr4_test/20170930/stack_1004122.fit','<PYAPS_DATA>/opr4_test/20170930/stack_1004121.fit',
    #     '--classfile' , '<PYAPS_DIR>/PyAPS_results/20171001/4407/zbest_stack_1004122__stack_1004121_test.fits',
    #     '--config_dir','<PYAPS_DIR>/configs/Exgal_configs/',
    #     '--templates_dir','<PYAPS_DIR>/PyAPS_templates/templates_ExGal/',
    #     '--outpath', '<PYAPS_DIR>/PyAPS_results/20170223/3800/',
    #     '--aps_ids', '2', # or 'None' to run for all availble fibreids
    #     '--wlranges', '3900.0,4900', '7000.0,8100', # or 'None' to deploy the whole available wlranges
    #     '--targsrvy', 'None',
    #     '--targclass', 'None',
    #     '--mask_aps_ids', 'None',
    #     '--area', 'None',
    #     '--mask_areas', 'None',
    #     '--headname', 'stack_1004122__stack_1004121_test',
    #     '--params', '<PYAPS_DIR>/configs/ExGal_configs/MOSLIFULR11.json',
    #     '--PPXF', 'True',
    #     '--EMIPPXF', 'False',
    #     '--EMIPPXF_LEVEL', 'BIN',
    #     '--LS', 'False',
    #     '--LS_MODE' ,'1',
    #     '--LS_RES', 'ADAPTED',
    #     '--sens_corr' , 'True',
    #     '--mp' ,'1',
    #     '--safe_mask_gaps', 'True',
    #     '--mask_gaps', 'True',
    #     '--tellurics', 'False',
    #     '--vacuum', 'False',
    #     '--fill_gap', 'False',
    #     '--arms_ratio', '1.0, 0.83',
    #     '--join_arms', 'True',
    #     '--fig', 'True',
    #     '--caldir', '<PYAPS_DATA>/CAL',
    #     '--catdir', '<PYAPS_DATA>/CAT',
    #     '--configdir', '<PYAPS_DIR>/configs/ExGal_configs'
    # ]

    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    mosExGal_weave(options=debug_demo)
