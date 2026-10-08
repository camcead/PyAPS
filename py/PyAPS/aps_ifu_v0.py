"""
=====================

APS tools for WEAVE IFU analysis
"""


import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob
import json
import logging
import traceback
import urllib.request
from copy import deepcopy
from multiprocessing import Process, Queue
from pathlib import Path

import astropy.units as u
import matplotlib
import pandas as pd
import sep
from astropy.coordinates import FK5, ICRS, SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table, hstack, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import simple_norm
from astropy.wcs import WCS, utils
from astropy_healpix import HEALPix

# matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

# for photutils > 1.5
from photutils.aperture import CircularAperture, aperture_photometry
from reproject import reproject_interp
from scipy.interpolate import interp1d

import PyAPS
from PyAPS import aps_constants
from PyAPS.aps_ferre import proc_ferre
from PyAPS.aps_L2merge import ifuExGalL2merge, ifuGalL2merge
from PyAPS.aps_rvs import proc_rvs
from PyAPS.aps_utils import (
    APSOB,
    apply_redshift_to_fwhm_corrected,
    aps_ids_class,
    fix_non_unicode_string,
    gen_targlist,
    index_in_class_patch,
    l1_fileinfo,
    makeR,
    none_or_str,
    print_args,
    str2bool,
)
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import IFUExGalEMIPPXF, IFUExGalLS, IFUExGalPPXF
from PyAPS import IFUExGalPrepare as IFUExGalPrepare
from PyAPS import MOSExGalPrepare as MOSExGalPrepare
from PyAPS import ExGalPrepare as ExGalPrepare
from PyAPS import ExGalutil

# for photutils <=1.5
# from photutils import CircularAperture, aperture_photometry







APSVERS = PyAPS.__version__

#################################################
"""
aps_IFU
Python code for analysing APS IFU data





versions:
 3.1 By A. Molaeinezhad (IAC, July 2020)
 3.2 By A. Molaeinezhad (IAC, October 2020)
 3.3 By A. Molaeinezhad (CASU, May 2021)
 3.5 by A. Molaeinezhad (CASU, June 2021) - New version (fully reshuffled and modified)
 4.0 By A. Molaeinezhad (CASU, Aug 2022)
 4.1 By A. Molaeinezhad (CASU, May 2023)
 4.2 By A. Molaeinezhad (CASU, Feb 2024)
 4.3 By A. Molaeinezhad (CASU, Feb 2024)
 4.4 By A. Molaeinezhad (CASU, Feb 2024)
 4.5 By A. Molaeinezhad (CASU, April 2024)
 4.6 By A. Molaeinezhad (CASU, June 2024)
 4.7 By A. Molaeinezhad (CASU, August 2025)
NOTES:


TODO list:

########### BAISC APS PARAM ###################################################################################################
|    param                |    APS default  |  RRWEAVE EQUIVALENT          |  IFUGAL DEFAULT   |   available options/notes      |
infiles (Required)        |        -        |  --infiles (Required)        |        -          |                               |
aps_ids (Optional)        |     None        |  --aps_ids (Optional)        |        None       |                               |
targsrvy (Optional)       |     None        |  --targsrvy (Optional)       |        None       |                               |
targclass (Optional)      |     None        |  --targclass (Optional)      |        None       |                               |
mask_aps_ids (Optional)   |     None        |  --mask_aps_ids (Optional)   |        None       |                               |
wlranges (Optional)       |     None        |  --wlranges (Optional)       |        None       |                               |
sens_corr (Optional)      |     True        |  --sens_corr (Optional)      |        True       |                               |
mask_gaps (Optional)      |     True        |  --mask_gaps (Optional)      |        True       |                               |
safe_mask_gaps (Optional) |     True        |  --safe_mask_gaps (Optional) |        True       |                               |
vacuum (Optional)         |     False       |  --vacuum (Optional)         |        False      |                               |
tellurics (Optional)      |     False       |  --tellurics (Optional)      |        False      |                               |
fill_gap (Optional)       |     False       |  --fill_gap (Optional)       |        False      |                               |
arms_ratio (Optional)     |     None        |  --arms_ratio (Optional)     |        None       | for R band in OPR3B  is 0.83  |
join_arms (Optional)      |     False       |  --join_arms (Optional)      |        True       |                               |
funit (Optional)          |     1.0e18      |  USE DEFAULT APS value       |        AS APS     |                               |
offset_gap_pix (Optional) |     10          |  USE DEFAULT APS value       |        AS APS     |                               |
                          |                 |                              |                   |                               |
######### DEDICATED APS_IFU PARAM ##############################################################################################
                          |                 |  --IFU_config_dir  (Opt)   |          -        |                               |
                          |                 |  --ExGal_templates (Opt)     |          -        |                               |
                          |                 |  --IFU_params    (optional)|          -        |  set if not available         |
                          |                 |  --outpath   (Required)      |          -        |                               |
                          |                 |  --headname  (Optional)      |     'headname'    |                               |
                          |                 |  --fig       (Optional)      |        False      |                               |
                          |                 |  --PPXF      (Optional)      |        True       |                               |
                          |                 |  --EMIPPXF   (Optional)      |        True       |                               |
                          |                 |  --LS        (Optional)      |        True       |                               |
                          |                 |  --mp        (Optional)      |         2         |                               |
                          |                 |  --patch_file (Optional)     |          None   | see format sample in configs dir|
                          |                 |  --seg2d      (Optional)     |        True       |                               |
                          |                 |  --seg2d_white_images (Opt)  |          None     |  list of images               |
                          |                 |  --seg2d_white_src    (Opt)  |     'weave'       |options: 'weave', 'PS1' or PANN|
                          |                 |  --seg2d_search_gaia  (Opt)  |        True       |                               |
                          |                 |  --seg2d_extract      (Opt)  |        True       |                               |
                          |                 |  --seg2d_exclude_ctarg (Opt) |        True       |                               |
                          |                 |  --class_ntop   (Opt)  |        1          |                               |
                          |                 |  --seg2d_mask_stars   (Opt)  |        True       |                               |
                          |                 |  --seg2d_ext_thresh   (Opt)  |        2.5        |                               |
                          |                 |  --seg2d_minarea      (Opt)  |        400        |          pixel                |
                          |                 |  --seg2d_deblend_nthresh(Opt)|        4          |                               |
                          |                 |  --seg2d_radii_factor   (Opt)|        7          |   ~ 3 sigma                   |
                          |                 |  --class_patch  (Optional)   |         True      |                               |
                          |                 |  --class_z_rad  (Optional)   |         1.5       |      arcs                     |
                          |                 |  --class_templates  (Opt)    |          -    |                               |
                          |                 |  --class_templates_ARC  (Opt)    |          -    |                               |
                          |                 |  --user_patch   (Optional)   |         False     |      TBA                      |
################################################################################################################################


Example:

Running IFU Preparatory Module:
aps_ifu.py --infiles <PYAPS_DATA>/APERTIF/stackcube_1002022.fit <PYAPS_DATA>/APERTIF/stackcube_1002021.fit --headname WA --outpath <PYAPS_DATA>/APERTIF/ --wlranges None --sens_corr True --mp 2 --safe_mask_gaps True --mask_gaps True --tellurics False --vacuum False --fill_gap False --join_arms True --fig True --patch_file None --aps_ids None --targsrvy None --targclass None --mask_aps_ids None --arms_ratio 1.0,0.83 --seg2d True --class_ntop 1 --Gal_run False --ExGal_run False --seg2d_white_images None --seg2d_white_src weave --seg2d_search_gaia True --seg2d_extract True --seg2d_exclude_ctarg True --seg2d_mask_stars True --seg2d_ext_thresh 2.5 --seg2d_minarea 400 --seg2d_deblend_nthresh 4 --seg2d_radii_factor 7.0 --class_patch True --class_templates <PYAPS_DIR>/PyAPS_templates/templates_RR/ --class_templates_ARC <PYAPS_DIR>/PyAPS_templates/templates_ARC_RR/ --class_z_rad 1.5 --user_patch False

Running ExGal Module:
aps_ifu.py --infiles <PYAPS_DATA>/APERTIF/stackcube_1002022.fit <PYAPS_DATA>/APERTIF/stackcube_1002021.fit --headname WA --outpath <PYAPS_DATA>/APERTIF/ --wlranges None --sens_corr True --mp 2 --safe_mask_gaps True --mask_gaps True --tellurics False --vacuum False --fill_gap False --join_arms True --fig True  --aps_ids None --targsrvy None --targclass None --mask_aps_ids None --arms_ratio 1.0,0.83 --IFU_config_dir <PYAPS_DIR>/configs/Exgal_configs/ --IFU_params <PYAPS_DIR>/configs/Exgal_configs/LIFUHR11.json --seg2d False --class_ntop 1 --patch_file <PYAPS_DATA>/APERTIF/WA_targets.dat --Gal_run False --ExGal_run True --user_patch False --ExGal_templates <PYAPS_DIR>/PyAPS_templates/templates_ExGal/ --PPXF True --EMIPPXF True --LS True

Running Galactic Module:
aps_ifu.py --infiles <PYAPS_DATA>/APERTIF/stackcube_1002022.fit <PYAPS_DATA>/APERTIF/stackcube_1002021.fit --headname WA --outpath <PYAPS_DATA>/APERTIF/ --wlranges None --sens_corr True --mp 2 --safe_mask_gaps True --mask_gaps True --tellurics False --vacuum False --fill_gap False --join_arms False --fig True  --aps_ids None --targsrvy None --targclass None --mask_aps_ids None --arms_ratio 1.0,0.83 --seg2d False --class_ntop 1 --patch_file <PYAPS_DATA>/APERTIF/WA_targets.dat --Gal_run True --ExGal_run False --user_patch False



History:

11 Feb 2020: Update velscale_ratio for both Gandalf and PPXF to 1 (in IFUExGalPPXF and IFUExGalGANALF)
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

11 Feb 2020: Please note, as error array is affected by Voronoi Binning
 and the error after Voronoi binning is av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
 we use 0.95 * np.sqrt(error_limit) as our error limit to mask CCD gaps
 where 0.95 is 2 sigma around this value
 In MOS mode, in opposite to IFU, we usually do not run VORONOI
 So, we do not expect the error to be affected.
 Just to make sure we are on the safe side, for MOS mode we also use sqrt(error_limit) instead
 of error_limit.

 29 July 2020: Added redshift measurement to the code. A bunch of changes have been made in aps_utils and aps_rr to support this.

29 July 2020: converted from old aps_ifu.py (function mode) into this general APS-friendly structure (By A. Molaeinezhad)


31 August 2020: Since PPXF (VERSION > 7.0), ppxf does not include bvls_solve function. Therefore, to make sure the code is compatible with the latest version
            of PPXF, we added this function to the gandalf_util.py
31 August 2020: Added support to PPXF version> 7.0
31 August 2020: Now PyAPS is independent of the GITS pipeline.
16 October 2020: Now, velscale is a free parameter to be set by user (through the configuration file). If empty/None/Null, if will be
                estimated by the code itself (Suggested by Stefano).
20 Oct 2020: in-line documentation updated
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
** [NEW VERSION] **
20 July 2021: New version (including 2d segmentation and classification)
** [NEW VERSION] **
09 Dec 2021: prepareSpectralTemplateLibrary module has been fully upgraded:
            1- It accepts WEAVE FORMAT (.npy) templates
            2- SSP parameters can be extracted
            3- PCA compressed templates added. Now you can use PCA compressed grids for kinematics
            4- The way the code handle wlranges of templates have been significantly improved
09 Dec 2021: A keyword called PCA added to the config file, to trigger PCA mode
09 Dec 2021: Again we have separated ExGalPrepare module for IFU and MOS. From now on, we have two separate IFUExGalPrepare and MOSExGalPrepare
27 Jan 2022: Confits files for all possible binning modes added
27 Jan 2022: configs file directory updated to configs/ExGal_configs (which contains all necessary configs for EXGAL modules)
27 Jan 2022: [CAUTION: NOT TESTED PROPERLY] A new definition for velscale has been added. Now we have an array of velscale instead of a single value. The way we measure it, has been extracted from Capellari's codes.
30 Jan 2022: new velscale method [previous tweak on 27 Jan 2022] tested.
30 Jan 2022: Spotted an error due to exception handling of prepareSpectralTemplateLibrary in the IFUExGalGANDALF. Now the except only catches the very specific errors. Not all warnings and errors.
            The same has been added to MOSExGalGandalf (Not fully tested yet).
01 Feb 2022: And return has been added to the end of worker functions to avoid getting stuck while the job is running through SLURM and it sticks after it finished.
05 Feb 2022: in ifu_class, we replace missing values of the patch_table with nan (since ASTROPY 5.0.1 missing values are non NAN by default and are masked )
10 Feb 2022: To make sure plotting routine works properly, you should have the following packages (dvipng texlive-latex-extra texlive-fonts-recommended cm-super)
            On Ubuntu: sudo apt-get install dividing texlive-latex-extra texlive-fonts-recommended cm-super
16 Feb 2022: A new parameter (exgal_only) added. If set to True, if only proceed with Exgal sources. Otherwise, it (temporarily) treats Galactic sources like ExGal ones.
16 Feb 2022: the worker code updated. From now on, even if we have a galactic source, it always create a fits L2 file (but with empty results ext).
16 March 2022: A bug fixed in MOSExGalGANDALF and IFUExGalGANDAL. Now, EBMV and ERR_EBMV only measured once, and not for each index.
16 March 2022: For_error set to 1 in all config files.
30 Aug 2022: Galactic module (ifu_Gal) added. It runs RVSPECFIT and FERRE on each pixel set to be galactic (in param file or based on classifier)
30 Aug 2022: ifu_ExGal module updated. Several new keywords and params added to the function. Manual updated.
19 Dec 2022: We temporarily set crr (CosmicRay rejection) to True. (for the FL data)
2 May 2023: Following the discussion with Scott, LS mode for LS module updated to ADAPTED to broadening the spectra to LIS resolution
3 May 2023: LS module broadening the spectra to LIS resolution but WITHOUT considering the sigma coming out of PPXF
16 Oct 2023: Support for multiple z/class enteries added. index_in_class function added to the aps_utils to extract z/calss info based on the targeted_class
02 Feb 2024: UAPSID added to this to be reflected in the primary header of the final products(APS fits files)
09 Feb 2024: seg2d_exclude_ctarg keyword added to control if the central target after extraction of at least one target source must be excluded or not
11 Feb 2024: update_patch_table function added to update/break the patch_table in case mpre that one class is presented for a target
11 Feb 2024: seg2d_exclude_ctarg is not presented in both prepare and run mode in the script generator
15 Feb 2024: counting number of scuessfull and failed runs for each module (Gal/ExGal) to check if all jobs failed or not and report it back to slurm
23 Feb 2024: skip central target in case of being classified as 'Galactic source' and at least one other target is detected (it means we deal with seg2d results not just the central field)
23 Feb 2024: fix_non_unicode_string function added to the aps_util to handle  possible not unicode strings, especially in targ[class]
26 Feb 2024: using mean instead of median to calculate signal in each cube elements (in faint targets, most of pixels are zero so median is always zero).
26 Feb 2024: If the first try to do votonoi binning failed, we continue with another run without voronoi binning by setting min_snr to 0.0
26 Feb 2024: LSF files for IFU mode updated
19 Feb 2024: Following WEAVEQAG-34 ticket reported on 16 April 2024 by Scott about wrong SNR calculation for IFU. we made the following changes
            given that aps_util already masked bad ivars (zero and negatives) and changed them to 1.0/(large_error**2), we consider those pixels
             with values < 90% of this 1.0/(large_error**2 as our mask (mask_tgs) we only use the nomask targets for SNR calculation
19 Feb 2024: two new functions called read/write_ascii_patchfile added to properly read/write the ascii patch files
22 Dec 2022: set crr (CosmicRay rejection) to False. (check history on 19 Dec 2022)
22 Dec 2022: update the SNR below target secianrio. Fixing a bug in related to binNum in ifuExGal
08 June 2024: A dynamic target_snr introduced to deal with large number of spaxels, in cases that SNR is very high
              There is also a worse-case-senario handler that set the target_snr to 0.0 to avoid breaking the code
13 June 2024: update_patch_table function updated to start the new ID from 1, if the central target is already excluded
25 June 2024: fixed a major bug in the SNR calculation. Now use RMS for error calculation.
06 August 2024: in voronoi_2d_binning function, we set cvt to False, as it was introducing some errorr in the tesselation process:
        File "<VENV_DIR>/lib/python3.11/site-packages/vorbin/voronoi_2d_binning.py", line 283, in _cvt_equal_mass
            diff2 = np.sum((xnode - xnode_old)**2 + (ynode - ynode_old)**2)
                            ~~~~~~^~~~~~~~~~~
        ValueError: operands could not be broadcast together with shapes (4169,) (4170,)
06 Aug 2024: in case we are dealing with collapsed spectra for classification and redishift determination,
        it is recommended to deploy cosmic ray cleaning code, as stacking too many spectra together, increase the chance of dealing with cosmic rays
12 Sept 2024: reference_SNR updated. Also now we use the 3rd quantile instead of the mean to find the best SNR or the voronoi
12 Sept 2024: In case of highredshift targets, we modify the working wavelenght ranges for PPXF and GANDALF, introding new parameters called HZ_LMIN/LMAX_PPXF/GANDALF
5 Nov 2024: In case of no overlap between spectra and template, we geneate mock PPXF outputs, readable by Gandalf and LS
5 Nov 2024: In case of no overlap between spectra and template, we geneate mock Gandalf outputs, readable by LS
5 Nov 2024: Gandalf is now supporting no ppxf output scenario.
29 Nov 2024: added resort keyword to update_patch_table to make sure the final output is always sorted based on area of the path to make sure always the biggest patches are lower id
29 Nov 2024: also update the path_area for ifu_ExGal to make sure it double the radius if the path is the only target or if the area is more than half the area of the field
29 Nov 2024: reference_SNR updated to start from 20 instead of 10 (following the discussion with Jesus on Nov 2024)
02 Jul 2024: Spatial binning added to the code. It is now possible to spatially bin the spectra before the voronoi binning, if SPBIN_SIZE param in config is set.
1 Aug 2025: Major upgrade to the code. Now it is fully adopted PPXF for the emi emission line analysis, instead of GANDALF.
            This is a major change, as it is now using the same PPXF code for both
14 Aug 2025: caldir, catdir and IFU_config_dir added to inputs. So it can now generate LSF from the L1 calibrations insated of default resolution
14 Aug 2025: New LSF implementation based on L1 calibrations added
24 Oct 2024: New Surface brighness adoptive filter added to both ifu_Gal/ExGal to ged rid of junks in outer parts with very low SB
05 June 2026: class_templates_ARC added to the config file to allow using arc lamp spectra as templates for classification and redshift measurement. This is specially useful for high redshift targets, where we do not have enough overlap between the spectra and the templates.
"""

########################################

# PHYSICAL CONSTANTS and global parameters
Clight = 299792.458  # km/s

## set default value for error of badpixels/badspaxels
large_error = aps_constants.large_error


# reference values for TARGET_SNR
reference_SNR = [20.0, 30.0, 40.0]


########################################

# Function to convert multidimensional column to CSV
def convert_to_csv(column):
    if isinstance(column[0], (list, np.ndarray)):
        return [','.join(map(str, arr)).replace('"', '') for arr in column]
    else:
        return column
########################################
def write_ascii_patchfile(patch_table, patch_output):
    # Create a new table with multidimensional columns converted to CSV
    ready_to_csv_patch_table_temp = {col_name: convert_to_csv(patch_table[col_name]) for col_name in patch_table.columns}
    ready_to_csv_patch_table = Table(ready_to_csv_patch_table_temp)
    ascii.write(ready_to_csv_patch_table, Path(patch_output) , overwrite=True,quotechar=' ', delimiter=' ', format="commented_header")

########################################

def read_ascii_patchfile(patch_file):
    #read ascii patch files
    # Define the column names
    column_names = ['id', 'RA_icrs', 'DEC_icrs', 'A_world', 'B_world', 'angle', 'flag', 'type', 'Z', 'ZERR', 'ZWARN', 'CLASS']


    # Read the file into a pandas DataFrame
    df = pd.read_csv(patch_file, sep='\\s+', comment='#', header=None,names=column_names)

    # Convert the last column to a list
    df['CLASS'] = df['CLASS'].str.split(',')
    df['Z'] = df['Z'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    df['ZERR'] = df['ZERR'].apply(lambda x: np.array(list(map(float, x.split(',')))))
    df['ZWARN'] = df['ZWARN'].apply(lambda x: np.array(list(map(float, x.split(',')))))

    # Convert the DataFrame to an Astropy Table
    patch_table_temp = Table.from_pandas(df)


    # Given that normal fits file cannot handle arrays as object, we first create empty arrays for Z,ZERR,ZWARN,CLASS
    # and fill them with values from the converted df and then, stack them into that to make a proper table which is
    # convertible to fits file

    # get number of rows
    lpt = len(patch_table_temp)
    # and get number of elements in those entries which are array (object)
    _class_ntop=len(patch_table_temp[0]['CLASS'])

    # create the placeholder for these new part of the table
    patch_table_class = Table([ [[np.nan] * _class_ntop for _ in range(lpt)], [[np.nan] * _class_ntop for _ in range(lpt)] , [[np.nan] * _class_ntop for _ in range(lpt)], [[''] * _class_ntop for _ in range(lpt)] ], names=('Z','ZERR', 'ZWARN', 'CLASS'), dtype=('float','float', 'float', 'U18'))

    # fill the patch_table_class with values from patch_table_temp
    for colnames in ['CLASS', 'Z', 'ZWARN', 'ZERR']:
        for row_i in range(0,lpt):
            for col_i in range(0,_class_ntop):
                patch_table_class[row_i][colnames][col_i] = patch_table_temp[row_i][colnames][col_i]

    # and stack part of patch_table_temp with patch_table_class
    patch_table = hstack([patch_table_temp[['id', 'RA_icrs', 'DEC_icrs', 'A_world', 'B_world', 'angle', 'flag', 'type']], patch_table_class])

    return patch_table

##########################################
def update_patch_table(patch_table, class_lists=[ ['GALAXY','QSO'], ['STAR','WD'] ] , debug=True, ctarg_excluded = False, resort=True):
    """
        update/break the patch_table in case more that one class is presented for a target.
        for each row in original patch table, we loop over each group of classes in class_lists
        and generate a new row for each claas-item for each row in original patch_table
        Note: You can use the  class_lists=[ ['GALAXY'],['QSO'], ['STAR','WD'] ] to seperate QSO and GALAXIES
    """

    index_in_class_patch_dict= dict()
    for i_cl, cl in enumerate(class_lists):
        index_in_class_patch_dict[i_cl] = index_in_class_patch(patch_table, cl, ids=None, ncchar=2)


    # given that id =0 is reserved for Central target with no segmentation, we check if excluding ctarg is already happend or not
    # if so, we start from 1 for the new id. Otherwise, if C target is present, we start from 0
    if ctarg_excluded:
        new_id =1
    else:
        new_id=0

    new_path_list = []
    # first loop over each row in the original patch_table
    for i_pte, pte in enumerate(patch_table):

        if debug:
            print(f"DEBUG: Processing the row={i_pte} id={pte['id']} of the original patch table")
        # now for each row in original patch table, we loop over each group of classes in class_lists
        # and generate a new row for each claas-item for each row in original patch_table
        for index_in_classX_key, index_in_classX_value in index_in_class_patch_dict.items():
            if debug:
                print(f"DEBUG: Processing all classtypes={class_lists[int(index_in_classX_key)]}")

            targX  = index_in_classX_value[pte['id']]

            new_row = dict()
            # we only proceed with flag= 0 from index_in_class_patch: means found that specific classtype in the class array
            # also for Mask type, as we do not care about z info, we ignore this check
            if (targX['flag'] == 0) or (pte['type']=='M'):

                new_row['id'] = new_id
                for fix_colname in ['RA_icrs', 'DEC_icrs', 'A_world', 'B_world', 'angle', 'flag', 'type']:
                    new_row[fix_colname] = pte[fix_colname]
                for arr_colname in ['Z', 'ZERR', 'ZWARN', 'CLASS']:
                    new_row[arr_colname] = targX[arr_colname]
                # also update the new_id
                new_id +=1

                new_path_list.append(new_row)
    updated_patch_table = Table(new_path_list)

    # also making sure the biggest item is always the highest id
    if resort:
        # Sort by A_world * B_world in descending order
        updated_patch_table["area"] = updated_patch_table["A_world"] * updated_patch_table["B_world"]  # Compute area as a new column
        updated_patch_table.sort("area", reverse=True)  # Sort by the area column in descending order

        # Reassign ids based on the sorted order, but using the original range of ids
        sorted_ids = sorted(updated_patch_table["id"])  # Get the sorted range of original ids
        updated_patch_table["id"] = sorted_ids  # Assign the sorted range to the id column

        # Remove the area column if no longer needed
        updated_patch_table.remove_column("area")


    return updated_patch_table
##########################################

def gen_parms(infiles, IFU_config_dir, wlranges):
    l1_info        = l1_fileinfo(infiles, wlranges=wlranges, arms_ratio = None)

    of_infiles        = l1_info['infiles']
    of_obsmode        = l1_info['obsmode']
    of_res_mode       = l1_info['res_mode']
    of_camera         = l1_info['camera']
    of_xbin           = l1_info['xbin']
    of_ybin           = l1_info['ybin']
    of_mode           = l1_info['mode']
    of_setups         = l1_info['setups']
    of_srvys          = l1_info['srvys']
    of_wlranges_def   = l1_info['wlranges_def']
    of_wlranges_out   = l1_info['wlranges']
    of_resolution     = l1_info['resolution']
    of_obid           = l1_info['obid']
    of_obsdate        = l1_info['obsdate']
    # of_arms_ratio     = l1_info['arms_ratio']
    # Generate exgal_param filename (example MOSLR11.json)

    exgal_param = (of_mode+of_res_mode[0]+str(of_xbin[0])+str(of_ybin[0])+'.json').replace(" ",'')
    exgal_param = str(Path(IFU_config_dir).joinpath(exgal_param))

    return exgal_param


#############################################################
def ICRS2FK5(ra, dec, log=True):

    coords_Gaia = SkyCoord(ra=ra,dec=dec, unit='deg', frame=ICRS())
    coords_PS1 = coords_Gaia.transform_to(FK5(equinox='J2000'))
    ra_fk5 = coords_PS1.ra.value
    dec_fk5 = coords_PS1.dec.value

    if log:
        print("RA, DEC [ICRS, J2015.5] = %14.8f , %14.8f" %(ra, dec))
        print("RA, DEC [FK5, J2000]    = %14.8f , %14.8f" %(ra_fk5, dec_fk5))

    return ra_fk5, dec_fk5

#############################################################

def update_coord(infile, extn_list, keyword, value):

    for ext_i in extn_list:
        fits.setval(infile, keyword, value=value, ext=ext_i)
    return



#############################################################
# Following the email exchange with Scott about wrong coordinates of WA and WC IFU in OPR3b


def update_OPR3_coord(infile, extn_list):

    hdu = fits.open(infile)[6]
    wcs = WCS(hdu.header)

    wcs.wcs.CRVAL1
    wcs.wcs.CRVAL2

    c = SkyCoord(ra=wcs.wcs.CRVAL1, dec=wcs.wcs.CRVAL2, frame='fk5', equinox='J2015.5')
    c_icrs=c.transform_to('icrs')


    for ext_i in extn_list:
        fits.setval(infile, 'CRVAL1', value=c_icrs.ra.value, ext=ext_i)
        fits.setval(infile, 'CRVAL2', value=c_icrs.dec.value, ext=ext_i)

    return


##################################################################################################################
def plot_region(infile, headname, figdir, ext=1,  area=None, mask_areas=None):

    ## Check if figdir exist. Otherwise, it will create it.
    if not Path(figdir).exists():
        Path(figdir).mkdir(parents=True, exist_ok=True)
        print("FIGDIR: %s Created!" %(figdir))


    figfname = Path(figdir).joinpath(headname+'_reg.png')
    hdu = fits.open(infile)[ext]
    wcs = WCS(hdu.header)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)
    norm = simple_norm(hdu.data, 'log')
    ax.imshow(hdu.data, origin='lower',norm=norm,  cmap='gray',aspect=0.83)
    ax.coords.grid(True, color='white', ls='dotted')

    ax.set_xlabel('Right Ascension ICRS [hh:mm:ss]')
    ax.set_ylabel('Declination ICRS [hh:mm:ss]')
    if area is not None:
        c = Circle((area[0], area[1]), area[2]/3600.0, edgecolor='yellow', facecolor='none', transform=ax.get_transform('icrs'))
        ax.add_patch(c)

    if mask_areas is not None:
        for each_mask in mask_areas:
                c = Circle((each_mask[0], each_mask[1]), each_mask[2]/3600.0, edgecolor='red', facecolor='none', transform=ax.get_transform('icrs'))
                ax.add_patch(c)

    overlay = ax.get_coords_overlay('icrs')
    overlay[0].set_axislabel('Right Ascension [deg.]')
    overlay[1].set_axislabel('Declination [deg.]')

    plt.savefig(figfname, bbox_inches='tight')
    plt.close()


#############################################################

def geturl(ra, dec, size=240, output_size=None, filters="giryz",
           format="jpg", color=False):
    """
    Get URL for images in the table.

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    colour = if True, creates a colour image (only for jpg or png format).
            Default is return a list of URL for single-filter greyscale images.
    Returns a string with the URL
    """
    if color and format == "fits":
        raise ValueError(
            "color images are available only for jpg or png formats")
    if format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra, dec, size=size, filters=filters)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0, len(table)//2, len(table)-1]]
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + "&{}={}".format(param, table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url

#############################################################


def getimages(ra, dec, size=240, filters="giryz"):
    """Query ps1filenames.py service to get a list of images.

    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}").format(**locals())
    table = Table.read(url, format='ascii')
    return table

#############################################################


#############################################################
class ifu_seg2d ():

    def __init__(self, infiles, headname, outpath, white_images = None, white_src = 'PS1', search_gaia=True, extract = True, mask_stars = True, ext_thresh =2.5, minarea =400 , deblend_nthresh=4, radii_factor=7.0, class_ntop=3):


        self.gaia_parallax_over_error = 1.0
        self.mask_rad_arcsec = 3.0

        self.infiles = infiles
        self.mask_stars = mask_stars
        self.ext_thresh = ext_thresh
        self.minarea = minarea
        self.deblend_nthresh = deblend_nthresh
        self.radii_factor = radii_factor

        # We take the first infile as reference
        assert isinstance(self.infiles, list) , 'infiles must be a list'

        self.__collapse_extnum = 6
        self.headname = headname
        self.collapse_fits = []
        self.whiteimage = None
        self.gaia_stars = None
        self.objects = None
        self.segmap = None
        self.extc_mask = None
        self.extc_wcs = None
        self.patch_table = None

        self.targets_fname  = None
        self.segments_fname = None
        self._class_ntop= class_ntop


        ## Check if outpath and figdir exist. Otherwise, it will create them.
        self.outpath = outpath

        if not Path(self.outpath).exists():
            Path(self.outpath).mkdir(parents=True, exist_ok=True)
            print("outpath: %s Created!" %(self.outpath))


        self.figdir = Path(self.outpath).joinpath('figs')
        if not Path(self.figdir).exists():
            Path(self.figdir).mkdir(parents=True, exist_ok=True)
            print("FIGDIR: %s Created!" %(self.figdir))



        self.wcs, self.fk5_param = self.wcs_info(frame='fk5')
        self.wcs, self.icrs_param = self.wcs_info(frame='icrs')


        if white_images is not None:
            assert isinstance(white_images, list), "white_images must be a list"

            print("Using the white_image list provided by user as reference. Note: white_src param will be ignored")

            try:
                self.collapse_fits = deepcopy(white_images)
                self.create_white_image(extnum=0)
                white_src = 'using_white_image_list' # in this way, the next if conditions (checking white_src) will be ignored
            except:
                print('Something goes wrong with the white_images param. Try2: Using white_src input, instead...')


        ### Creating white-light image: 2 possible scenarios
        if (white_src).upper().replace(" ", "") in ['PANSTARR1', 'PS1', 'PAN', 'PS']:
            print('Using PS1 to create white-light image')
            try:
                self.get_ps1()
                self.create_white_image(extnum=0)
            except:
                print('Something goes wrong with the PS1. Try2: Using WEAVE collapsed data, instead...')
                self.collapse_fits = deepcopy(infiles)
                self.create_white_image(extnum=self.__collapse_extnum)


        elif (white_src).upper().replace(" ", "") in ['WEAVE']:
            print('Using WEAVE (collapsed) to create white-light image')
            self.collapse_fits = deepcopy(infiles)
            self.create_white_image(extnum=self.__collapse_extnum)



        ## check if search_gaia starts is requested
        if search_gaia:
            self.search_gaia(parallax_over_error= self.gaia_parallax_over_error)

        ## in case use set extract = False, then, the code will only generate the table for the central target and apply masking (if set)
        if extract:
            self.object_extract(ext_thresh =self.ext_thresh, minarea =self.minarea , deblend_nthresh=self.deblend_nthresh , mask_stars = self.mask_stars , patch_table = True, fig=True)
        else:
            self.create_aps_targets()



    def wcs_info(self, frame = 'fk5'):
        wcs_param={}
        hdu = fits.open(self.infiles[0])[self.__collapse_extnum]
        icrs_wcs = WCS(hdu.header)

        # We only accept ICRS frame.
        assert icrs_wcs.wcs.radesys.lower().replace(" ", "") == 'icrs'

        # extract the footprint of this field (ra, dec of all 4 corners)
        footprint = icrs_wcs.calc_footprint()

        if frame.lower().replace(" ", "") == 'fk5':
            # Convert coordinates from ICRS (WEAVE) to FK5 (PS1)
            sky_range_WCS = SkyCoord(footprint, frame=ICRS(), unit="deg").transform_to(FK5(equinox='J2000'))
        elif frame.lower().replace(" ", "") == 'icrs' :
            sky_range_WCS = SkyCoord(footprint, frame=ICRS(), unit="deg")

        else:
            sys.exit('Unsupported Coordinarte franme')

        # Extract ranges to found boundaries of the field
        ra_range  = [sky_range_WCS.ra.value.min(), sky_range_WCS.ra.value.max()]
        dec_range = [sky_range_WCS.dec.value.min(), sky_range_WCS.dec.value.max()]

        wcs_param['RA_CENT']  = (ra_range[0] + ra_range[1])/2
        wcs_param['DEC_CENT'] = (dec_range[0] + dec_range[1])/2
        wcs_param['RA_WIDTH']    = (ra_range[1] - ra_range[0]) % 360
        wcs_param['DEC_WIDTH']   = dec_range[1] - dec_range[0]
        wcs_param['MAX_WIDTH']   = np.max([wcs_param['RA_WIDTH']*np.sin(np.radians(wcs_param['DEC_CENT'])), wcs_param['DEC_WIDTH']])

        return icrs_wcs, wcs_param



    def get_ps1(self, filters = 'gir', out_ratio= 1.5, color=True, overwrite=False):

        # define basic params for PS1
        PS1_PIXEL = 0.25  # arcsec

        size_pix = int(self.fk5_param['MAX_WIDTH'] * out_ratio  * 3600. / PS1_PIXEL) # if out_ratio = 1.5 it adds 50% buffer

        #Get PS1 images in different filters for this field
        __ps1_fits = []
        for filter_i in filters:
            ps1_fname  = Path(self.outpath).joinpath(self.headname+'_PS1_'+filter_i+'.fits')

            if (ps1_fname.exists() and overwrite is True) or (ps1_fname.exists() is False):
                fitsurl = geturl(self.fk5_param['RA_CENT'], self.fk5_param['DEC_CENT'], size=size_pix, filters=filter_i, format="fits")
                hdu = fits.open(fitsurl[0])
                hdu.writeto(ps1_fname, overwrite=True)
                __ps1_fits.append(str(ps1_fname))

            else:
                print("A copy of the %s bands image found in the %s directory." %(filter_i, self.outpath))
                __ps1_fits.append(str(ps1_fname))

        if color:
            #Get color PS1 images for this field
            ps1_color  = Path(self.figdir).joinpath(self.headname+'_PS1_color.jpeg')

            if (ps1_color.exists() and overwrite is True) or (ps1_color.exists() is False):
                print("Download color image for this field...")
                jpgurl = geturl(self.fk5_param['RA_CENT'], self.fk5_param['DEC_CENT'], size=size_pix, color=True, format="jpg")
                urllib.request.urlretrieve(jpgurl, ps1_color)
            else:
                print('color image already exists!')

        self.collapse_fits = __ps1_fits
        return self.collapse_fits



    def create_white_image(self, extnum = 0, overwrite=True):

        assert len(self.collapse_fits) > 0, 'No input fits file to create white-ligth image'
        # We take the first image in the ps1_fits list as reference
        hdu_ref = fits.open(self.collapse_fits[0])[extnum]
        data_ref = hdu_ref.data
        data_ref[data_ref == 0] = np.nan # or use np.nan
        final_image_list = [data_ref]

        for file_id, collapse_file in enumerate(self.collapse_fits[1:]):

            hdu_top = fits.open(collapse_file)[extnum]
            data_top = hdu_top.data
            data_top[data_top == 0] = np.nan

            new_hdu_top, footprint = reproject_interp(hdu_top, hdu_ref.header)

            final_image_list.append(new_hdu_top.data)

        final_image = np.mean(final_image_list, axis=0)

        outfile  = Path(self.outpath).joinpath(self.headname+'_white'+'.fits')
        fits.writeto(outfile, final_image, header = hdu_ref.header, overwrite=overwrite)
        print("a white-light image created: %s" %(outfile))
        del final_image_list
        del hdu_ref
        del final_image

        self.whiteimage = str(outfile)
        return self.whiteimage


    def search_gaia(self, out_ratio= 1.5, parallax_over_error=None):

        from astroquery.gaia import Gaia

        # We only accept ICRS frame.
        assert self.wcs.wcs.radesys.lower().replace(" ", "") == 'icrs'

        if parallax_over_error is None:
            parallax_over_error = 1.0
            print('parallax_over_error (automatically) set to %s' %(parallax_over_error))

        # serach Gaia eDR3
        central_coord = SkyCoord(ra=self.wcs.wcs.crval[0], dec=self.wcs.wcs.crval[1], unit=(u.degree,u.degree), frame='icrs')
        radius = u.Quantity(self.fk5_param['MAX_WIDTH'] * 0.5 * out_ratio, u.deg)


        cone_search = Gaia.cone_search_async(central_coord, radius, table_name='gaiaedr3.gaia_source')
        gaia_sources = cone_search.get_results()
        gaia_stars = gaia_sources[gaia_sources['parallax_over_error'] > parallax_over_error]

        self.gaia_stars = gaia_stars

        return self.gaia_stars


    def object_extract(self, ext_thresh =2.5, minarea =400 , deblend_nthresh=4, mask_stars = True, patch_table = True , fig=True):

        assert self.whiteimage is not None, 'No white_light image found to feed extractor code'

        hdu = fits.open(self.whiteimage)[0]
        extc_data = hdu.data
        extc_header = hdu.header
        extc_wcs = WCS(hdu.header)

        self.extc_wcs = extc_wcs


        extc_data[extc_data == 0] = np.nan # or use np.nan

        # numpy 2+ compability
        extc_data = extc_data.byteswap().view(extc_data.dtype.newbyteorder())


        # Create a mask array
        extc_mask=np.zeros(extc_data.shape)
        extc_mask[np.isnan(extc_data)] = np.nan

        if mask_stars and self.gaia_stars is not None:

            # Here we first define the mask radius in arcsec
            # self.mask_rad_arcsec = 3.0

            # create an ogrid array
            X_mask, Y_mask  = np.ogrid[:np.shape(extc_mask)[0], :np.shape(extc_mask)[1]]

            # assuming pixel scales in both x and y direction are identical and therefore we only use the x one
            mask_rad_pix = np.ceil(self.mask_rad_arcsec / (extc_wcs.proj_plane_pixel_scales()[0].value*3600))


            if extc_wcs.wcs.radesys.lower().replace(" ", "") == 'fk5':
                coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())
                coords_gaia_wcs= coords_gaia_wcs.transform_to(FK5(equinox='J2000'))

            elif extc_wcs.wcs.radesys.lower().replace(" ", "") == 'icrs':
                coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())

            for cgw in coords_gaia_wcs:
                # Check if the coordinate coming from gaia is covered by the current field
                if extc_wcs.footprint_contains(cgw):
                    cgw_indx = extc_wcs.world_to_array_index(cgw)
                    dist_from_center = np.sqrt((X_mask - cgw_indx[0])**2 + (Y_mask-cgw_indx[1])**2)
                    mask = dist_from_center <= mask_rad_pix
                    extc_mask[mask] = 1

            self.extc_mask = extc_mask


        # measure a spatially varying background on the image
        extc_bkg = sep.Background(extc_data, mask = extc_mask, bw=64,  bh=64)
        # bkg = sep.Background(data, mask=mask, bw=64, bh=64, fw=3, fh=3)

        # get a "global" mean and noise of the image background:
        # print(extc_bkg.globalback)
        # print(extc_bkg.globalrms)

        extc_bkg_image = extc_bkg.back()

        # evaluate the background noise as 2-d array, same size as original image
        extc_bkg_rms = extc_bkg.rms()

        # subtract the background
        extc_data_sub = extc_data - extc_bkg

        objects, segmap = sep.extract(extc_data_sub, ext_thresh ,minarea = minarea, deblend_nthresh= deblend_nthresh , err=extc_bkg.globalrms, mask=extc_mask, segmentation_map=True)


        self.objects = objects
        self.segmap = segmap


        if patch_table:
            self.create_aps_targets()

        if fig:

            try:
                matplotlib.rc('font', size=5)
                figure = plt.figure()

                # # show the image
                # ax1 = figure.add_subplot(3,2,1)
                # m, s = np.nanmean(extc_data), np.nanstd(extc_data)
                # cax1 = ax1.imshow(extc_data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
                # ax1.set_title('White Light')
                # figure.colorbar(cax1, ax=ax1)


                # show the background
                ax1 = figure.add_subplot(2,3,1, projection=extc_wcs)
                cax1 = ax1.imshow(extc_bkg_image, interpolation='nearest', cmap='gray', origin='lower')
                ax1.set_title('Background')
                ax1.set_xlabel('RA')
                ax1.set_ylabel('DEC')
                figure.colorbar(cax1, ax=ax1, orientation= 'horizontal')



                # show the background noise
                ax2 = figure.add_subplot(2,3,2, projection=extc_wcs)
                cax2 = ax2.imshow(extc_bkg_rms, interpolation='nearest', cmap='gray', origin='lower')
                ax2.set_xlabel('RA')
                ax2.set_ylabel('DEC')
                ax2.set_title('Background RMS')
                figure.colorbar(cax2, ax=ax2, orientation= 'horizontal')


                from matplotlib.patches import Ellipse


                if self.gaia_stars is not None:

                    if extc_wcs.wcs.radesys.lower().replace(" ", "") == 'fk5':
                        coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())
                        coords_gaia_wcs = coords_gaia_wcs.transform_to(FK5(equinox='J2000'))
                    elif extc_wcs.wcs.radesys.lower().replace(" ", "") == 'icrs':
                        coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())

                    else:
                         sys.exit('This coordinate system: %s is not supported by PyAPS' %(extc_wcs.wcs.radesys.lower().replace(" ", "")))


                    coords_gaia_pix = extc_wcs.world_to_pixel(coords_gaia_wcs)

                    ax3 = figure.add_subplot(2,3,3 , projection=extc_wcs)
                    ax3.set_title('GAIA sources [to be masked!]')
                    ax3.set_xlabel('RA')
                    ax3.set_ylabel('DEC')

                    m, s = np.nanmean(extc_data_sub), np.nanstd(extc_data_sub)
                    cax3 = ax3.imshow(extc_data_sub, interpolation='nearest', cmap='gray',
                                   vmin=m-s, vmax=m+s, origin='lower')

                    figure.colorbar(cax3, ax=ax3, orientation= 'horizontal')

                    # plot an ellipse for each object to be masked

                    # computing the mask width and height in pix (pxsc is pixle scale)
                    pxsc = extc_wcs.proj_plane_pixel_scales()[0].value
                    mask_dim_pix = (self.mask_rad_arcsec*2.0/3600.0)/pxsc

                    for i in range(np.shape(coords_gaia_pix)[1]):
                        e = Ellipse(xy=(coords_gaia_pix[0][i], coords_gaia_pix[1][i]), width=mask_dim_pix, height=mask_dim_pix, angle=0.0)
                        e.set_facecolor('none')
                        e.set_edgecolor('yellow')
                        ax3.add_artist(e)





                ax4 = figure.add_subplot(2,3,4, projection=extc_wcs)
                cax4 = ax4.imshow(extc_mask, interpolation='nearest', cmap='gray', origin='lower')
                ax4.set_title('Mask')
                ax4.set_xlabel('RA')
                ax4.set_ylabel('DEC')
                figure.colorbar(cax4, ax=ax4, orientation= 'horizontal')



                ax5 = figure.add_subplot(2,3,5, projection=extc_wcs)
                ax5.set_title('Sources')
                ax5.set_xlabel('RA')
                ax5.set_ylabel('DEC')
                m, s = np.nanmean(extc_data_sub), np.nanstd(extc_data_sub)
                cax5 = ax5.imshow(extc_data_sub, interpolation='nearest', cmap='gray',
                               vmin=m-s, vmax=m+s, origin='lower')

                figure.colorbar(cax5, ax=ax5, orientation= 'horizontal')

                # plot an ellipse for each object
                # The ellipse’s major and minor axes (by default) are multiplied by radii_factor = 7 (which corresponds roughly to twice the size of the isophotal footprint on each axis).

                for i in range(len(objects)):
                    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                                width=self.radii_factor*objects['a'][i],
                                height=self.radii_factor*objects['b'][i],
                                angle=objects['theta'][i] * 180. / np.pi)
                    e.set_facecolor('none')
                    e.set_edgecolor('red')
                    ax5.add_artist(e)



                ax6 = figure.add_subplot(2,3,6, projection=extc_wcs)
                cax6 = ax6.imshow(segmap, interpolation='nearest', cmap='gray', origin='lower')
                ax6.set_title("Segmentation Map")
                ax6.set_xlabel('RA')
                ax6.set_ylabel('DEC')
                figure.colorbar(cax6, ax=ax6, orientation= 'horizontal')



                plt.tight_layout()
                objects_figname  = Path(self.figdir).joinpath(self.headname+'_source_detection.pdf')
                plt.savefig(objects_figname)
                plt.close()
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(exc_value)
                sys.stdout.flush()



    def create_aps_targets(self):


        if (self.objects is None) or len(self.objects) == 0:

            print('WARNING: This module needs at least one target (Only the full-field info will be generated)')


        ### STEP1: Create a Table structure and fill it with info of the whole field (original field)
        ## The main reason behind defining this is that, if extractor code failed to extract any objects, we have this patch_table
        ## that contains the full-field and information of mask stars


        patch_table = Table()
        patch_table['id'] = [0]
        patch_table['RA_icrs'] = [self.icrs_param['RA_CENT']] * u.degree
        patch_table['DEC_icrs'] = [self.icrs_param['DEC_CENT']] * u.degree
        patch_table['A_world'] = [self.icrs_param['RA_WIDTH']] * u.degree
        patch_table['B_world'] = [self.icrs_param['DEC_WIDTH']] * u.degree
        patch_table['angle'] = [0.0] * u.degree
        patch_table['flag'] = [0.0]
        patch_table['type'] = ['C'] # type M:mask, T: Target, C: Centre



        ### STEP2: Create a Table structure out of objects with structure similar to patch_table
        if (self.objects is not None) and len(self.objects) > 0 :

            extc_table=Table(self.objects)

            extc_table.sort('flux', reverse=True)

            ## IMPORTANT: ID here starts from 1. Because we have dedicate id=0 to the whole image
            ## And please note, this IDs are not matched with the ids from the segment map

            extc_table.add_column(np.arange(len(extc_table), dtype=np.int32)+1, index=0, name='id')

            ## Evaluate pixel_scale in degree
            pxsc = self.extc_wcs.proj_plane_pixel_scales()[0].value
            xy_world_icrs = self.extc_wcs.pixel_to_world(extc_table['x'],extc_table['y'] ).transform_to('icrs')
            xy_world_fk5_2000 = self.extc_wcs.pixel_to_world(extc_table['x'],extc_table['y'] ).transform_to(FK5(equinox='J2000'))
            extc_table.add_column(xy_world_icrs.ra, name='RA_icrs')
            extc_table.add_column(xy_world_icrs.dec, name='DEC_icrs')


            ## Following the source extractor manual:
            # The ellipse’s major and minor axes are multiplied by radii_factor = 7 (which corresponds roughly to twice the size of the isophotal footprint on each axis).
            extc_table.add_column(extc_table['a']*self.radii_factor*pxsc, name='A_world')
            extc_table['A_world'].unit = u.degree
            extc_table.add_column(extc_table['b']*self.radii_factor*pxsc, name='B_world')
            extc_table['B_world'].unit = u.degree
            extc_table.add_column(extc_table['theta']*57.2958, name='angle')
            extc_table['angle'].unit = u.degree
            extc_table['angle'].info.format = '9.3f'
            extc_table['theta'].unit = u.radian
            extc_table.add_column('T', name='type')

            patch_table_objects = extc_table['id', 'RA_icrs', 'DEC_icrs', 'A_world', 'B_world', 'angle', 'flag', 'type']


            ### STEP3: Vertically Join two tables (objects if available and original patch_table (full_field))
            patch_table = vstack([patch_table,patch_table_objects], join_type='exact')
        else:
            extc_table = None

        ### STEP4: Add MASK stars to the table
        if self.mask_stars and self.gaia_stars is not None:

            if self.extc_wcs.wcs.radesys.lower().replace(" ", "") == 'fk5':
                coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())
                coords_gaia_wcs= coords_gaia_wcs.transform_to(FK5(equinox='J2000'))

            elif self.extc_wcs.wcs.radesys.lower().replace(" ", "") == 'icrs':
                coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())


            for cgw in coords_gaia_wcs:
                # Check if the coordinate coming from gaia is covered by the current field
                if self.extc_wcs.footprint_contains(cgw):

                    patch_table.add_row([np.max(patch_table['id'])+1, # id (stars from the last id of the main targets)
                                    cgw.transform_to('icrs').ra, # RA_icrs
                                    cgw.transform_to('icrs').dec, # DEC_icrs
                                    self.mask_rad_arcsec*2.0/3600.0, # a_world (in degree)
                                    self.mask_rad_arcsec*2.0/3600.0, # b_world (in degree)
                                    0.0, # angle
                                    0, #flag
                                    'M' # type M:mask, T: Target, C: Centre
                        ])


        ### Step 5 (method 1): Create empty columns for redshift/classification info
        # patch_table.add_column(np.nan, name='Z')
        # patch_table.add_column(np.nan, name='ZERR')
        # patch_table.add_column(np.nan, name='ZWARN')
        # patch_table.add_column('%10s'%('nan'), name='CLASS')


        ### Step 5 (method 2): Create empty columns for redshift/classification info
        ## Note: We had to create a table with the same number of rows as the patch_table
        ## and then fill it with nan values and then stack it with the original patch_table
        ## We could not simply use add_column to the patch_table as that way, we could not define dtype for each column
        ## especially here that we should set dtype='object' for the CLASS columns to be able to deal with arbitrary size strings

        lpt = len(patch_table)
        patch_table_class = Table([ [[np.nan] * self._class_ntop for _ in range(lpt)], [[np.nan] * self._class_ntop for _ in range(lpt)] , [[np.nan] * self._class_ntop for _ in range(lpt)], [[''] * self._class_ntop for _ in range(lpt)] ], \
            names=('Z','ZERR', 'ZWARN', 'CLASS'), dtype=('float','float', 'float', 'U18'))
        patch_table = hstack([patch_table,patch_table_class])

        ### STEP6: write outputs
        targets_fits  = Path(self.outpath).joinpath(self.headname+'_targets'+'.fits')
        patch_table_hdu = fits.table_to_hdu(patch_table)
        patch_table_hdu.writeto(targets_fits, overwrite=True, checksum=True)
        self.targets_fname  = targets_fits

        try:

            segments_fits = Path(self.outpath).joinpath(self.headname+'_segments'+'.fits')
            segments_hdu = fits.PrimaryHDU(self.segmap, header = self.extc_wcs.to_header())
            segments_hdu.writeto(segments_fits, overwrite=True, checksum=True)
            self.segments_fname = segments_fits
        except:
            print('Warning: No segmentation map has been generated. If you need it, set seg2d_extract to True')


        self.patch_table = patch_table
        self.extc_table  = extc_table
        ## keep filenames for targets and segments (fits files)


        return patch_table



    # return the original WCS (infile)
    def get_wcs(self):
        return self.wcs


    # return the APS-RADY table of objects
    def get_patch_table(self):
        return self.patch_table

    # return the original extraction table with no modification
    def get_extc_table(self):
        return self.extc_table


    # return segments filename (full path)
    def get_seg_fname(self):
        return self.segments_fname

    # return targets filename (full path)
    def get_targets_fname(self):
        return self.targets_fname


# Define the custom converter for columns with arrays
def str_to_array(s):
    return np.array([float(value) for value in s.split(',')])


def ifu_class(infiles, headname, patch_file, class_templates, class_templates_ARC=None, aps_ids=None, mask_aps_ids=None ,z_rad=1.5, wlranges= None,figdir= None, ncpus=2, arms_ratio=None, class_ntop=1, catdir=None, caldir=None, IFU_config_dir=None ):


    from pathlib import Path

    from astropy.table import Table

    from PyAPS.aps_rr import rrweave_worker

    ## check if PCA REDROCK templates exist
    assert Path(class_templates).is_dir(), 'No Redrock templates directory found'



    # ----------------------------------------------------------------
    # TEMPORARY: Auto-derive class_templates_ARC from class_templates
    # if not provided.
    # TODO: Replace this with explicit class_templates_ARC path in
    # all configs once updated.
    # ----------------------------------------------------------------
    if class_templates_ARC is None and class_templates is not None:
        templates_str = str(class_templates).strip()
        if 'templates_RR' in templates_str:
            class_templates_ARC = templates_str.replace(
                'templates_RR', 'templates_ARC_RR')
            print(f"\n{'='*70}")
            print(f"  *** TEMPORARY WARNING ***")
            print(f"  class_templates_ARC not set — auto-derived from "
                f"class_templates path:")
            print(f"  class_templates    : {templates_str}")
            print(f"  class_templates_ARC: {class_templates_ARC}")
            print(f"  Please update your config/script to set "
                f"class_templates_ARC explicitly.")
            print(f"  This auto-derivation will be removed in a future version.")
            print(f"{'='*70}\n")

            # Verify derived path exists
            from pathlib import Path
            if not Path(class_templates_ARC).exists():
                print(f"  *** WARNING: derived class_templates_ARC path "
                    f"does not exist:")
                print(f"  {class_templates_ARC}")
                print(f"  Continuing without archetypes — results may be wrong!")
                class_templates_ARC = None
        else:
            print(f"\n  WARNING: class_templates_ARC not set and cannot be "
                f"auto-derived")
            print(f"  (class_templates path does not contain 'templates_RR')")
            print(f"  class_templates: {templates_str}")
            print(f"  Continuing without archetypes — results may be wrong!\n")


    if class_templates_ARC is not None:
        assert Path(class_templates_ARC).is_dir(), 'No Redrock ARC templates directory found'

    ## Check if the patch file (fits table) is available
    assert Path(patch_file).exists(), 'No Patch file (%s) found' %(patch_file)

    ## This code can handle both fits and ASCII patch_file

    if Path(patch_file).suffix == '.fits':
        patch_table = Table.read(patch_file)

        ## replacing missing values with nan (since astropy 5.0.1 , we should manually check and handle missing values)
        for ptcl in patch_table.colnames:
            if hasattr(patch_table[ptcl], 'mask'):
                patch_table[ptcl] = patch_table[ptcl].filled(np.nan)

    else:
        patch_table = read_ascii_patchfile(patch_file)
        assert patch_table.colnames == ['id','RA_icrs','DEC_icrs','A_world','B_world','angle','flag','type','Z','ZERR','ZWARN', 'CLASS'], 'Error: inconsistent patch_file column names'
        assert patch_table[0]['type']=='C', 'type of the target in the first row of the patch_file must be C [Central target]'


    assert len(patch_table) > 0 , 'Patch table must have at least 1 row (%d rows found)' %(len(patch_table))

    ## Loop over all targets in the patch_table (including mask targets, if any)

    for trgs in patch_table:
    #     if trgs['type']=='C':
    #         continue



        ## We ignore those trgs that already have redshift
        if not np.all(np.isnan(trgs['Z'])):
            continue
        collapse_fname = headname+"_"+ ('P%04d' %(trgs['id']))

        ## Here we use the minimum of A, B (both in deg, converted to arcsec) and z_rad (in arcsec) to make the Z_diam (collapse and search diametre)
        Z_diam=np.min([trgs['A_world']*3600.0, trgs['B_world']*3600.0, z_rad *2.0 ])

        area = [trgs['RA_icrs'],trgs['DEC_icrs'],Z_diam,Z_diam,0]

        print('**************************************************************')
        print('Adding class and redshift info to the patch %d' %(trgs['id']))
        print('RA = %f DEC = %f A= %f (arcsec.) B= %f (arcsec.) theta = %f (deg.) collapse_diam = %d (arcsec) type = %s'\
         %(trgs['RA_icrs'],trgs['DEC_icrs'], trgs['A_world']*3600.0, trgs['B_world']*3600.0,trgs['angle'] , Z_diam, trgs['type'] ))


        # try:
        scandata, zbest, zspec, zfitall = rrweave_worker(infiles, class_templates, aps_ids=aps_ids, mask_aps_ids=mask_aps_ids, ntop=class_ntop,
            area=area, mask_areas=None, wlranges=wlranges, sens_corr=True, mask_gaps=True, safe_mask_gaps=True , vacuum=True, tellurics=False, fill_gap=False,
            arms_ratio=arms_ratio, join_arms=False, figdir=figdir, return_outputs=True, collapse=True, collapse_fname = collapse_fname, ncpus=ncpus ,nminima=3,
            gpu=False, max_gpuprocs = None, catdir=catdir, caldir=caldir, configdir=IFU_config_dir, archetypes=class_templates_ARC)



        # del scandata, zspec, zfitall

        trgs['Z'] = np.array([[targz_i] for targz_i in zbest['Z']]) if zbest['Z'].ndim > 1 else np.array(zbest['Z'])
        trgs['ZERR'] = np.array([[targz_i] for targz_i in zbest['ZERR']]) if zbest['ZERR'].ndim > 1 else np.array(zbest['ZERR'])
        trgs['ZWARN'] = np.array([[targz_i] for targz_i in zbest['ZWARN']]) if zbest['ZWARN'].ndim > 1 else np.array(zbest['ZWARN'])
        # for CLASS we do extra check to make sure it is not bytes type
        trgs_class_list_dummy = np.array([[targz_i] for targz_i in zbest['CLASS']]) if zbest['CLASS'].ndim > 1 else np.array(zbest['CLASS'])
        trgs_class_list = [clslist_i.decode() if isinstance(clslist_i, bytes) else clslist_i for clslist_i in trgs_class_list_dummy]
        trgs['CLASS'] = np.array(trgs_class_list)

        print('Patch info updated')
        # except:
        #     print('failed to add class and redshift info to the patch %d' %(trgs['id']))

    ## if the input patch_file is fits or ASCII, we update that as well (now with the updated class params).
    if Path(patch_file).suffix == '.fits':
        ## Update Patch file (fits format)
        patch_table_hdu = fits.table_to_hdu(patch_table)
        patch_table_hdu.writeto(patch_file, overwrite=True, checksum=True)
    else:
        # write the updated table in a format similar to the input one
        write_ascii_patchfile(patch_table, Path(patch_file))
    return patch_table

############################################################################

def ifu_Gal(
    infiles,
    headname,
    IFU_params,
    outpath,
    wlranges=None,
    aps_ids=None,
    targsrvy=None,
    targclass=None,
    mask_aps_ids=None,
    area=None,
    mask_areas=None,
    nthreads=1,
    sens_corr=True,
    mask_gaps=True,
    safe_mask_gaps=True,
    vacuum=False,
    tellurics=False,
    fill_gap=False,
    arms_ratio=None,
    join_arms=True,
    z_input=None,
    overwrite=True,
    rvs_config=None,
    ferre_templates=None,
    ferre_grid_ids=None,
    ferre_grid_prefix=None,
    ferre_exe=None,
    spbin_size_gal=None,
    min_snr_gal=None,
    voronoi_gal=None,
    target_snr_gal=None,
    catdir=None,
    caldir=None,
    IFU_config_dir=None,
):
    """
    Updated Galactic mode with consistent spatial binning and Voronoi support.
    Matches ExGal mode's approach but keeps linear wavelengths.
    """

    # Force join_arms to True for Galactic mode
    join_arms = True
    print("GALACTIC MODE: Processing with joined arms (join_arms=True)")

    ## Check if figdir exist (and create it if necessary)
    figdir = Path(outpath).joinpath("figs_Gal")
    if not Path(figdir).exists():
        Path(figdir).mkdir(parents=True, exist_ok=True)
        print("FIGDIR: %s Created!" % (figdir))

    # read config file for Galactic mode
    if not os.path.exists(IFU_params):
        sys.exit("No parameter file found! in the %s" % (IFU_params))
    configs_gal = json.load(open(IFU_params))
    configs_gal["CONFIG_FILE"] = os.path.basename(IFU_params)

    figdir = f"{figdir}{os.sep}"

    # Set spatial binning parameters
    if spbin_size_gal is not None:
        configs_gal["SPBIN_SIZE_GAL"] = spbin_size_gal
    elif "SPBIN_SIZE_GAL" in configs_gal and configs_gal["SPBIN_SIZE_GAL"] is not None:
        spbin_size_gal = configs_gal["SPBIN_SIZE_GAL"]
    else:
        spbin_size_gal = -1.0
    print(f"DEBUG: SPBIN for GAL set to {spbin_size_gal}")

    # Set minimum SNR
    if min_snr_gal is not None:
        configs_gal["MIN_SNR"] = min_snr_gal
    elif "MIN_SNR_GAL" in configs_gal and configs_gal["MIN_SNR_GAL"] is not None:
        configs_gal["MIN_SNR"] = configs_gal["MIN_SNR_GAL"]
    else:
        configs_gal["MIN_SNR"] = 10.0
    print(f'DEBUG: MIN SNR for GAL set to {configs_gal["MIN_SNR"]}')

    # Set Voronoi parameters
    if voronoi_gal is not None:
        configs_gal["VORONOI_GAL"] = int(voronoi_gal)
    elif "VORONOI" not in configs_gal:
        configs_gal["VORONOI_GAL"] = 0  # Default to no Voronoi for Galactic
    print(f'DEBUG: VORONOI_GAL for GAL set to {configs_gal["VORONOI_GAL"]}')

    # Set TARGET_SNR_GAL
    if target_snr_gal is not None:
        configs_gal["TARGET_SNR_GAL"] = target_snr_gal
    elif "TARGET_SNR_GAL" not in configs_gal:
        configs_gal["TARGET_SNR_GAL"] = 20.0  # Default target SNR
    print(f'DEBUG: TARGET_SNR_GAL for GAL set to {configs_gal["TARGET_SNR_GAL"]}')

    # Set rvs_config
    if rvs_config is not None:
        configs_gal["RVS_CONFIG"] = rvs_config
    elif "RVS_CONFIG" in configs_gal and configs_gal["RVS_CONFIG"] is not None:
        rvs_config = configs_gal["RVS_CONFIG"]
    else:
        configs_gal["RVS_CONFIG"] = "<PYAPS_DIR>/configs/rvs_config.yaml"
        rvs_config = configs_gal["RVS_CONFIG"]
    print(f'DEBUG:RVS_CONFIG for GAL set to {configs_gal["RVS_CONFIG"]}')

    # Set ferre_exe
    if ferre_exe is not None:
        configs_gal["FERRE_EXE"] = ferre_exe
    elif "FERRE_EXE" in configs_gal and configs_gal["FERRE_EXE"] is not None:
        ferre_exe = configs_gal["FERRE_EXE"]
    else:
        configs_gal["FERRE_EXE"] = "<PYAPS_DIR>/externals/ferre/bin/ferre.x"
        ferre_exe = configs_gal["FERRE_EXE"]
    print(f'DEBUG:FERRE_EXE for GAL set to {configs_gal["FERRE_EXE"]}')

    # Set ferre_grid_ids
    if ferre_grid_ids is not None:
        configs_gal["FERRE_GRID_IDS"] = ferre_grid_ids
    elif "FERRE_EXE" in configs_gal and configs_gal["FERRE_GRID_IDS"] is not None:
        ferre_grid_ids = [
            str(x).replace(" ", "") for x in configs_gal["FERRE_GRID_IDS"].split(",")
        ]
    else:
        configs_gal["FERRE_GRID_IDS"] = "1,2,3,4,5,6,7,8,9"
        ferre_grid_ids = [
            str(x).replace(" ", "") for x in configs_gal["FERRE_GRID_IDS"].split(",")
        ]
    print(f'DEBUG:FERRE_GRID_IDS for GAL set to {configs_gal["FERRE_GRID_IDS"]}')

    # Set ferre_templates
    if ferre_templates is not None:
        configs_gal["FERRE_TEMPLATES"] = ferre_templates
    elif (
        "FERRE_TEMPLATES" in configs_gal and configs_gal["FERRE_TEMPLATES"] is not None
    ):
        ferre_templates = configs_gal["FERRE_TEMPLATES"]
    else:
        configs_gal["FERRE_TEMPLATES"] = "<PYAPS_DIR>/PyAPS_templates/templates_FR/"
        ferre_templates = configs_gal["FERRE_TEMPLATES"]
    print(f'DEBUG:FERRE_TEMPLATES for GAL set to {configs_gal["FERRE_TEMPLATES"]}')

    # Set ferre_grid_prefix
    if ferre_grid_prefix is not None:
        configs_gal["FERRE_GRID_PREFIX"] = ferre_grid_prefix
    elif (
        "FERRE_GRID_PREFIX" in configs_gal
        and configs_gal["FERRE_GRID_PREFIX"] is not None
    ):
        ferre_grid_prefix = configs_gal["FERRE_GRID_PREFIX"]
    else:
        configs_gal["FERRE_EXE"] = "n"
        ferre_grid_prefix = configs_gal["FERRE_GRID_PREFIX"]
    print(f'DEBUG:FERRE_GRID_PREFIX for GAL set to {configs_gal["FERRE_GRID_PREFIX"]}')

    configs_gal["COVAR_VOR"] = configs_gal.get("COVAR_VOR", 0.0)

    print(f'VORONOI for GAL set to {configs_gal["VORONOI_GAL"]}')
    if configs_gal["VORONOI_GAL"] == 1:
        print(f'TARGET_SNR_GAL for Voronoi set to {configs_gal["TARGET_SNR_GAL"]}')

    ## Check the APS_IDs belong to this patch
    aps_ids_in_class, _, _, _, _ = gen_targlist(
        infiles[0],
        "IFU",
        aps_ids=aps_ids,
        targsrvy=targsrvy,
        targclass=targclass,
        mask_aps_ids=mask_aps_ids,
        area=area,
        mask_areas=mask_areas,
        la_out=False,
    )

    if len(aps_ids_in_class) == 0:
        print("WARNING: No valid APS IDs found for this Galactic patch. Skipping...")
        return

    # Create APSOBJ with join_arms=True
    APSOBJ = APSOB(
        infiles,
        targsrvy=targsrvy,
        targclass=targclass,
        mask_aps_ids=mask_aps_ids,
        area=area,
        mask_areas=mask_areas,
        wlranges=wlranges,
        aps_ids=aps_ids_in_class,
        sens_corr=sens_corr,
        mask_gaps=mask_gaps,
        safe_mask_gaps=safe_mask_gaps,
        vacuum=vacuum,
        tellurics=tellurics,
        fill_gap=fill_gap,
        arms_ratio=arms_ratio,
        join_arms=True,
        catdir=catdir,
        caldir=caldir,
        configdir=IFU_config_dir,
    )

    targs = APSOBJ.data()
    targs_infiles = APSOBJ.infiles()
    targs_id = APSOBJ.id()
    targs_idfx = APSOBJ.idfx()
    targs_idxf = APSOBJ.idxf()
    targs_nbands = APSOBJ.nbands()
    targs_funits = APSOBJ.funits()
    targs_wavelist = APSOBJ.wavelist()
    targs_mode = APSOBJ.mode()
    targs_join_arms = APSOBJ.join_arms()
    targs_origin = APSOBJ.origin()
    targs_setups = APSOBJ.setups()
    setups_original = APSOBJ.setups_original()
    wlranges_original = APSOBJ.wlranges_original()
    targs_lsf = APSOBJ.get_fwhm(aps_id=None, fwhm_key="gfwhm")

    if targs_lsf is not None:
        if len(targs_lsf) > 1:
            targs_lsf = targs_lsf[0]

    original_wave = targs[0].spectra[0].wave
    nwave = len(original_wave)

    # Storing everything into a structure
    targ_len = len(targs_id)
    gal_targ = {
        "aps_id": np.zeros((targ_len), dtype=np.int32),
        "targid": np.empty(targ_len, dtype="U40"),
        "cname": np.empty(targ_len, dtype="U40"),
        "x": np.zeros((targ_len)),
        "y": np.zeros((targ_len)),
        "z": np.zeros((targ_len)),
        "zerr": np.zeros((targ_len)),
        "healpix": np.zeros((targ_len), dtype=np.int64),
        "x_0": np.zeros((targ_len)),
        "y_0": np.zeros((targ_len)),
        "wave": np.zeros((targ_len)),
        "spec": np.zeros((nwave, targ_len)),
        "error": np.zeros((nwave, targ_len)),
        "snr": np.zeros((targ_len)),
        "signal": np.zeros((targ_len)),
        "noise": np.zeros((targ_len)),
        "velscale": 0.0,
        "pixelsize": 0.0,
    }

    ## Update configs_gal dict
    configs_gal["sens_corr"] = sens_corr
    configs_gal["mask_gaps"] = mask_gaps
    configs_gal["safe_mask_gaps"] = safe_mask_gaps
    configs_gal["vacuum"] = vacuum
    configs_gal["tellurics"] = tellurics
    configs_gal["fill_gap"] = fill_gap
    configs_gal["arms_ratio"] = arms_ratio
    configs_gal["funits"] = targs_funits
    configs_gal["stitched"] = True
    configs_gal["infiles"] = targs_infiles
    configs_gal["join_arms"] = True
    configs_gal["targs_mode"] = targs_mode

    configs_gal["orig_setups"] = setups_original
    configs_gal["targs_setups"] = targs_setups
    configs_gal["orig_wlranges"] = wlranges_original

    ## Call HEALPix
    hp = HEALPix(nside=1024, order="nested", frame=ICRS())

    # Get reference coordinates for tangent plane projection
    ref_ra = targs_origin[0]  # degrees
    ref_dec = targs_origin[1]  # degrees

    # Calculate spherical correction factor
    cos_dec_correction = np.cos(np.deg2rad(ref_dec))

    print(f"\n{'='*70}")
    print("SPHERICAL PROJECTION CORRECTION")
    print(f"{'='*70}")
    print(f"Reference coordinates:")
    print(f"  RA:  {ref_ra:.6f} deg")
    print(f"  DEC: {ref_dec:.6f} deg")
    print(f"Correction factors:")
    print(f"  cos(DEC) = {cos_dec_correction:.6f}")
    print(f"  X-scale factor: {cos_dec_correction:.6f}")
    print(f"  Y-scale factor: 1.0")
    print(f"\nThis corrects for spherical sky geometry on tangent plane.")
    print(
        f"Without this, maps would be elongated in X by factor {1.0/cos_dec_correction:.2f}"
    )
    print(f"{'='*70}\n")

    for ntgs, tgs in enumerate(targs_id):
        tgs_indx = targs_idfx[tgs]
        assert (
            targs[targs_idfx[tgs]].id == tgs
        ), "It should never happen. Something wrong in APSOB"

        gal_targ["targid"][ntgs] = targs[tgs_indx].targid
        gal_targ["aps_id"][ntgs] = targs[tgs_indx].aps_id
        gal_targ["cname"][ntgs] = targs[tgs_indx].cname
        gal_targ["spec"][:, ntgs] = targs[tgs_indx].spectra[0].flux

        ivar_tgs = targs[tgs_indx].spectra[0].ivar
        ivar_mask_value = 1.0 / (large_error**2)
        mask_tgs = ivar_tgs <= 10 * ivar_mask_value
        nomask_tgs = ivar_tgs > 10 * ivar_mask_value
        ivar_tgs[mask_tgs] = ivar_mask_value
        espec_tgs = 1.0 / (ivar_tgs**0.5)
        gal_targ["error"][:, ntgs] = espec_tgs

        gal_targ["signal"][ntgs] = np.nanmean(
            targs[tgs_indx].spectra[0].flux[nomask_tgs], axis=0
        )
        gal_targ["noise"][ntgs] = np.sqrt(
            np.nanmean(espec_tgs[nomask_tgs] ** 2, axis=0)
        )

        if gal_targ["noise"][ntgs] > 0.0:
            gal_targ["snr"][ntgs] = gal_targ["signal"][ntgs] / gal_targ["noise"][ntgs]
        else:
            gal_targ["snr"][ntgs] = 0.0

        # ===== CORRECTED: Apply tangent plane projection =====
        # Calculate angular offsets in degrees
        dx_deg = targs[tgs_indx].targra - ref_ra
        dy_deg = targs[tgs_indx].targdec - ref_dec

        # Apply tangent plane projection with spherical correction
        gal_targ["x"][ntgs] = np.float64(
            -1.0 * dx_deg * 3600.0 * cos_dec_correction
        )  # arcsec
        gal_targ["y"][ntgs] = np.float64(dy_deg * 3600.0)  # arcsec

        gal_targ["z"][ntgs] = z_input[0]
        gal_targ["zerr"][ntgs] = z_input[1]

        coord = SkyCoord("%fd %fd" % (targs[tgs_indx].targra, targs[tgs_indx].targdec))
        gal_targ["healpix"][ntgs] = hp.skycoord_to_healpix(coord)

        gal_targ["x_0"][ntgs] = targs_origin[0]
        gal_targ["y_0"][ntgs] = targs_origin[1]

    # and for those parameters that are constant across spaxels
    gal_targ["wave"] = targs[0].spectra[0].wave
    gal_targ["velscale"] = 0
    gal_targ["pixelsize"] = 0.0  # Set appropriate pixel size if needed

    if targs_lsf is not None:
        lsf_values = targs_lsf[0](gal_targ["wave"])
    else:
        lsf_values = None

    ##################################################################################################################
    # Check for spatial binning
    if spbin_size_gal > 0:
        print(
            f"Applying spatial binning with size {spbin_size_gal} arcsec for Galactic sources"
        )
        # Apply spatial binning
        spatial_bins, aps_id_to_spatial_bin = ExGalPrepare.spatial_bin_with_provenance(
            gal_targ, spbin_size_gal,
            min_snr=configs_gal["MIN_SNR"],
            apply_flux_filter=False,
            verbose=True,
        )

        idx_inside = np.where(spatial_bins["flag"] == 1)[0]
        idx_outside = np.where(spatial_bins["flag"] == 0)[0]

        n_valid = len(idx_inside)
        n_total = len(spatial_bins["snr"])

        print(
            f"SNR filtering: {n_valid}/{n_total} bins have SNR >= {configs_gal['MIN_SNR']}"
        )

        if n_valid == 0:
            print(f"WARNING: No spatial bins meet the SNR threshold")
            return

        # Define Voronoi bins (may be applied on top of spatial bins)
        binNum = ExGalPrepare.define_voronoi_bins(
            configs_gal["VORONOI_GAL"],
            spatial_bins["aps_ids"],
            spatial_bins["targid"],
            spatial_bins["cname"],
            spatial_bins["x"],
            spatial_bins["y"],
            spatial_bins["z"],
            spatial_bins["zerr"],
            spatial_bins["healpix"],
            spatial_bins["x_0"],
            spatial_bins["y_0"],
            spatial_bins["signal"],
            spatial_bins["noise"],
            gal_targ["pixelsize"],
            spatial_bins["snr"],
            configs_gal["TARGET_SNR_GAL"],
            configs_gal["COVAR_VOR"],
            idx_inside,
            idx_outside,
            headname,
            outpath,
            configs_gal,
        )

        # Check if we need Voronoi binning on top of spatial bins
        if configs_gal["VORONOI_GAL"] == 1:
            # Apply Voronoi binning to spatially binned data
            print(
                f"Applying Voronoi binning (target SNR={configs_gal['TARGET_SNR_GAL']}) on top of spatial bins"
            )
            ExGalPrepare.apply_voronoi_bins(
                binNum,
                spatial_bins["spec"][:, idx_inside],
                spatial_bins["error"][:, idx_inside],
                headname,
                outpath,
                spatial_bins["wave"],
                "lin",  # LINEAR for Galactic
                "IFU",
            )
        else:
            # No Voronoi - save spatial bins directly
            print(
                "VORONOI=0 with spatial binning: saving spatially binned spectra directly"
            )
            ExGalPrepare.save_binned_spectra_novor(
                spatial_bins["spec"][:, idx_inside],
                spatial_bins["error"][:, idx_inside],
                spatial_bins["wave"],
                headname,
                outpath,
                flag="lin",
                binNum=binNum,
            )

        print(
            f"Created {n_total} spatial bins ({n_valid} valid) from {targ_len} spaxels"
        )

        try:
            ExGalPrepare.plot_spatial(
                gal_targ,
                spatial_bins=spatial_bins,
                color_field="snr",
                bin_size_label=spbin_size_gal,
                figdir=figdir,
                headname=headname,
            )
        except Exception as e:
            print(f"Error plotting spatial bins: {e}")

        try:
            table_file = os.path.join(outpath, f"{headname}_table.fits")
            voronoi_data = ExGalPrepare.read_voronoi_fits_table(table_file)
            ExGalPrepare.voronoi_visualization(
                gal_targ,
                voronoi_data,
                save_path=os.path.join(figdir, f"{headname}_voronoi_cube.png"),
                dpi=72,
                quantity="snr",
                show_filtered=False,
            )
        except Exception as e:
            print(f"Error reading/plotting Voronoi table: {e}")

    else:
        # No spatial binning - process original spaxels
        print("No spatial binning - processing original spaxels")

        # Apply SNR threshold
        idx_inside = np.where(gal_targ["snr"] >= configs_gal["MIN_SNR"])[0]
        idx_outside = np.where(gal_targ["snr"] < configs_gal["MIN_SNR"])[0]

        n_valid = len(idx_inside)

        if n_valid == 0:
            print(f"WARNING: No spaxels meet the SNR threshold")
            return

        print(
            f"Processing {n_valid}/{targ_len} spaxels with SNR >= {configs_gal['MIN_SNR']}"
        )

        # Define Voronoi bins
        binNum = ExGalPrepare.define_voronoi_bins(
            configs_gal["VORONOI_GAL"],
            gal_targ["aps_id"],
            gal_targ["targid"],
            gal_targ["cname"],
            gal_targ["x"],
            gal_targ["y"],
            gal_targ["z"],
            gal_targ["zerr"],
            gal_targ["healpix"],
            gal_targ["x_0"],
            gal_targ["y_0"],
            gal_targ["signal"],
            gal_targ["noise"],
            gal_targ["pixelsize"],
            gal_targ["snr"],
            configs_gal["TARGET_SNR_GAL"],
            configs_gal["COVAR_VOR"],
            idx_inside,
            idx_outside,
            headname,
            outpath,
            configs_gal,
        )

        if configs_gal["VORONOI_GAL"] == 1:
            # Apply Voronoi binning
            print(
                f"Applying Voronoi binning with target SNR={configs_gal['TARGET_SNR_GAL']}"
            )
            ExGalPrepare.apply_voronoi_bins(
                binNum,
                gal_targ["spec"][:, idx_inside],
                gal_targ["error"][:, idx_inside],
                headname,
                outpath,
                gal_targ["wave"],
                "lin",  # LINEAR for Galactic
                "IFU",
            )
        else:
            # No Voronoi - save directly
            print("VORONOI=0 without spatial binning: saving spectra directly")
            ExGalPrepare.save_binned_spectra_novor(
                gal_targ["spec"][:, idx_inside],
                gal_targ["error"][:, idx_inside],
                gal_targ["wave"],
                headname,
                outpath,
                flag="lin",
                binNum=binNum,
            )

    ##################################################################################################################
    # In ifu_Gal function, replace the proc_rvs call with:

    ## Run RVS on binned data
    print("\n" + "=" * 70)
    print("Running RVS on binned Galactic data...")
    print("=" * 70)

    # Import the new IFU RVS module
    from PyAPS.aps_ifu_rvs import run_rvs_for_ifu_gal

    # Simply call the function - it will automatically construct the filename
    rvs_success = run_rvs_for_ifu_gal(
        headname=headname,
        outpath=outpath,
        rvs_config=rvs_config,
        figdir=str(figdir),
        nthreads=nthreads,
        configs_gal=configs_gal,
        overwrite=True,
        lsf=lsf_values,
    )

    if not rvs_success:
        print("WARNING: RVS processing failed for Galactic sources")

    ## Run FERRE on binned data
    if ferre_exe is not None and ferre_templates is not None:
        print("\n" + "=" * 70)
        print("Running FERRE on binned Galactic data...")
        print("=" * 70)

        # Import the new IFU FERRE module
        from PyAPS.aps_ifu_ferre import run_ferre_for_ifu_gal
    else:
        ferre_success = False

    # Call the function - it will automatically construct the filename
    ferre_success = run_ferre_for_ifu_gal(
        headname=headname,
        outpath=outpath,
        ferre_exe=ferre_exe,
        ferre_templates=ferre_templates,
        ferre_grid_ids=ferre_grid_ids,
        ferre_grid_prefix=ferre_grid_prefix,
        configs_gal=configs_gal,
        nthreads=nthreads,
        overwrite=True,
        outspec=True,
        lsf=lsf_values,
    )

    if not ferre_success:
        print("WARNING: FERRE processing failed for Galactic sources")
    else:
        print(
            "INFO: FERRE processing skipped (ferre_exe or ferre_templates not provided)"
        )

    return

############################################################################

def find_nearest_snr(reference_values, value):
    """
    This function takes a list of reference values and a target value,
    and returns the nearest value from the reference values to the target value.

    :param reference_values: List of numerical reference values
    :param value: The target value to find the nearest reference value for
    :return: The nearest reference value
    """
    nearest_value = min(reference_values, key=lambda x: abs(x - value))
    return nearest_value



############################################################################
def ifu_ExGal(infiles, headname, IFU_params, outpath, IFU_config_dir, templates_dir, wlranges = None,
    aps_ids = None, targsrvy=None, targclass=None, mask_aps_ids=None, area=None, mask_areas=None, PPXF=False, EMIPPXF=False, LS= False, EMIPPXF_LEVEL = None,
    LS_MODE = None, LS_RES=None, nthreads=1, sens_corr=True, mask_gaps=True, safe_mask_gaps=True, vacuum=False, tellurics=False,
    fill_gap=False, arms_ratio=None, join_arms=False, z_input=None, catdir=None, caldir=None):
    """
    Updated ExGal processing with consistent spatial binning handling.
    When spatial binning is used with VORONOI=0, saves binned spectra directly.
    """

    # read APSExGal config file
    if not os.path.exists(IFU_params):
        sys.exit('No parameter file found! in the %s' %(IFU_params))
    configs = json.load(open(IFU_params))
    configs['CONFIG_FILE'] = os.path.basename(IFU_params)


    # Extract basic ExGAL settings from the config file if presented
    if EMIPPXF_LEVEL is not None:
        configs['EMIPPXF_LEVEL'] = EMIPPXF_LEVEL
    elif 'EMIPPXF_LEVEL' in configs and configs['EMIPPXF_LEVEL'] is not None:
        EMIPPXF_LEVEL = configs['EMIPPXF_LEVEL']
    else:
        EMIPPXF_LEVEL = 'BIN'
    print(f'DEBUG: EMIPPXF_LEVEL for ExGAL set to {EMIPPXF_LEVEL}')


    if LS_MODE is not None:
        configs['LS_MODE'] = LS_MODE
    elif 'LS_MODE' in configs and configs['LS_MODE'] is not None:
        LS_MODE = configs['LS_MODE']
    else:
        LS_MODE = 1
    print(f'DEBUG: LS_MODE for ExGAL set to {LS_MODE}')



    if LS_RES is not None:
        configs['LS_RES'] = LS_RES
    elif 'LS_RES' in configs and configs['LS_RES'] is not None:
        LS_RES = configs['LS_RES']
    else:
        LS_RES = 'ADAPTED'
    print(f'DEBUG: LS_RES for ExGAL set to {LS_RES}')







    ## Check if figdir exist (and create it if necessary)
    figdir = Path(outpath).joinpath('figs_ExGal')
    if not Path(figdir).exists():
        Path(figdir).mkdir(parents=True, exist_ok=True)
        print("FIGDIR: %s Created!" %(figdir))
    figdir= f'{figdir}{os.sep}'

    ## Just quickly check the APS_IDs belong to this patch
    aps_ids_in_class, _, _, _,_ = gen_targlist( infiles[0], 'IFU' , aps_ids=aps_ids, targsrvy= targsrvy, targclass = targclass,
        mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas, la_out=False)

    if len(aps_ids_in_class) ==0:
        print("WARNING: No valid APS IDs found for this ExGal patch. Skipping...")
        return

    APSOBJ = APSOB(infiles, targsrvy= targsrvy, targclass = targclass, aps_ids=aps_ids_in_class, mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics,
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, catdir=catdir, caldir=caldir, configdir=IFU_config_dir)

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
    targs_origin = APSOBJ.origin()
    targs_centroid = APSOBJ.centroid()
    targs_lsf = APSOBJ.get_fwhm(aps_id=None, fwhm_key='gfwhm')

    if len(targs_lsf) > 1:
        sys.exit('LSF results show arm stitching has not happened or we are still dealing with more than one arm')
    targs_lsf = targs_lsf[0]

    targs_setups = APSOBJ.setups()
    setups_original = APSOBJ.setups_original()

    if targs_join_arms and (targs_nbands ==1) and (len(setups_original) > 1):
        orig_setups = ['_'.join(setups_original)]
    else:
        orig_setups = targs_setups

    original_wave = targs[0].spectra[0].wave
    nwave = len(original_wave)
    pixel_scale = original_wave[1]-original_wave[0]

    # Storing everything into a structure
    targ_len = len(targs_id)
    cube = {'aps_id':np.zeros((targ_len), dtype= np.int32), 'targid' : np.empty(targ_len,dtype='U40'),
    'cname' : np.empty(targ_len,dtype='U40'), 'x':np.zeros((targ_len)), 'y':np.zeros((targ_len)),
    'z':np.zeros((targ_len)),'zerr':np.zeros((targ_len)),'healpix':np.zeros((targ_len),dtype=np.int64) ,
    'x_0':np.zeros((targ_len)), 'y_0':np.zeros((targ_len)),'wave': np.zeros((targ_len)),
    'spec': np.zeros((nwave, targ_len)), 'error' : np.zeros((nwave, targ_len)),'snr':np.zeros((targ_len)),
    'signal':np.zeros((targ_len)), 'noise':np.zeros((targ_len)), 'velscale':0.0, 'pixelsize':0.0, 'pixel_scale':0.0}

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

    # Get reference coordinates for tangent plane projection
    ref_ra = targs_origin[0]   # degrees
    ref_dec = targs_origin[1]  # degrees

    # Calculate spherical correction factor
    cos_dec_correction = np.cos(np.deg2rad(ref_dec))

    print(f"\n{'='*70}")
    print("SPHERICAL PROJECTION CORRECTION")
    print(f"{'='*70}")
    print(f"Reference coordinates:")
    print(f"  RA:  {ref_ra:.6f} deg")
    print(f"  DEC: {ref_dec:.6f} deg")
    print(f"Correction factors:")
    print(f"  cos(DEC) = {cos_dec_correction:.6f}")
    print(f"  X-scale factor: {cos_dec_correction:.6f}")
    print(f"  Y-scale factor: 1.0")
    print(f"\nThis corrects for spherical sky geometry on tangent plane.")
    print(f"Without this, maps would be elongated in X by factor {1.0/cos_dec_correction:.2f}")
    print(f"{'='*70}\n")



    for ntgs, tgs in enumerate(targs_id):
        tgs_indx = targs_idfx[tgs]
        assert targs[targs_idfx[tgs]].id == tgs, 'It should never happen. Something wrong in APSOB'

        cube['aps_id'][ntgs] = targs[tgs_indx].aps_id
        cube['targid'][ntgs] = targs[tgs_indx].targid
        cube['cname'][ntgs] = targs[tgs_indx].cname
        cube['spec'][:,ntgs] = targs[tgs_indx].spectra[0].flux

        ivar_tgs = targs[tgs_indx].spectra[0].ivar
        ivar_mask_value = 1.0/(large_error**2)
        mask_tgs = ( ivar_tgs <= 10 * ivar_mask_value)
        nomask_tgs  = ( ivar_tgs > 10 * ivar_mask_value)
        ivar_tgs[mask_tgs] = ivar_mask_value
        espec_tgs = 1. / (ivar_tgs**.5)
        cube['error'][:,ntgs] = espec_tgs

        cube['signal'][ntgs]  = np.nanmean(targs[tgs_indx].spectra[0].flux[nomask_tgs],axis=0)
        cube['noise'][ntgs] = np.sqrt(np.nanmean(espec_tgs[nomask_tgs]**2, axis=0))

        if cube['noise'][ntgs] > 0.0:
            cube['snr'][ntgs] = cube['signal'][ntgs] / cube['noise'][ntgs]
        else:
            cube['snr'][ntgs] = 0.0


        # ===== CORRECTED: Apply tangent plane projection =====
        # Calculate angular offsets in degrees
        dx_deg = targs[tgs_indx].targra - ref_ra
        dy_deg = targs[tgs_indx].targdec - ref_dec

        # Apply tangent plane projection with spherical correction
        cube['x'][ntgs] = np.float64(-1.0 * dx_deg * 3600.0 * cos_dec_correction)  # arcsec
        cube['y'][ntgs] = np.float64(dy_deg * 3600.0)  # arcsec

        cube['x_0'][ntgs] = targs_origin[0]
        cube['y_0'][ntgs] = targs_origin[1]
        cube['z'][ntgs] = z_input[0]
        cube['zerr'][ntgs] = z_input[1]

        coord = SkyCoord('%fd %fd' %(targs[tgs_indx].targra,targs[tgs_indx].targdec))
        cube['healpix'][ntgs] = hp.skycoord_to_healpix(coord)

    # De-redshift spectra
    cube['wave'] = targs[0].spectra[0].wave / (1 + z_input[0])

    # Update for high-redshift targets
    if z_input[0] >= 1.0:
        configs['LMIN_PPXF'] = configs['HZ_LMIN_PPXF']
        configs['LMAX_PPXF'] = configs['HZ_LMAX_PPXF']
        configs['LMIN_EMI'] = configs['HZ_LMIN_EMI']
        configs['LMAX_EMI'] = configs['HZ_LMAX_EMI']

    # Set velscale
    if str(configs['VELSCALE']).replace(" ",'').lower() in ['none', 'null' , '']:
        s_sampling = 1.0
        lam_range = [cube['wave'][0],cube['wave'][-1]]
        s_lam = len(cube['wave'])
        dlam = (lam_range[1]-lam_range[0])/(s_lam-1.)
        lim = lam_range/dlam + [-0.5,0.5]
        loglim = np.log(lim)
        cube['velscale']= float(np.diff(loglim)/(s_sampling*s_lam)*Clight)
    else:
        cube['velscale'] = configs['VELSCALE']

    cube['pixelsize'] = configs['PIXELSIZE']
    cube['pixel_scale'] = pixel_scale

    ##################################################################################################################
    # Reject defunct spaxels and apply SNR threshold
    configs['MIN_SNR'] = 1.e-3

    # Check for spatial binning
    if configs.get('SPBIN_SIZE_EXGAL', -1.0) > 0:
        spbin_size_exgal = configs['SPBIN_SIZE_EXGAL']

        ExGalutil.prettyOutput_Done(f"Spatial Binning Mode with size {spbin_size_exgal} arcsec", progressbar=True)


        # Apply spatial binning
        spatial_bins, aps_id_to_spatial_bin = ExGalPrepare.spatial_bin_with_provenance(
            cube, spbin_size_exgal,
            min_snr=configs['MIN_SNR'],
            verbose=True,
            apply_flux_filter=True if configs["SB_FILTER"] == 1 else False,
            flux_filter_mode=configs['SB_FILTER_MODE'],
            flux_filter_snr_min=configs.get('SB_FILTER_SNR_MIN', 3.0),    # NEW
            flux_filter_delta=configs.get('SB_FILTER_DELTA', 2.0),         # NEW
            flux_filter_min_spaxels=configs.get('SB_FILTER_MIN_SPAXELS', 10),  # NEW
            flux_filter_percentile=configs.get('SB_FILTER_PERCENT', 10),
            flux_filter_min_keep_fraction=configs['SB_FILTER_MIN_FRAC'],
        )

        idx_inside = np.where(spatial_bins['flag'] == 1)[0]
        idx_outside = np.where(spatial_bins['flag'] == 0)[0]

        n_valid = len(idx_inside)
        n_total = len(spatial_bins['snr'])

        print(f"SNR/SB filtering: {n_valid}/{n_total} bins have SNR >= {configs['MIN_SNR']}")

        if n_valid == 0:
            print("WARNING: No spatial bins meet the SNR/SB threshold")
            return

        # Update TARGET_SNR if needed
        if str(configs['TARGET_SNR']).replace(" ",'').lower() in ['none', 'null' , '']:
            print(f"3rd quantile of SNR: {np.nanquantile(spatial_bins['snr'],0.75):.2f}")
            configs['TARGET_SNR'] = find_nearest_snr(reference_SNR, np.nanquantile(spatial_bins['snr'], 0.9))
            print(f"TARGET_SNR set to {configs['TARGET_SNR']}")

        # Define Voronoi bins
        binNum = ExGalPrepare.define_voronoi_bins(
            configs['VORONOI'], spatial_bins['aps_ids'], spatial_bins['targid'],
            spatial_bins['cname'], spatial_bins['x'], spatial_bins['y'],
            spatial_bins['z'], spatial_bins['zerr'], spatial_bins['healpix'],
            spatial_bins['x_0'], spatial_bins['y_0'], spatial_bins['signal'],
            spatial_bins['noise'], cube['pixelsize'], spatial_bins['snr'],
            configs['TARGET_SNR'], configs['COVAR_VOR'], idx_inside, idx_outside,
            headname, outpath, configs
        )

        # Log-rebin the spatially binned spectra
        log_spec_spbin, log_error_spbin, logLam_spbin = IFUExGalPrepare.log_rebinning(
            spatial_bins, configs, headname, outpath, save=False
        )

        # Check if we need Voronoi binning on top of spatial bins
        if configs['VORONOI'] == 1:
            # Apply Voronoi binning to spatially binned data
            ExGalPrepare.apply_voronoi_bins(
                binNum, log_spec_spbin[:, idx_inside], log_error_spbin[:, idx_inside],
                headname, outpath, logLam_spbin, 'log', 'IFU'
            )
        else:
            # No Voronoi binning - save spatial bins directly
            print("VORONOI=0 with spatial binning: saving spatially binned spectra directly")
            ExGalPrepare.save_binned_spectra_novor(
                log_spec_spbin[:, idx_inside], log_error_spbin[:, idx_inside],
                logLam_spbin, headname, outpath, flag='log', binNum=binNum
            )

        try:
            ExGalPrepare.plot_spatial(cube, spatial_bins=spatial_bins, color_field='snr',
                                    bin_size_label=spbin_size_exgal, figdir=figdir, headname=headname)
        except Exception as e:
            print(f"Error plotting spatial bins: {e}")


        try:
            table_file = os.path.join(outpath, f"{headname}_table.fits")
            voronoi_data = ExGalPrepare.read_voronoi_fits_table(table_file)
            ExGalPrepare.voronoi_visualization(
                cube, voronoi_data, save_path=os.path.join(figdir, f"{headname}_voronoi_cube.png"),
                dpi=72, quantity='snr', show_filtered = False
            )
        except Exception as e:
            print(f"Error reading/plotting Voronoi table: {e}")


    else:
        # No spatial binning mode
        ExGalutil.prettyOutput_Done("No Spatial Binning Mode", progressbar=True)

        idx_inside, idx_outside = IFUExGalPrepare.rejectDefunctSpaxels_applySNRThreshold(cube, configs)

        # Update TARGET_SNR if needed
        if str(configs['TARGET_SNR']).replace(" ",'').lower() in ['none', 'null' , '']:
            print(f"3rd quantile of SNR: {np.nanquantile(cube['snr'],0.75):.2f}")
            configs['TARGET_SNR'] = find_nearest_snr(reference_SNR, np.nanquantile(cube['snr'], 0.9))
            print(f"TARGET_SNR set to {configs['TARGET_SNR']}")

        # Define Voronoi bins
        binNum = ExGalPrepare.define_voronoi_bins(
            configs['VORONOI'], cube['aps_id'], cube['targid'], cube['cname'],
            cube['x'], cube['y'], cube['z'], cube['zerr'], cube['healpix'],
            cube['x_0'], cube['y_0'], cube['signal'], cube['noise'],
            cube['pixelsize'], cube['snr'], configs['TARGET_SNR'],
            configs['COVAR_VOR'], idx_inside, idx_outside, headname, outpath, configs
        )

        # Log-rebin the original spectra
        log_spec, log_error, logLam = IFUExGalPrepare.log_rebinning(cube, configs, headname, outpath)

        if configs['VORONOI'] == 1:
            # Apply Voronoi binning
            ExGalPrepare.apply_voronoi_bins(
                binNum, log_spec[:, idx_inside], log_error[:, idx_inside],
                headname, outpath, logLam, 'log', 'IFU'
            )
        else:
            # No Voronoi binning - save directly
            print("VORONOI=0 without spatial binning: saving log-rebinned spectra directly")
            ExGalPrepare.save_binned_spectra_novor(
                log_spec[:, idx_inside], log_error[:, idx_inside],
                logLam, headname, outpath, flag='log', binNum=binNum
            )

        try:
            table_file = os.path.join(outpath, f"{headname}_table.fits")
            voronoi_data = ExGalPrepare.read_voronoi_fits_table(table_file)
            ExGalPrepare.voronoi_visualization(
                cube, voronoi_data, save_path=os.path.join(figdir, f"{headname}_voronoi_cube.png"),
                dpi=72, quantity='snr'
            )
        except Exception as e:
            print(f"Error reading/plotting Voronoi table: {e}")

    # LSF setup
    LSF_Data = apply_redshift_to_fwhm_corrected(targs_lsf, z_input[0])

    # Also preparing the LSF for templates
    LSF = np.genfromtxt(IFU_config_dir+'LSF-Config_'+configs['SSP_LIB'], comments='#')
    LSF_Templates = interp1d(LSF[:,0], LSF[:,1], 'linear', fill_value = 'extrapolate')

    # Run analysis modules
    str_to_id_emilevel = {'None':0, 'BIN':1, 'SPAXEL':2}
    str_to_id_LS_mod = {'None':0, 'LS':1 , 'SSP':2 }

    if PPXF:
        IFUExGalPPXF.runModule_PPXF(nthreads, configs, cube['velscale'], LSF_Data, LSF_Templates,
                                   outpath, IFU_config_dir, templates_dir, figdir, headname,
                                   z_input[0], z_input[1], large_error, debug=False)

    if EMIPPXF:
        IFUExGalEMIPPXF.runModule_EMIPPXF(nthreads, configs, cube['velscale'], LSF_Data,
                                                     LSF_Templates, outpath, IFU_config_dir, templates_dir,
                                                     figdir, headname, z_input[0], z_input[1], large_error,
                                                     debug=False, diag_plots=True, tie_mode='optimised',
                                                     save_all_plots=True)

    if LS:
        IFUExGalLS.runModule_LINESTRENGTH(LS_MODE, LS_RES, nthreads, configs, cube['velscale'],
                                          LSF_Data, outpath, IFU_config_dir, templates_dir, figdir,
                                          headname, debug=False)

    return
##################################################################################################################################################
def patch_table_needs_modification(patch_table):
    """
    Check if a patch table needs modification by looking for rows with multiple classes or Z values.

    The update_patch_table function splits rows that have multiple classes/redshifts into
    separate rows, each with a single class and corresponding Z value.

    Parameters:
    -----------
    patch_table : astropy.table.Table
        The patch table to check

    Returns:
    --------
    needs_modification : bool
        True if the table has rows that need to be split, False otherwise
    reason : str
        Description of why modification is needed (or not needed)
    """

    if patch_table is None or len(patch_table) == 0:
        return False, "Empty or None patch table"

    # Check each row for multiple classes or Z values
    for row in patch_table:
        # Skip mask entries as they don't get modified
        if row['type'] == 'M':
            continue

        # Check CLASS column
        row_classes = row['CLASS']
        if isinstance(row_classes, (list, np.ndarray)):
            # Count valid (non-empty, non-nan) classes
            valid_classes = []
            for c in row_classes:
                c_str = str(c).strip().upper()
                if c_str and c_str not in ['', 'NAN', 'NONE']:
                    valid_classes.append(c)

            if len(valid_classes) > 1:
                return True, f"Row ID={row['id']} has multiple valid classes: {valid_classes}"

        # Check Z column (redshift values)
        row_z = row['Z']
        if isinstance(row_z, (list, np.ndarray)):
            # Count valid (non-NaN) Z values
            valid_z = [z for z in row_z if not np.isnan(float(z))]

            if len(valid_z) > 1:
                return True, f"Row ID={row['id']} has multiple valid Z values: {valid_z}"

    return False, "All rows have single class and Z values - no modification needed"



##################################################################################################################################################

def ifu_worker(infiles, headname, outpath, IFU_config_dir=None, ExGal_templates=None, class_templates=None, class_templates_ARC=None, patch_file = None, IFU_params = None, user_patch=False, seg2d=False, \
    class_patch=False, wlranges = None, aps_ids = None, targsrvy=None, targclass=None,mask_aps_ids=None, PPXF=True, EMIPPXF=True, LS= True,mp_prep=1, mp_ExGal=1, mp_Gal=1, sens_corr=True, mask_gaps=True, safe_mask_gaps=True, vacuum=False, tellurics=False, fill_gap=False, arms_ratio=None, \
    join_arms=True, seg2d_white_images = None, seg2d_white_src = 'weave', seg2d_search_gaia = True, seg2d_extract=True, seg2d_exclude_ctarg=True ,seg2d_mask_stars = True, seg2d_ext_thresh=2.5, class_ntop=3,  \
    seg2d_minarea=400,seg2d_deblend_nthresh=4, seg2d_radii_factor=8.0, class_z_rad=1.5, Gal_run=True, ExGal_run=True, overwrite=True, UAPSID=None, catdir=None, caldir=None, config_dir=None):
    """
    Main worker function for IFU data processing pipeline.

    This function operates in two distinct modes:

    1. PREPARATION MODE (ExGal_run=False and Gal_run=False):
       - Performs 2D segmentation (if seg2d=True)
       - Runs classification and redshift determination (if class_patch=True)
       - Creates/updates patch table files
       - Writes outputs to disk for use in subsequent analysis runs

    2. ANALYSIS MODE (ExGal_run=True and/or Gal_run=True):
       - Reads existing patch files (must be provided via patch_file parameter)
       - Processes each target according to its classification
       - Does NOT write/modify patch files to avoid conflicts when running in parallel
       - ExGal and Gal modules can run simultaneously on the same patch file

    IMPORTANT: When running analysis in parallel:
    - First run preparation mode to generate patch files
    - Then run ExGal_run and Gal_run separately or in parallel, both reading the same patch file
    - The function handles multi-class splitting internally without file conflicts

    Parameters:
    -----------
    [existing parameter documentation...]
    """

    ### Preparatory steps
    ## set default value (None) for the patch_table
    patch_table = None

    ## Check if figdir exist (and create it if necessary)
    figdir = Path(outpath).joinpath('figs')
    if not Path(figdir).exists():
        Path(figdir).mkdir(parents=True, exist_ok=True)
        print("FIGDIR: %s Created!" %(figdir))

    # Determine operational mode based on inputs
    preparation_mode = (patch_file is None)
    analysis_mode = (ExGal_run or Gal_run)

    if preparation_mode:
        # PREPARATION MODE: No patch file given, disable analysis modules
        print("INFO: Running in PREPARATION MODE - generating patch files only")
        ExGal_run = False
        Gal_run = False
    else:
        # ANALYSIS MODE: Patch file must exist
        print("INFO: Running in ANALYSIS MODE - using existing patch file")
        assert Path(patch_file).exists(), f'Patch file not found: {patch_file}'

    # Handle user_patch mode (not implemented yet)
    if user_patch:
        patch_file = None
        seg2d = False
        sys.exit('This module (user_patch) has not been implemented yet')

    # Load existing patch file if provided (ANALYSIS MODE)
    if patch_file is not None:
        assert Path(patch_file).exists(), f'No patch_file ({patch_file}) found'

        if Path(patch_file).suffix == '.fits':
            patch_table = Table.read(patch_file)
        else:
            patch_table = read_ascii_patchfile(patch_file)
        assert patch_table.colnames == ['id','RA_icrs','DEC_icrs','A_world','B_world','angle','flag','type','Z','ZERR','ZWARN', 'CLASS'], 'Error: inconsistent patch_file column names'

        # Disable seg2d when patch file is provided
        seg2d = False
        print(f"INFO: Loaded patch table with {len(patch_table)} entries")

    # Run 2D segmentation if requested (PREPARATION MODE only)
    if seg2d:
        print("INFO: Running 2D segmentation...")
        ifu_seg2d_out = ifu_seg2d(infiles, headname, outpath, white_images=seg2d_white_images, white_src=seg2d_white_src, search_gaia=seg2d_search_gaia, \
            extract=seg2d_extract, mask_stars = seg2d_mask_stars, ext_thresh = seg2d_ext_thresh, minarea =seg2d_minarea , deblend_nthresh=seg2d_deblend_nthresh, \
            radii_factor=seg2d_radii_factor, class_ntop=class_ntop)
        patch_file = ifu_seg2d_out.get_targets_fname()
        # Ensure classification runs after segmentation
        class_patch = True

    # Run classification if requested
    if class_patch:
        print("INFO: Running classification and redshift determination...")
        assert class_templates is not None, 'No Classifier PCA templates directory found'
        assert Path(class_templates).is_dir(), 'No Classifier PCA templates directory found'
        patch_table = ifu_class(infiles, headname, patch_file, class_templates,class_templates_ARC=class_templates_ARC , aps_ids=aps_ids, mask_aps_ids=mask_aps_ids,
                               z_rad=class_z_rad, wlranges=wlranges, figdir=figdir, ncpus=mp_prep, arms_ratio=arms_ratio,
                               class_ntop=class_ntop, catdir=catdir, caldir=caldir, IFU_config_dir=IFU_config_dir)

    # Verify patch table exists
    assert patch_table is not None, 'No option set to generate patch_table. Please Modify the input params'
    assert (patch_table['type']).dtype in ['U1','S1'], 'TYPE column format must be S1'

    # Handle central target exclusion
    ctarg_excluded = False
    if seg2d_exclude_ctarg:
        if len(patch_table[patch_table['type'] == 'T']) > 0:
            print('DEBUG: seg2d_exclude_ctarg is True, removing central target as other targets are available')
            wp_table = patch_table[patch_table['type'] != 'C']
            ctarg_excluded = True
        else:
            print('DEBUG: seg2d_exclude_ctarg is True, but keeping central target (no other targets available)')
            wp_table = deepcopy(patch_table)
            ctarg_excluded = False
    else:
        wp_table = deepcopy(patch_table)
        print('DEBUG: seg2d_exclude_ctarg is False. Central target will remain in patch table.')
        ctarg_excluded = False

    # CRITICAL: Only write files in PREPARATION MODE to avoid parallel access conflicts
    if preparation_mode:
        print("INFO: PREPARATION MODE - Writing patch files to disk...")

        # Save the base patch table (after central target handling)
        if Path(patch_file).suffix in ['.fit', '.fits']:
            wp_table_hdu = fits.table_to_hdu(wp_table)
            wp_table_hdu.writeto(str(patch_file), overwrite=True, checksum=True)
        else:
            write_ascii_patchfile(wp_table, Path(patch_file))
        print(f'DEBUG: Base patch table saved: {patch_file}')

        # Check if multi-class splitting is needed
        needs_mod, reason = patch_table_needs_modification(wp_table)
        print(f'DEBUG: Modification check: {reason}')

        if needs_mod:
            print(f'DEBUG: Splitting multi-class entries in patch table...')
            print(f'DEBUG: Original patch table length: {len(wp_table)}')

            # Split multi-class entries
            wp_table_modified = update_patch_table(wp_table,
                                                  class_lists=[['GALAXY','QSO'], ['STAR','WD']],
                                                  debug=False,
                                                  ctarg_excluded=ctarg_excluded,
                                                  resort=True)

            # Save modified version with _mod suffix
            mod_patch_file = Path(patch_file).with_name(Path(patch_file).stem + "_mod" + Path(patch_file).suffix)

            if mod_patch_file.suffix in ['.fit', '.fits']:
                wp_table_mod_hdu = fits.table_to_hdu(wp_table_modified)
                wp_table_mod_hdu.writeto(str(mod_patch_file), overwrite=True, checksum=True)
            else:
                write_ascii_patchfile(wp_table_modified, mod_patch_file)

            print(f'DEBUG: Modified patch table saved: {mod_patch_file}')
            print(f'DEBUG: Modified patch table length: {len(wp_table_modified)}')

            # Use modified table for any subsequent processing in this run
            wp_table = wp_table_modified
        else:
            print('DEBUG: No multi-class splitting needed')

        print("INFO: PREPARATION MODE complete - patch files written to disk")

    else:  # ANALYSIS MODE
        print("INFO: ANALYSIS MODE - Reading existing patch files, NOT writing to disk")

        # Check if modification is needed but DON'T write to disk
        needs_mod, reason = patch_table_needs_modification(wp_table)

        if needs_mod:
            print(f'DEBUG: Applying multi-class splitting in memory only: {reason}')
            wp_table = update_patch_table(wp_table,
                                         class_lists=[['GALAXY','QSO'], ['STAR','WD']],
                                         debug=False,
                                         ctarg_excluded=ctarg_excluded,
                                         resort=True)
            print(f'DEBUG: Split table has {len(wp_table)} entries (in memory only)')

        # Check for existing _mod file from preparation run
        mod_patch_file = Path(patch_file).with_name(Path(patch_file).stem + "_mod" + Path(patch_file).suffix)
        if mod_patch_file.exists():
            print(f"INFO: Found existing modified patch file: {mod_patch_file}")
            print("INFO: Both ExGal and Gal modules can safely read this file in parallel")


    if ExGal_run:
        ## check if GALAXY templates exist
        assert ExGal_templates is not None, 'ExGal templates in necessary for the ExGal module'
        assert Path(ExGal_templates).is_dir(), 'No ExGal templates directory found'

        ## check if IFU_config_dir exist
        assert IFU_config_dir is not None, 'IFU_config_dir in necessary for the ExGal module'
        assert Path(IFU_config_dir).is_dir(), 'No IFU_config_dir directory found'

        ## Checking if IFU_params file is available. Otherwise use the default one
        if (IFU_params is None) or (not Path(str(IFU_params)).exists()):
            IFU_params = gen_parms(infiles, IFU_config_dir,  wlranges)
            print('Warning: No IFU_params file found. Using default params file: %s' %(IFU_params))
        assert Path(IFU_params).exists(), 'No default IFU_params file found on the system'


        ## Now loop over all targets and select EXGAL targets
        # index_all_patch_ExGAL = index_in_class_patch(wp_table, ['GALAXY','QSO'], ids=None, ncchar=2)

        # counting number of scuessfull and failed runs for each module
        proper_ExGal_targets = len(wp_table)
        fault_counter_ExGal = 0

        for trgs in wp_table:
            ## Skip Masked sources
            if str(trgs['type']).replace(' ', '').upper() == 'M' :
                print('** NOTE: EX-GAL module: [Mask target found] skip <Mask> target id = %d' %( trgs['id']))
                proper_ExGal_targets -= 1
                continue

            ## Skip unclassified sources
            # class_value = str(index_all_patch_ExGAL[trgs['id']]['CLASS']).strip().upper()
            class_value = str(trgs['CLASS']).strip().upper()
            if (class_value not in ['GALAXY','QSO'] ) or class_value == 'NAN' or len(class_value) == 0:
                print('** NOTE2: ExGAL module: [NON-EXGAL or unassigned class found] skip target id = %d' %(trgs['id']))
                proper_ExGal_targets -= 1
                continue


            ## create the Target area
            patch_headname = headname+"_"+ ('P%04d' %(trgs['id']))


            # update 8 July 2024
            # in case the C target is excluded and only one main target is available, we double the radius of the area to
            # to make sure nothing is missed

            path_area = (trgs['A_world'] * trgs['B_world'])
            if (len(wp_table[wp_table['type'] == 'T']) == 1 and ctarg_excluded) or (path_area > 0.4 * 0.00054 and ctarg_excluded):

                enhanced_rad_factor = 3.0
                print(f"we are dealing with a single ExGal target in the field or a very large target so we multiply the radius of the area by {enhanced_rad_factor} to make sure nothing is missed")
            else:
                enhanced_rad_factor = 1.0


            patch_area = [trgs['RA_icrs'],trgs['DEC_icrs'],trgs['A_world']*3600.0 * enhanced_rad_factor,trgs['B_world']*3600.0 * enhanced_rad_factor,trgs['angle']]

            ## create the mask areas
            mask_patch_area = []
            for msk_trgs in wp_table[wp_table['type'] == 'M']:
                mask_patch = [msk_trgs['RA_icrs'],msk_trgs['DEC_icrs'],msk_trgs['A_world']*3600.0,msk_trgs['B_world']*3600.0,msk_trgs['angle']]
                mask_patch_area.append(mask_patch)

            ## reset mask_patch_area to None if empty
            if len(mask_patch_area) == 0: mask_patch_area=None


            # z_input_patch = [index_all_patch_ExGAL[trgs['id']]['Z'], index_all_patch_ExGAL[trgs['id']]['ZERR']]
            z_input_patch = [trgs['Z'], trgs['ZERR']]

            if np.isnan(z_input_patch).any():
                print('Cannot proceed with target with unknown redshift (or uncertainty in redshift). Target skipped!')
                print('We recommend to set class_patch=True')
                proper_ExGal_targets -= 1
                continue

            # print('** Note: Running ifu_ExGal module on the target id = %d Z =%f CLASS=%s' %(trgs['id'], index_all_patch_ExGAL[trgs['id']]['Z'], index_all_patch_ExGAL[trgs['id']]['CLASS'] ))
            print('** Note: Running ifu_ExGal module on the target id = %d Z =%f CLASS=%s' %(trgs['id'], trgs['Z'], trgs['CLASS'] ))
            print('******************************************************************************************************')

            try:

                ifu_ExGal(infiles, patch_headname, IFU_params, outpath, IFU_config_dir, ExGal_templates, wlranges = wlranges, aps_ids = aps_ids, targsrvy=targsrvy, targclass=targclass,\
                        mask_aps_ids=mask_aps_ids, area=patch_area, mask_areas=mask_patch_area, PPXF=PPXF, EMIPPXF=EMIPPXF, LS= LS, \
                        nthreads=mp_ExGal, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio=arms_ratio, \
                        join_arms=join_arms, z_input=z_input_patch, catdir=catdir, caldir=caldir)

                ifuExGalL2merge(infiles, outpath, patch_headname, wlranges = wlranges, outfile_suffix='_APS', EMIPPXF_LEVEL= 'BIN', LS_RES = 'ADAPTED', patch_area = patch_area, patch_id = trgs['id'], UAPSID=UAPSID)

            except ValueError as e:
                if "No valid spaxels remaining after filters!" in str(e):
                    # This is expected in some cases, just log and continue without counting as fault
                    print('** Target id = %d has no valid spaxels remaining after filters - skipping' % (trgs['id']))
                    sys.stdout.flush()
                else:
                    # Other ValueError - count as fault
                    fault_counter_ExGal += 1
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print('** Failed to run ifu_ExGal module on the target id = %d' % (trgs['id']))
                    print(exc_value)
                    sys.stdout.flush()
            except:
                fault_counter_ExGal += 1
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print('** Failed to run ifu_ExGal module on the target id = %d' % (trgs['id']))
                print(exc_value)
                sys.stdout.flush()

        # in case all tries failed for ExGal sources
        if proper_ExGal_targets > 0 and proper_ExGal_targets == fault_counter_ExGal:
            sys.exit('None of the targets passed the IFU_ExGAL module')


    if Gal_run:
        ## Now loop over Galactic targets
        # index_all_patch_GAL = index_in_class_patch(wp_table, ['STAR','WD'], ids=None, ncchar=2)
        fault_counter_Gal = 0
        proper_Gal_targets = len(wp_table)

        ## check if IFU_config_dir exist
        assert IFU_config_dir is not None, 'IFU_config_dir in necessary for the Gal module'
        assert Path(IFU_config_dir).is_dir(), 'No IFU_config_dir directory found'

        ## Checking if IFU_params file is available. Otherwise use the default one
        if (IFU_params is None) or (not Path(str(IFU_params)).exists()):
            IFU_params = gen_parms(infiles, IFU_config_dir,  wlranges)
            print('Warning: No IFU_params file found. Using default params file: %s' %(IFU_params))
        assert Path(IFU_params).exists(), 'No default IFU_params file found on the system'



        ## For Galactic sources we shrink the the radii of to half)
        enhanced_rad_factor = 0.5
        for trgs in wp_table:

            ## Skip Masked sources
            if str(trgs['type']).replace(' ', '').upper() == 'M' :
                print('** NOTE1: GAL module: [Masked target] skip target id = %d' %( trgs['id']))
                proper_Gal_targets -=1
                continue

            # skip central target in case of classified as 'Galactic source' and at least one other target is detected (it means we deal with seg2d results not just the central field)
            if (str(trgs['type']).replace(' ', '').upper() == 'C') and ('T' in list(wp_table['type'])):
                print('** NOTE1: GAL module: [Central target] skip target id = %d' %( trgs['id']))
                proper_Gal_targets -=1
                continue


            ## Skip ExGalactic sources
            # class_value = str(index_all_patch_GAL[trgs['id']]['CLASS']).strip().upper()
            class_value = str(trgs['CLASS']).strip().upper()
            # also make sure the class value is a simple string and not a nonunicode one
            class_value = fix_non_unicode_string(class_value)



            if (class_value in ['GALAXY','QSO'] ) or class_value == 'NAN' or len(class_value) == 0:
                print('** NOTE2: GAL module: [NON-GAL or unassigned class found] skip target id = %d' %(trgs['id']))
                proper_Gal_targets -=1
                continue

            patch_headname = headname+"_"+ ('P%04d' %(trgs['id']))
            patch_area = [trgs['RA_icrs'],trgs['DEC_icrs'],trgs['A_world']*3600.0 * enhanced_rad_factor ,trgs['B_world']*3600.0 * enhanced_rad_factor ,trgs['angle']]

            # z_input_patch = [index_all_patch_GAL[trgs['id']]['Z'], index_all_patch_GAL[trgs['id']]['ZERR']]
            z_input_patch = [trgs['Z'], trgs['ZERR']]
            if np.isnan(z_input_patch).any():
                print('Cannot proceed with target with unknown redshift (or uncertainty in redshift). Target skipped!')
                print('We recommend to set class_patch=True')
                proper_Gal_targets -=1
                continue
            mask_patch_area=None

            print('** Note: Running ifu_Gal module on the target id = %d Z =%f CLASS=%s' %( trgs['id'], trgs['Z'], trgs['CLASS'] ))
            print('******************************************************************************************************')


            ## Note: for Gal sources, we deliberately set the join_arms to False
            try:
                ifu_Gal(infiles, patch_headname, IFU_params, outpath, wlranges = wlranges, aps_ids = aps_ids, targsrvy=targsrvy,  targclass=targclass, mask_aps_ids=mask_aps_ids, area=patch_area, \
                    mask_areas=mask_patch_area, nthreads=mp_Gal, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, \
                    arms_ratio=arms_ratio, join_arms=False, z_input=z_input_patch, overwrite=overwrite, \
                    catdir=catdir, caldir=caldir, IFU_config_dir=IFU_config_dir)

                ifuGalL2merge(infiles, outpath, patch_headname, wlranges = wlranges, outfile_suffix='_APS', patch_area = patch_area, patch_id = trgs['id'], UAPSID=UAPSID)

            except:
                fault_counter_Gal +=1
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print('** Failed to run ifu_Gal module on the target id = %d Z =%f CLASS=%s' %( trgs['id'], trgs['Z'], trgs['CLASS'] ))
                print(exc_value)
                sys.stdout.flush()

        # in case all tries failed for Gal sources
        if proper_Gal_targets > 0 and proper_Gal_targets == fault_counter_Gal:
            sys.exit('None of the targets passed the IFU_GAL module')

    return

##################################################################################################################################################


def ifu_runner(options=None):

    # Shared registry -- see aps_common_args.py's own module docstring.
    # No "spatial_selection" group -- this script selects by patch, not
    # by --area/--mask_areas.
    parser = build_common_parser(
        description="RUN aps_worker [IFU Mode].",
        groups=["target_selection", "wavelength", "l1_processing", "caldirs", "output"],
        overrides={
            "headname": {"help": "Output headname. The output filenames will be generated based on this headname"},
            "vacuum": {"default": False},
            "join_arms": {"default": True},
            "overwrite": {"help": "If enabled the code will overwrite the existing products, otherwise it will skip them"},
            "configdir": {"help": "Directory contains APSExGal configuration files"},
        },
        extra_args=[
            (("--class_templates",), dict(type=none_or_str, default=None, required=False,
                                           help="Parent directory contains REDDROCK(CLASSIFIER) PCA templates")),
            (("--class_templates_ARC",), dict(type=none_or_str, default=None, required=False,
                                               help="Parent directory contains REDDROCK(CLASSIFIER) ARC templates")),
            (("--IFU_config_dir",), dict(type=none_or_str, default=None, required=False,
                                          help="Directory contains APSExGal configuration files")),
            (("--ExGal_templates",), dict(type=none_or_str, default=None, required=False,
                                           help="Parent directory contains working templates for ExGal")),
            (("--IFU_params",), dict(type=none_or_str, default=None, required=False,
                                      help="parameter (.json) file (full path)")),
            (("--ExGal_run",), dict(type=str2bool, default=True, required=False,
                                     help="If True, run ifu_ExGal for all ex-galactic sources")),
            (("--Gal_run",), dict(type=str2bool, default=True, required=False,
                                   help="If True, run ifu_Gal for all galactic sources")),
            (("--PPXF",), dict(type=str2bool, default=False, required=False, help="Run PPXF?")),
            (("--EMIPPXF",), dict(type=str2bool, default=False, required=False, help="RUN EMIPPXF?")),
            (("--LS",), dict(type=str2bool, default=False, required=False, help="RUN LS?")),
            (("--class_ntop",), dict(type=int, default=1, required=False,
                                      help="number of guesses return by classifier module")),
            (("--mp_prep",), dict(type=int, default=1, required=False,
                                   help="The number of threads to run the IFU preparatory steps")),
            (("--mp_ExGal",), dict(type=int, default=1, required=False,
                                    help="The number of threads to run the IFU ExGal modules")),
            (("--mp_Gal",), dict(type=int, default=1, required=False,
                                  help="The number of threads to run the IFU Galactic modules")),
            (("--fig",), dict(type=str2bool, default=False, required=False,
                               help="if True, the code also produces plots of the best fitted model")),
            (("--patch_file",), dict(type=none_or_str, default=None, required=False,
                                      help="full path to the patch file contain desired extraction info (ellipse and class params)")),
            (("--z_input",), dict(type=none_or_str, default=None, required=False,
                                   help="initial estimation for redshift of this patch")),
            (("--seg2d",), dict(type=str2bool, default=False, required=False,
                                 help="Run auto 2D segmentation code?")),
            (("--seg2d_white_images",), dict(type=none_or_str, default=None, required=False,
                                              help="comma-separated list of input white_images to be deployed for 2D segmentation")),
            (("--seg2d_white_src",), dict(type=none_or_str, default='weave', required=False,
                                           help="Which white image type to be used [options: weave, PANN, PS1, etc]")),
            (("--seg2d_search_gaia",), dict(type=str2bool, default=True, required=False,
                                             help="if True, it search Gaia catalogue for all sources within the field")),
            (("--seg2d_extract",), dict(type=str2bool, default=True, required=False,
                                         help="if True, extract all sources using source extractor. Otherwise, "
                                              "it only returns the central source (type=C)")),
            (("--seg2d_exclude_ctarg",), dict(type=str2bool, default=True, required=False,
                                               help="if True, if a target (type=T) found in segmentation, it "
                                                    "excludes the Central target (whole field) (type=C)")),
            (("--seg2d_mask_stars",), dict(type=str2bool, default=True, required=False,
                                            help="if True, mask all stellar sources find in the Gaia catalogue (type=M)")),
            (("--seg2d_ext_thresh",), dict(type=float, default=2.5, required=False,
                                            help="source extraction threshold")),
            (("--seg2d_minarea",), dict(type=float, default=400.0, required=False,
                                         help="min area for each source to be considered as a target (type = T)")),
            (("--seg2d_deblend_nthresh",), dict(type=float, default=4, required=False,
                                                 help="source extraction de-blending threshold")),
            (("--seg2d_radii_factor",), dict(type=float, default=7.0, required=False,
                                              help="factor to set the ellipse radii, with respect to the FWHM "
                                                   "of the detected source (default = 7 ~ 3 sigma)")),
            (("--class_patch",), dict(type=str2bool, default=False, required=False,
                                       help="if True, run classifier to measure redshift and class type for all targets (C, T, M)")),
            (("--class_z_rad",), dict(type=float, default=1.5, required=False,
                                       help="radius of the circular aperture to collapse all spectra within to run the classifier)")),
            (("--user_patch",), dict(type=str2bool, default=False, required=False,
                                      help="if True, generates the patch_file from all L1 provinces.")),
            (("--uapsid",), dict(type=none_or_str, default=None, required=False,
                                  help="unique id to be used for the jobnames in APS database/repository ")),
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

    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)

    seg2d_white_images = None
    if args.seg2d_white_images is not None:
        args.seg2d_white_images = [str(x) for x in args.seg2d_white_images.split(",")]

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ', '')


    if (len(args.infiles) > 1) and (args.join_arms is False):
        sys.exit('No way to proceed with more than one input file without joining arms')


    if (args.PPXF is False) and ((args.EMIPPXF is True) or (args.LS is True)):
        print('Warning: Running EMIPPXF or LS without running PPXF in the same run')
        print('may cause inconsistency among results and order of results in the outputs')
        print('Make sure you are using the same aps_ids list as you used for PPXF, earlier')


    # print args and assigned/default values on the screen
    print_args(args,module='aps_ifu', version= aps_constants.__aps_ifu_version__ , path=outpath, headname=args.headname)



    ifu_worker(args.infiles, args.headname, outpath, IFU_config_dir=args.IFU_config_dir, ExGal_templates=args.ExGal_templates, \
     class_templates=args.class_templates, class_templates_ARC=args.class_templates_ARC, patch_file = args.patch_file,IFU_params = args.IFU_params, user_patch=args.user_patch, \
     seg2d=args.seg2d, class_patch=args.class_patch, wlranges = args.wlranges, aps_ids = aps_ids, targsrvy=targsrvy, \
     targclass=targclass, mask_aps_ids=mask_aps_ids, PPXF=args.PPXF, EMIPPXF=args.EMIPPXF, \
     LS= args.LS, mp_prep=args.mp_prep, mp_ExGal=args.mp_ExGal, \
     mp_Gal=args.mp_Gal, sens_corr=args.sens_corr, mask_gaps=args.mask_gaps, safe_mask_gaps=args.safe_mask_gaps, vacuum=args.vacuum, \
     tellurics=args.tellurics, fill_gap=args.fill_gap, arms_ratio=args.arms_ratio, join_arms=args.join_arms, overwrite=args.overwrite, \
     seg2d_white_images = args.seg2d_white_images, seg2d_white_src = args.seg2d_white_src, seg2d_search_gaia = args.seg2d_search_gaia, \
     seg2d_extract=args.seg2d_extract, seg2d_exclude_ctarg=args.seg2d_exclude_ctarg, seg2d_mask_stars = args.seg2d_mask_stars, seg2d_ext_thresh=args.seg2d_ext_thresh, class_ntop =args.class_ntop, \
     seg2d_minarea=args.seg2d_minarea, seg2d_deblend_nthresh=args.seg2d_deblend_nthresh, seg2d_radii_factor=args.seg2d_radii_factor, \
     class_z_rad=args.class_z_rad, Gal_run=args.Gal_run, ExGal_run=args.ExGal_run, UAPSID=args.uapsid, catdir=args.catdir, caldir=args.caldir, config_dir=args.configdir)


############################################################################

if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).





    debug_LIFU= [

    '--infiles', "<PYAPS_DATA>/L1/<night>/stackcube_<runid>.fit","<PYAPS_DATA>/L1/<night>/stackcube_<runid>.fit",
    '--headname', 'LWVE_<target>_01_GR_H1',
    '--outpath','<PYAPS_DATA>/L2/<night>/<obid>/',
    '--wlranges', 'None',
    '--arms_ratio', '1.0,1.0',
    '--IFU_config_dir','<PYAPS_DIR>/configs/ExGal_configs/',
    '--ExGal_templates', '<PYAPS_DIR>/PyAPS_templates/templates_ExGal/',
    '--IFU_params', '<PYAPS_DIR>/configs/ExGal_configs/LIFUHR11_marc_test.json',
    '--PPXF','True',
    '--EMIPPXF','True',
    '--LS','True',
    '--sens_corr','True',
    '--mp_prep', '12',
    '--mp_Gal', '1',
    '--mp_ExGal', '6',
    '--safe_mask_gaps','True',
    '--mask_gaps','True',
    '--tellurics','False',
    '--vacuum','False',
    '--fill_gap','False',
    '--join_arms', 'True',
    '--fig', 'True',
    '--patch_file', '<PYAPS_DATA>/L2/<night>/<obid>/LWVE_<target>_01_GR_H1_targets.fits',
    '--aps_ids', 'None',
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--overwrite', 'True',
    '--uapsid' , 'None',
    '--seg2d', 'True',
    '--class_ntop', '3',
    '--seg2d_white_images', 'None',
    '--seg2d_white_src', 'weave',
    '--seg2d_search_gaia', 'False',
    '--seg2d_extract','True',
    '--seg2d_mask_stars', 'False',
    '--seg2d_ext_thresh', '2.5',
    '--seg2d_minarea', '400',
    '--seg2d_deblend_nthresh', '4',
    '--seg2d_radii_factor', '7.0',
    '--class_patch' , 'True',
    '--class_templates','<PYAPS_DIR>/PyAPS_templates/templates_RR/',
    '--class_templates_ARC','<PYAPS_DIR>/PyAPS_templates/templates_ARC_RR/',
    '--class_z_rad', '1.5',
    '--user_patch' , 'False',
    '--ExGal_run', 'True',
    '--Gal_run', 'False',
    '--seg2d_exclude_ctarg', 'True',
    '--caldir', '<PYAPS_DATA>/CAL',
    '--catdir', '<PYAPS_DATA>/CAT',
    '--configdir', '<PYAPS_DIR>/configs/ExGal_configs'
    ]



    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    ifu_runner(options=debug_LIFU)
