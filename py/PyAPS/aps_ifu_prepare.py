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

import astropy.units as u
import pandas as pd
import sep
from astropy.coordinates import FK5, ICRS, SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table, hstack, vstack
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS, utils
from astropy_healpix import HEALPix

# for photutils > 1.5
from photutils.aperture import CircularAperture, aperture_photometry
from reproject import reproject_interp
from scipy.interpolate import interp1d

import PyAPS
from PyAPS import aps_constants
from PyAPS.aps_L2merge import ifuExGalL2merge, ifuGalL2merge
from PyAPS.aps_rr import rrweave_worker
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
from PyAPS import IFUExGalPrepare as IFUExGalPrepare
from PyAPS import MOSExGalPrepare as MOSExGalPrepare
from PyAPS import ExGalPrepare as ExGalPrepare
from PyAPS import ExGalutil
from PyAPS.apsPlot.region_plot import plot_region
from PyAPS.apsPlot.source_detection import build_figure as _build_source_detection_figure
from PyAPS.apsPlot.source_detection_3d import build_figure as _build_source_detection_3d_figure
from PyAPS import aps_ifu_seg3d

# for photutils <=1.5
# from photutils import CircularAperture, aperture_photometry



APSVERS = PyAPS.__version__

# PHYSICAL CONSTANTS and global parameters
Clight = 299792.458  # km/s
## set default value for error of badpixels/badspaxels
large_error = aps_constants.large_error

#################################################
"""
aps_ifu_prepare.py
Python code for preparing APS IFU data





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
 5.0 By A. Molaeinezhad (CASU, June 2026)
NOTES:


TODO list:

Example:

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
12 June 2026: the IFU code split into Gal, ExGal, prepare and utils parts. It makes things easier to debug and also to imporve
12 June 2026: Several new parameters for adaptive set of params added to the iFU pipeline
"""

##################################################################################################################################################
"""

Usage for interactive debugging
--------------------------------
  # Step 1: run preparation only
  prep = ifu_worker_prepare(
      infiles=[...], headname='WA', outpath='/tmp/out/',
      patch_file=None,          # triggers seg2d + classification
      seg2d=True, class_patch=True,
      class_templates='/templates/RR/', ...
  )

  # Step 2: inspect / tweak
  print(prep['wp_table'])
  prep['wp_table']['Z'][0] = 0.05   # manual override example

"""



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
    df = pd.read_csv(patch_file, sep='\s+', comment='#', header=None,names=column_names)

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

    new_patch_list = []
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

                new_patch_list.append(new_row)
    updated_patch_table = Table(new_patch_list)

    # also making sure the biggest item is always the highest id
    if resort:
        _CLASS_PRIORITY = {"GALAXY": 0, "QSO": 1, "STAR": 2, "WD": 3}

        # Compute area
        area = np.array(
            updated_patch_table["A_world"] * updated_patch_table["B_world"],
            dtype=float,
        )

        # Compute class priority per row (first valid class slot)
        class_priority = []
        for row in updated_patch_table:
            cl = np.atleast_1d(row["CLASS"])
            c  = ""
            for v in cl:
                s = str(v).strip()
                if s.startswith("b'") and s.endswith("'"):
                    s = s[2:-1]
                s = s.upper()
                if s and s not in ("NAN", "NONE"):
                    c = s
                    break
            class_priority.append(_CLASS_PRIORITY.get(c, 4))

        class_priority = np.array(class_priority, dtype=int)

        # Primary: area DESC (-area ASC), secondary: class priority ASC
        # np.lexsort sorts by last key first
        sort_idx = np.lexsort((class_priority, -area))
        updated_patch_table = updated_patch_table[sort_idx]

        # Reassign ids in the new sorted order
        sorted_ids = sorted(updated_patch_table["id"])
        updated_patch_table["id"] = sorted_ids


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
# plot_region now lives in PyAPS.apsPlot.region_plot, built on the shared
# WCS-aware image primitives (PyAPS.apsPlot.wcs_image) used across the
# PyAPS.apsPlot platform. Imported at module top as
# `from PyAPS.apsPlot.region_plot import plot_region`.

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

        ## ext_thresh/minarea = None -> estimate them from the white-light
        ## image itself instead of trusting a fixed guess. Motivated by a
        ## real-data finding: the previous fixed default minarea=300 found
        ## only 3 objects on a field where the data-driven value (measured
        ## PSF-disk area) found 51 on the identical image. See
        ## aps_ifu_seg3d.estimate_seg2d_params for the method (PSF FWHM
        ## measured from the image's own compact sources -> minarea; then
        ## ext_thresh calibrated via the same sign-flipped-image empirical
        ## purity check seg3d uses for its own detections).
        ## deblend_nthresh = None also falls back cleanly here, but to a
        ## FIXED default (not a data-driven estimate) -- deliberately not
        ## auto-tuned, per explicit request.
        if extract and (self.ext_thresh is None or self.minarea is None):
            print('INFO: seg2d ext_thresh and/or minarea requested as auto (None) -- '
                  'estimating from the white-light image ...')
            white_data = fits.open(self.whiteimage)[0].data.astype(np.float64)
            est = aps_ifu_seg3d.estimate_seg2d_params(white_data)
            if self.ext_thresh is None:
                self.ext_thresh = est['ext_thresh']
                print(f"  auto ext_thresh = {self.ext_thresh} "
                      f"(purity-calibrated, see estimate_seg2d_params)")
            if self.minarea is None:
                self.minarea = est['minarea']
                print(f"  auto minarea = {self.minarea} (from measured PSF "
                      f"FWHM={est['fwhm_pix']:.2f}px, {est['fwhm_source']}, "
                      f"{est['n_psf_sources_used']} sources used)")
        if self.deblend_nthresh is None:
            self.deblend_nthresh = 4
            print('INFO: seg2d_deblend_nthresh requested as auto (None) -- '
                  'using fixed default = 4 (not data-driven by design)')

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

            # make_figure/write_image now live in PyAPS.apsPlot.source_detection,
            # built on the shared WCS-aware image primitives
            # (PyAPS.apsPlot.wcs_image) used across the PyAPS.apsPlot platform.
            try:
                coords_gaia_pix = None
                mask_dim_pix = None

                if self.gaia_stars is not None:
                    if extc_wcs.wcs.radesys.lower().replace(" ", "") == 'fk5':
                        coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())
                        coords_gaia_wcs = coords_gaia_wcs.transform_to(FK5(equinox='J2000'))
                    elif extc_wcs.wcs.radesys.lower().replace(" ", "") == 'icrs':
                        coords_gaia_wcs = SkyCoord(ra=self.gaia_stars['ra'], dec= self.gaia_stars['dec'], unit='deg', frame=ICRS())
                    else:
                         sys.exit('This coordinate system: %s is not supported by PyAPS' %(extc_wcs.wcs.radesys.lower().replace(" ", "")))

                    coords_gaia_pix = extc_wcs.world_to_pixel(coords_gaia_wcs)

                    # computing the mask width and height in pix (pxsc is pixle scale)
                    pxsc = extc_wcs.proj_plane_pixel_scales()[0].value
                    mask_dim_pix = (self.mask_rad_arcsec*2.0/3600.0)/pxsc

                source_fig = _build_source_detection_figure(
                    extc_wcs, extc_bkg_image, extc_bkg_rms, extc_data_sub, extc_mask,
                    objects, segmap, gaia_pix=coords_gaia_pix, mask_dim_pix=mask_dim_pix,
                    radii_factor=self.radii_factor, headname=self.headname,
                )
                objects_figname = Path(self.figdir).joinpath(self.headname+'_source_detection.pdf')
                source_fig.write_image(str(objects_figname))
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


def ifu_make_central_patch(infiles, headname, outpath, class_ntop=1):
    """
    Create a minimal single-target patch file from the IFU field centre.

    Used when seg2d=False but class_patch=True: no segmentation is run, so
    there is no initial target list.  This function reads the WCS of the
    collapsed white-light extension (ext 6) from the first input file,
    computes the field centre and angular extent, and writes a one-row patch
    table (type='C') to disk.

    Returns the Path of the written FITS file.
    """
    _COLLAPSE_EXT = 6

    hdu   = fits.open(infiles[0])[_COLLAPSE_EXT]
    wcs   = WCS(hdu.header)

    assert wcs.wcs.radesys.lower().strip() == 'icrs', \
        "Expected ICRS coordinate system in collapsed extension WCS"

    footprint = wcs.calc_footprint()                          # (4, 2) corners
    sky       = SkyCoord(footprint, frame=ICRS(), unit="deg")
    ra_vals   = sky.ra.value
    dec_vals  = sky.dec.value

    ra_cent  = (ra_vals.min()  + ra_vals.max())  / 2.0
    dec_cent = (dec_vals.min() + dec_vals.max()) / 2.0
    ra_width  = (ra_vals.max()  - ra_vals.min())  % 360
    dec_width = dec_vals.max() - dec_vals.min()

    patch_table = Table()
    patch_table['id']       = [0]
    patch_table['RA_icrs']  = [ra_cent]  * u.degree
    patch_table['DEC_icrs'] = [dec_cent] * u.degree
    patch_table['A_world']  = [ra_width]  * u.degree
    patch_table['B_world']  = [dec_width] * u.degree
    patch_table['angle']    = [0.0]      * u.degree
    patch_table['flag']     = [0.0]
    patch_table['type']     = ['C']

    patch_table_class = Table(
        [
            [[np.nan] * class_ntop],
            [[np.nan] * class_ntop],
            [[np.nan] * class_ntop],
            [['']     * class_ntop],
        ],
        names=('Z', 'ZERR', 'ZWARN', 'CLASS'),
        dtype=('float', 'float', 'float', 'U18'),
    )
    patch_table = hstack([patch_table, patch_table_class])

    targets_fits = Path(outpath) / (headname + '_targets.fits')
    fits.table_to_hdu(patch_table).writeto(
        str(targets_fits), overwrite=True, checksum=True
    )

    print(f"INFO: Central-field patch created: {targets_fits}")
    print(f"      RA={ra_cent:.5f} deg  Dec={dec_cent:.5f} deg  "
          f"A={ra_width*3600:.1f}\"  B={dec_width*3600:.1f}\"")

    return targets_fits


def ifu_class(infiles, headname, patch_file, class_templates, class_templates_ARC=None,\
    aps_ids=None, mask_aps_ids=None ,z_rad=1.5, wlranges= None,figdir= None, ncpus=2, arms_ratio=None,join_arms=False, class_ntop=1, catdir=None, caldir=None, IFU_config_dir=None,targclass=None,targsrvy=None,
    spaxel_weighted_lsf=True ):
    """
    spaxel_weighted_lsf : bool, optional
        Per-spaxel LSF for the collapsed classification/redshift spectrum
        this function builds -- **default `True`** as of v1.9 (all IFU
        pipeline code defaults to spaxel-weighted LSF now; pass `False`
        explicitly to fall back to the old flat-global-curve behaviour).
        Threaded down through `rrweave_worker`/`read_spectra` into the
        collapsing `APSOB`'s own `spaxel_weighted_lsf` -- see
        `aps_utils.py`'s collapse block for how the collapsed spectrum's
        own 'fwhm' becomes a real flux-weighted average of its
        contributing spaxels' per-spaxel LSF instead of the flat global
        curve.
    """




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
    if patch_file is None:
        raise ValueError(
            "ifu_class requires a patch file with target positions (patch_file=None). "
            "This should have been created automatically — check that class_patch=True "
            "and that ifu_make_central_patch() ran successfully."
        )
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


        scandata, zbest, zspec, zfitall = rrweave_worker(infiles, class_templates, aps_ids=aps_ids, mask_aps_ids=mask_aps_ids, ntop=class_ntop,
            area=area, mask_areas=None, wlranges=wlranges, sens_corr=True, mask_gaps=True, safe_mask_gaps=True , vacuum=True, tellurics=False, fill_gap=False,
            arms_ratio=arms_ratio, join_arms=join_arms, figdir=figdir, return_outputs=True, collapse=True, collapse_fname = collapse_fname, ncpus=ncpus ,nminima=3,
            gpu=False, max_gpuprocs = None, catdir=catdir, caldir=caldir, configdir=IFU_config_dir, archetypes=class_templates_ARC,targclass=targclass,targsrvy=targsrvy,
            spaxel_weighted_lsf=spaxel_weighted_lsf)

        if zbest is None:
            print('WARNING: patch %d (RA=%.6f DEC=%.6f) returned no spectra — '
                  'likely outside the WEAVE footprint. Skipping.' % (trgs['id'], trgs['RA_icrs'], trgs['DEC_icrs']))
            continue

        trgs['Z'] = np.array([[targz_i] for targz_i in zbest['Z']]) if zbest['Z'].ndim > 1 else np.array(zbest['Z'])
        trgs['ZERR'] = np.array([[targz_i] for targz_i in zbest['ZERR']]) if zbest['ZERR'].ndim > 1 else np.array(zbest['ZERR'])
        trgs['ZWARN'] = np.array([[targz_i] for targz_i in zbest['ZWARN']]) if zbest['ZWARN'].ndim > 1 else np.array(zbest['ZWARN'])
        # for CLASS we do extra check to make sure it is not bytes type
        trgs_class_list_dummy = np.array([[targz_i] for targz_i in zbest['CLASS']]) if zbest['CLASS'].ndim > 1 else np.array(zbest['CLASS'])
        trgs_class_list = [clslist_i.decode() if isinstance(clslist_i, bytes) else clslist_i for clslist_i in trgs_class_list_dummy]
        trgs['CLASS'] = np.array(trgs_class_list)

        print('Patch info updated')

    ## if the input patch_file is fits or ASCII, we update that as well (now with the updated class params).
    if Path(patch_file).suffix == '.fits':
        ## Update Patch file (fits format)
        patch_table_hdu = fits.table_to_hdu(patch_table)
        patch_table_hdu.writeto(patch_file, overwrite=True, checksum=True)
    else:
        # write the updated table in a format similar to the input one
        write_ascii_patchfile(patch_table, Path(patch_file))
    return patch_table



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


# =========================================================================== #
#  STAGE 1 — preparation                                                      #
# =========================================================================== #

def ifu_worker_prepare(
    infiles,
    headname,
    outpath,
    # --- patch / seg2d / classification ---
    patch_file=None,
    user_patch=False,
    seg2d=False,
    seg2d_white_images=None,
    seg2d_white_src="weave",
    seg2d_search_gaia=True,
    seg2d_extract=True,
    seg2d_exclude_ctarg=True,
    seg2d_mask_stars=True,
    seg2d_ext_thresh=2.5,
    seg2d_minarea=400,
    seg2d_deblend_nthresh=4,
    seg2d_radii_factor=8.0,
    # --- seg3d: optional 3D matched-filter emission-line detection ---
    # (see aps_ifu_seg3d.py module docstring). Opt-in, additive: never
    # touches patch_file/patch_table/classification. Runs independently of
    # seg2d (default seg2d output is unchanged either way); when both run
    # in the same call, the diagnostic figure compares them.
    #
    # sens_corr/mask_gaps/safe_mask_gaps/tellurics/vacuum: these are the
    # SAME --sens_corr/--mask_gaps/--safe_mask_gaps/--tellurics/--vacuum
    # flags aps_ifu_prepare.py's CLI already accepts (shared l1_processing
    # arg group) -- previously parsed but never actually consumed by this
    # function. seg3d's cube loading (aps_ifu_seg3d.load_wavelength_range,
    # built on aps_utils.APSOB) is the first consumer; wired here as
    # regular parameters rather than seg3d_-prefixed duplicates, so a
    # single existing flag controls L1 processing consistently for
    # whichever stage ends up using it.
    sens_corr=True,
    mask_gaps=True,
    safe_mask_gaps=True,
    tellurics=False,
    vacuum=False,
    seg3d=False,
    # seg3d_lmin/lmax: LOWRES (LIFULR/MIFULR) blue+red combined native
    # coverage, trimmed ~100A per edge -- see aps_ifu_seg3d.run_seg3d's
    # docstring. HIGHRES surveys override via IFU_params JSON (real gap
    # between HR arms, no overlap with this LOWRES-derived window).
    seg3d_lmin=3700.0,
    seg3d_lmax=9390.0,
    seg3d_threshold=6.0,
    seg3d_spatial_bin=4,
    seg3d_min_npix=3,
    seg3d_sky_mask_min_A=None,
    seg3d_continuum_method="pca",
    seg3d_ivar_calibration="per-wave",
    seg3d_fig=True,
    # seg3d_merge: auto-merge 'obvious' (SNR >= seg3d_merge_min_snr) 3D-only
    # candidates into the real patch table as auxiliary targets. Requires
    # seg2d=True (there must be a patch table to merge into) -- a no-op
    # with a warning otherwise. Default False: 2D SExtractor alone remains
    # the pipeline default; seg3d and seg3d_merge are both opt-in.
    seg3d_merge=False,
    seg3d_merge_min_snr=10.0,
    seg3d_merge_min_purity=0.9,
    seg3d_merge_aperture_arcsec=2.0,
    # seg3d_merge_group_*: an ADDITIONAL, independent path into seg3d_merge
    # alongside the SNR/purity path above -- a candidate with >=
    # seg3d_merge_group_min_lines OTHER no_continuum_counterpart candidates
    # at the exact same detection-grid pixel (different wavelengths, i.e.
    # plausibly several lines of the same source) is merged if the group's
    # COMBINED significance (quadrature sum over ALL its members' S/N, see
    # aps_ifu_seg3d.combined_group_snr -- not just its strongest member) is
    # >= the effective group threshold. That threshold is field-calibrated
    # from the group-level sign-flip purity self-check (aps_ifu_seg3d.
    # post_veto_group_purity_scan / resolve_group_min_snr, run_seg3d's
    # returned group_purity_scan), the same self-calibration relationship
    # the single-line path above already has with seg3d_merge_min_snr/
    # seg3d_merge_min_purity -- seg3d_merge_group_min_snr is only the floor
    # that calibration cannot go below, and seg3d_merge_group_min_n_trust
    # is deliberately much lower than the single-line path's implicit
    # min_n_trust=15 (multi-line groups are intrinsically rarer). Does NOT
    # verify the lines are at a physically consistent redshift -- that is
    # left to the Redrock classification pass every merged target (single-
    # or multi-line) already goes through right after this step. The
    # exact-pixel grouping default was validated against the sign-flipped
    # (noise) cube before being set -- see aps_ifu_seg3d.merge_into_
    # patch_table's docstring and doc/aps_ifu_prepare.md before loosening
    # it. Set seg3d_merge_group_min_lines=None to disable this path.
    seg3d_merge_group_min_lines=2,
    seg3d_merge_group_min_snr=10.0,
    seg3d_merge_group_min_n_trust=3,
    seg3d_group_radius_px=0.0,
    # seg3d_merge_extreme_*: a THIRD, independent path into seg3d_merge for
    # a candidate that clears neither path above -- typically a genuinely
    # extreme-S/N SINGLETON (no corroborating line, so the group path can't
    # help) whose S/N is so high that the single-line purity self-check
    # never gets measured with enough samples that far out to certify it
    # either. Compares the candidate's S/N directly against the single
    # loudest fluctuation the FULL sign-flipped search ever produced
    # (aps_ifu_seg3d.loudest_event_snr / run_seg3d's returned
    # loudest_neg_event -- every voxel, no veto applied), requiring it to
    # clear that loudest event by seg3d_merge_extreme_safety_margin (10%
    # margin by default) -- see aps_ifu_seg3d.resolve_extreme_snr_min /
    # merge_into_patch_table's docstring path (c). Opt-in and independent
    # of the other two paths in both directions. Set
    # seg3d_merge_extreme_safety_margin=None to disable this path.
    seg3d_merge_extreme_safety_margin=1.1,
    class_patch=False,
    class_templates=None,
    class_templates_ARC=None,
    class_ntop=3,
    class_z_rad=1.5,
    # --- spectral flags (needed for classification) ---
    wlranges=None,
    aps_ids=None,
    mask_aps_ids=None,
    arms_ratio=None,
    join_arms=False,
    IFU_config_dir=None,
    # --- multiprocessing ---
    mp_prep=1,
    # --- calibration dirs ---
    catdir=None,
    caldir=None,
    IFU_params=None,
    UAPSID=None,
    targclass=None,
    targsrvy=None,
) -> dict:
    """
    Preparation stage for the IFU worker pipeline.

    Handles:
      - Optional 2D segmentation (``seg2d=True``)
      - Optional classification / redshift determination (``class_patch=True``)
      - Loading an existing patch file (when ``patch_file`` is provided)
      - Central-target exclusion
      - Multi-class splitting (in-memory when in analysis mode,
        written to disk when in preparation mode)

    Parameters
    ----------
    infiles : list of str
        Input L1 FITS files.
    headname : str
        Output file prefix.
    outpath : str
        Output directory.
    patch_file : str or None
        Path to an existing patch file.  When ``None`` the function runs in
        *preparation mode* (seg2d + classification) and writes output files.
        When provided it runs in *analysis mode* and does **not** write to
        disk.
    user_patch : bool
        Not yet implemented — reserved for future use.
    seg2d : bool
        Run 2D source extraction.
    seg2d_white_images : list of str or None
        Pre-computed white-light images to use instead of downloading PS1.
    seg2d_white_src : str
        Source for white-light image (``'weave'``, ``'PS1'``, etc.).
    seg2d_search_gaia : bool
        Query Gaia for stellar sources to mask.
    seg2d_extract : bool
        Run source extractor.  If ``False`` only the central target is kept.
    seg2d_exclude_ctarg : bool
        Exclude the central (whole-field) target when at least one
        extracted source is available.
    seg2d_mask_stars : bool
        Add Gaia stars as mask entries in the patch table.
    seg2d_ext_thresh : float
        Source-extraction detection threshold (sigma).
    seg2d_minarea : float
        Minimum source area in pixels.
    seg2d_deblend_nthresh : float
        Deblending threshold for source extraction.
    seg2d_radii_factor : float
        Ellipse radii scale factor (~3 sigma = 7).
    class_patch : bool
        Run the Redrock classifier to assign redshifts and classes.
    class_templates : str or None
        Directory containing Redrock PCA templates.
    class_templates_ARC : str or None
        Directory containing Redrock archetype templates.
    class_ntop : int
        Number of classification guesses to store per target.
    class_z_rad : float
        Aperture radius in arcsec for collapsing spectra before classification.
    wlranges : list of [float, float] or None
        Per-arm wavelength ranges (forwarded to the classifier).
    aps_ids : list of int or None
        APS IDs to include.
    mask_aps_ids : list of int or None
        APS IDs to exclude.
    arms_ratio : list of float or None
        Per-arm flux scale factors.
    join_arms : bool
        Whether to join arms before classification.
    IFU_config_dir : str or None
        Directory containing APSExGal configuration files.
    mp_prep : int
        Number of threads for the classification step.
    catdir, caldir : str or None
        Catalogue / calibration directories.

    Returns
    -------
    dict with keys:

    ``wp_table``
        Final processed patch table (Astropy Table, in memory).
    ``patch_file``
        Path to the base patch file on disk (str).
    ``ctarg_excluded``
        Whether the central target was removed (bool).

    Returns ``None`` on fatal errors (missing patch table, etc.).
    """

    # ------------------------------------------------------------------
    # 0. Basic guards
    # ------------------------------------------------------------------
    figdir = Path(outpath) / "figs"
    if not figdir.exists():
        figdir.mkdir(parents=True, exist_ok=True)
        print("FIGDIR: %s Created!" % figdir)

    if user_patch:
        sys.exit("The user_patch module has not been implemented yet")


    if IFU_params is not None and Path(str(IFU_params)).exists():
        configs = json.load(open(IFU_params))
        configs["CONFIG_FILE"] = os.path.basename(IFU_params)
    else:
        IFU_params = gen_parms(infiles, IFU_config_dir, wlranges)
        if IFU_params is None or not Path(str(IFU_params)).exists():
            sys.exit(f"ERROR: No IFU_params provided and auto-derivation from "
                    f"IFU_config_dir={IFU_config_dir} failed.")
        print(f"WARNING: IFU_params not found — auto-derived: {IFU_params}")
        configs = json.load(open(IFU_params))
        configs["CONFIG_FILE"] = os.path.basename(IFU_params)

    # ------------------------------------------------------------------
    # 1. Determine operational mode
    # ------------------------------------------------------------------
    preparation_mode = (patch_file is None)

    if preparation_mode:
        print("INFO: PREPARATION MODE — generating patch files")
    else:
        print("INFO: ANALYSIS MODE — using existing patch file")
        assert Path(patch_file).exists(), f"Patch file not found: {patch_file}"

    # ------------------------------------------------------------------
    # 2. Load existing patch file (analysis mode)
    # ------------------------------------------------------------------
    patch_table = None

    if patch_file is not None:
        if Path(patch_file).suffix == ".fits":
            patch_table = Table.read(patch_file)
        else:
            patch_table = read_ascii_patchfile(patch_file)

        assert patch_table.colnames == [
            "id", "RA_icrs", "DEC_icrs", "A_world", "B_world",
            "angle", "flag", "type", "Z", "ZERR", "ZWARN", "CLASS",
        ], "Error: inconsistent patch_file column names"

        seg2d = False
        print(f"INFO: Loaded patch table with {len(patch_table)} entries")

    # ------------------------------------------------------------------
    # 3. 2D segmentation (preparation mode only)
    # ------------------------------------------------------------------
    if seg2d:
        print("INFO: Running 2D segmentation...")
        ifu_seg2d_out = ifu_seg2d(
            infiles, headname, outpath,
            white_images=seg2d_white_images,
            white_src=seg2d_white_src,
            search_gaia=seg2d_search_gaia,
            extract=seg2d_extract,
            mask_stars=seg2d_mask_stars,
            ext_thresh=seg2d_ext_thresh,
            minarea=seg2d_minarea,
            deblend_nthresh=seg2d_deblend_nthresh,
            radii_factor=seg2d_radii_factor,
            class_ntop=class_ntop,
        )
        patch_file  = ifu_seg2d_out.get_targets_fname()
        class_patch = True   # always classify after segmentation

    # ------------------------------------------------------------------
    # 3b. 3D matched-filter segmentation (optional, opt-in, additive --
    #     see aps_ifu_seg3d.py). Runs independently of seg2d and, by
    #     default, never touches patch_file/patch_table/classification --
    #     it only writes its own candidate table + (optional) comparison
    #     figure. If seg3d_merge=True as well (also opt-in, also requires
    #     seg2d=True), 'obvious' seg3d-only candidates are additionally
    #     merged into the real patch table as auxiliary targets -- see
    #     aps_ifu_seg3d.merge_into_patch_table's docstring for exactly
    #     what qualifies and how provenance is tracked.
    # ------------------------------------------------------------------
    if seg3d:
        print("INFO: Running 3D matched-filter segmentation (seg3d)...")
        try:
            seg3d_out = aps_ifu_seg3d.run_seg3d(
                infiles, headname, outpath,
                lmin=seg3d_lmin,
                lmax=seg3d_lmax,
                sens_corr=sens_corr,
                mask_gaps=mask_gaps,
                safe_mask_gaps=safe_mask_gaps,
                tellurics=tellurics,
                vacuum=vacuum,
                arms_ratio=arms_ratio,
                threshold=seg3d_threshold,
                spatial_bin=seg3d_spatial_bin,
                sky_mask_min_A=seg3d_sky_mask_min_A,
                continuum_method=seg3d_continuum_method,
                ivar_calibration=seg3d_ivar_calibration,
                min_npix=seg3d_min_npix,
                group_radius_px=seg3d_group_radius_px,
            )
            candidates_fname = Path(outpath) / f"{headname}_seg3d_candidates.fits"
            seg3d_out["table"].write(str(candidates_fname), overwrite=True)
            print(f"INFO: seg3d candidate table written to {candidates_fname} "
                  f"({len(seg3d_out['table'])} candidates)")

            # Merge runs BEFORE the figure so the figure can mark exactly
            # which candidates were actually added to the seg2d patch
            # table (merged_ids), rather than the figure guessing at a
            # duplicate of this same filtering logic.
            merged_ids = np.array([], dtype=int)
            if seg3d_merge:
                if not seg2d:
                    print("WARNING: seg3d_merge=True requires seg2d=True (need an existing "
                          "patch table to merge into) -- skipping merge.")
                else:
                    # purity_scan supplied -> the effective SNR threshold is
                    # derived from THIS field's own purity self-check
                    # (resolve_merge_min_snr), not trusted blindly at
                    # seg3d_merge_min_snr; that value is only the floor.
                    # If no threshold in the scan is both
                    # purity>=seg3d_merge_min_purity and backed by enough
                    # candidates to trust (n_pos>=15), merging is skipped
                    # for this field entirely -- by design: an empty merge
                    # is preferred over adding a candidate that isn't
                    # actually statistically distinguishable from noise.
                    merged_table, n_merged, n_rejected, merged_ids = aps_ifu_seg3d.merge_into_patch_table(
                        ifu_seg2d_out.patch_table, seg3d_out["table"],
                        min_snr=seg3d_merge_min_snr, purity_scan=seg3d_out["purity_scan"],
                        purity_target=seg3d_merge_min_purity,
                        aperture_radius_arcsec=seg3d_merge_aperture_arcsec, class_ntop=class_ntop,
                        group_min_lines=seg3d_merge_group_min_lines,
                        group_min_snr=seg3d_merge_group_min_snr,
                        group_purity_scan=seg3d_out.get("group_purity_scan"),
                        group_min_n_trust=seg3d_merge_group_min_n_trust,
                        extreme_snr_loudest_event=(
                            seg3d_out.get("loudest_neg_event")
                            if seg3d_merge_extreme_safety_margin is not None else None),
                        extreme_snr_safety_margin=(seg3d_merge_extreme_safety_margin or 1.1),
                    )
                    if n_merged > 0:
                        fits.table_to_hdu(merged_table).writeto(str(patch_file), overwrite=True,
                                                                 checksum=True)
                        ifu_seg2d_out.patch_table = merged_table
                        print(f"INFO: seg3d_merge added {n_merged} auxiliary target(s) to "
                              f"{patch_file} ({n_rejected} candidate(s) rejected as duplicates "
                              f"of existing targets); marked with "
                              f"flag={aps_ifu_seg3d.SEG3D_PROVENANCE_FLAG} for later identification")
                    else:
                        print(f"INFO: seg3d_merge added 0 targets ({n_rejected} rejected as "
                              f"duplicates) -- either no candidate cleared the reliability bar, "
                              f"or none had no continuum counterpart")

            if seg3d_fig:
                fig3d = _build_source_detection_3d_figure(
                    seg3d_out["wcs"], seg3d_out["collapse"], seg3d_out["objects"],
                    seg3d_out["candidates"], seg3d_out["purity_scan"],
                    radii_factor=seg2d_radii_factor, headname=headname,
                    merged_ids=merged_ids, aperture_arcsec=seg3d_merge_aperture_arcsec,
                    pixscale_arcsec=seg3d_out["pixscale_arcsec"],
                    group_purity_scan=seg3d_out.get("group_purity_scan"),
                )
                fig3d_fname = figdir / f"{headname}_seg3d_compare.pdf"
                fig3d.write_image(str(fig3d_fname))
                print(f"INFO: seg3d comparison figure written to {fig3d_fname}")
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f"WARNING: seg3d failed, continuing without it: {exc_value}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. Classification (preparation mode only)
    # ------------------------------------------------------------------
    if class_patch:
        print("INFO: Running classification and redshift determination...")
        assert class_templates is not None, \
            "class_templates directory is required for classification"
        assert Path(class_templates).is_dir(), \
            "class_templates directory not found: %s" % class_templates

        if patch_file is None:
            print(
                "INFO: seg2d=False and no patch_file provided — "
                "creating a single central-field patch automatically."
            )
            if seg2d_exclude_ctarg:
                print(
                    "WARNING: seg2d_exclude_ctarg=True is ignored when seg2d=False; "
                    "the central target is the only target and will be kept."
                )
            patch_file = ifu_make_central_patch(
                infiles, headname, outpath, class_ntop=class_ntop
            )

        patch_table = ifu_class(
            infiles, headname, patch_file, class_templates,
            class_templates_ARC=class_templates_ARC,
            aps_ids=aps_ids, mask_aps_ids=mask_aps_ids,
            z_rad=class_z_rad, wlranges=wlranges,
            figdir=str(figdir), ncpus=mp_prep,
            arms_ratio=arms_ratio, class_ntop=class_ntop,join_arms=False,
            catdir=catdir, caldir=caldir,
            IFU_config_dir=IFU_config_dir,targclass=targclass,targsrvy=targsrvy
        )

    # ------------------------------------------------------------------
    # 5. Verify we have a patch table
    # ------------------------------------------------------------------
    assert patch_table is not None, (
        "No patch table available. "
        "Provide patch_file, or set seg2d=True / class_patch=True."
    )
    assert (patch_table["type"]).dtype in ["U1", "S1"], \
        "TYPE column format must be S1"

    # ------------------------------------------------------------------
    # 6. Central target exclusion
    # ------------------------------------------------------------------
    ctarg_excluded = False

    if seg2d_exclude_ctarg:
        if len(patch_table[patch_table["type"] == "T"]) > 0:
            print(
                "DEBUG: seg2d_exclude_ctarg=True — removing central target "
                "(other targets available)"
            )
            wp_table       = patch_table[patch_table["type"] != "C"]
            ctarg_excluded = True
        else:
            print(
                "DEBUG: seg2d_exclude_ctarg=True — keeping central target "
                "(no other targets available)"
            )
            wp_table = deepcopy(patch_table)
    else:
        wp_table = deepcopy(patch_table)
        print("DEBUG: seg2d_exclude_ctarg=False — central target remains")

    # ------------------------------------------------------------------
    # 7. Multi-class splitting and disk writes
    # ------------------------------------------------------------------
    needs_mod, reason = patch_table_needs_modification(wp_table)
    print(f"DEBUG: Modification check: {reason}")

    if preparation_mode:
        # Write base patch table to disk
        if Path(patch_file).suffix in [".fit", ".fits"]:
            fits.table_to_hdu(wp_table).writeto(
                str(patch_file), overwrite=True, checksum=True
            )
        else:
            write_ascii_patchfile(wp_table, Path(patch_file))
        print(f"DEBUG: Base patch table saved: {patch_file}")

        if needs_mod:
            print("DEBUG: Splitting multi-class entries...")
            print(f"DEBUG: Original length: {len(wp_table)}")

            wp_table = update_patch_table(
                wp_table,
                class_lists=[["GALAXY", "QSO"], ["STAR", "WD"]],
                debug=False,
                ctarg_excluded=ctarg_excluded,
                resort=True,
            )

            mod_patch_file = Path(patch_file).with_name(
                Path(patch_file).stem + "_mod" + Path(patch_file).suffix
            )
            if mod_patch_file.suffix in [".fit", ".fits"]:
                fits.table_to_hdu(wp_table).writeto(
                    str(mod_patch_file), overwrite=True, checksum=True
                )
            else:
                write_ascii_patchfile(wp_table, mod_patch_file)

            print(f"DEBUG: Modified patch table saved: {mod_patch_file}")
            print(f"DEBUG: Modified length: {len(wp_table)}")
        else:
            print("DEBUG: No multi-class splitting needed")

        print("INFO: PREPARATION MODE complete — patch files written to disk")

    else:
        # Analysis mode: split in memory only, never write
        if needs_mod:
            print(f"DEBUG: Applying multi-class split in memory only: {reason}")
            wp_table = update_patch_table(
                wp_table,
                class_lists=[["GALAXY", "QSO"], ["STAR", "WD"]],
                debug=False,
                ctarg_excluded=ctarg_excluded,
                resort=True,
            )
            print(f"DEBUG: Split table has {len(wp_table)} entries (in memory only)")

        mod_patch_file = Path(patch_file).with_name(
            Path(patch_file).stem + "_mod" + Path(patch_file).suffix
        )
        if mod_patch_file.exists():
            print(f"INFO: Found existing modified patch file: {mod_patch_file}")

    # ------------------------------------------------------------------
    # 8. Return
    # ------------------------------------------------------------------
    return {
        "wp_table":       wp_table,
        "patch_file":     str(patch_file),
        "ctarg_excluded": ctarg_excluded,
    }


##################################################################################################################################################

"""
aps_ifu_prepare.py  —  WEAVE IFU pipeline: PREPARATION stage
=====================================================
Runs seg2d, classification, patch table generation.
Writes {headname}_targets.fits  (and _mod if multi-class).

After this script completes, run in parallel:
    aps_ExGal_worker.py  --patch_file {headname}_targets_mod.fits ...
    aps_Gal_worker.py    --patch_file {headname}_targets_mod.fits ...
"""

import argparse


def ifu_runner(options=None):

    parser = build_common_parser(
        description="aps_ifu_prepare — IFU Preparation stage",
        groups=["target_selection", "wavelength", "l1_processing",
                "caldirs", "output"],
        overrides={
            "vacuum": {"default": False},
            "join_arms": {"default": True},
            "targsrvy": {"help": "Comma-separated survey filter passed to the classifier"},
            "targclass": {"help": "Comma-separated class filter passed to the classifier"},
        },
        # this script has no --configdir and no --overwrite flag at all
        exclude=["configdir", "overwrite"],
        extra_args=[
            (("--patch_file",), dict(type=none_or_str, default=None)),
            (("--seg2d",), dict(type=str2bool, default=False)),
            (("--seg2d_white_images",), dict(type=none_or_str, default=None)),
            (("--seg2d_white_src",), dict(type=none_or_str, default="weave")),
            (("--seg2d_search_gaia",), dict(type=str2bool, default=True)),
            (("--seg2d_extract",), dict(type=str2bool, default=True)),
            (("--seg2d_exclude_ctarg",), dict(type=str2bool, default=True)),
            (("--seg2d_mask_stars",), dict(type=str2bool, default=True)),
            (("--seg2d_ext_thresh",), dict(type=none_or_str, default=2.5)),
            (("--seg2d_minarea",), dict(type=none_or_str, default=400.0)),
            (("--seg2d_deblend_nthresh",), dict(type=none_or_str, default=4)),
            (("--seg2d_radii_factor",), dict(type=float, default=7.0)),
            (("--seg3d",), dict(type=str2bool, default=False)),
            (("--seg3d_lmin",), dict(type=float, default=3700.0)),
            (("--seg3d_lmax",), dict(type=float, default=9390.0)),
            (("--seg3d_threshold",), dict(type=float, default=6.0)),
            (("--seg3d_spatial_bin",), dict(type=int, default=4)),
            (("--seg3d_min_npix",), dict(type=int, default=3)),
            (("--seg3d_sky_mask_min_A",), dict(type=none_or_str, default=None)),
            (("--seg3d_continuum_method",), dict(type=none_or_str, default="pca")),
            (("--seg3d_ivar_calibration",), dict(type=none_or_str, default="per-wave")),
            (("--seg3d_fig",), dict(type=str2bool, default=True)),
            (("--seg3d_merge",), dict(type=str2bool, default=False)),
            (("--seg3d_merge_min_snr",), dict(type=float, default=10.0)),
            (("--seg3d_merge_min_purity",), dict(type=float, default=0.9)),
            (("--seg3d_merge_aperture_arcsec",), dict(type=float, default=2.0)),
            (("--seg3d_merge_group_min_lines",), dict(type=none_or_str, default=2)),
            (("--seg3d_merge_group_min_snr",), dict(type=float, default=10.0)),
            (("--seg3d_merge_group_min_n_trust",), dict(type=int, default=3)),
            (("--seg3d_group_radius_px",), dict(type=float, default=0.0)),
            (("--seg3d_merge_extreme_safety_margin",), dict(type=none_or_str, default=1.1)),
            (("--class_patch",), dict(type=str2bool, default=False)),
            (("--class_templates",), dict(type=none_or_str, default=None)),
            (("--class_templates_ARC",), dict(type=none_or_str, default=None)),
            (("--class_ntop",), dict(type=int, default=1)),
            (("--class_z_rad",), dict(type=float, default=1.5)),
            (("--IFU_config_dir",), dict(type=none_or_str, default=None)),
            (("--mp_prep",), dict(type=int, default=1)),
            (("--user_patch",), dict(type=str2bool, default=False)),
            (("--fig",), dict(type=str2bool, default=False)),
            (("--uapsid",), dict(type=none_or_str, default=None)),
            (("--IFU_params",), dict(type=none_or_str, default=None)),
        ],
    )

    # ------------------------------------------------------------------
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print("No command-line arguments — running DEMO/DEBUG mode")
        args = parser.parse_args(options)

    # ---- normalise lists ----
    # resolve_common_args now also writes the normalized wlranges/arms_ratio
    # back onto args (for print_args()'s own reporting, previously
    # absent here) and gives the arms_ratio length-mismatch assert a real
    # message (previously bare) — both disclosed, non-functional
    # improvements over the pre-migration code.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio
    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)

    seg2d_white_images = None
    if args.seg2d_white_images is not None:
        seg2d_white_images = [str(x) for x in args.seg2d_white_images.split(",")]

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" % args.outpath)
    outpath = (args.outpath + os.sep).replace(" ", "")

    print_args(args, module="aps_ifu_prepare",
               version=aps_constants.__aps_ifu_version__,
               path=outpath, headname=args.headname)

    seg3d_sky_mask_min_A = (float(args.seg3d_sky_mask_min_A)
                             if args.seg3d_sky_mask_min_A is not None else None)

    # seg2d_ext_thresh/minarea/deblend_nthresh: None -> auto-estimate (see
    # ifu_seg2d.__init__ and aps_ifu_seg3d.estimate_seg2d_params); a given
    # value arrives here as a string (none_or_str doesn't itself parse
    # numbers) and needs converting.
    def _none_or_float(x):
        return None if x is None else float(x)

    seg2d_ext_thresh_val = _none_or_float(args.seg2d_ext_thresh)
    seg2d_minarea_val = _none_or_float(args.seg2d_minarea)
    seg2d_deblend_nthresh_val = _none_or_float(args.seg2d_deblend_nthresh)
    seg3d_merge_group_min_lines_val = (None if args.seg3d_merge_group_min_lines is None
                                        else int(args.seg3d_merge_group_min_lines))
    seg3d_merge_extreme_safety_margin_val = (None if args.seg3d_merge_extreme_safety_margin is None
                                              else float(args.seg3d_merge_extreme_safety_margin))

    # ------------------------------------------------------------------
    prep = ifu_worker_prepare(
        infiles=args.infiles,
        headname=args.headname,
        outpath=outpath,
        patch_file=args.patch_file,
        user_patch=args.user_patch,
        sens_corr=args.sens_corr,
        mask_gaps=args.mask_gaps,
        safe_mask_gaps=args.safe_mask_gaps,
        tellurics=args.tellurics,
        vacuum=args.vacuum,
        seg2d=args.seg2d,
        seg2d_white_images=seg2d_white_images,
        seg2d_white_src=args.seg2d_white_src,
        seg2d_search_gaia=args.seg2d_search_gaia,
        seg2d_extract=args.seg2d_extract,
        seg2d_exclude_ctarg=args.seg2d_exclude_ctarg,
        seg2d_mask_stars=args.seg2d_mask_stars,
        seg2d_ext_thresh=seg2d_ext_thresh_val,
        seg2d_minarea=seg2d_minarea_val,
        seg2d_deblend_nthresh=seg2d_deblend_nthresh_val,
        seg2d_radii_factor=args.seg2d_radii_factor,
        seg3d=args.seg3d,
        seg3d_lmin=args.seg3d_lmin,
        seg3d_lmax=args.seg3d_lmax,
        seg3d_threshold=args.seg3d_threshold,
        seg3d_spatial_bin=args.seg3d_spatial_bin,
        seg3d_min_npix=args.seg3d_min_npix,
        seg3d_sky_mask_min_A=seg3d_sky_mask_min_A,
        seg3d_continuum_method=args.seg3d_continuum_method,
        seg3d_ivar_calibration=args.seg3d_ivar_calibration,
        seg3d_fig=args.seg3d_fig,
        seg3d_merge=args.seg3d_merge,
        seg3d_merge_min_snr=args.seg3d_merge_min_snr,
        seg3d_merge_min_purity=args.seg3d_merge_min_purity,
        seg3d_merge_aperture_arcsec=args.seg3d_merge_aperture_arcsec,
        seg3d_merge_group_min_lines=seg3d_merge_group_min_lines_val,
        seg3d_merge_group_min_snr=args.seg3d_merge_group_min_snr,
        seg3d_merge_group_min_n_trust=args.seg3d_merge_group_min_n_trust,
        seg3d_group_radius_px=args.seg3d_group_radius_px,
        seg3d_merge_extreme_safety_margin=seg3d_merge_extreme_safety_margin_val,
        class_patch=args.class_patch,
        class_templates=args.class_templates,
        class_templates_ARC=args.class_templates_ARC,
        class_ntop=args.class_ntop,
        class_z_rad=args.class_z_rad,
        wlranges=wlranges,
        aps_ids=aps_ids,
        mask_aps_ids=mask_aps_ids,
        targsrvy=targsrvy,
        targclass=targclass,
        arms_ratio=arms_ratio,
        IFU_config_dir=args.IFU_config_dir,
        mp_prep=args.mp_prep,
        catdir=args.catdir,
        caldir=args.caldir,
        IFU_params=args.IFU_params,
        UAPSID=args.uapsid,
    )

    if prep is None:
        sys.exit("Preparation failed — no patch table generated")

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print(f"  Rows       : {len(prep['wp_table'])}")
    print("  Ready for aps_ExGal_worker.py  and  aps_Gal_worker.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    debug_LIFU = [
        "--infiles",
        "<PYAPS_DATA>/L1/<night>/stackcube_<runid>.fit",
        "<PYAPS_DATA>/L1/<night>/stackcube_<runid>.fit",
        "--headname",  "LWVE_<target>_01_GR_H1",
        "--outpath",   "<PYAPS_DATA>/L2/<night>/<obid>/",
        "--wlranges",  "None",
        "--arms_ratio","1.0,1.0",
        "--IFU_config_dir", "<PYAPS_DIR>/configs/ExGal_configs/",
        "--mp_prep",   "12",
        "--sens_corr", "True",
        "--mask_gaps", "True",
        "--safe_mask_gaps", "True",
        "--tellurics", "False",
        "--vacuum",    "False",
        "--fill_gap",  "False",
        "--seg2d",     "True",
        "--class_ntop","3",
        "--seg2d_white_images", "None",
        "--seg2d_white_src",    "weave",
        "--seg2d_search_gaia",  "False",
        "--seg2d_extract",      "True",
        "--seg2d_mask_stars",   "False",
        "--seg2d_ext_thresh",   "2.5",
        "--seg2d_minarea",      "400",
        "--seg2d_deblend_nthresh", "4",
        "--seg2d_radii_factor", "7.0",
        "--class_patch",        "True",
        "--class_templates",    "<PYAPS_DIR>/PyAPS_templates/templates_RR/",
        "--class_templates_ARC","<PYAPS_DIR>/PyAPS_templates/templates_ARC_RR/",
        "--class_z_rad",        "1.5",
        "--seg2d_exclude_ctarg","True",
        "--caldir", "<PYAPS_DATA>/CAL",
        "--catdir", "<PYAPS_DATA>/CAT",
    ]

    ifu_runner(options=debug_LIFU)
