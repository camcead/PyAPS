from __future__ import absolute_import, division, print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import re
import warnings
import traceback
import argparse
import json
import numpy as np
import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, l1_fileinfo, gen_targlist, l1_to_image
from PyAPS import aps_constants
import scipy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.io.fits import getdata, getheader
from astropy.io.fits.verify import VerifyWarning






def make_joint_cube(infiles, output_fname, aps_ids=None,skysub=True, targsrvy= None, targclass = None, mask_aps_ids=None , area=None, mask_areas=None,
    wlranges=None, sens_corr=None, mask_gaps=None,safe_mask_gaps=None, vacuum=False, tellurics=False, 
    fill_gap=False, arms_ratio=None, join_arms=False, funit= 1.e18, offset_gap_pix = None, collapse=False, crr=False):

    """
    Note: This is Beta version and only a few header keywords will be updated in the new joint file.
    Make a Joint cube with structure similar to the original L1 file.

    """


    print(f"NOTE: This is a Beta version of this code and the final output may have many faulty keywords in the header")
    print("")



    #  (sky subtracted)
    print(f"Reading/Preparing the L1 file (Sky_subtracted)")
    targs_joint =APSOB(infiles, aps_ids=aps_ids,skysub=True, targsrvy= targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps,safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, 
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, funit= funit, offset_gap_pix = offset_gap_pix, collapse=False, crr=crr)

    #  (NO sky subtracted)
    print(f"Reading/Preparing the L1 file (Sky_subtracted)")
    targs_joint_nss =APSOB(infiles, aps_ids=aps_ids,skysub=False, targsrvy= targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps,safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, 
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, funit= funit, offset_gap_pix = offset_gap_pix, collapse=False, crr=crr)


    # get first (bluest) arm as reference # and make a copy to be filled in
    infile_0 = targs_joint._infiles[0]


    # Preparing the header
    print(f"Preparing headers")
    h0 = getheader(infile_0, 0) # PRIMARY header
    h0_joint = h0.copy()
    h1 = getheader(infile_0, 1) # ARM0_DATA header
    h1_joint = h1.copy()
    h2 = getheader(infile_0, 2) # ARM0_IVAR header
    h2_joint = h2.copy()
    h3 = getheader(infile_0, 3) # ARM0_DATA_NOSS header
    h3_joint = h3.copy()
    h4 = getheader(infile_0, 4) # ARM0_IVAR_NOSS header
    h4_joint = h4.copy()
    h5 = getheader(infile_0, 5) # ARM0_SENSFUNC header
    h5_joint = h5.copy() 
    h6 = getheader(infile_0, 6) # ARM0_DATA_COLLAPSE header
    h6_joint = h6.copy() 
    h7 = getheader(infile_0, 7) # ARM0_IVAR_COLLAPSE header
    h7_joint = h7.copy() 


    ## loop over all available collapse data in EXT6 and EXT7 and make a stacked one
    collapse_data_list=[]
    collapse_ivar_list=[]
    for infile_i in targs_joint._infiles:
        collapse_data_list.append(getdata(infile_i, 6)) # collapse Data
        collapse_ivar_list.append(getdata(infile_i, 7)) # collapse ivar 

    collapse_data = np.nanmean(collapse_data_list, axis=0)
    collapse_ivar = np.nansum(collapse_ivar_list, axis=0)

    # read the WCS of the the arm0 data (L1) and create a matrix of APS_IDs identical to xy matrix
    wcs_h1 = wcs.WCS(h1)
    cube_shape = wcs_h1.pixel_shape  # [xpix, ypix, zpix]
    # APS_ID as we have in the original L2 file
    APS_ID_ALL = np.arange(cube_shape[0] * cube_shape[1]) + 1
    # APS_ID in the format of L1 file
    APS_CUBE = np.reshape(APS_ID_ALL, (cube_shape[1] , cube_shape[0]))

    # get info about the length of the joint spectra
    new_fl_len = len(targs_joint._targetlist[0].spectra[0].wave)

    # read flux  matrix of L1 to get shape info
    fl = getdata(infile_0, 1)


    # prepare some empty arrays for the updated spectra
    fl_new = np.empty((new_fl_len,fl.shape[1],fl.shape[2]))
    fl_nss_new = np.empty((new_fl_len,fl.shape[1],fl.shape[2]))
    ivar_new = np.empty((new_fl_len,fl.shape[1],fl.shape[2]))
    ivar_nss_new = np.empty((new_fl_len,fl.shape[1],fl.shape[2]))

    # And create an updated sensitivity function with all values = 1.0
    sens_new = np.ones((new_fl_len,))

    # loop over all spaxels and fill the new values
    for c0 in np.arange(cube_shape[0]):
        for c1 in np.arange(cube_shape[1]):
            aps_id_i = APS_CUBE[c1,c0]

            if aps_id_i in targs_joint.id():
                id_in_aps = targs_joint.apstoid(aps_id_i)
                fl_new[:,c1,c0] = targs_joint._targetlist[id_in_aps].spectra[0].flux
                ivar_new[:,c1,c0] = targs_joint._targetlist[id_in_aps].spectra[0].ivar
                fl_nss_new[:,c1,c0] = targs_joint_nss._targetlist[id_in_aps].spectra[0].flux
                ivar_nss_new[:,c1,c0] = targs_joint_nss._targetlist[id_in_aps].spectra[0].ivar


    ## Update headers info
    # h0_joint #TODO: no update on h0 at the moment
    h1_joint['NAXIS3'] = (int(new_fl_len), 'Updated by APS')
    h1_joint['EXTNAME'] = ('JOINT_DATA', 'Updated by APS')
    h2_joint['NAXIS3'] = (int(new_fl_len), 'Updated by APS')
    h2_joint['EXTNAME'] = ('JOINT_IVAR', 'Updated by APS')
    h3_joint['NAXIS3'] = (int(new_fl_len), 'Updated by APS')
    h3_joint['EXTNAME'] = ('JOINT_DATA_NOSS', 'Updated by APS')
    h4_joint['NAXIS3'] = (int(new_fl_len), 'Updated by APS')
    h4_joint['EXTNAME'] = ('JOINT_IVAR_NOSS', 'Updated by APS')
    h5_joint['NAXIS1'] = (int(new_fl_len), 'Updated by APS')
    h6_joint['EXTNAME'] = ('JOINT_DATA_COLLAPSE', 'Updated by APS')
    h7_joint['EXTNAME'] = ('JOINT_DATA_COLLAPSE', 'Updated by APS')



    # preparing HDUs of the output
    print(f"Writing the final output to {output_fname}")
    hdu0 = fits.PrimaryHDU(header = h0_joint)
    hdu1 = fits.ImageHDU(data=fl_new, header=h1_joint, name = 'JOINT_DATA')
    hdu2 = fits.ImageHDU(data=ivar_new, header=h2_joint, name = 'JOINT_IVAR')
    hdu3 = fits.ImageHDU(data=fl_nss_new, header=h3_joint, name = 'JOINT_DATA_NOSS')
    hdu4 = fits.ImageHDU(data=ivar_nss_new, header=h4_joint, name = 'JOINT_IVAR_NOSS')
    hdu5 = fits.ImageHDU(data=sens_new, header=h5_joint, name = 'JOINT_SENSFUNC')
    hdu6 = fits.ImageHDU(data=collapse_data, header=h6_joint, name = 'JOINT_DATA_COLLAPSE')
    hdu7 = fits.ImageHDU(data=collapse_ivar, header=h7_joint, name = 'JOINT_IVAR_COLLAPSE')

    new_hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
    new_hdul.writeto(output_fname, overwrite=True)

# ------------------------------------------------------------------------------------------


if __name__ == "__main__":




    # local
    # infiles = ['<PYAPS_DATA>/FL/20221025/stackcube_2963103.fit','<PYAPS_DATA>/FL/20221025/stackcube_2963102.fit']
    # output_fname= '<OUTPUT_DIR>/example.fits'

    # dama
    infiles = ['/scratch/aps/PyAPS/PyAPS_data/FL/20221025/stackcube_2963103.fit','/scratch/aps/PyAPS/PyAPS_data/FL/20221025/stackcube_2963102.fit']
    output_fname= '/scratch/aps/PyAPS/PyAPS_results/20221025/joint_stackcube_2963103__stackcube_2963102.fits'

    wlranges = None
    aps_ids=None
    targsrvy = None
    targclass = None
    mask_aps_ids= None
    area=None
    mask_areas = None
    sens_corr = True
    mask_gaps = False
    safe_mask_gaps = False
    vacuum  = False
    tellurics = False
    fill_gap = False
    arms_ratio = [1.0,1.0]
    join_arms= True
    funit = 1.0e18
    offset_gap_pix= 10
    crr = False


    make_joint_cube(infiles, output_fname, aps_ids=aps_ids,skysub=True, targsrvy= targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps,safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, 
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, funit= funit, offset_gap_pix = offset_gap_pix, collapse=False, crr=crr)

