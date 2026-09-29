
import os
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import sys
import argparse
import time
import itertools
import multiprocessing as mp
from collections import OrderedDict
import pandas
from astropy.io import fits
import astropy.wcs as pywcs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import  Table, Column, vstack
import datetime
import yaml
#import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
#from PyAPS import aps_constants
from astropy import units
from copy import deepcopy
import numpy as np
from scipy import interpolate
import scipy
import astropy
import matplotlib
from datetime import datetime
from scipy.stats.mstats import mquantiles
# from parameters import *
from astropy.table import Table
from ptemcee import Sampler as PTSampler
startTime = datetime.now()

__aps_amy_version__ = 1.3




"""
aps_amy: A Python wrapper to run the AMY code, as a WEAVE Contributed software


versions:
 1.0 Original version by Maria Monguio (June 2020)
 1.2 By A. Molaeinezhad(CASU, December 2020)- Code fully reshuffled with all required changes to make it compatible with PyAPS platform and the latest weave data model (v 8.0)
 1.3 By A. Molaeinezhad(CASU, May 2021)- Minor updates
 1.4 By M.Monguio - Paralelising the code
1.5 By A. Molaeinezhad (CASU/APS October 2023) 

TODO LIST:
1 Many details after the new changes needed to be discussed with Maria
2 Code is not compatible with the latest version of the ptemcee code. It Only support the version provided by Maria inside the package (ptemcee-1.0.0)
3 The code is versy sensitive to the wavelength range selection. Needed to be sorted out
4 No proper error/exception handling
5 Very slow (analysis time per target ~ 13 min). 
6 A better parallelisation approach is needed to be deployed. A.M. removed the method introduced in the original version
7 Code fully reshuffled with all required changes to make it compatible with PyAPS platform and the latest weave data model (v 8.0). But all details must be discussed with Maria
8 The templates generation code must be completed. Already implemented but not tested
9 This header (general code header) must be completed
10 and many more details to discuss
11 Output structure depends on wavelength range (e.g. formats=['I','I','20A','30A','1880E','1880E']). This method is not acceptable and must be improved. Structure must be independent of the input params.

** (May 2021) What about mask_gaps or safe_mask_gaps or tellurics corrections??????? Why you set them all to False????




Example: 
Mode1: Run through the command line:

python3 aps_amy.py --infiles <PYAPS_DATA>/AMY/4365/stacked_1003689.fit --aps_ids 184,185,187 --targsrvy None --targclass None --wlranges 8470.0,8940.0 --exclude_region 8470,8610 8630.0,8940.0 --sens_corr True --mask_gaps True --safe_mask_gaps True --tellurics False --vacuum False --fill_gap False --arms_ratio 1.0 --join_arms False --outpath <PYAPS_DIR>/PyAPS_results/20170911/4365/ --templates_dir <PYAPS_DIR>/PyAPS_templates/templates_AMY/ --template_flux template_grid.npy --template_wave wavelength_data.dat --headname stacked_1003689 --overwrite True --mproc 2 --progress_bar True

Mode2 (debug/develop mode): 
By updating the parameters at the bottom of this file and directly call the aps_amy code




HISTORY:

June 2020: Version 1.0 (Original version) by Maria Monguio 
December 2020 : (By A. Molaeinezhad) - Code fully reshuffled with all required changes to make it compatible with PyAPS platform and the latest weave data model (v 8.0)
29 January 2021: (By A. Molaeinezhad)- fixed a bug where the code generates the noisespec (noise spectrum)
29 January 2021: (By A. Molaeinezhad)- progress_bar returned as an option
07 May 2021: (By A. Molaeinezhad) - aps_utils (APSOB) updated to solve the issue we had with deepcopy of wlranges
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
18 May 2021: update mask_gaps and safe_mask_gaps to True. (Not sure how it affects the results)
09 Oct 2023: mask_aps_ids, area, mask_areas arguments feeders added. But not in used. Just to keep the consistency
"""






###############################################################################

# The following parameters are suitable for the default CaT region case using 
# the templates provided and LR WEAVE spectra. The following can be tweaked if
# desired, but generally they are satisfactory.

### MCMC SAMPLE PARAMETERS ### 

nwalkers = 200      # number of walkers
burn = 250          # number of burn-in steps
runs = 2000         # number of steps
ntemps = 5          # number of temperatures

### PARAMETER BOUNDARIES ###

# Teff, logg, vsini must be same as template grid
Teff = np.arange(5000, 18500, 500)
logg = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
vsini = np.array([0, 50, 100, 150, 200, 250, 300])

drv = np.arange(-500., 500., 0.1)
slopes = np.arange(-0.3e-5, 2.0e-5, 1.0e-10)
intercepts = np.arange(-1.0e-1, 1.0e-1, 1.0e-6)

# pixels-per-resolution element value (PPRE = resolution/wavelength sampling)
PPRE = 5.2

# grid of parameters for walker initial positions
ini_grid = [Teff, logg, vsini, drv, slopes, intercepts]
ndim=6



###############################################################################
# processing (cropping, smoothing, re-binning, rotational broadening) templates if required 
def prepare_templates():

    # list of templates to be processed. Listed objects should include paths
    template_list_file = 'your_own_templates/template_list'


    # directory to write processed templates and template grid to
    templates_dir = 'your_own_templates/processed/'


    # index of wavelength and un-normalised flux/counts columns
    wav_ind = 0
    flux_ind = 2



    # wavelengths to crop template spectrum to. Should be sufficiently larger than
    # ptmcmc wavelength region to allow for Doppler shifting.
    template_crop_min_wav = 6000.   
    template_crop_max_wav = 9000.   

    # The vsini values if rotational broadening is desired. In this case, the
    # provided templates should be unbroadened. The vsini values should be given in
    # array form i.e. '[50.0, 100.0, 200.0]'. If rotational broadening is not
    # desired, enter 'None'. Remember to enter appropriate vsini values in the
    # PARAMETER BOUNDARIES section above.
    rotbroads = None#[50.0, 100.0, 150.0, 200.0, 250.0, 300.0]

    # The width (i.e., standard deviation) of the Gaussian profile used to
    # broaden/smooth. Enter 'None' for no smoothing.
    sigma = 0.48

    # The desired wavelength sampling. Enter 'None' for no rebinning.  
    sampling = 0.25

    # The starting and ending indexed of temperature, logg, vsini information in
    # the listed templates. E.g. for a listed template called 
    # 'C:/Users/me/Documents/templates/t05000g3.00-vsini000.dat', the indices would
    # be (33, 38), (39,43), (49, 52) respectively.
    temp_ind = (30, 36)
    logg_ind = (37, 41)
    vsini_ind = (46, 49)    # if templates are not being rotationally broadened


    print('Beginning processing of templates')
    from PyAstronomy import pyasl
    import scipy.stats
    templatelist = np.genfromtxt(template_list_file, dtype=None, encoding=None)

    # Ensuring the number of templates to be processed matches the number of points on the grid. 
    notemps = len(templatelist)
    d_size = len(Teff) * len(logg) * len(vsini)
    if notemps < d_size:
        sys.exit('ERROR: The amount of templates to be processed is less than the amount of points in the grid. Add more templates to the list or reduce the grid as necessary (see PARAMETER BOUNDARIES in parameters file).')
    if notemps > d_size: 
        sys.exit('ERROR: The amount of templates to be processed is greater than the amount of points in the grid. Remove templates from the list or expand the grid as necessary (see PARAMETER BOUNDARIES in parameters file).')

    def restrict_range(w, f):       # to crop the templates
        our_range = (w > template_crop_min_wav) & (w < template_crop_max_wav)
        w, f = w[our_range], f[our_range]
        return w, f

    def smooth(w, f, sig):      # to smooth/broaden templates to match resolution of observed spectrum
        f = pyasl.broadGaussFast(w, f, sig)
        return f

    def rotbroad(w, f, vsini):      # to rotationally broaden the templates
        f = pyasl.rotBroad(w, f, 0.6, float(vsini))
        return f

    def rebin(w, f, samp):      # to rebin the templates to match the sampling of the observed spectrum
        f, bin_edges, binnumber = scipy.stats.binned_statistic(w, f, statistic = 'mean', bins = (w[-1] - w[0]) / samp)
        bin_width = bin_edges[1] - bin_edges[0]
        w = bin_edges[1:] - bin_width/2
        return w, f

    def process_template(w, f, sig=sigma, samp=sampling):
        if sigma != None:
            f = smooth(w, f, sig)
        if sampling != None:
            w, f = rebin(w, f, samp)
        file = open(templates_dir + '/'+ os.path.splitext(os.path.basename(i))[0] + '_processed', "w")
        for index in range(len(w)):
            file.write(str(w[index]) + " " + str(f[index]) + "\n")
        file.close()
        return w, f

    def process_template_vsini(w, f, vsini, sig=sigma, samp=sampling):
        f = rotbroad(w, f, vsini)
        if sigma != None:
            f = smooth(w, f, sig)
        if sampling != None:
            w, f = rebin(w, f, samp)
        file = open(templates_dir + '/' + os.path.splitext(os.path.basename(i))[0] + '_processed_vsini' + str(int(vsini)), "w")
        for index in range(len(w)):
            file.write(str(w[index]) + " " + str(f[index]) + "\n")
        file.close()
        return w, f

    def write_to_grid(templatename, f, t_ind=temp_ind, l_ind=logg_ind, v_ind=vsini_ind, vsini0=None):
        # finding the teff and logg information from the template name, to write to corresponding grid point
        t1 = float(templatename[temp_ind[0]:temp_ind[1]])
        l1 = float(templatename[logg_ind[0]:logg_ind[1]])
        if vsini0==None:
            v1 = float(templatename[vsini_ind[0]:vsini_ind[1]])
        else: 
            v1 = float(vsini0)

        print('Teff: ' + str(t1) + ', logg: ' + str(l1) + ', vsini: ' + str(v1))

        d[np.where(Teff == t1)[0][0], np.where(logg == l1)[0][0], np.where(vsini == v1)[0][0]] = f

    t_wavelength_0, t_flux_0 = np.genfromtxt(templatelist[0], unpack=True, usecols=(wav_ind, flux_ind), dtype=None, encoding=None)
    t_wavelength_0, t_flux_0 = restrict_range(t_wavelength_0, t_flux_0)
    if sampling != None:
        t_wavelength_0, t_flux_0 = rebin(t_wavelength_0, t_flux_0, sampling)

    d = np.zeros((len(Teff), len(logg), len(vsini), len(t_wavelength_0)))
    d_filled = 0

    for i in templatelist:
        print('Processing: ' + i)
        t_wavelength, t_flux = np.genfromtxt(i, unpack=True, usecols=(wav_ind, flux_ind), dtype=None, encoding=None)
        t_wavelength, t_flux = restrict_range(t_wavelength, t_flux)
        if rotbroads == None:
            _, t_flux = process_template(t_wavelength, t_flux, sigma, sampling)
            print('template processed, writing to grid at:')
            write_to_grid(i, t_flux)
            d_filled += 1
            print('grid is ' + str("{0:.1f}".format(float(d_filled)/float(d_size) * 100.)) + ' percent full')



        else:
            print('processing for vsini: 0')
            _, t_flux_v0 = process_template(t_wavelength, t_flux, sig=sigma, samp=sampling)
            print('template processed, writing to grid at:')
            write_to_grid(i, t_flux_v0, vsini0=0.)
            d_filled += 1
            print('grid is ' + str("{0:.1f}".format(float(d_filled)/float(d_size) * 100.)) + ' percent full')

            for j in rotbroads:
                if j == 0:
                    continue
                print('processing for vsini: ' + str(j))
                _, t_flux_vj = process_template_vsini(t_wavelength, t_flux, j, sig=sigma, samp=sampling)
                print('template processed, writing to grid at:')
                write_to_grid(i, t_flux_vj, vsini0=j)
                d_filled += 1
                print('grid is ' + str("{0:.1f}".format(float(d_filled)/float(d_size) * 100.)) + ' percent full')


    np.save(templates_dir + 'template_grid.npy', d)

###############################################################################

#################################################

def modheader(hdul):
    hdul[1].header.comments['TTYPE1']='The number of the spectrum'
    hdul[1].header.comments['TTYPE2']='Fibre id'
    hdul[1].header.comments['TTYPE3']='WEAVE object name from coordinates'
    hdul[1].header.comments['TTYPE4']='Identifier of the target assigned by survey'
    hdul[1].header.comments['TTYPE5']='Mean fraction of proposed walker jumps'
    hdul[1].header.comments['TTYPE6']='Effective temperature'
    hdul[1].header.comments['TTYPE7']='1-sigma negative uncertainty on Teff'
    hdul[1].header.comments['TTYPE8']='1-sigma positive uncertainty on Teff'
    hdul[1].header.comments['TTYPE9']='Surface gravity log(g)'
    hdul[1].header.comments['TTYPE10']='1-sigma negative uncertainty on logg'
    hdul[1].header.comments['TTYPE11']='1-sigma positive uncertainty on logg'
    hdul[1].header.comments['TTYPE12']='Projected rotational velocity'
    hdul[1].header.comments['TTYPE13']='1-sigma negative uncertainty on vsini'
    hdul[1].header.comments['TTYPE14']='1-sigma positive uncertainty on vsini'
    hdul[1].header.comments['TTYPE15']='Radial/line-of-sight velocity'
    hdul[1].header.comments['TTYPE16']='1-sigma negative uncertainty on RV'
    hdul[1].header.comments['TTYPE17']='1-sigma positive uncertainty on RV'
    hdul[1].header.comments['TTYPE18']='Slope of mapping function '
    hdul[1].header.comments['TTYPE19']='1-sigma negative uncertainty on slope'
    hdul[1].header.comments['TTYPE20']='1-sigma positive uncertainty on slope'
    hdul[1].header.comments['TTYPE21']='Intercept of mapping function'
    hdul[1].header.comments['TTYPE22']='1-sigma negative uncertainty on intercept'
    hdul[1].header.comments['TTYPE23']='1-sigma positive uncertainty on intercept'


    hdul[1].header.comments['TFORM1']='data format of field: integer'
    hdul[1].header.comments['TFORM2']='data format of field: integer'
    hdul[1].header.comments['TFORM3']='data format of field: ASCII Character'
    hdul[1].header.comments['TFORM4']='data format of field: ASCII Character'
    hdul[1].header.comments['TFORM5']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM6']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM7']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM8']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM9']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM10']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM11']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM12']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM13']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM14']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM15']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM16']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM17']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM18']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM19']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM20']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM21']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM22']='data format of field: 4-byte REAL'
    hdul[1].header.comments['TFORM23']='data format of field: 4-byte REAL'

    hdul[1].header.set('TDISP1','I4',after='TFORM1')
    hdul[1].header.set('TDISP2','I4',after='TFORM2')
    hdul[1].header.set('TDISP3','A20',after='TFORM3')
    hdul[1].header.set('TDISP4','A30',after='TFORM4')
    hdul[1].header.set('TDISP5','F7.3',after='TFORM5')
    hdul[1].header.set('TDISP6','F7.3',after='TFORM6')
    hdul[1].header.set('TDISP7','F7.3',after='TFORM7')
    hdul[1].header.set('TDISP8','F7.3',after='TFORM8')
    hdul[1].header.set('TDISP9','F7.3',after='TFORM9')
    hdul[1].header.set('TDISP10','F7.3',after='TFORM10')
    hdul[1].header.set('TDISP11','F7.3',after='TFORM11')
    hdul[1].header.set('TDISP12','F7.3',after='TFORM12')
    hdul[1].header.set('TDISP13','F7.3',after='TFORM13')
    hdul[1].header.set('TDISP14','F7.3',after='TFORM14')
    hdul[1].header.set('TDISP15','F7.3',after='TFORM15')
    hdul[1].header.set('TDISP16','F7.3',after='TFORM16')
    hdul[1].header.set('TDISP17','F7.3',after='TFORM17')
    hdul[1].header.set('TDISP18','F7.3',after='TFORM18')
    hdul[1].header.set('TDISP19','F7.3',after='TFORM19')
    hdul[1].header.set('TDISP20','F7.3',after='TFORM20')
    hdul[1].header.set('TDISP21','F7.3',after='TFORM21')
    hdul[1].header.set('TDISP22','F7.3',after='TFORM22')
    hdul[1].header.set('TDISP23','F7.3',after='TFORM23')


    hdul[1].header.set('TUCD1','meta.id',after='TDISP1')
    hdul[1].header.set('TUCD2','meta.id',after='TDISP2')
    hdul[1].header.set('TUCD3','meta.id;meta.main',after='TDISP3')
    hdul[1].header.set('TUCD4','meta.id',after='TDISP4')
    hdul[1].header.set('TUCD5','obs.param',after='TDISP5')
    hdul[1].header.set('TUCD6','phys.temperature.effective',after='TDISP6')
    hdul[1].header.set('TUCD7','stat.error;phys.temperature.effective',after='TDISP7')
    hdul[1].header.set('TUCD8','stat.error;phys.temperature.effective',after='TDISP8')
    hdul[1].header.set('TUCD9','phys.gravity',after='TDISP9')
    hdul[1].header.set('TUCD10','stat.error;phys.gravity',after='TDISP10')
    hdul[1].header.set('TUCD11','stat.error;phys.gravity',after='TDISP11')
    hdul[1].header.set('TUCD12','phys.veloc.rotat',after='TDISP12')
    hdul[1].header.set('TUCD13','stat.error;phys.veloc.rotat',after='TDISP13')
    hdul[1].header.set('TUCD14','stat.error;phys.veloc.rotat',after='TDISP14')
    hdul[1].header.set('TUCD15','spect.dopplerVeloc',after='TDISP15')
    hdul[1].header.set('TUCD16','stat.error;spect.dopplerVeloc',after='TDISP16')
    hdul[1].header.set('TUCD17','stat.error;spect.dopplerVeloc',after='TDISP17')
    hdul[1].header.set('TUCD18','obs.param',after='TDISP18')
    hdul[1].header.set('TUCD19','stat.error;obs.param',after='TDISP19')
    hdul[1].header.set('TUCD20','stat.error;obs.param',after='TDISP20')
    hdul[1].header.set('TUCD21','obs.param',after='TDISP21')
    hdul[1].header.set('TUCD22','stat.error;obs.param',after='TDISP22')
    hdul[1].header.set('TUCD23','stat.error;obs.param',after='TDISP23')


    hdul[1].header.set('TUNIT6','K',after='TUCD6')
    hdul[1].header.set('TUNIT7','K',after='TUCD7')
    hdul[1].header.set('TUNIT8','K',after='TUCD8')
    hdul[1].header.set('TUNIT12','km/s',after='TUCD12')
    hdul[1].header.set('TUNIT13','km/s',after='TUCD13')
    hdul[1].header.set('TUNIT14','km/s',after='TUCD14')
    hdul[1].header.set('TUNIT15','km/s',after='TUCD15')
    hdul[1].header.set('TUNIT16','km/s',after='TUCD16')
    hdul[1].header.set('TUNIT17','km/s',after='TUCD17')

    hdul[1].header.comments['TDISP1']='Display format for column'
    hdul[1].header.comments['TDISP2']='Display format for column'
    hdul[1].header.comments['TDISP3']='Display format for column'
    hdul[1].header.comments['TDISP4']='Display format for column'
    hdul[1].header.comments['TDISP5']='Display format for column'
    hdul[1].header.comments['TDISP6']='Display format for column'
    hdul[1].header.comments['TDISP7']='Display format for column'
    hdul[1].header.comments['TDISP8']='Display format for column'
    hdul[1].header.comments['TDISP9']='Display format for column'
    hdul[1].header.comments['TDISP10']='Display format for column'
    hdul[1].header.comments['TDISP11']='Display format for column'
    hdul[1].header.comments['TDISP12']='Display format for column'
    hdul[1].header.comments['TDISP13']='Display format for column'
    hdul[1].header.comments['TDISP14']='Display format for column'
    hdul[1].header.comments['TDISP15']='Display format for column'
    hdul[1].header.comments['TDISP16']='Display format for column'
    hdul[1].header.comments['TDISP17']='Display format for column'
    hdul[1].header.comments['TDISP18']='Display format for column'
    hdul[1].header.comments['TDISP19']='Display format for column'
    hdul[1].header.comments['TDISP20']='Display format for column'
    hdul[1].header.comments['TDISP21']='Display format for column'
    hdul[1].header.comments['TDISP22']='Display format for column'
    hdul[1].header.comments['TDISP23']='Display format for column'

    hdul[1].header.comments['TUCD1']='UCD for column'
    hdul[1].header.comments['TUCD2']='UCD for column'
    hdul[1].header.comments['TUCD3']='UCD for column'
    hdul[1].header.comments['TUCD4']='UCD for column'
    hdul[1].header.comments['TUCD5']='UCD for column'
    hdul[1].header.comments['TUCD6']='UCD for column'
    hdul[1].header.comments['TUCD7']='UCD for column'
    hdul[1].header.comments['TUCD8']='UCD for column'
    hdul[1].header.comments['TUCD9']='UCD for column'
    hdul[1].header.comments['TUCD10']='UCD for column'
    hdul[1].header.comments['TUCD11']='UCD for column'
    hdul[1].header.comments['TUCD12']='UCD for column'
    hdul[1].header.comments['TUCD13']='UCD for column'
    hdul[1].header.comments['TUCD14']='UCD for column'
    hdul[1].header.comments['TUCD15']='UCD for column'
    hdul[1].header.comments['TUCD16']='UCD for column'
    hdul[1].header.comments['TUCD17']='UCD for column'
    hdul[1].header.comments['TUCD18']='UCD for column'
    hdul[1].header.comments['TUCD19']='UCD for column'
    hdul[1].header.comments['TUCD20']='UCD for column'
    hdul[1].header.comments['TUCD21']='UCD for column'
    hdul[1].header.comments['TUCD22']='UCD for column'
    hdul[1].header.comments['TUCD23']='UCD for column'

    hdul[1].header.comments['TUNIT6']='physical unit of field'
    hdul[1].header.comments['TUNIT7']='physical unit of field'
    hdul[1].header.comments['TUNIT8']='physical unit of field'
    hdul[1].header.comments['TUNIT12']='physical unit of field'
    hdul[1].header.comments['TUNIT13']='physical unit of field'
    hdul[1].header.comments['TUNIT14']='physical unit of field'
    hdul[1].header.comments['TUNIT15']='physical unit of field'
    hdul[1].header.comments['TUNIT16']='physical unit of field'
    hdul[1].header.comments['TUNIT17']='physical unit of field'


    hdul[1].header.set('TPROP1',0,after='TUCD1')
    hdul[1].header.set('TPROP2',0,after='TUCD2')
    hdul[1].header.set('TPROP3',0,after='TUCD3')
    hdul[1].header.set('TPROP4',0,after='TUCD4')
    hdul[1].header.set('TPROP5',0,after='TUCD5')
    hdul[1].header.set('TPROP6',0,after='TUCD6')
    hdul[1].header.set('TPROP7',0,after='TUCD7')
    hdul[1].header.set('TPROP8',0,after='TUCD8')
    hdul[1].header.set('TPROP9',0,after='TUCD9')
    hdul[1].header.set('TPROP10',0,after='TUCD10')
    hdul[1].header.set('TPROP11',0,after='TUCD11')
    hdul[1].header.set('TPROP12',0,after='TUCD12')
    hdul[1].header.set('TPROP13',0,after='TUCD13')
    hdul[1].header.set('TPROP14',0,after='TUCD14')
    hdul[1].header.set('TPROP15',0,after='TUCD15')
    hdul[1].header.set('TPROP16',0,after='TUCD16')
    hdul[1].header.set('TPROP17',0,after='TUCD17')
    hdul[1].header.set('TPROP18',0,after='TUCD18')
    hdul[1].header.set('TPROP19',0,after='TUCD19')
    hdul[1].header.set('TPROP20',0,after='TUCD20')
    hdul[1].header.set('TPROP21',0,after='TUCD21')
    hdul[1].header.set('TPROP22',0,after='TUCD22')
    hdul[1].header.set('TPROP23',0,after='TUCD23')


    hdul[1].header.comments['TPROP1']='Public column'
    hdul[1].header.comments['TPROP2']='Public column'
    hdul[1].header.comments['TPROP3']='Public column'
    hdul[1].header.comments['TPROP4']='Public column'
    hdul[1].header.comments['TPROP5']='Public column'
    hdul[1].header.comments['TPROP6']='Public column'
    hdul[1].header.comments['TPROP7']='Public column'
    hdul[1].header.comments['TPROP8']='Public column'
    hdul[1].header.comments['TPROP9']='Public column'
    hdul[1].header.comments['TPROP10']='Public column'
    hdul[1].header.comments['TPROP11']='Public column'
    hdul[1].header.comments['TPROP12']='Public column'
    hdul[1].header.comments['TPROP13']='Public column'
    hdul[1].header.comments['TPROP14']='Public column'
    hdul[1].header.comments['TPROP15']='Public column'
    hdul[1].header.comments['TPROP16']='Public column'
    hdul[1].header.comments['TPROP17']='Public column'
    hdul[1].header.comments['TPROP18']='Public column'
    hdul[1].header.comments['TPROP19']='Public column'
    hdul[1].header.comments['TPROP20']='Public column'
    hdul[1].header.comments['TPROP21']='Public column'
    hdul[1].header.comments['TPROP22']='Public column'
    hdul[1].header.comments['TPROP23']='Public column'


    hdul[1].header.set('TDMIN1',1,after='TUCD1')
    hdul[1].header.set('TDMIN2',1,after='TUCD2')
    hdul[1].header.set('TDMIN5',0,after='TUCD5')
    hdul[1].header.set('TDMIN6',0,after='TUCD6')
    hdul[1].header.set('TDMIN7',0,after='TUCD7')
    hdul[1].header.set('TDMIN8',0,after='TUCD8')
    hdul[1].header.set('TDMIN9',0,after='TUCD9')
    hdul[1].header.set('TDMIN10',0,after='TUCD10')
    hdul[1].header.set('TDMIN11',0,after='TUCD11')
    hdul[1].header.set('TDMIN12',0,after='TUCD12')
    hdul[1].header.set('TDMIN13',0,after='TUCD13')
    hdul[1].header.set('TDMIN14',0,after='TUCD14')
    hdul[1].header.set('TDMIN16',0,after='TUCD16')
    hdul[1].header.set('TDMIN17',0,after='TUCD17')
    hdul[1].header.set('TDMIN18',-1,after='TUCD18')
    hdul[1].header.set('TDMIN19',0,after='TUCD19')
    hdul[1].header.set('TDMIN20',0,after='TUCD20')
    hdul[1].header.set('TDMIN22',0,after='TUCD22')
    hdul[1].header.set('TDMIN23',0,after='TUCD23')

    hdul[1].header.comments['TDMIN1']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN2']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN5']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN6']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN7']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN8']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN9']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN10']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN11']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN12']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN13']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN14']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN16']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN17']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN18']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN19']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN20']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN22']='Minimum value expected for field'
    hdul[1].header.comments['TDMIN23']='Minimum value expected for field'


    hdul[1].header.set('TDMAX1',960,after='TDMIN1')
    hdul[1].header.set('TDMAX2',1100,after='TDMIN2')
    hdul[1].header.set('TDMAX5',1,after='TDMIN5')
    hdul[1].header.set('TDMAX18',1,after='TDMIN18')

    hdul[1].header.comments['TDMAX1']='Maximum value expected for field'
    hdul[1].header.comments['TDMAX2']='Maximum value expected for field'
    hdul[1].header.comments['TDMAX5']='Maximum value expected for field'
    hdul[1].header.comments['TDMAX18']='Maximum value expected for field'



    hdul[2].header.comments['TTYPE1']='The number of the spectrum'
    hdul[2].header.comments['TTYPE2']='Fibre id'
    hdul[2].header.comments['TTYPE3']='WEAVE object name from coordinates'
    hdul[2].header.comments['TTYPE4']='Identifier of the target assigned by survey'
    hdul[2].header.comments['TTYPE5']='Input Spectrum normalised'
    hdul[2].header.comments['TTYPE6']='Fit Spectrum normalised'

    hdul[2].header.comments['TFORM1']='data format of field: integer'
    hdul[2].header.comments['TFORM2']='data format of field: integer'
    hdul[2].header.comments['TFORM3']='data format of field: ASCII Character'
    hdul[2].header.comments['TFORM4']='data format of field: ASCII Character'
    hdul[2].header.comments['TFORM5']='data format of field: 4-byte REAL'
    hdul[2].header.comments['TFORM6']='data format of field: 4-byte REAL'

    hdul[2].header.set('TDISP1','I4',after='TFORM1')
    hdul[2].header.set('TDISP2','I4',after='TFORM2')
    hdul[2].header.set('TDISP3','A20',after='TFORM3')
    hdul[2].header.set('TDISP4','A30',after='TFORM4')

    hdul[2].header.comments['TDISP1']='Display format for column'
    hdul[2].header.comments['TDISP2']='Display format for column'
    hdul[2].header.comments['TDISP3']='Display format for column'
    hdul[2].header.comments['TDISP4']='Display format for column'

    hdul[2].header.set('TUCD1','meta.id',after='TDISP1')
    hdul[2].header.set('TUCD2','meta.id',after='TDISP2')
    hdul[2].header.set('TUCD3','meta.id;meta.main',after='TDISP3')
    hdul[2].header.set('TUCD4','meta.id',after='TDISP4')
    hdul[2].header.set('TUCD5','phot.count',after='TFORM5')
    hdul[2].header.set('TUCD6','phot.count',after='TFORM6')
    hdul[2].header.comments['TUCD1']='UCD for column'
    hdul[2].header.comments['TUCD2']='UCD for column'
    hdul[2].header.comments['TUCD3']='UCD for column'
    hdul[2].header.comments['TUCD4']='UCD for column'
    hdul[2].header.comments['TUCD5']='UCD for column'
    hdul[2].header.comments['TUCD6']='UCD for column'

    hdul[2].header.set('TUNIT5','counts','physical unit of field',after='TUCD5')
    hdul[2].header.set('TUNIT6','counts','physical unit of field',after='TUCD6')

    hdul[2].header.set('TDMIN1',1,after='TUCD1')
    hdul[2].header.set('TDMIN2',1,after='TUCD2')
    hdul[2].header.comments['TDMIN1']='Minimum value expected for field'
    hdul[2].header.comments['TDMIN2']='Minimum value expected for field'
    hdul[2].header.set('TDMAX1',960,after='TUCD1')
    hdul[2].header.set('TDMAX2',1100,after='TUCD2')
    hdul[2].header.comments['TDMAX1']='Maximum value expected for field'
    hdul[2].header.comments['TDMAX2']='Maximum value expected for field'
    hdul[2].header.set('TPROP1',0,after='TUCD1')
    hdul[2].header.set('TPROP2',0,after='TUCD2')
    hdul[2].header.set('TPROP3',0,after='TUCD3')
    hdul[2].header.set('TPROP4',0,after='TUCD4')
    hdul[2].header.set('TPROP5',0,after='TUCD5')
    hdul[2].header.set('TPROP6',0,after='TUCD6')
    hdul[2].header.comments['TPROP1']='Public column'
    hdul[2].header.comments['TPROP2']='Public column'
    hdul[2].header.comments['TPROP3']='Public column'
    hdul[2].header.comments['TPROP4']='Public column'
    hdul[2].header.comments['TPROP5']='Public column'
    hdul[2].header.comments['TPROP6']='Public column'

    hdul[2].header.set('TCTYP5','AWAV','Coordinate type',after='TPROP5')
    hdul[2].header.set('TCTYP6','AWAV','Coordinate type',after='TPROP6')
    hdul[2].header.set('TCUNI5','Angstrom','Coordinate unit',after='TCTYP5')
    hdul[2].header.set('TCUNI6','Angstrom','Coordinate unit',after='TCTYP6')
    hdul[2].header.set('TCRPX5',1,'Pixel coordinate of the reference point',after='TCUNI5')
    hdul[2].header.set('TCRPX6',1,'Pixel coordinate of the reference point',after='TCUNI6')
    hdul[2].header.set('TCRVL5',8470.25,'Coordinate value at reference point',after='TCRPX5')
    hdul[2].header.set('TCRVL6',8470.25,'Coordinate value at reference point',after='TCRPX6')
    hdul[2].header.set('TCDLT5',0.25,'Coordinate increment at reference point',after='TCRVL5')
    hdul[2].header.set('TCDLT6',0.25,'Coordinate increment at reference point',after='TCRVL6')

#   hdul[2].header['XTENSION']='BINSPEC'
    return hdul
#################################################

def createDAT(outpath, headname, infiles, aps_ids, overwrite):
    tab = open(outpath+'/amy_wd/'+ headname+'_AMY.dat', "w")
    tab.write("Date of creation: {}\n".format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')))
    tab.write("\n")
    tab.write("######## OB info and inputs ###########\n")
    tab.write("OB: {}\n".format(os.path.splitext(os.path.basename(infiles[0]))[0].split('_')[1]))
    tab.write("input file: {}\n".format(infiles[0]))
    tab.write("output directory: {}\n".format(outpath))
    tab.write("output file: {}\n".format(headname+'_AMY.fits'))
    tab.write("targlist: {}\n".format(aps_ids))
    tab.write("apsclassification: {}\n".format("None"))
    tab.write("override: {}\n".format(overwrite))

    tab.write("\n")
    tab.write("######## libraries version ###########\n")

    tab.write("numpy version: {}\n".format(np.version.version))
    tab.write("astropy version: {}\n".format(astropy.version.version))
    tab.write("scipy version: {}\n".format(scipy.version.version))
    tab.write("matplotlib version: {}\n".format(matplotlib.__version__))
#   tab.write(": {}\n".format())
    tab.write("\n")
    tab.write("######## parameters.py file: ###############\n")
    tab.close()
    # os.system('more parameters.py >>{}'.format(outpath+'/amy_wd/'+ headname+'_AMY.dat'))


#################################################
def make_output_fits(outpath, headname, infiles, aps_ids, overwrite=True):

    #PHU
    hdr = fits.Header()
    hdr['COMMENT'] = "WEAVE Contributed Software: AMY"
    hdr['DATAMVER'] = 7.60
    hdr.comments['DATAMVER']='WEAVE Data Model Version'
    hdr['CS_CODE'] = 'AMY'
    hdr.comments['CS_CODE']='CS code name'
    hdr['CS_VER'] = 'May20'
    hdr.comments['CS_VER']='CS version '
    hdr['CS_NME1'] = 'Amy, Maria'
    hdr.comments['CS_NME1']='CS author forename'
    hdr['CS_NME2'] = 'Harris, Monguio'
    hdr.comments['CS_NME2']='CS author surname(s)'
    hdr['CS_MAIL'] = 'm.monguio@icc.ub.edu'
    hdr.comments['CS_MAIL']='CS author email'
    hdr['PROV1001'] = os.path.splitext(os.path.basename(infiles[0]))[0]+'.fit'
    hdr.comments['PROV1001']='L1 file used'
    hdr['PROV2001'] = ''
    hdr.comments['PROV2001']='L2 file used'
    hdr['DATETIME'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    hdr.comments['DATETIME']='Datetime file created'
    print(hdr['PROV1001'],hdr['DATETIME'])
    empty_primary = fits.PrimaryHDU(header=hdr)


    #bin,PARAM
    output_files = glob.glob(outpath + '/amy_wd/*_results')
    all_res = []
    for i in output_files:
        with open(i) as outf:
            all_res.append(outf.read().split())
    t = Table(rows=all_res, names=('Nspec', 'FIBREID', 'CNAME', 'TARGID', 'AMY_ACCEPTANCE', 'AMY_TEFF', 'AMY_TEFF_minus', 'AMY_TEFF_plus', 'AMY_LOGG', 'AMY_LOGG_minus', 'AMY_LOGG_plus', 'AMY_VSINI', 'vsini_minus', 'AMY_VSINI_plus', 'AMY_RV', 'AMY_RV_minus', 'AMY_RV_plus', 'AMY_SLOPE', 'AMY_SLOPE_minus', 'AMY_SLOPE_plus', 'AMY_INTERCEPT', 'AMY_INTERCEPT_minus', 'INTERCEPT_plus'))

    cols=['Nspec', 'FIBREID', 'CNAME', 'TARGID', 'AMY_ACCEPTANCE', 'AMY_TEFF', 'AMY_TEFF_minus','AMY_TEFF_plus', 'AMY_LOGG', 'AMY_LOGG_minus', 'AMY_LOGG_plus', 'AMY_VSINI','vsini_minus', 'AMY_VSINI_plus', 'AMY_RV', 'AMY_RV_minus', 'AMY_RV_plus', 'AMY_SLOPE','AMY_SLOPE_minus', 'AMY_SLOPE_plus', 'AMY_INTERCEPT', 'AMY_INTERCEPT_minus', 'INTERCEPT_plus']
    formats=['I','I','20A','30A','E','E','E','E','E','E','E', 'E','E','E','E','E','E','E','E','E','E','E','E']

    columns=[]
    for i in range(len(cols)):
        columns.append(fits.Column(name=cols[i], array=t[cols[i]],format=formats[i]))

    bintable = fits.BinTableHDU.from_columns(columns)
    


    #binspec
    output_files = glob.glob(outpath + '/amy_wd/*_spec')
    all_spec = []
    col0=[]
    col1=[]
    col2=[]
    col3=[]
    col4=[]
    col5=[]
    for i in output_files:
        with open(i) as outf:
            line=outf.read().split(';')
            col0.append(line[0])
            col1.append(line[1])
            col2.append(line[2])
            col3.append(line[3])
            col4.append(np.array(np.matrix(line[4])).ravel())
            col5.append(np.array(np.matrix(line[5])).ravel())
    t=[col0,col1,col2,col3,col4,col5]


    cols=['Nspec', 'FIBREID', 'CNAME', 'TARGID','AMY_SPECTRAIN','AMY_SPECTRAFIT']
    formats=['I','I','20A','30A','1880E','1880E']

    columns=[]
    for i in range(len(cols)):
        columns.append(fits.Column(name=cols[i], array=t[i],format=formats[i]))

    spectable = fits.BinTableHDU.from_columns(columns)
    
    hdul = fits.HDUList([empty_primary,bintable,spectable])
    hdul = modheader(hdul)

    hdul.writeto(outpath+'/'+headname+'_AMY.fits', overwrite=True, checksum=True)

    #ascii file with info for reproducibility:
    createDAT(outpath, headname, infiles, aps_ids, overwrite)


#################################################

def amy_serial_worker(infiles, outpath, headname, targsrvy= None, targclass = None, aps_ids=None, exclude_region = None, 
    templates_dir=None, template_flux=None,template_wave=None, wlranges=None , mproc=1, overwrite=True, config=None, 
    sens_corr=True, mask_gaps = True, safe_mask_gaps = True, tellurics=None, vacuum=False, fill_gap=False, arms_ratio=None, join_arms=True, progress_bar=True):

    startTime = datetime.now()


    ####### READING DATA into the APSOB format

    # Sky subtracted APS targets
    APSOBJ     = APSOB(infiles, skysub=True,  targsrvy= targsrvy, targclass = targclass, wlranges=wlranges, aps_ids=aps_ids, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms)

    # NSS (No Sky subtraction)
    APSOBJ_NSS = APSOB(infiles, skysub=False, targsrvy= targsrvy, targclass = targclass, wlranges = wlranges,  aps_ids=aps_ids, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms)


    # READ targets in the class
    targs_list=APSOBJ.data()
    targs_list_nss=APSOBJ_NSS.data()

    # Just a dummy check
    assert len(targs_list) == len(targs_list_nss), 'Something goes wrong with APSOB. Number of Sky-Subtracted and Non-Sky-Subtracted spectra must be identical'


    # Extract basic info from the APSOBBJ
    # Note, here we extract the wavelength info from the APSOB, as it could be different compare to the input wlranges (user input)
    wav_list = APSOBJ.wlranges()
    wav_list = [wav_item for wav_sublist in wav_list for wav_item in wav_sublist]
    min_wav = np.min(wav_list)
    max_wav = np.max(wav_list)



    #### Preparing templates and the parameter space

    # loading the template flux grid
    flux_data_all = np.load(templates_dir + template_flux)
    #loading the wavelength data

    templatewavelength = np.loadtxt(templates_dir + template_wave)
    # templatewavelength = t_wavelength_0

    template_min_wav = min_wav*(1.0 + (min(drv)/299792.458)) - 10.
    template_max_wav = max_wav*(1.0 + (max(drv)/299792.458)) + 10.

    templatemask = (templatewavelength > template_min_wav) & (templatewavelength < template_max_wav)
    flux_data = np.zeros((len(Teff), len(logg), len(vsini), len(templatewavelength[templatemask])))
    templatewavelength = templatewavelength[templatemask]

    for ii in range(len(Teff)):
        for jj in range(len(logg)):
            for kk in range(len(vsini)):
                flux_data[ii,jj,kk] = flux_data_all[ii,jj,kk][templatemask]

    ipo = interpolate.RegularGridInterpolator((Teff, logg, vsini), flux_data, method='linear')


    # define edges of parameter space
    teffmin, teffmax, loggmin, loggmax, vsinimin, vsinimax, rvmin, rvmax, slopemin, slopemax, interceptmin, interceptmax = min(Teff), max(Teff), min(logg), max(logg), min(vsini), max(vsini), min(drv), max(drv), min(slopes), max(slopes), min(intercepts), max(intercepts)

    ## INTERNAL FUNCTION #####################################

    def model(X, wavelength):
        i, j, k, l, m, n = X 
        # interpolating template grid with teff (i), logg (j), vsini (k) trial parameter
        templateflux = ipo([i, j, k])[0]
        # interpolating on wavelength axis for trial RV (l)
        fi = interpolate.interp1d(templatewavelength*(1.0 + l/299792.458), templateflux)
        return fi(wavelength)

    ## INTERNAL FUNCTION #####################################

    def lnprior(X):
        i, j, k, l, m, n = X
        # flat prior, edges should correspond to template grid
        if (teffmin <= i <= teffmax) & (loggmin <= j <= loggmax) & (vsinimin <= k <= vsinimax) & (rvmin <= l <= rvmax) & (slopemin <= m <= slopemax) & (interceptmin <= n <= interceptmax):
            return 0.0
        else:
            return -np.inf

    ## INTERNAL FUNCTION #####################################

    def lnlike(X, wavelength, flux, noisespec, mask):
        i, j, k, l, m, n = X
        z = m, n 
        f = np.poly1d(z)
        if exclude_region is None:
            return -(np.sum((flux - (model(X, wavelength)*f(wavelength)))**2/(2*PPRE*noisespec**2)))
        else:
            return -(np.sum((flux[mask] - (model(X, wavelength)*f(wavelength))[mask])**2/(2*PPRE*noisespec[mask]**2)))


    # Now loop over all available targets and ...
    outdict = OrderedDict()
    for tg_i, tg in enumerate(targs_list):
        # Extract basic indexing info for this target
        TARGID  = tg.targid
        FIBREID = tg.aps_id
        CNAME   = tg.cname
        NSPEC   = tg.meta[0]['NSPEC']


        # Make sure loop over targ_list and targs_list_nss properly works and provides consistent results
        assert targs_list_nss[tg_i].aps_id == tg.aps_id, 'sky-subtracted and non-sky-subtracted spectra are not consistent'


        # checking if spectrum file exists, and if result already written
        if os.path.exists(outpath + '/amy_wd/' + str(TARGID) + '_results'):
            if overwrite:
                os.remove(outpath + '/amy_wd/' + str(TARGID) + '_results')
                os.remove(outpath + '/amy_wd/' + str(TARGID) + '_spec')
            else:
                print('WARNING: result for '+ str(TARGID) +' already exists, skipping')
                return


        print("Processing target: %s" %(TARGID))
        targ_start = datetime.now()


        # extract data for this target
        wavelength  = tg.spectra[0].wave
        flux        = tg.spectra[0].flux
        flux_nss    = targs_list_nss[tg_i].spectra[0].flux


        # mask for excluding a wavelength region
        if exclude_region is not None:
            mask = np.zeros(len(tg.spectra[0].wave), dtype=bool)
            for line in exclude_region:
                mask |= (wavelength >= line[0]) & (wavelength <= line[1])
        else:
            mask = np.ones(len(wavelength), dtype=bool)


        # create corresponding noise spectrum
        noisespec = np.sqrt((2.0* flux_nss) - flux) 

        loglargs=[wavelength, flux, noisespec, mask]

        # choose initial walker positions

        pos = [[[np.random.choice(i) for i in ini_grid] for i in range(nwalkers)] for i in range(ntemps)]

        # initialise MCMC sampler
        sampler = PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=ndim, logl=lnlike, logp=lnprior, Tmax=np.inf, loglargs=[wavelength, flux, noisespec, mask])


        # run MCMC sampler for burn period

        print("running burn for %s" %(TARGID))
        pos, prob, state = sampler.run_mcmc(pos, burn, adapt=True, progress_bar=progress_bar)
        # reset sampler, run MCMC sampler for run period with walkers starting at their positions at the end of burn
        sampler.reset()

        print("running runs for %s" %(TARGID))
        sampler.run_mcmc(pos, runs, adapt=True, progress_bar=progress_bar)
        samples=sampler.chain[0, :, :, :].reshape((-1, ndim))

        ## plot walker paths
        #ylabels = ['Teff', 'logg', 'vsini', 'RV', 'slope', 'intercept']
        #for m in range(ndim):
        #    plt.subplot(ndim,1,m+1)
        #    plt.plot(sampler.chain[0,:,:,m].transpose(), alpha=0.2)
        #    plt.ylabel(ylabels[m])
        #plt.xlabel('Step')
        #plt.savefig(outpath + '/amy_figs/' + t + '_walkers.png', bbox_inches='tight')
        #plt.close()

        # calculate 16th, 50th, 84th quantiles of the parameter samples
        quantiles = mquantiles(samples, prob=[0.16, 0.50, 0.84], axis=0)

        # targetname = os.path.splitext(os.path.basename(TARGID))[0]
        acceptance_r = np.mean(sampler.acceptance_fraction)

        # print acceptance fraction, should be between 0.2-0.5 for efficient sampling
        print("Mean acceptance fraction: {0:.3f}"
                    .format(acceptance_r))

        # The parameter results
        Teff_r = quantiles[1][0]
        Teffminus_r = quantiles[1][0] - quantiles[0][0]
        Teffplus_r = quantiles[2][0] - quantiles[1][0]
        logg_r = quantiles[1][1]
        loggminus_r = quantiles[1][1] - quantiles[0][1]
        loggplus_r = quantiles[2][1] - quantiles[1][1]
        vsini_r = quantiles[1][2]
        vsiniminus_r = quantiles[1][2] - quantiles[0][2]
        vsiniplus_r = quantiles[2][2] - quantiles[1][2]
        RV_r = quantiles[1][3]
        RVminus_r = quantiles[1][3] - quantiles[0][3]
        RVplus_r = quantiles[2][3] - quantiles[1][3]
        slope_r = quantiles[1][4]
        slopeminus_r = quantiles[1][4] - quantiles[0][4]
        slopeplus_r = quantiles[2][4] - quantiles[1][4]
        intercept_r = quantiles[1][5]
        interceptminus_r = quantiles[1][5] - quantiles[0][5]
        interceptplus_r = quantiles[2][5] - quantiles[1][5]

        #fig = corner.corner(samples, quantiles=[0.16, 0.50, 0.84], labels=['Teff', 'log(g)', 'vsini', 'RV', 'slope', 'intercept'], show_titles=True, title_kwargs={"fontsize": 10}, plot_datapoints=True, plot_contours=True, auto_bars=True, data_kwargs={"alpha": 0.005})
        # fig.savefig(outpath + t + '_cornerplot.png', bbox_inches='tight')
        # plt.close()

        # parameters of best fit (for plotting)
        Xp = Teff_r, logg_r, vsini_r, RV_r, slope_r, intercept_r
        fp = np.poly1d(Xp[-2:])
        fitp = fp(wavelength)

        ## plot spectrum and best-fit
        #plt.plot(wavelength, flux, wavelength, model(Xp, wavelength) * fitp)
        #if exclude_region is not None:
        #    for n,line in enumerate(exclude_region):
        #        if n == 0:
        #            plt.vlines([line[1]], flux.min()*0.8, flux.max()*1.2, colors='r', linestyles='dashed')
        #        elif n == (len(exclude_region)-1):
        #            plt.vlines([line[0]], flux.min()*0.8, flux.max()*1.2, colors='r', linestyles='dashed')
        #        else:
        #            plt.vlines([line[0], line[1]], flux.min()*0.8, flux.max()*1.2, colors='r', linestyles='dashed')

        #plt.xlim(min_wav, max_wav)
        #plt.ylim(flux.min()*0.8, flux.max()*1.2)
        #plt.xlabel(r'Wavelength ($\AA$)')
        #plt.ylabel('Calibrated counts')
        #plt.savefig(outpath + '/amy_figs/' + t + '_spectrum.png', bbox_inches='tight')
        #plt.close()

        ## plot mapping function
        #plt.plot(wavelength, fitp, wavelength, flux / model(Xp, wavelength))
        #plt.xlim(min_wav, max_wav)
        #plt.xlabel(r'Wavelength ($\AA$)')
        #plt.ylabel('Calibrated counts')
        #plt.savefig(outpath + '/amy_figs/' + t + '_mapping_function.png', bbox_inches='tight')
        #plt.close()

        # write results
        tab = open(outpath + '/amy_wd/' + str(TARGID) +  '_results', "w")
        tab.write(str(NSPEC) + " " + str(FIBREID) + " " + str(CNAME) + " " + str(TARGID) + " " + str(acceptance_r) + " " + str(Teff_r) + " " + str(Teffminus_r) + " " + str(Teffplus_r) + " " + str(logg_r) + " " + str(loggminus_r) + " " + str(loggplus_r) + " " + str(vsini_r) + " " + str(vsiniminus_r) + " " + str(vsiniplus_r) + " " + str(RV_r) + " " + str(RVminus_r) + " " + str(RVplus_r) + " " + str(slope_r) + " " + str(slopeminus_r) + " " + str(slopeplus_r) + " " + str(intercept_r) + " " + str(interceptminus_r) + " " + str(interceptplus_r) + "\n")
        tab.close()

        tab = open(outpath + '/amy_wd/' + str(TARGID) +  '_spec', "w")
        tab.write("{0} ; {1} ; {2}; {3} ; {4} ; {5}\n".format(NSPEC,FIBREID,CNAME,str(TARGID),list(flux),list(model(Xp, wavelength) * fitp)))
        tab.close()


        targ_end = datetime.now() - targ_start


    # Write the results into the final fits files
#    make_output_fits(outpath, headname, infiles, aps_ids, overwrite=True)
#################################################


def amy_many(infiles, outpath, headname, targsrvy= None, targclass = None, aps_ids=None, exclude_region = None, 
    templates_dir=None, template_flux=None,template_wave=None, wlranges=None , mproc=1, overwrite=True, config=None, 
    sens_corr=True, mask_gaps = True, safe_mask_gaps = True, tellurics=None, vacuum=False, fill_gap=False, arms_ratio=None, join_arms=True, progress_bar=True):

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
    # Here we first run a dummy test using aps_ids_class, with nthreads=1 and threadid=0

#    aps_ids_satis_class = aps_ids_class(infiles, ['STAR'], aps_ids=aps_ids, nthreads=1, threadid=0)
    
#    ## Check if aps_ids_satis_class is OK or not. If not, it will force the code to exit (back to one level up)
#    if aps_ids_satis_class is None:
#        return
    
    ## After the check above (if aps_ids_satis_class is None: return) we made sure aps_ids_satis_class is a list with > 0 elements
    ## Now we just want to make sure that nthreads and number of targets are consistent.
    ## And we do not assign an empty job to a thread
#    if len(aps_ids_satis_class) < nthreads: 
#        nthreads = len(aps_ids_satis_class)
#        print('Update nthreads param to %d'%(nthreads))

    
    nthreads=mproc
    if nthreads > 1:
        parallel = True
    else:
        parallel = False

    
    # Updated by ALireza on 30-Jan-2020
    res = []

    if (not overwrite) :
        print('skipping, products already exist')
        sys.exit()


    if parallel:
            APSOBJ = APSOB(infiles, skysub=True,  targsrvy= targsrvy, targclass = targclass, wlranges=wlranges, aps_ids=aps_ids, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms)
            
            targs_list=APSOBJ.data()
            if len(targs_list) < nthreads: 
                nthreads = len(targs_list)
            print('Update nthreads param to %d'%(nthreads))

            pool = mp.Pool(nthreads)

            for tg_i, tg in enumerate(targs_list):
                # Extract basic indexing info for this target
                TARGID  = tg.targid
                FIBREID = tg.aps_id
                res.append( pool.apply_async(
#               res.append(pool.map_async(
                proc_weave_wrapper,(infiles, outpath, headname),{'targsrvy': targsrvy, 'targclass': targclass, 'aps_ids':FIBREID, 'exclude_region':exclude_region, 'templates_dir':templates_dir,'template_flux':template_flux,'template_wave':template_wave, 'wlranges':wlranges , 'mproc':mproc, 'overwrite':overwrite, 'config':config, 'sens_corr':sens_corr, 'mask_gaps': mask_gaps, 'safe_mask_gaps': safe_mask_gaps, 'tellurics':tellurics, 'vacuum':vacuum,         'fill_gap':fill_gap, 'arms_ratio':arms_ratio, 'join_arms':join_arms, 'progress_bar':progress_bar}))
               

    if parallel:
            pool.close()
            pool.join()

    else:
         amy_serial_worker(infiles, outpath, headname , targsrvy= targsrvy, targclass = targclass, aps_ids=aps_ids, exclude_region=exclude_region,         templates_dir=templates_dir,template_flux=template_flux,template_wave=template_wave, wlranges=wlranges , mproc=mproc, overwrite=overwrite, config=config,         sens_corr=sens_corr, mask_gaps = mask_gaps, safe_mask_gaps = safe_mask_gaps, tellurics=tellurics, vacuum=vacuum,         fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, progress_bar=progress_bar)



    make_output_fits(outpath, headname, infiles, aps_ids, overwrite=True)
    
    
def proc_weave_wrapper(*args, **kwargs):
    try:
        ret = amy_serial_worker(*args, **kwargs)
        return ret
    except:
        print('failed with these arguments', args, kwargs)
        raise


proc_weave_wrapper.__doc__ = amy_serial_worker.__doc__
#################################################################    
def track_job(job, update_interval=3):
    while job._number_left > 0:
        print("Tasks remaining = {0}".format(
        job._number_left * job._chunksize))
        time.sleep(update_interval)

################################################################################

def amy_runner(options=None):

    parser = build_common_parser(
        description=None,
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "output"],
        # this script has no --catdir/--caldir/--configdir at all, so the
        # "caldirs" group is simply not included above.
        overrides={
            # this script's own wording differs from the canonical text,
            # preserved verbatim rather than silently switched to canonical
            "mask_aps_ids": {"help": "[Not In used] comma-separated list of APS_IDs to be masked"},
            "area": {"help": "[Not In used] The area in [RA(deg), DEC(deg), Radius(arcsec)] to be considered in analysis"},
            "mask_areas": {"help": "[Not In used] The multi area(s) in [RA(deg), DEC(deg), Radius(arcsec)] to be excluded from analysis"},
            "vacuum": {"default": False},
            "outpath": {"help": "Directory to keep WEAVE_RVS outputs"},
            "overwrite": {"default": True,
                          "help": "If enabled the code will overwrite the existing products, otherwise it will skip them"},
        },
        extra_args=[
            (("--exclude_region",), dict(type=none_or_str, default=None,
                required=False, help="The desired EXCLUDE wavelength ranges within the range set by wlranges", nargs='*')),
            (("--config",), dict(help='The filename of the configuration file (RESERVED: not applicable yet!!!)',
                type=none_or_str, default=None, required=False)),
            (("--templates_dir",), dict(
                help='Full path of the directory contains templates', type=none_or_str, default=None, required=True)),
            (("--template_flux",), dict(
                help='flux template filename', type=none_or_str, default=None, required=True)),
            (("--template_wave",), dict(
                help='wavelength template filename', type=none_or_str, default=None, required=True)),
            (("--progress_bar",), dict(
                help='Set True if you like to have a progress bar, showing the progress of the code (Turn it off(False) for batch processing )',
                type=str2bool, default=True)),
            (("--mproc",), dict(help='Number of threads for the fits',
                type=int, default=1)),
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


    # resolve_common_args iterates over args.wlranges directly (rather than
    # indexing it by infiles-count) — functionally identical when the two
    # lists have matching lengths, as they always should. Everything else
    # (l1_fileinfo normalization, wlranges/arms_ratio write-back,
    # join_arms<2 correction, arms_ratio length assert) is unchanged.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio

    # --- exclude_region: script-specific, kept exactly as before ---
    exclude_region = None
    if args.exclude_region[0] is not None:

        exclude_region=[]
        for i in range(len(args.exclude_region)):
            exclude_region.append([float(x) for x in args.exclude_region[i].split(",")])

        for exr_pain in exclude_region:
            assert len(exclude_region) == 2 , 'each exclude_region element must contains 2 elements '
    # Update this param to be shown by print_args
    args.exclude_region = exclude_region


    ## Not using config any more, as all parameters has been moved either to the main body of this code or the input parameters
    # if not os.path.exists(args.config):
    #     sys.exit('No config file found in %s' %(args.config))

    if not os.path.exists(args.templates_dir):
        sys.exit('No template directory found in %s' %(args.templates_dir))

    templates_dir = args.templates_dir


    if not os.path.exists(templates_dir+'/'+args.template_flux):
        sys.exit('No flux template found in %s' %(args.templates_dir))

    template_flux = args.template_flux



    if not os.path.exists(templates_dir+'/'+args.template_wave):
        sys.exit('No wavelength template found in %s' %(args.templates_dir))

    template_wave = args.template_wave



    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ','')



    ####### creating /amy_figs and /results folders in outpath if they don't already exist 
    if not os.path.exists(outpath + '/amy_fig'):
        os.mkdir(outpath + '/amy_fig')
        print(outpath + 'amy_fig directory created')
    if not os.path.exists(outpath + '/amy_wd/'):
        os.mkdir(outpath + '/amy_wd/')
        print(outpath + '/amy_wd/  directory created')



    # resolve_common_args uses np.int32 for mask_aps_ids (matching
    # aps_rr.py/aps_rvs.py and this script's own aps_ids handling above)
    # instead of the previous plain int() — a disclosed dtype fix.
    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
    area, mask_areas = resolved.area, resolved.mask_areas

    # print args and assigned/default values on the screen
    print_args(args,module='APS_AMY', version= __aps_amy_version__, path=outpath, headname=args.headname)


    amy_many(args.infiles, outpath, args.headname , targsrvy= targsrvy, targclass = targclass, aps_ids=aps_ids, exclude_region=exclude_region, templates_dir=templates_dir,template_flux=template_flux,template_wave=template_wave, wlranges=wlranges , mproc=args.mproc, overwrite=args.overwrite, config=args.config, sens_corr=args.sens_corr, mask_gaps = args.mask_gaps, safe_mask_gaps = args.safe_mask_gaps, tellurics=args.tellurics, vacuum=args.vacuum, fill_gap=args.fill_gap, arms_ratio=arms_ratio, join_arms=args.join_arms, progress_bar=args.progress_bar)

#################################################################################################
if __name__ == '__main__':

# READ DATA and put them in the APSOBJ OBJECT


#    debug_demo= ['--infiles', '<PYAPS_DATA>/AMY/4365/stacked_1003689.fit',
    debug_demo= ['--infiles', '<PYAPS_DATA>/AMY/4365/stacked_1003690.fit','<PYAPS_DATA>/AMY/4365/stacked_1003689.fit',
    '--aps_ids', '184,185,187,192', # or 'None' to run for all available fibreids
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--wlranges', ' 0.0,0.0', '8470.0,8940.0',
    '--exclude_region' , '8470,8610' , '8630.0,8940.0' , 
    '--sens_corr', 'True',
    '--mask_gaps', 'True',
    '--safe_mask_gaps', 'True',
    '--tellurics', 'False',
    '--vacuum', 'False',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0,1.0',
    '--join_arms', 'False',
    # '--config', '<PYAPS_DIR>/CS/AMY/AMY/parameters.py', ## at the moment, we ignore this, but kept it as may be used later
    '--outpath', '<PYAPS_DIR>/PyAPS_results/20170911/4365/',
    '--templates_dir', '<PYAPS_DIR>/PyAPS_templates/templates_AMY/',
    '--template_flux', 'template_grid.npy',
    '--template_wave','wavelength_data.dat',
    '--headname', 'stacked_1003689',
    '--overwrite', 'True',
    '--progress_bar', 'True',
    '--mproc' ,'4' ]

    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    amy_runner(options=debug_demo)

