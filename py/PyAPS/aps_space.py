import sys, os, re
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict

import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist, islandinfo
from PyAPS import aps_constants

from astropy.io import fits
import astropy.table as atpy
from astropy.modeling import models
from astropy import units as u

import pdb
from functools import reduce
import warnings
import subprocess
import time

from multiprocessing import Pool, Manager

from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
import matplotlib.pyplot as plt


__aps_space_version__=1.3
APSVERS = PyAPS.__version__


#######################################
#limit the number of threads to one for each job
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
#######################################


"""
Python wrapper to run SP_ACE code on WEAVE DATA and generate output tables

versions:
 1.0 By Corrado Boeche First version (June 2020)
 1.2 Modified By APS team (A. Molaeinezhad, December 2020)
 1.3 Modified By APS team (A. Molaeinezhad, March 2021)
 1.35 By A. Molaeinezhad (CASU, May 2021)


NOTES:
To Corrado: Why do you change the working directory before running the main executable

TODO list:


########### BAISC APS PARAM ###################################################################################################
|    param                |    APS default  |  SP_ACE EQUIVALENT          |  SP_ACE DEFAULT  |   availble options/notes      |
infiles (Required)        |        -        |  --infiles (Required)        |        -          |                               |
aps_ids (Optional)        |     None        |  --aps_ids (Optional)        |        None       |                               |
targsrvy (Optional)       |     None        |  --targsrvy (Optional)       |        None       |                               |
######### RRWEAVE INITAIL PARAM ################################################################################################
cache_Rcsr  (Optional)    |     -           |  --cache_Rcsr  (Optional)    |         True      |                               |
                          |                 |                              |                   |                               |
######### DEDICATED RRWEAVE PARAM ##############################################################################################
                          |                 |  --outpath   (Required)      |          -        |                               |
################################################################################################################################



Example:

python3 <PYAPS_DIR>/py/PyAPS/aps_space.py --infiles <PYAPS_DATA>/SPACE/3191/stacked_1002214.fit <PYAPS_DATA>/SPACE/3191/stacked_1002213.fit --outpath <PYAPS_DIR>/PyAPS_results/20160908/3191/ --headname stacked_1002214__stacked_1002213 --wlranges 4915.0,6200.0 5800.0,6860.0 --aps_ids 347,343,332,930,307 --targsrvy GA-OC --targclass None --apsfile <PYAPS_DIR>/PyAPS_results/20160908/3191/stacked_1002214__stacked_1002213_APS.fits  --working_dir <PYAPS_DIR>/PyAPS_results/20160908/3191/space_wd/ --space_exe <PYAPS_DIR>/CS/SPACE/SPACE_v1.4W --outspec False --fig True --overwrite True --mp 2 --GCOG_dir <PYAPS_DIR>/PyAPS_templates/templates_SPACE --error_est False --Salaris_MH True --RV_ini True --ABD_loop True --SN_sp_file True --norm_rad 30.0 --wave_separ True

History:

11 JUN 2020: First version, developed by Corrado Boeche (Developer)
...
11 DEC 2020: fixed a bug. It wrote FWHM as F6.1 instead of F6.2 (By Corrado Boeche)

29 DEC 2020: Modified/Updated to V1.2: PyAPS compatibility and performance optimization (By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: compatibility upgrade considering the latest WEAVE DATA model (v 8.0) and APS output (By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: Error and exception handling improved (By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: The way code deals with FITS files improved (By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: Redundant features/functions removed (By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: The way code run the main SP_ACE executable (Fortran code) code updated (By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: A new command-line keyword (--space_exe) added pointing to the full path of the main SP_ACE executable path(By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: A new command-line keyword (--space_exe) added pointing to the full path of the main SP_ACE executable path(By A. Molaeinezhad)
29 DEC 2020: Modified/Updated to V1.2: redundant codes to check fibre_status and targsrvy removed from the main body of the code as APS_UTILS already handles it (By A. Molaeinezhad)
29 DEC 2020: Added an example to run the code through command line (By A. Molaeinezhad)
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
16 Feb 2022: Bugs fixed
20 Feb 2022: use CNAME instead of TARGID file_params_root, as TARGID could be blank in some configurations. (TBC by the developer)
"""


#######################################
#def measure_time_process(start_time,FIBREID, TARGID, CNAME):
#    process_time = str(int(time.time()- start_time))
#    file=open('time_process_r224_HR',"a+")
#    file.write(','.join([str(FIBREID), TARGID, CNAME,process_time]))
#    file.write('\n')
#    file.close()
#######################################
def remove_blueh_infiles(infiles, wlranges):
    #this function reject the 'BLUEH' setup from the infiles list.
    #this is because the GCOG library do not cover wavelengths<4800A
    infiles_out = []
    wlranges_out = []

    infiles_info = l1_fileinfo(infiles, wlranges=wlranges)
    for i,file in enumerate(infiles_info['infiles']):
        if re.search('BLUEH',infiles_info['setups'][i]):
            pass
        else:
            infiles_out.append(file)
            wlranges_out.append(infiles_info['wlranges'][i])

    return infiles_out, wlranges_out
#######################################
def set_sn_telluric(wave,flux_sn):
    #between 6270-6310A there are tellutic lines
    #this routine set to zero the weigths on this region.
    bool = (wave>=6270) & (wave<6310)
    if bool.sum()>0:
        flux_sn[bool] = 0.

    return flux_sn
#######################################
def find_wave_separ(aps_obj, FIBREID):
    #SPAce uses two FWHMs and two RVs for the two arms.
    #It needs to know the wavelength that separate the two arms.
    #this routine finds the wavelength.           

    #prepare a list containing the setups
    setups_list = aps_obj.setups_original()
    #prepare a list containing 'HR' or 'LR'
    res_mode_list = aps_obj.res_mode()
    #prepare the dictionaries
    wave_dict = {}
    flux_dict = {}
    #for each setup
    for i,setup in enumerate(setups_list):
        #put the wave and fluy into the dictionary
        wave_dict[setup] = aps_obj.data()[aps_obj.idfx()[FIBREID]].spectra[i].wave
        flux_dict[setup] = aps_obj.data()[aps_obj.idfx()[FIBREID]].spectra[i].flux    
        #set the strings red_setup_str and blue_setup_str. Note that in case the
        #setups are two, both of them will be set. In case the setups is just one, only
        #one will be set but in this case it will not be used.
        if re.search('RED', setup):
            red_setup_str = setup
        else:
            blue_setup_str = setup

    #if only one arm is going to measured (the setups is just one)
    if len(setups_list)==1:
        if re.search('RED',setups_list[0]):
            wave_separ = wave_dict[setup][0]
        else:
            wave_separ = wave_dict[setup][-1]
        #return here
        return wave_separ

    #if the setups are two, then execute the following code

    #if the resolution mode is HR
    if res_mode_list[0]=='HR':
        #for the two setups, set the lower wavelength of the red arm
        #and the upper wavelength of the blue/green arm
        for i,setup in enumerate(setups_list):
            if setup==red_setup_str:
                wave_inf_red = wave_dict[setup][0]
            else:
                wave_sup_green = wave_dict[setup][-1]
        #define the wave_separ as the average
        wave_separ = int((wave_inf_red + wave_sup_green)/2.)
    #if the resolution mode is LR
    elif res_mode_list[0]=='LR':
        #the following boolean set "True" the wavelengths of the red arm are 
        #lower than the upper limit of the blue/green arm
        bool_overlap_wave_red = wave_dict[red_setup_str] <= wave_dict[blue_setup_str][-1]
        #the following boolean set "True" the wavelengths of the blue/green arm are 
        #higher than the lower limit of the red arm
        bool_overlap_wave_blue = wave_dict[blue_setup_str] >= wave_dict[red_setup_str][0]
        #set the blue and red fluxes
        flux_blue_over = flux_dict[blue_setup_str][bool_overlap_wave_blue]
        flux_red_over = flux_dict[red_setup_str][bool_overlap_wave_red]
        #the next boolean set "True" the wavelength where the blue flux is higher than the red flux
        bool_higher_blue_flux = flux_blue_over > flux_red_over
        #set as wave_separ the higher wavelength for which the blue flux is larger than the red flux
        wave_separ = wave_dict[blue_setup_str][bool_overlap_wave_blue][bool_higher_blue_flux][-1]

    return wave_separ
#######################################
def initial_fwhm(setup_list):
    #set an initial guess of fwhm depending on the spectral resolution
    #the guess must not be precise and an approximation is good enough.

    if setup_list[0] == 'LR':
        fwhm = 0.8 #this is an initial guess good enough for SPAce
    elif setup_list[0] == 'HR':
        fwhm = 0.2 #this is an initial guess good enough for SPAce
    else:
        print('*** WARNING ***: unknown setup. I give fwhm = 0.2')
        fwhm = 0.2

    return fwhm
#######################################
def save_plot(file_params_root, setups, wave_lims_dict, norm_spec_dict, figdir, NSPEC, TARGID, FIBREID, CNAME):
    

    SPAce_file_spec = file_params_root + '_model.dat'
    #open the file, if SP_Ace converged, otherwise return
    if os.path.isfile(SPAce_file_spec):
        spec_wave,spec_input,spec_norm,spec_model,spec_cont,spec_weight,spec_sn=np.genfromtxt(SPAce_file_spec,usecols=[0,1,2,3,4,5,6],unpack=True)
    else:
        return

    bool_dict = {}
    for setup in setups:
        bool_ = (spec_wave>=wave_lims_dict[setup][0]) & (spec_wave<=wave_lims_dict[setup][1])
        bool_dict[setup] = bool_


    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=False,
                figsize=(12,12), constrained_layout=False)
    fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.93,wspace=0.,hspace=0.0)

    #define the wavelength range limits
    wlranges_flatten = [item for sublist in list(wave_lims_dict.values()) for item in sublist]
    x_inf = min(wlranges_flatten)
    x_sup = max(wlranges_flatten)
    #set title
#    pdb.set_trace()
    title_txt = 'NSPEC= ' + str(NSPEC) + ', FIBREID=' + str(FIBREID) + ', TARGID=' +  TARGID + ', CNAME=' + CNAME
    axes[0].set_title(title_txt)

    #plot spec orig
    for i,setup in enumerate(setups):
        if i==0:
            axes[0].plot(norm_spec_dict[setup][0], norm_spec_dict[setup][1], color='gray', label='original WEAVE spectrum')
        else:
            axes[0].plot(norm_spec_dict[setup][0], norm_spec_dict[setup][1], color='gray')

    axes[0].set_ylabel('counts')
    axes[0].legend(bbox_to_anchor=(0.5, 0.8,0.5,0.2), loc=1, ncol=1,borderaxespad=0.5)
    axes[0].set_ylim(-10.,np.median(norm_spec_dict[setup][1])+3*np.std(norm_spec_dict[setup][1]))
    axes[0].set_xlim(x_inf-20., x_sup+20.)
    #plot spec input and SPAce continuum

    for i,setup in enumerate(setups):
        if i==0:
            axes[1].plot(spec_wave[bool_dict[setup]], spec_input[bool_dict[setup]], 'r-', label='WEAVE normalized spectrum')
            axes[1].plot(spec_wave[bool_dict[setup]], spec_cont[bool_dict[setup]], 'b-', label='continuum chosen by SP_Ace')
        else:
            axes[1].plot(spec_wave[bool_dict[setup]], spec_input[bool_dict[setup]], 'r-')
            axes[1].plot(spec_wave[bool_dict[setup]], spec_cont[bool_dict[setup]], 'b-')

    axes[1].set_ylabel('norm flux')
    axes[1].legend(bbox_to_anchor=(0.5, 0.8,0.5,0.2), loc=1, ncol=2,borderaxespad=0.5)
    axes[1].set_ylim(-0.1,1.59)

    #plot normalized spec and SPAce best model
    #shades
    for i,setup in enumerate(setups):
        bool_shade = (spec_weight[bool_dict[setup]]<0.01)
        if i==0:
            axes[2].fill_between(spec_wave[bool_dict[setup]], bool_shade, 0, facecolor="gray", alpha=0.8,label='wavelength rejected by SPAce')
            axes[2].plot(spec_wave[bool_dict[setup]], spec_norm[bool_dict[setup]], 'b-',label='spectrum normalized by SP_Ace')
            axes[2].plot(spec_wave[bool_dict[setup]], spec_model[bool_dict[setup]], 'g-',label='best matching model')
        else:
            axes[2].fill_between(spec_wave[bool_dict[setup]], bool_shade, 0, facecolor="gray", alpha=0.8)
            axes[2].plot(spec_wave[bool_dict[setup]], spec_norm[bool_dict[setup]], 'b-')
            axes[2].plot(spec_wave[bool_dict[setup]], spec_model[bool_dict[setup]], 'g-')

    axes[2].set_ylabel('norm flux')
    axes[2].legend(bbox_to_anchor=(0., 0.8,1.0,0.2), loc=1, ncol=3,borderaxespad=0.5)
    axes[2].set_xlabel('angstrom')
    axes[2].set_ylim(-0.1,1.59)

    name_plot = figdir + '_%s_%s_%s.png' % (NSPEC, CNAME, FIBREID)

    try:
        plt.savefig(name_plot)
    except:
       print('***** WARNING: I cannot write the diagnostic plot for CNAME', str(CNAME), '! *****')
#    plt.show()

    #clean the axes
    for ax in axes:
        ax.clear()
    plt.close()
#############################################
def collect_results(file_params_root, NSPEC, TARGID, FIBREID, CNAME):
    #once SPAce wrote the results (such as derived stellar parameters and abundances)
    #this subroutine read the results and pass them as list.

    #set the name of the file where SPAce wrote the derived stellar params and abundances
    results_TGM_file = file_params_root + '_TGM_ABD.dat'

    #if the file exist do the following, otherwise return None.
    if os.path.isfile(results_TGM_file):

        #open the file and collect the header and the values
        file=open(results_TGM_file,"r")
        header=file.readline()
        line=file.readline()
        file.close()

        #format the values in line
        list_line = line.split()
        list_cols = header.split()
        line_formatted = []

        for i, item in enumerate(list_line):
            if item == 'null':
                line_formatted.append(np.nan)
            elif list_cols[i] == 'conv' or re.search('_N',list_cols[i]) or list_cols[i] == 'Teff':
                line_formatted.append(int(item))
#            elif list_cols[i] == 'SNR' or re.search('RV',list_cols[i]) or re.search('FWHM',list_cols[i]):
            elif list_cols[i] == 'SNR' or re.search('RV',list_cols[i]):
                line_formatted.append(np.round(float(item),1))
            else:
                line_formatted.append(np.round(float(item),2))

        #the SPAce header is different from the one requested.
        #here we format the header columns as described in the ICD document
        list_cols_formatted = []
        suffix = 'SPACE_'
        for item in list_cols:
            item_ = item
            if re.search('T_',item):
                item_ = item.replace('T_','TEFF_')
            if re.search('L_',item):
                item_ = item.replace('L_','LOGG_')

            if re.search('_l',item_):
                list_cols_formatted.append(suffix + item_.replace('_l','_LOW').upper())
            elif re.search('_h',item_):
                list_cols_formatted.append(suffix + item_.replace('_h','_UPP').upper())
            else:
                list_cols_formatted.append(suffix + item_.upper())

        #prepare the results line
        line_header = [NSPEC, str(FIBREID), CNAME, TARGID]
        results_line = line_header + line_formatted
        #prepare the header (column names)
        header_list = ['NSPEC', 'FIBREID', 'CNAME', 'TARGID']  + list_cols_formatted
        return header_list,results_line
    else:
        return None, None
##############################################
def collect_spectra(file_params_root,wave_original_dict):
    #once SPAce wrote the spectra (such as normalized spectrum, best matching model, etc.)
    #this subroutine read these spectra and pass them as structured array

    #set the name where the spectra has been saved by SPAce
    results_model_file = file_params_root + '_model.dat'
    #set the names of the columns
    fields_names = ['LAMBDA_INPUT', 'SPACE_FLUX_N_IN','SPACE_FLUX_N_OUT','SPACE_FLUX_MODEL','SPACE_CONTINUUM','SPACE_WEIGHT','SPACE_SN']
    dtype_list = [(item, '<f8') for item in fields_names]

    #set the list that contain the wavelength array. We decided that
    #all the spectra will be returned with wavelength interval 4800-6860A 
    #(GCOG library limits)
    wave_list = []
    #check first array in wave_original_dict to extract the wave increment
    array_wave = list(wave_original_dict.values())[0]
    wave_incr = array_wave[1] - array_wave[0]
    wave_list = np.arange(4800.0, 6860.+wave_incr,wave_incr).round(2)
    #prepare a dataframe that will contain the spectra with wavelength extension 4800-6860A
    df_ = pd.DataFrame(wave_list, columns=[fields_names[0]])

    if os.path.isfile(results_model_file):
        #load the spectra from the file results_model_file
        result_df = pd.read_csv(results_model_file, header=None, index_col=None, names=fields_names, sep='\s+')
        #put the loaded spectra into the dataframe df_ (this fit the SPAce spectrum into the original wave array)
        #in this way, the resulting array will be always the same for all the stacked spectra
        df_space_spectra = pd.merge(df_,result_df,on=fields_names[0], how='left')
        #transform it into a structured array
        struct_array = np.array([tuple(x) for x in df_space_spectra.values], dtype=dtype_list)

        return struct_array
    else:
        return None

##############################################
def prepare_space_pars(file_params_root, GCOG_dir, RV, wave_lims_dict, fwhm, wave_separ, space_args):
    #unpack space_args
    args_Salaris_MH, args_RV_ini, args_ABD_loop, args_SN_sp_file, args_norm_rad, args_wave_separ, args_error_est = space_args 

    #prepare the parameters needed by SPAce. 
    #These will be written in the file_params_name read by SPAce.
    list_space_commands=[]
    spectrum_file_address = file_params_root + '.asc'

    list_space_commands.append('obs_sp_file ' + "'" + spectrum_file_address + "'")
    list_space_commands.append('GCOGlib ' + "'" + GCOG_dir + "'")
    list_space_commands.append('fwhm ' + str(fwhm))
    string_range = 'wave_lims '
    for setup, range_list in wave_lims_dict.items():
        string_range = string_range + ' ' + str(range_list[0]) + ' ' + str(range_list[1])

    list_space_commands.append(string_range)
    if args_Salaris_MH:
        list_space_commands.append('Salaris_MH')
    else:
        print('***** WARNING: no Salaris_MH option given! *****')

    if args_RV_ini:
        list_space_commands.append('RV_ini ' + str(RV))
    else:
        print('***** WARNING: no RV_ini option given! *****')

    if args_ABD_loop:
        list_space_commands.append('ABD_loop')
    else:
        print('***** WARNING: no ABD_loop option given! *****')

    if args_SN_sp_file:
        list_space_commands.append('SN_sp_file')
    else:
        print('***** WARNING: no SN_sp_file option given! *****')

    if args_norm_rad:
        list_space_commands.append('norm_rad ' + str(30.0))
    else:
        print('***** WARNING: no norm_rad option given! *****')

    if args_wave_separ:
        list_space_commands.append('wave_separ ' + str(wave_separ))
    else:
        print('***** WARNING: no wave_separ option given! *****')

    if args_error_est:
        list_space_commands.append('error_est')
    else:
        print('***** WARNING: no error_est option given! *****')


#prepare the name of the files to write. These files are read by SPAce.
    file_params_name = file_params_root + '.par'
    #write the file_params_name that contains the SPAce parameters
    file=open(file_params_name,"w")
    file.writelines( "%s\n" % item for item in list_space_commands )        
    file.close()

    return file_params_name
################################################
def write_spectrum(file_params_root, wave, flux_norm, flux_sn, lsf):
    #prepare the name of the files to write. These files are read by SPAce.
    spectrum_file_address = file_params_root + '.asc'
    #
    df = pd.DataFrame(data={'wave': wave.round(3), 'flux': flux_norm.round(5),'sn': flux_sn.round(1), 'lsf': lsf.round(2)})
    df.to_csv(spectrum_file_address, sep=' ', header=False, index=False)
################################################
def prepare_sn_array(flux_ivar):
    #prepare the flux_sn array
    #first, select boolean where flux_ivar>0.
    bool_non_zero = (flux_ivar>1e-12)
    #prepare the flux_sn array = 0.001
    flux_sn = np.zeros(len(flux_ivar)) + 0.001
    #set the sn where flux_ivar>0.
    flux_sn[bool_non_zero] = np.sqrt(1./flux_ivar[bool_non_zero])
    return flux_sn
################################################
def set_wave_limits(setups, wlranges, norm_spec_dict):
    #this function defines the limits of the wavelength intervals
    #These intervals are set by considering the requests of the user
    #and the limits of the input spectrum. Once these limits are passed
    #to SPAce, this will also consider the limits of the GCOG library.
    wave_lims_dict = {}

    for i,setup in enumerate(setups):
        w_min_spec = norm_spec_dict[setup][0][0]
        w_max_spec = norm_spec_dict[setup][0][-1]
        w_inf = np.max([float(wlranges[i][0]),w_min_spec])
        w_sup = np.min([float(wlranges[i][1]),w_max_spec])
        wave_lims_dict[setup] = [w_inf, w_sup]

    return wave_lims_dict
################################################
def concatenate_norm_spectra(list_setups, norm_spec_dict, wave_separ):

    wave_list = []
    flux_list = []
    flux_norm_list = []
    ivar_list = []
    lsf_list = []


    for setup in list_setups:
        #select the wavelengths lower and higher than wave_separ
        #this is important for low resolution setup. For high resolution
        # does not make any difference
        if re.search('RED', setup):
            bool_to_append = norm_spec_dict[setup][0] > wave_separ
        else:
            bool_to_append = norm_spec_dict[setup][0] < wave_separ

        wave_list.append(norm_spec_dict[setup][0][bool_to_append])
        flux_list.append(norm_spec_dict[setup][1][bool_to_append])
        flux_norm_list.append(norm_spec_dict[setup][2][bool_to_append])
        ivar_list.append(norm_spec_dict[setup][3][bool_to_append])
        lsf_list.append(norm_spec_dict[setup][4][bool_to_append])

    wave = np.concatenate(wave_list)
    flux_orig = np.concatenate(flux_list)
    flux_norm = np.concatenate(flux_norm_list)
    flux_ivar = np.concatenate(ivar_list)
    lsf = np.concatenate(lsf_list)

    return wave, flux_orig, flux_norm, flux_ivar, lsf
#################################################
def normalize_spectrum(wave,flux,RV):

    wave_u = wave * u.Angstrom
    flux_u = flux * u.dimensionless_unscaled
    spectrum_u = Spectrum1D(data=flux_u,spectral_axis=wave_u)
    #working copies of these arrays
    wave_ = wave_u
    flux_ = flux_u
    #RV in term of fraction of wavelength
    c=1.0+RV/299792.0
    #wavelength of Ha and Hb, given the RV
    wave_Ha = 6563.*c
    wave_Hb = 4861.*c

    #initialize the  polynomial coefficients
    c0,c1,c2,c3,c4,c5,c6,c7 = [1.,0.,0.,0.,0.,0.,0.,0.]
    #find the continuum, for 5 times
    for i in range(5):
        spectrum = Spectrum1D(data=flux_,spectral_axis=wave_)
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            #fit the continuum using 7th degree Legendre1D function
            cont_fit = fit_generic_continuum(spectrum,model=models.Legendre1D(7,c0=c0,c1=c1,c2=c2,c3=c3,c4=c4,c5=c5,c6=c6,c7=c7))

        #set the just found coefficients
        c0,c1,c2,c3,c4,c5,c6,c7 = cont_fit.parameters
        #set the found continuum (clip it to a minimum=1 to avoid negative values.
        #Remember that the flux is expressed in counts).
        y_continuum_fitted = np.clip(cont_fit(wave_),1.,np.inf)
        #normalized the flux
        flux_norm = spectrum.flux/y_continuum_fitted

# used during debug only
#        plt.subplot(2,1,1)
#        plt.plot(wave_,flux_)
#        plt.plot(wave_,y_continuum_fitted)
#        plt.subplot(2,1,2)
#        plt.plot(wave_,flux_norm)
#        plt.show()
##################################

        #set booleans that identify intervals +-3A around Ha and Hb
        bool_Hb = (wave_<(wave_Hb-3)* u.Angstrom) | (wave_>(wave_Hb+3)* u.Angstrom)
        bool_Ha = (wave_<(wave_Ha-3)* u.Angstrom) | (wave_>(wave_Ha+3)* u.Angstrom)
        #remove the cores of Ha and Hb lines
        #also, remove the points that have flux that deviates more that 4sigma and less than 1sigma from the flux_norm.mean()
        bool=(flux_norm>(flux_norm.mean()-flux_norm.std())) & (flux_norm<(flux_norm.mean()+4.*flux_norm.std())) & bool_Ha & bool_Hb
        wave_ = wave_[bool]
        flux_ = flux_[bool]

    y_continuum_fitted = cont_fit(wave_u)
    flux_norm = spectrum_u.flux/y_continuum_fitted

    return flux_norm.value
################################################
def extract_norm_spectra(wave,flux,flux_ivar,lsf,pix_lims_list, RV):
    pix_spec_low, pix_spec_hi, pix_gap_low, pix_gap_hi = pix_lims_list

    #if pix_gap_low = pix_gap_hi = None means that the spectrum arm has no gaps
    # then put a flag
    if pix_gap_low == None:
        no_gaps_flag = True
    else:
        no_gaps_flag = False

    #initialize arrays
    flux_ = np.copy(flux[pix_spec_low:pix_spec_hi])
    ivar_ = np.copy(flux_ivar[pix_spec_low:pix_spec_hi])
    wave_ = np.copy(wave[pix_spec_low:pix_spec_hi])
    lsf_ = np.copy(lsf[pix_spec_low:pix_spec_hi])
    flux_norm = np.zeros(len(flux_ ))

    #identify the two pieces of the spectrum separated by the gap
    if no_gaps_flag == False:
        bool_blue_ = (wave_ < wave_[pix_gap_low])
        bool_red_ = (wave_ > wave_[pix_gap_hi])

        #cut the spectrum in the blue and red parts
        #do it for the blue side of the spectrum
        blue_flux = flux_[bool_blue_]
        blue_wave = wave_[bool_blue_]

        #do it for the red side of the spectrum
        red_flux = flux_[bool_red_]
        red_wave = wave_[bool_red_]

        #normalize the blue part
        blue_flux_norm = normalize_spectrum(blue_wave,blue_flux,RV)
        #normalize the red part
        red_flux_norm = normalize_spectrum(red_wave,red_flux,RV)

        #put together the two pieces
        flux_norm[bool_red_] = red_flux_norm
        flux_norm[bool_blue_] = blue_flux_norm
    else:
        #normalize
        flux_norm = normalize_spectrum(wave_,flux_,RV)

    return wave_,flux_,flux_norm,ivar_,lsf_
#######################################
def plotta(aps_obj):
#used during debug only
        plt.subplot(2,1,1)
        plt.plot(aps_obj.data()[0].spectra[0].wave,aps_obj.data()[0].spectra[0].flux) 
        plt.plot(aps_obj.data()[0].spectra[1].wave,aps_obj.data()[0].spectra[1].flux) 
        plt.subplot(2,1,2)
        plt.plot(aps_obj.data()[0].spectra[0].wave,aps_obj.data()[0].spectra[0].ivar) 
        plt.plot(aps_obj.data()[0].spectra[1].wave,aps_obj.data()[0].spectra[1].ivar) 
        plt.show()   
#######################################
# def extract_info_tables(FIBTABLE, STARTABLE,nfib):
#     #this routine extract information from the APS L2 table

#     #row for 'FIBTABLE'
#     i_row = np.argwhere(FIBTABLE['FIBREID'] == nfib)[0][0]
#     FIBREID = FIBTABLE[i_row]['FIBREID']

#     # Just to make sure we are getting the right object
#     assert FIBREID == nfib, 'Failed to matched the requested FIBREID'
    

#     NSPEC      = str(FIBTABLE[i_row]['NSPEC'])
#     CNAME      = str(FIBTABLE[i_row]['CNAME'])
#     TARGID   = FIBTABLE[i_row]['TARGID']
#     targsrvy   = FIBTABLE[i_row]['TARGSRVY']
#     TARGNAME = FIBTABLE[i_row]['TARGNAME']

#     #row for 'STELLAR_TABLE_RVS'
#     i_row = np.argwhere(STARTABLE['APS_ID'] == nfib)

#     #if nfib has a RV (it is listed in STELLAR_TABLE_RVS), assign it
#     if len(i_row)>0:
#         RV = np.round(STARTABLE[i_row[0][0]]['VRAD'],2)
#     else: #otherwise assign zero
#         RV = 0.0

#     return FIBREID, TARGID, CNAME, NSPEC, RV, targsrvy, TARGNAME
######################################
def find_gaps(flux_ivar):

    #we assume here that there is at least one ccd gap
    gap_isl, gap_len = islandinfo(flux_ivar, trigger_val=0.0)

    #check if there are pixels at the borders that have ivar=0.
    #if so, report them so that only the pixels where the flux>0.
    #will be considered

    #prepare a boolean which length is equal to the number of gaps found
    bool_gap = np.ones(len(gap_isl)).astype(bool)

    #if there are no gaps, return here
    if bool_gap.sum() == 0:
        pix_spec_low = 0
        pix_spec_hi = len(flux_ivar)
        pix_gap_low = None
        pix_gap_hi = None

        return [pix_spec_low, pix_spec_hi, pix_gap_low, pix_gap_hi]

    #otherwise, if there are gaps, do the following
    if gap_isl[0][0]==0: #if the gap is at the blue border
        #pix_spec_low is the bluer pixel where flux>0
        #i.e. at the blue border the flux is not zero
        pix_spec_low = gap_isl[0][1]+1
        #set false, that mean this gap is not a real gap
        #but zero flux at the spectrum border
        bool_gap[0] = False 
    else: #if the gap is *not* at the blue border
        #then, the bluer pixel has flux>0 
        pix_spec_low = 0        

    #do the same for the red border.
    #check if at the red border the flux is zero or not
    if gap_isl[-1][1]==len(flux_ivar)-1: #if the gap is at the red border
        #pix_spec_hi is the redder pixel where flux>0
        #i.e. at the red border the flux is not zero
        pix_spec_hi = gap_isl[-1][0]
        #set false, that mean this gap is not a real gap
        #but zero flux at the spectrum border
        bool_gap[-1] = False
    else: #if the gap is *not* at the red border
        #then, the redder pixel has flux>0 
        pix_spec_hi = len(flux_ivar)

    #now bool_gap is equal to True if there are gaps.
    #if there is more than one gap, choose the larger one as the ccd one
    if bool_gap.any():
        gap_isl_ = [gap_isl[i] for i in range(len(gap_len)) if bool_gap[i]] 
        gap_len_ = [gap_len[i] for i in range(len(gap_len)) if bool_gap[i]]
        i_max = np.argmax(gap_len_)
        pix_gap_low = gap_isl_[i_max][0]
        pix_gap_hi = gap_isl_[i_max][1]
    else: #if there are no gaps
        pix_gap_low = None
        pix_gap_hi = None

    return [pix_spec_low, pix_spec_hi, pix_gap_low, pix_gap_hi]
######################################
def plot(wave, flux, flux_norm, flux_ivar, FIBREID, TARGID, CNAME):
    #this function was useful only during debug

    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=False,
                figsize=(12,12), constrained_layout=False)
    fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.93,wspace=0.,hspace=0.0)

    title_txt = 'FIBREID=' + str(FIBREID) + ', TARGID=' +  TARGID + ', CNAME=' + CNAME
    axes[0].set_title(title_txt)

    axes[0].plot(wave, flux, color='grey') #blue part
    axes[0].set_ylabel('counts')

    axes[1].plot(wave,flux_norm, color='grey') #blue part
    axes[1].set_ylabel('norm flux')

    axes[2].plot(wave,flux_ivar, color='grey') #blue part
    axes[2].set_ylabel('ivar')
    axes[2].set_xlabel('angstrom')

    plt.show()
########################################

########################################
def func_parallel(spectrum_info, aps_obj, fwhm, space_args,space_exe):
    FIBREID, TARGID, CNAME, NSPEC, RV, wlranges, working_dir, GCOG_dir, figdir, fig, spec_fits= spectrum_info

    #measure time
    start_time = time.time()
    #define the general name of the spectrum/parameters files
    file_params_root = working_dir + str(CNAME)

    #initialize the dictionary that will hold the arrays of the normalized spectrum
    norm_spec_dict = {}
    #initialize the dictionary that will hold the inf and sup limits of wavelengths
    wave_original_dict = {}

#    plotta(aps_obj) #used during debug

    #find the the separating wavelength (the wavelength that separates the two arms)
    wave_separ = find_wave_separ(aps_obj,FIBREID)

    for i,setup in enumerate(aps_obj.setups_original()):
        wave = aps_obj.data()[aps_obj.idfx()[FIBREID]].spectra[i].wave
        flux = aps_obj.data()[aps_obj.idfx()[FIBREID]].spectra[i].flux
        flux_ivar = aps_obj.data()[aps_obj.idfx()[FIBREID]].spectra[i].ivar
        #here we assign the LSF. To date (March 2020), this is not available
        #in the tables. Therefore, we temporary assign a np.ones() array.
        try:
            lsf = aps_obj.data()[aps_obj.idfx()[FIBREID]].spectra[i].lsf
        except:
            print('**** WARNING: the LSF was not found in aps_obj  ****')
            print('**** WARNING: I set LSF=1  ****')
            lsf = np.ones(len(wave))
###for debug only
        #add a linear law for the lsf. Test for debug
#        bool_red_lsf = wave >= wave_separ
#        bool_blue_lsf = wave < wave_separ
#        lsf[bool_blue_lsf] = -0.0004*(wave[bool_blue_lsf]-4800)+1
#        lsf[bool_red_lsf] = -0.0004*(wave[bool_red_lsf]-5900)+1
#################
        #find the gaps limits
        pix_lims_list = find_gaps(flux_ivar)

        wave_norm, flux_orig, flux_norm, flux_ivar_norm, lsf_norm = extract_norm_spectra(wave,flux,flux_ivar,lsf,pix_lims_list, RV)
        norm_spec_dict[setup] = [wave_norm, flux_orig, flux_norm, flux_ivar_norm, lsf_norm]
        wave_original_dict[setup] = wave

# used during debug only        
#        plt.subplot(2,1,1)
#        plt.plot(wave_norm,flux_norm)
#        plt.subplot(2,1,2)
#        plt.plot(wave_norm,flux_ivar_norm)
#        plt.show()
##########################
    #normalize and concatenate the green and red spectra
    wave, flux_orig, flux_norm, flux_ivar, lsf = concatenate_norm_spectra(aps_obj.setups_original(), norm_spec_dict, wave_separ)


# used during debug only        
#    #to plot every spectrum (for debug only, to be commented for processing)
#    plot(wave, flux_orig, flux_norm, flux_ivar, FIBREID, TARGID, CNAME)
#    s = input('continue? (y/n)')
#    if s=='n':
#        os.chdir(current_dir)
#        sys.exit()
############################
    #set the wavelength limits keeping in account the limits given by the user
    wave_lims_dict = set_wave_limits(aps_obj.setups_original(), aps_obj.wlranges(), norm_spec_dict)

    #prepare the SN array
    flux_sn = prepare_sn_array(flux_ivar)
    #set S/N=0 the wavelengths 6270-6310A because affected by telluric lines
    flux_sn = set_sn_telluric(wave,flux_sn)

    #write the spectrum in a file that will be read by SPAce
    spectrum_file_address = write_spectrum(file_params_root, wave, flux_norm, flux_sn, lsf)


    #prepare the SPAce parameter file
    file_params_name = prepare_space_pars(file_params_root, GCOG_dir, RV, wave_lims_dict, fwhm, wave_separ, space_args)



    #######################################   run SPAce    #######################################
    # command = space_exe+' ' + file_params_name
    # sys_msg=os.system(command)

    cmd = space_exe + ' ' + file_params_name
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        print(stdout_line, end='') 
    popen.stdout.close()
    return_code = popen.wait()
    # if return_code:
    #     raise subprocess.CalledProcessError(return_code, cmd)
    ################################################################################################

    #measure time
#    measure_time_process(start_time,FIBREID, TARGID, CNAME)

    #collect the results
    header, results_line = collect_results(file_params_root, NSPEC, TARGID, FIBREID, CNAME)

    #collect spectra model
    if spec_fits is not None:
        df_space_spectra = collect_spectra(file_params_root, wave_original_dict)

    #make the plot and store it
    if fig == True:
        save_plot(file_params_root, aps_obj.setups_original(), wave_lims_dict, norm_spec_dict, figdir, NSPEC, TARGID, FIBREID, CNAME)
    #clean up the working directory from the just measured spectrum and its parameter/results files
    # command = 'rm ' + file_params_root + '*'
    # os.system(command)


    if spec_fits is None:
        return header, results_line, None
    else:
        return header, results_line, df_space_spectra
##############################

#######################################
def create_fits_table(param_fits, results):

    #read the name of the columns of the SPAce results
    columns = results[0][0]

    #create a ordered dictionary with the columns name
    outdict = OrderedDict()
    for c in columns:
        outdict[c]=[]

    #put in outdict the value corresponding to the column name
    for res in results:
        head = res[0]
        line = res[1]
        for i,col in enumerate(head):
            outdict[col].append(line[i])

    #create a table
    outtab = atpy.Table(outdict)
    #set the units
    for item in columns[5:]:
        if re.search('TEFF',item):
            outtab[item].unit = 'K'
        elif re.search('RV',item):
            outtab[item].unit = 'km/s'
        elif re.search('FWHM',item):
            outtab[item].unit = 'Angstrom'
        elif re.search('CONV',item) or re.search('CHISQ',item) or re.search('_N',item):
            None
        else:
            outtab[item].unit = 'dex'

    outtab.meta['EXTNAME'] = 'SPAce_TABLE'
    outtab.meta['APSVER'] = (APSVERS,'PYAPS version')
    outtab.meta['COMMENT'] = 'This result has been obtained with the software SP_Ace v1.4W (WEAVE version) \
and the GCOG library described in the paper Boeche, Vallenari, and Lucatello (A&A submitted 2020) released under GPL licence'
    outtab.sort('NSPEC')

    #return the table
    return outtab
#######################################
def create_fits_spec(spec_fits, results):

    columns = results[0][0][0:4] + list(results[0][2].dtype.names)
    wave_incr = np.round(results[0][2]['LAMBDA_INPUT'][1] - results[0][2]['LAMBDA_INPUT'][0],2)
    leng_wave_array = len(results[0][2]['LAMBDA_INPUT'])

    outdict = OrderedDict()
    for c in columns:
        outdict[c]=[]

    for head,line,spec in results:
        if spec is None:
            continue
        for i,col in enumerate(columns):
            if col in head:
                outdict[col].append(line[i])
            else:
                outdict[col].append(spec[col])

    outtab = atpy.Table(outdict)

    outtab.meta['EXTNAME'] = 'SPACE_SPECTRA'
    outtab.meta['APSVER'] = (APSVERS,'PYAPS version')
    outtab.meta['COMMENT'] = 'This result has been obtained with the software SP_Ace v1.4W (WEAVE version) \
and the GCOG library described in the paper Boeche, Vallenari, and Lucatello (A&A submitted 2020) released under GPL licence'
    outtab.meta['TCRVL'] = '4800.'
    outtab.meta['TCLDT'] = str(wave_incr)
    outtab.meta['TFORM'] = str(leng_wave_array)
    outtab.sort('NSPEC')

    return outtab
######################################
def write_HDUlist(param_fits,spec_fits,results, overwrite_flag):

    #prepare HDUlist
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    #create fits table
    param_space_fits = create_fits_table(param_fits, results)
    #append it
    hdul.append(fits.convenience.table_to_hdu(param_space_fits))
    #create spectra fits table, if requested by the user
    if spec_fits is not None:
        spec_space_fits = create_fits_spec(spec_fits, results)
        #append it
        hdul.append(fits.convenience.table_to_hdu(spec_space_fits))

    #write the HUDlist
    if overwrite_flag:
        hdul.writeto(param_fits, overwrite=overwrite_flag, checksum=True)
    else:
        print(' *** WARNING: the user has chosen overwrite=False *** ')
        print(' ***       the results has no been written        *** ')
######################################

######################################
def proc_many(infiles, apsfile, param_fits, outpath, working_dir, GCOG_dir, space_exe, figdir=None, 
 spec_fits=None, aps_ids=None, setups=None, wlranges=None , nthreads=None,
 overwrite=True, fig=False, targsrvy=None, targclass=None, space_args=None):

    try:

        #upload the first L1 table in order to get the fibreID and the STATUS
        # FIBTABLE=fits.getdata(infiles[0], extname='FIBTABLE')
        # table_L1 = fits.open(infiles[0])

        #upload the APS L2 table
        STARTABLE=fits.getdata(apsfile, extname='STAR_TABLE')
        # table_L2 = fits.open(apsfile)

    except:
        print('ERROR : %s' %(sys.exc_info()[1]))
        return



    # ## put in a list only the fibres that are chosen by the user that also have RVS
    # ## Note: Fibre status, later will be checked by APSOB and fibres with status other than 'A' will be automatically rejected  
    # list_fibre=reduce(np.intersect1d,(STARTABLE['APS_ID'], aps_ids)).tolist()


    #load the object
    aps_obj = APSOB(infiles, wlranges=wlranges, aps_ids=aps_ids, sens_corr=False, mask_gaps=True, safe_mask_gaps=True,
        vacuum=False, tellurics=True, fill_gap=False, join_arms=False,
        targsrvy=targsrvy, targclass=targclass)

    # print('***** WARNING: As in the current WEAVE data model, we do not have targclass yet, we still have to stick to the APS_ID to target our objects. *****')
    # print('***** Later, we can set targclass (already set to None) to choose our interested targets *****')



    #initialize list info
    spectra_list_info = []
    fibs_in_class = []
    
    #we take NSPEC from the RV table, that means take only the objects
    #that have RVs
    for targ in aps_obj.data():

        status = targ.fib_status
        #if "status" is not 'A' (active), skip the fibre
        if status.upper() != 'A':
            continue
        #set fibre number
        nfib = targ.id

        #extract info of the i-th row
        # FIBREID, TARGID, CNAME, NSPEC, RV, targsrvy_l2, TARGNAME = extract_info_tables(FIBTABLE, STARTABLE,nfib)

        FIBREID = targ.aps_id
        TARGID = targ.targid
        CNAME    = targ.cname
        NSPEC    = targ.meta[0]["NSPEC"]
        TARGNAME = targ.meta[0]["TARGNAME"]
        targsrvy_l2 = targ.targsrvy

        # Extract RV from the stellar_table L2
        #row for 'STELLAR_TABLE_RVS'
        i_row = np.argwhere(STARTABLE['APS_ID'] == nfib)

        #if nfib has a RV (it is listed in STELLAR_TABLE_RVS), assign it
        if len(i_row)>0:
            RV = np.round(STARTABLE[i_row[0][0]]['VRAD'],2)
        else: #otherwise assign zero
            RV = 0.0



        #if TARGID = '', then skip it

        ## Why? 19 Feb 2022
        # if len(TARGID)==0:
        #     continue
#        print(NSPEC,FIBREID,CNAME,TARGID,TARGNAME)
        list_to_append = [FIBREID, TARGID, CNAME, NSPEC, RV, wlranges, working_dir, GCOG_dir, figdir, fig, spec_fits]
        spectra_list_info.append(list_to_append)
        fibs_in_class.append(FIBREID)



    # #delete the FIBTABLE to free space
    # del FIBTABLE

    #delete the STARTABLE to free space
    # del STARTABLE
    # print('fibs_in_class ', fibs_in_class)
    # print(spectra_list_info)

    #assign a first guess of FWHM. SPAce will refine it later.
    fwhm = initial_fwhm(aps_obj.res_mode())

    with Pool(processes=int(nthreads)) as pool:
        list_process = [pool.apply_async(func_parallel, args=(pars, aps_obj, fwhm, space_args, space_exe)) for pars in spectra_list_info]
        results = [p.get() for p in list_process]
###### for debug only
#    results = func_parallel(spectra_list_info[0], aps_obj, fwhm)
#    sys.exit()
####################

    #write the results table (and the spectra table, if requested)
    write_HDUlist(param_fits,spec_fits,results,overwrite)
    
######################################################################

def spaceweave(options=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--infiles", type=none_or_str, default=None,
        required=True, help="input files", nargs='*')

    parser.add_argument("--apsfile", help='The input fits file, contains the aps table',
        type=none_or_str, default=none_or_str, required=True)

    parser.add_argument('--aps_ids',help='comma-separated list of WEAVE APS_IDs',
                            type=none_or_str, default=None, required=False)

    parser.add_argument("--wlranges", type=none_or_str, default=None,
        required=False, help="wavelenght range array for each elements of the setup", nargs='*')

    parser.add_argument("--targsrvy", type=none_or_str, default=None,
        required=True, help="comma-separated list of surveys to be considered")
            
    parser.add_argument("--targclass", type=none_or_str, default=None,
        required=False, help="comma-separated list of classtypes to be considered")

    parser.add_argument("--outpath", 
        help='Directory to keep WEAVE_SP_Ace outputs', type=none_or_str, default=None, required=True)

    parser.add_argument("--working_dir", 
        help='Directory where SP_Ace can write its outputs', type=none_or_str, default=None, required=True)

    parser.add_argument('--GCOG_dir', 
        help='Directory of the GCOG library', type=none_or_str, default=None, required=True)

    parser.add_argument('--space_exe', 
        help='Full path of the SP_Ace executable file', type=none_or_str, default=None, required=True)

    parser.add_argument('--headname', help='Output headname. The output filenames will be generated based on this',
        type=none_or_str, default='headname', required=True)

    parser.add_argument("--outspec", default=False, type=str2bool,
        required=False, help="if True, the code return the spectra and best fitted model [FITS file]")

    parser.add_argument("--fig", default=False, type=str2bool,
        required=False, help="if True, the code also produces diagnostic plots")

    parser.add_argument('--overwrite', help='If enabled the code will overwrite the existing products, otherwise it will skip them',
        type=str2bool, default=False)

    parser.add_argument('--mp', help='Number of threads for the fits',
        type=int, default=1, required=False)

    parser.add_argument('--Salaris_MH', 
        help='Use the SP_Ace internal metallicity given by Salaris', type=none_or_str, default='True', required=False)

    parser.add_argument('--RV_ini', 
        help='Set the SP_Ace initial RV found inside the APS L2 table', type=none_or_str, default='True', required=False)

    parser.add_argument('--ABD_loop', 
        help='Set the SP_Ace ABD_loop keyword', type=none_or_str, default='True', required=False)

    parser.add_argument('--SN_sp_file', 
        help='Inform SP_Ace that the spectrum file provided has S/N array', type=none_or_str, default='True', required=False)

    parser.add_argument('--norm_rad', 
        help='Set the SP_Ace norm_rad keyword', type=none_or_str, default='30.0', required=False)

    parser.add_argument('--wave_separ', 
        help='Set the SP_Ace waev_separ keyword', type=none_or_str, default='True', required=False)

    parser.add_argument('--error_est', 
        help='Set the SP_Ace error_est keyword', type=none_or_str, default='True', required=True)


    ## Check if any command-line argument has been passed to the module. It counts the number of system arguments to check this.
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')

        args = parser.parse_args(options)


    if args.infiles is None:
        raise Exception('You need to specify the spectra you want to fit')
    infiles = args.infiles

    #this must be the APS file results from where I can read the CLASS and the RVs of the objects
    if args.apsfile is None:
        raise Exception('You need to specify the APS file to proceed')
    apsfile = args.apsfile

    #working directory where SP_Ace write its outputs (that will be deleted after being read).
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)
        print("WORKING DIRECTORY: %s Created!" %(args.working_dir))
    working_dir=args.working_dir+os.path.sep
    working_dir=working_dir.replace(' ','')

    #GCOG directory
    if not os.path.exists(args.GCOG_dir):
        raise Exception('The GCOG LIBRARY path seems wrong!')
    GCOG_dir=args.GCOG_dir+os.path.sep
    GCOG_dir=GCOG_dir.replace(' ','')


    # Check SP_Ace executable
    if not os.path.exists(args.space_exe):
        sys.exit('No CP_Ace executable found in %s' %(args.space_exe))


    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ','')

    param_fits=os.path.join(args.outpath,str(args.headname)+'_SPACE'+'.fits')
    param_fits=param_fits.replace(' ', '')

    spec_fits=None #save best fitting model
    if args.outspec:
        spec_fits=os.path.join(args.outpath,'space_spec_'+str(args.headname)+'.fits')
        spec_fits.replace(' ', '')

    figdir=None
    if args.fig:
        figdir=args.outpath + '/space_fig/'
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            print("FIGDIR: %s Created!" %(figdir))
        figdir=figdir + str(args.headname)
        figdir=figdir.replace(' ', '')

    aps_ids = None
    if args.aps_ids is not None:
        aps_ids = np.array(args.aps_ids.split(","), dtype=np.int32).tolist()

    targsrvy = None
    if args.targsrvy is not None:
        targsrvy = [ str(x) for x in args.targsrvy.split(",") ]
    else:
        print('***** WARNING: targsrvy=None cannot be accepted! we assume here targsrvy=GA *****')
        targsrvy = ['GA']

    targclass = None
    if args.targclass is not None:
        targclass = [ str(x) for x in args.targclass.split(",") ]


    wlranges = None
    if args.wlranges[0] is not None:
        wlranges=[]
        for i in range(len(args.infiles)):
            wlranges.append([float(x) for x in args.wlranges[i].split(",")])







    ### Now we have both infiles and wlranges array. We use  l1_fileinfo function to update these two parameters and join_arms
    ### and puth them in the right order, if needed. However, we had similar test done by APSOB

    infiles_info = l1_fileinfo(infiles, wlranges=wlranges)
    infiles = infiles_info['infiles']
    wlranges     = infiles_info['wlranges']

    ## We also update the args.wlranges and args.infiles for reporting purpose
    args.infiles = infiles
    args.wlranges = wlranges

    # print args and assigned/default values on the screen
    print_args(args,module='SP_Ace', version= __aps_space_version__, path=outpath, headname=args.headname)



    ######### PHASE 1: PREPARATION ############

    ### Before running the main worker, we first make sure eveyrthing is OK
    try:
        infiles_check = l1_fileinfo(infiles)
        fcheck_targs, fcheck_idt, fcheck_info, fcheck_la, fcheck_wcs = gen_targlist(infiles_check['infiles'][0],
            infiles_check['mode'], aps_ids = aps_ids, targsrvy= targsrvy, targclass = targclass,
            mask_aps_ids = None, area=None, mask_areas=None, la_out=False)

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
    except:
        print('ERROR1 : %s' %(sys.exc_info()[1]))
        return

    ######### PHASE 2: MAIN ALANLYSES ############
    try:

        #put in a list the SP_Ace arguments
        space_args = [args.Salaris_MH, args.RV_ini, args.ABD_loop, args.SN_sp_file, args.norm_rad, args.wave_separ, args.error_est]

        proc_many(infiles, apsfile , param_fits, outpath, working_dir, GCOG_dir, args.space_exe,
            wlranges=wlranges, figdir=figdir, spec_fits=spec_fits, aps_ids=aps_ids, nthreads=args.mp,
            overwrite=args.overwrite, fig=args.fig, targsrvy=targsrvy, targclass=targclass, space_args=space_args)

    except:
        print('ERROR2 : %s' %(sys.exc_info()[1]))
        return

########################################################################
if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).



    option= [
    #APSOB options
    '--infiles', '<PYAPS_DATA>/opr4_jan2022/<night>/stack_<runid>.fit', '<PYAPS_DATA>/opr4_jan2022/<night>/stack_<runid>.fit',
    '--wlranges', '4915.0,6200.0','5800.0,6860.0',
    '--targsrvy', 'GA-OC',
    '--targclass', 'STAR', #with the present L1 there is no targclass specified.
    '--aps_ids', 'None',
    '--apsfile', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/stack_<runid>__stack_<runid>_APS.fits',
    '--working_dir', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/space_wd/',
    '--space_exe', '<PYAPS_DIR>/CS/SPACE/SPACE_v1.4W',
    '--outpath', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/',
    '--headname', 'stacked_<runid>__stacked_<runid>',
    '--outspec', 'False', #if True, the code also produces plots of the best fitted model
    '--fig', 'True', #if true, it output diagnostic plots in .png format
    '--overwrite', 'True', #overwrite the HDUlist result
    '--mp' ,'2', # Number of threads for the fits
    #option for SP_Ace. SP_Ace uses also infiles and wlranges.
    '--GCOG_dir', '<PYAPS_DIR>/PyAPS_templates/templates_SPACE',
    '--error_est', 'False',
    #the following options are not necessary and have default values
#    '--Salaris_MH', 'True',
#    '--RV_ini', 'True',
#    '--ABD_loop', 'True',
#    '--SN_sp_file', 'True',
#    '--norm_rad', '30.0',
#    '--wave_separ', 'True'
    ] 
    
    spaceweave(options=option)


