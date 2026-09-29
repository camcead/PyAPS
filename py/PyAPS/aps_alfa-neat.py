"""
ALFA+NEAT wrapper
Roger Wesson, Oct 2020
calls alfa (v2.0.53) and neat (v2.3.25) to analyse nebular spectra


versions:
 1.0 By Roger Wesson, Oct 2020
 1.2 By A. Molaeinezhad (CASU, December 2020)- Code fully reshuffled with all required changes to make it compatible with PyAPS platform and the latest weave data model (v 8.0)
 1.3 By A. Molaeinezhad (CASU, March 2021) - New keywords added to the input params (alfa_exe and neat_exe) and documentation updated. 
 1.4 By Roger Wesson, April 2021 (email received on 12 April 20201)
 1.5 By Roger Wesson, February 2022
 1.6 By Roger Wesson, May 2022
 1.7 By A. Molaeinezhad (CASU, December 2020) - IFU Compatibility
 1.8 By A. Molaeinezhad (CASU, October 2023)
 2.0 By A. Molaeinezhad (CASU, October 2023)

How to run:
Mode1.  Through the command line (bash mode):
python3 aps_alfa-neat.py --infiles <PYAPS_DATA>/AMY/4365/stacked_1003690.fit <PYAPS_DATA>/AMY/4365/stacked_1003689.fit --aps_ids 995,989,984,998,987,978 --targsrvy None --targclass None --mask_aps_ids None --area None --mask_areas None --wlranges None --sens_corr True --mask_gaps True --safe_mask_gaps True --tellurics False --vacuum False --fill_gap False --arms_ratio None --outpath <PYAPS_DIR>/PyAPS_results/20170911/4356/ --neat_null <PYAPS_DIR>/CS/ALFA-NEAT/neat_null.fits --alfa_exe <PYAPS_DIR>/CS/ALFA-NEAT/ALFA/alfa --neat_exe <PYAPS_DIR>/CS/ALFA-NEAT/NEAT/neat --columnnames <PYAPS_DIR>/configs/alfa_neat_columnnames.json --headname stacked_1003690__stacked_1003689 --overwrite True --mproc 1

Mode2.  (Develop/debug mode): Update the parameters at the bottom of this file and then, directly call this file from python




(IMPORTANT) How to install ALFA and NEAT:
Download the latest version of ALFA and NEAT (master repository, not the version provided on ALFA_NEAT repo)
To avoid permission issue, for both ALFA and NEAT in Makefile, change the value of PREFIX to $(HOME)/.local
cd ALFA(NEAT)
make
make install (it is necessary, because this steps provides all necessary libraries)

To uninstall them:
For both ALFA and NEAT
make clean
make uninstall

then run the following command to make sure your session has been reset and all symbolic links has been updated (e.g. if it still look for alfa in /usr/local/bin):
hash -r




todo list:
if HR, there are not 107 lines, need to pad with zeroes
test on LIFU
Multiprocessing at target level
Develop a function to handle NEAT null file for different configurations
To work on Issues related to different L1 configurations and different binning (2,4,) and also mIFU and LIFU
NEAT NULL file only compatible with the LR mode. It should be compatible with any subset of data (e.g. limited wavelength ranges)
Complete in-line comments
you Should consider the scenario, in which you only have access to a limited wavelength range (wavelength window)

** (May 2021) What about mask_gaps or safe_mask_gaps or tellurics corrections??????? Why you set them all to False????


# APS Notes:
1- A.M. reshuffled with all required changes to make it compatible with PyAPS platform and the latest weave data model (v 8.0).
2- NSPEC replaced by FIBREID (in some L1 configurations we do not have NSPEC column)
3- Now, the code accept targclass and targsrvy to find the appropriate targets (compatible with WEAVE data model V 8.0)
4- In order to test the current version (1.2) with OPR3b data, please use aps_id parameter to choose targets belong to Nebula or SKY classes.
5- Data preparation parts of the code replaced by robust tools we already developed as part of the PyAPS platform
6- Now the code, can be run on any desired subset of the data (specific set of fibres, different classes or different surveys)
7- colnames (.py) file replaced by a .JSON file. Now you can call it directly through the command line arguments. 
8- Need to address different binning modes?
9- fibre status, and all QC done through the APSOBJ super class. 
10- The new structure of the code (v 1.2, January 2021) is fully compatible with the PyAPS platform. If you need to make any changes, please follow the general APS guidlines. 
11- What about multiprocessing? 
12- On 24 March 2021  A. Molaeinezhad added --alfa_exe and --neat_exe added to the input arguments, represent the full path for ALFA and NEAT, respectively.
                Therefore, no root access is needed to install the source codes (Alfa and neat) and you can set the full path of these executable files in the configs.
13- to install alfa and neat source files, I simply used the make file provided in the ALFA-NEAT repo by simply "make" it. You may 
                need to run make clean and make uninstall before reinstalling the code



HISTORY:

June 2020: Version 1.0 (Original version) by Maria Monguio 
December 2020 : (By A. Molaeinezhad) - Code fully reshuffled with all required changes to make it compatible with PyAPS platform and the latest weave data model (v 8.0)
24 March 2021: (By A. Molaeinezhad) - --alfa_exe and --neat_exe added to the input arguments, represent the full path for ALFA and NEAT, respectively.
                Therefore, no root access is needed to install the source codes (alfa and neat) and you can set the full path of these executable files in the configs.
30 April 2021: file updated by Roger (please see the installation guide for alfa and neat)
18 May 2021: new parameters added (safe_mask_gaps) to mask gaps (read from lookup table)
18 May 2021: update mask_gaps and safe_mask_gaps to True. (Not sure how it affects the results)
10 Feb 2022: New parallelisation method added by Roger Wesson
04 My 2022: missing header keywords (Requested by WAS) added/sorted.
06 Sept 2022: Add mask_aps_ids, area and mask_areas into the criteria to select aps_ids/spaxels in IFU mode
09 Oct 2023: deep-catalogue paramters for alfa is now being generated from alfaexe path
17 Oct 2023: [Major upgrade] testing, debugging and fixing the parallelisation approach used by the code. 

**** Wrong parallelisation approach detected in this module. TODO: Check the aps_rvs code as reference 

"""

import argparse
import multiprocessing as mp
import numpy as np
import os
import glob
os.environ['OMP_NUM_THREADS'] = '1'
import subprocess
import sys
import time
from astropy.io import fits
# from columnnames import *
import json
from pathlib import Path

import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import aps_constants

__aps_alfa_neat_version__ = 1.3


######################################################################################################
def alfa_neat_serial_worker(infiles, outpath, working_dir, neat_null, columnnames, headname, alfa_exe, neat_exe,  aps_ids = None, targsrvy = None, targclass = None, mask_aps_ids = None, area=None, mask_areas=None,
    wlranges = None, sens_corr=True, mask_gaps=True, safe_mask_gaps=True, tellurics=False, vacuum=False, fill_gap=False, arms_ratio = None, overwrite = True, mproc =1 ):


    # Start timer
    starttime=time.time()

    # load columnnames file (in JSON format)
    colnames = json.load(open(columnnames))

    # Create default alfa input
    alfaopts=["--normalise","0","--deep-catalogue",str(Path(alfa_exe).parent.joinpath('linelists/optical_WEAVE.cat')),"--output-dir",working_dir,"--resolution-tolerance-1","1000","--bad-data","300"]



    # Read data into APSOB
    APSOBJ = APSOB(infiles, skysub=True,  targsrvy= targsrvy, targclass = targclass, wlranges=wlranges, aps_ids=aps_ids, mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio= arms_ratio, join_arms=False)

    # READ targets in the class
    targs_list=APSOBJ.data()
    setups=APSOBJ.setups()
    resolutions=APSOBJ.resolution()


    ## Prepare format code for the spectra in each band
    bluefmt = str(len(targs_list[0].spectra[0].wave))+"E"
    redfmt  = str(len(targs_list[0].spectra[1].wave))+"E"


    # resolution guess
    alfa_resolution_guess=resolutions[0]*2



    # If we are dealing with HR Mode (and also check binning in x and y)
    if setups[0] == 'BLUEH11':
        alfaopts=alfaopts+["--rebin","4"]
        alfa_resolution_guess = 0.25*alfa_resolution_guess

    alfaopts=alfaopts+["--resolution-guess",str(alfa_resolution_guess)]

    # Create an empty file-list to keep outfiles
    filelist = []


    # Now loop over all available targets and ...
    for tg_i, tg in enumerate(targs_list):

        # Extract basic indexing info for this target
        TARGID  = tg.targid
        FIBREID = tg.aps_id
        CNAME   = tg.cname
        NSPEC   = tg.meta[0]['NSPEC']


        outfile=working_dir+'/'+headname+"_"+str(FIBREID)+"_ALFA-NEAT.fits"

        if os.path.exists(outfile) and not overwrite:
            print(" [ALFA-NEAT] fibre "+str(FIBREID)+" output file exists, not overwriting")
            print("")
            continue
        else:
            os.system('rm -f '+working_dir+'/'+headname+"_"+str(FIBREID)+"_ALFA-NEAT.fits")
            print(" [ALFA-NEAT] analysing fibre "+str(FIBREID))
            print("")

        # clear working_dir directory
        os.system('rm -f '+working_dir+'/tmp_'+str(FIBREID)+'_*')

        # make BLUE spectrum
        blue_flux = tg.spectra[0].flux
        blue_wave = tg.spectra[0].wave
        blue_spectrum = np.stack((blue_wave,blue_flux),axis=1)

        np.savetxt(working_dir+"/tmp_"+str(FIBREID)+"_bluespectrum.dat",blue_spectrum)

        # make RED spectrum
        red_flux = tg.spectra[1].flux
        red_wave = tg.spectra[1].wave
        red_spectrum = np.stack((red_wave,red_flux),axis=1)
        np.savetxt(working_dir+"/tmp_"+str(FIBREID)+"_redspectrum.dat",red_spectrum)

        # analyse them

        try:
            subprocess.check_call([alfa_exe,working_dir+"/tmp_"+str(FIBREID)+"_bluespectrum.dat"]+alfaopts)
        except subprocess.CalledProcessError as e:
            if e.returncode>0:
                print("can't analyse this fibre, alfa returned error ",e.returncode)
                continue

        try:
            subprocess.check_call([alfa_exe,working_dir+"/tmp_"+str(FIBREID)+"_redspectrum.dat"]+alfaopts)
        except subprocess.CalledProcessError as e:
            if e.returncode>0:
                print("can't analyse this fibre, alfa returned error ",e.returncode)
                continue

        # read in the line lists, concatenate line list, write out

        blueALFA=fits.open(working_dir+"/tmp_"+str(FIBREID)+"_bluespectrum.dat_fit.fits")
        redALFA =fits.open(working_dir+"/tmp_"+str(FIBREID)+"_redspectrum.dat_fit.fits")

        lines=blueALFA["LINES"].data
        lines=np.append(lines,redALFA["LINES"].data)

        # we sill use this to format the output fits files
        n_lines = len(lines)
        n_lines_fmt = "107E"


        fits.writeto(working_dir+"/tmp_"+str(FIBREID)+"_lines.fits",lines,overwrite=True)
        fits.setval(working_dir+"/tmp_"+str(FIBREID)+"_lines.fits", 'EXTNAME', value='LINES', ext=1)

        # analyse with NEAT. check for no Hbeta found error

        # if no lines detected, copy the null neat results

        if blueALFA["QC"].data["NumberOfLines"]==0 and redALFA["QC"].data["NumberOfLines"]==0:
            NEAT=fits.open(neat_null)
            NEAT["LINES"].data["WlenObserved"][0:n_lines]=lines["WlenObserved"]
            NEAT["LINES"].data["WlenRest"][0:n_lines]=lines["WlenRest"]
            NEAT["LINES"].data["Flux"][0:n_lines]=lines["Flux"]
            NEAT["LINES"].data["Uncertainty"][0:n_lines]=lines["Uncertainty"]
            NEAT["LINES"].data["Ion"][0:n_lines]=lines["Ion"]

        # if some lines were detected, run neat
        else:
            try:
                subprocess.check_call([neat_exe,"-i",working_dir+"/tmp_"+str(FIBREID)+"_lines.fits","-u","--no-omp"])
                NEAT=fits.open(working_dir+"/tmp_"+str(FIBREID)+"_lines.fits")
            except subprocess.CalledProcessError as e:
                if e.returncode==201:
                    # open null output file, copy ALFA data into the lines table
                    NEAT=fits.open(neat_null)
                    NEAT["LINES"].data["WlenObserved"][0:n_lines]=lines["WlenObserved"]
                    NEAT["LINES"].data["WlenRest"][0:n_lines]=lines["WlenRest"]
                    NEAT["LINES"].data["Flux"][0:n_lines]=lines["Flux"]
                    NEAT["LINES"].data["Uncertainty"][0:n_lines]=lines["Uncertainty"]
                    NEAT["LINES"].data["Ion"][0:n_lines]=lines["Ion"]

        # fill the LINES extension with empty values if n_lines<107

        if n_lines<107:
            for linefill in range(n_lines,107):
                NEAT["LINES"].data[linefill]=[0,0,0,0,0,0,"",0,0,0,0,0]

        # compile the final file, fits from blue and red, lines from neat analysis, QC from both

        bluefit=blueALFA["FIT"].data
        redfit=redALFA["FIT"].data
        lines=NEAT["LINES"].data
        results=NEAT["RESULTS"].data
        blueQC=blueALFA["QC"].data
        redQC=redALFA["QC"].data
        neatQC=NEAT["QC"].data



        # compile the final file, fits from blue and red, lines from neat analysis, QC from both
      
        bluefit=blueALFA["FIT"].data
        redfit=redALFA["FIT"].data
        lines=NEAT["LINES"].data
        results=NEAT["RESULTS"].data
        blueQC=blueALFA["QC"].data
        redQC=redALFA["QC"].data
        neatQC=NEAT["QC"].data
      
        # create HDU
        hdr = fits.Header()
        hdr['COMMENT'] = "WEAVE Contributed Software: ALFA-NEAT"
        hdr['DATAMVER'] = "8.00"
        hdr['CS_CODE'] = "ALFA-NEAT"
        hdr['CS_VER'] = "1.5"
        hdr['CS_DOI'] = ""
        hdr['CS_NME1'] = "Roger"
        hdr['CS_NME2'] = "Wesson"
        hdr['CS_MAIL'] = "rw@nebulousresearch.org"
        hdr['PROV0001'] = ",".join(infiles)
        hdr['PROV0002'] = "-"
      
        phdu = fits.PrimaryHDU(header=hdr)
      
        # add extensions
      
        # FIT extension
      
        c1 = fits.Column(name="NSPEC",array=[NSPEC], format="I")
        c2 = fits.Column(name="FIBREID",array=[FIBREID], format="I")
        c3 = fits.Column(name="CNAME",array=[CNAME], format="A20")
      
        c4 = fits.Column(name="ALFA_BLUE_INPUT_SPECTRUM",array=[bluefit["InputSpec"]], format=bluefmt)
        c5 = fits.Column(name="ALFA_BLUE_CONTSUBBED",array=[bluefit["ContSubbedInput"]], format=bluefmt)
        c6 = fits.Column(name="ALFA_BLUE_FITTED_SPECTRUM",array=[bluefit["FittedSpec"]], format=bluefmt)
        c7 = fits.Column(name="ALFA_BLUE_CONTINUUM",array=[bluefit["Continuum"]], format=bluefmt)
        c8 = fits.Column(name="ALFA_BLUE_SKYLINES",array=[bluefit["SkyLines"]], format=bluefmt)
        c9 = fits.Column(name="ALFA_BLUE_RESIDUALS",array=[bluefit["Residuals"]], format=bluefmt)
        c10= fits.Column(name="ALFA_BLUE_UNCERTAINTY",array=[bluefit["Uncertainty"]], format=bluefmt)
        c11= fits.Column(name="ALFA_RED_INPUT_SPECTRUM",array=[redfit["InputSpec"]], format=redfmt)
        c12= fits.Column(name="ALFA_RED_CONTSUBBED",array=[redfit["ContSubbedInput"]], format=redfmt)
        c13= fits.Column(name="ALFA_RED_FITTED_SPECTRUM",array=[redfit["FittedSpec"]], format=redfmt)
        c14= fits.Column(name="ALFA_RED_CONTINUUM",array=[redfit["Continuum"]], format=redfmt)
        c15= fits.Column(name="ALFA_RED_SKYLINES",array=[redfit["SkyLines"]], format=redfmt)
        c16= fits.Column(name="ALFA_RED_RESIDUALS",array=[redfit["Residuals"]], format=redfmt)
        c17= fits.Column(name="ALFA_RED_UNCERTAINTY",array=[redfit["Uncertainty"]], format=redfmt)
      
        hdu_fit=fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17])
      
        # LINELIST extension
      
        # concatenate line IDs
      
        lineids=[x.ljust(12) for x in lines["Ion"]]
        lineids=np.array(["".join(lineids)])
      
        c1 = fits.Column(name="NSPEC",array=[NSPEC], format="I")
        c2 = fits.Column(name="FIBREID",array=[FIBREID], format="I")
        c3 = fits.Column(name="CNAME",array=[CNAME], format="A20")
      
        c4 = fits.Column(name="ALFA_WAVE_OBS",array=[lines["WlenObserved"]], format=n_lines_fmt)
        c5 = fits.Column(name="ALFA_WAVE_REST",array=[lines["WlenRest"]], format=n_lines_fmt)
        c6 = fits.Column(name="ALFA_FLUX",array=[lines["Flux"]], format=n_lines_fmt)
        c7 = fits.Column(name="ALFA_FLUX_E",array=[lines["Uncertainty"]], format=n_lines_fmt)
        c8 = fits.Column(name="ALFA_FWHM",array=[lines["FWHM"]], format=n_lines_fmt)
        c9 = fits.Column(name="NEAT_LINEID",array=[lineids], format="1284A12")
        c10= fits.Column(name="NEAT_DEREDDENEDFLUX",array=[lines["DereddenedFlux"]], format=n_lines_fmt)
        c11= fits.Column(name="NEAT_DEREDDENEDFLUX_E_UPPER",array=[lines["DereddenedFluxHi"]], format=n_lines_fmt)
        c12= fits.Column(name="NEAT_DEREDDENEDFLUX_E_LOWER",array=[lines["DereddenedFluxLo"]], format=n_lines_fmt)
        c13= fits.Column(name="NEAT_ABUNDANCE",array=[lines["Abundance"]], format=n_lines_fmt)
        c14= fits.Column(name="NEAT_ABUNDANCE_E_UPPER",array=[lines["AbundanceLow"]], format=n_lines_fmt)
        c15= fits.Column(name="NEAT_ABUNDANCE_E_LOWER",array=[lines["AbundanceHigh"]], format=n_lines_fmt)
      
        hdu_lines=fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15])
      
        # RESULTS extension
      
        c1 = fits.Column(name="NSPEC",array=[NSPEC], format="I")
        c2 = fits.Column(name="FIBREID",array=[FIBREID], format="I")
        c3 = fits.Column(name="CNAME",array=[CNAME], format="A20")
      
        # create columns for measurements which come from ALFA
      
        nlines=blueQC["NumberOfLines"][0]+redQC["NumberOfLines"][0]
      
        c4=fits.Column(name="ALFA_LINES_DETECTED",array=[nlines],format="I")

        # calculate RV and its standard deviation

        rvsum=0
        rvsqsum=0
        nvlines=0

        for line in range(len(lines)):
            if lines[line]["WlenObserved"]>0 and lines[line]["Flux"]>0:
                velocity=299792.*(lines[line]["WlenObserved"]-lines[line]["WlenRest"])/lines[line]["WlenRest"]
                rvsum=rvsum+velocity
                rvsqsum=rvsqsum+velocity**2
                nvlines=nvlines+1

        if nvlines>0:
            rvmean=rvsum/nvlines
            rvstddev=((1/nvlines)*((rvsqsum) - (((rvsum)**2)/nvlines)))**0.5
        else:
            rvmean=0
            rvstddev=0
      
        c5=fits.Column(name="ALFA_RV",array=[rvmean],format="E")
        c6=fits.Column(name="ALFA_RV_EL",array=[rvmean-rvstddev],format="E")
        c7=fits.Column(name="ALFA_RV_EU",array=[rvmean+rvstddev],format="E")
        columns = [c1,c2,c3,c4,c5,c6,c7]

        # copy the NEAT results

        for row in range(len(results)):
            # diagnostics which are never calculated are omitted
            # that is ones that rely on lines outside WEAVE range
            if results["Quantity"][row] not in colnames:
                continue
      
          # copy the rest
      
            colname=colnames[results["Quantity"][row]]
            value=results["Value"][row]
            uu=results["UpperUncertainty"][row]
            lu=results["LowerUncertainty"][row]
      
            c1=fits.Column(name=colname,array=[value],format="E")
            c2=fits.Column(name=colname+"_EL",array=[lu],format="E")
            c3=fits.Column(name=colname+"_EU",array=[uu],format="E")
      
            columns=columns+[c1,c2,c3]
      
      
      
        # RL reliability flags
        c1=fits.Column(name="NEAT_NIIRLRELIABLE",array=[neatQC["NIIRLsReliable"][0]],format="L")
        c2=fits.Column(name="NEAT_OIIRLRELIABLE",array=[neatQC["OIIRLsReliable"][0]],format="L")

        columns=columns+[c1,c2]

        hdu_results=fits.BinTableHDU.from_columns(columns)

        # name extensions

        hdu_fit.name="FIT"
        hdu_lines.name="LINELIST"
        hdu_results.name="RESULTS"

        # write file

        hdul = fits.HDUList([phdu,hdu_fit,hdu_lines,hdu_results])
        hdul.writeto(outfile, checksum=True, overwrite=True)

        continue # comment out to create null NEAT file containing linelist and results extensions with zeroes for all values
        
        c1 = fits.Column(name="WlenObserved",array=np.zeros(shape=(116)), format="E")
        c2 = fits.Column(name="WlenRest",array=np.zeros(shape=(116)), format="E")
        c3 = fits.Column(name="Flux",array=np.zeros(shape=(116)), format="E")
        c4 = fits.Column(name="Uncertainty",array=np.zeros(shape=(116)), format="E")
        c5 = fits.Column(name="FWHM",array=np.zeros(shape=(116)), format="E")
        c6 = fits.Column(name="Ion",array=lines["Ion"], format="A12")
        c7 = fits.Column(name="DereddenedFlux",array=np.zeros(shape=(116)), format="E")
        c8 = fits.Column(name="DereddenedFluxHi",array=np.zeros(shape=(116)), format="E")
        c9 = fits.Column(name="DereddenedFluxLo",array=np.zeros(shape=(116)), format="E")
        c10= fits.Column(name="Abundance",array=np.zeros(shape=(116)), format="E")
        c11= fits.Column(name="AbundanceLow",array=np.zeros(shape=(116)), format="E")
        c12= fits.Column(name="AbundanceHigh",array=np.zeros(shape=(116)), format="E")

        hdu_linelist=fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12])
        hdu_linelist.name="LINES"

        c1 = fits.Column(name="Quantity",array=results["Quantity"], format="A40")
        c2 = fits.Column(name="Value",array=np.zeros(shape=(166)), format="E")
        c3 = fits.Column(name="UpperUncertainty",array=np.zeros(shape=(166)), format="E")
        c4 = fits.Column(name="LowerUncertainty",array=np.zeros(shape=(166)), format="E")

        hdu_results=fits.BinTableHDU.from_columns([c1,c2,c3,c4])
        hdu_results.name="RESULTS"

        c1 = fits.Column(name="NIIRLsReliable",array=[False], format="L")
        c2 = fits.Column(name="OIIRLsReliable",array=[False], format="L")

        hdu_qc=fits.BinTableHDU.from_columns([c1,c2])
        hdu_qc.name="QC"

        hdul = fits.HDUList([phdu,hdu_linelist,hdu_results,hdu_qc])
        hdul.writeto("neat_null.fits", checksum=True, overwrite=True)
        sys.exit()


    endtime=time.time()
    m=(endtime-starttime)//60
    s=(endtime-starttime)%60

    print(" [ALFA-NEAT] analysis complete: processed %i fibres in %02dm%02ds"%(len(targs_list),m,s))

######################################################################################################

def make_output_fits(infiles, working_dir, outpath, headname, overwrite=True):

    # after all pixels have been processed, combine results into single FITS file
    print(" [ALFA-NEAT] writing final file")

    filelist = glob.glob(working_dir+'/'+headname+"_*_ALFA-NEAT.fits")
    if len(filelist) == 0:
        sys.exit(f"no file with {headname}_*_ALFA-NEAT.fits name structure found in the working directory: {working_dir}")

    # Start timer
    starttime=time.time()

    # start with first file
    fdata=fits.open(filelist[0])
    output_FIT=fdata["FIT"].data
    output_LINELIST=fdata["LINELIST"].data
    output_RESULTS=fdata["RESULTS"].data

    # del filelist[0]
    for f in filelist:
        # read the file
        fdata=fits.open(f)

        # append the table rows
        output_FIT=np.append(output_FIT,fdata["FIT"].data)
        output_LINELIST=np.append(output_LINELIST,fdata["LINELIST"].data)
        output_RESULTS=np.append(output_RESULTS,fdata["RESULTS"].data)

    # now write the final file
    # create primary HDU

    hdr = fits.Header()
    hdr['COMMENT'] = "WEAVE Contributed Software: ALFA-NEAT"
    hdr['DATAMVER'] = "8.00"
    hdr['CS_CODE'] = "ALFA-NEAT"
    hdr['CS_VER'] = "1.5"
    hdr['CS_DOI'] = ""
    hdr['CS_NME1'] = "Roger"
    hdr['CS_NME2'] = "Wesson"
    hdr['CS_MAIL'] = "rw@nebulousresearch.org"
    hdr['PROV0001'] = str(infiles[0])
    try:
        hdr['PROV0002'] = str(infiles[1])
    except:
        hdr['PROV0002'] = "-"

    phdu = fits.PrimaryHDU(header=hdr)

    # then each extension

    hdu_fit = fits.BinTableHDU(data=output_FIT,name="FIT")
    hdu_linelist = fits.BinTableHDU(data=output_LINELIST,name="LINELIST")
    hdu_results = fits.BinTableHDU(data=output_RESULTS,name="RESULTS")

    # write the file
    hdul = fits.HDUList([phdu,hdu_fit,hdu_linelist,hdu_results])
    hdul.writeto(outpath+"/"+headname+"_ALFA-NEAT.fits" , checksum=True, overwrite=overwrite)
    print(f"Writing a final fits file into {outpath}/{headname}_ALFA-NEAT.fits")

    # all done

    endtime=time.time()
    m=(endtime-starttime)//60
    s=(endtime-starttime)%60
    print(" [ALFA-NEAT] writing the final output fits file in %02dm%02ds"%(m,s))

    
######################################################################################################

def alfa_neat_many(infiles, outpath, working_dir, neat_null, columnnames, headname, alfa_exe, neat_exe,  aps_ids=None, targsrvy=None, targclass=None, mask_aps_ids=None, area=None, mask_areas=None, wlranges=None, sens_corr=True, mask_gaps=True, safe_mask_gaps=True, tellurics=None, vacuum=False, fill_gap=False, arms_ratio=None, overwrite=True, mproc=1 ):

# process many files
# this procedure is based on aps_amy.py (rw, 2022-02-08)

    nthreads=mproc
    if nthreads > 1:
        parallel = True
    else:
        parallel = False

    res = []

    if (overwrite) :
        # clear working_dir directory
        os.system('rm -f '+working_dir+'/*')


    if parallel:
            APSOBJ = APSOB(infiles, skysub=True,  targsrvy= targsrvy, targclass = targclass, wlranges=wlranges, aps_ids=aps_ids, mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=False)

            targs_list=APSOBJ.data()
            setups=APSOBJ.setups()
            resolutions=APSOBJ.resolution()

            if len(targs_list) < nthreads:
                nthreads = len(targs_list)
            print('Update nthreads param to %d'%(nthreads))

            pool = mp.Pool(nthreads)

            for tg_i, tg in enumerate(targs_list):
                # Extract basic indexing info for this target
                TARGID  = tg.targid
                FIBREID = tg.aps_id
                res.append( pool.apply_async(
                proc_weave_wrapper,(infiles, outpath, working_dir, neat_null, columnnames, headname, alfa_exe, neat_exe),{'targsrvy': targsrvy, 'targclass': targclass, 'aps_ids':FIBREID, 'mask_aps_ids': mask_aps_ids, 'area': area, 'mask_areas': mask_areas, 'wlranges':wlranges , 'mproc':mproc, 'overwrite':overwrite, 'sens_corr':sens_corr, 'mask_gaps': mask_gaps, 'safe_mask_gaps': safe_mask_gaps, 'tellurics':tellurics, 'vacuum':vacuum, 'fill_gap':fill_gap, 'arms_ratio':arms_ratio}))

    if parallel:
            pool.close()
            pool.join()

    else:
        alfa_neat_serial_worker(infiles, outpath, working_dir, neat_null, columnnames, headname, alfa_exe, neat_exe,  aps_ids = aps_ids, targsrvy = targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids, area=area, mask_areas=mask_areas, 
        wlranges = wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps, tellurics=tellurics, vacuum=vacuum, fill_gap=fill_gap, 
        arms_ratio = arms_ratio, overwrite = overwrite, mproc = mproc )

    # now make a single fits file out of all single fits (for individual fibre)
    make_output_fits(infiles, working_dir, outpath, headname, overwrite=overwrite)

def proc_weave_wrapper(*args, **kwargs):
    try:
        ret = alfa_neat_serial_worker(*args, **kwargs)
        return ret
    except:
        print('failed with these arguments', args, kwargs)
        raise




def alfa_neat_runner(options=None):
    
    parser = build_common_parser(
        description=None,
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "output"],
        # this script has no --catdir/--caldir/--configdir at all, so the
        # "caldirs" group is simply not included above.
        overrides={
            # this script's own wording differs from the canonical text,
            # preserved verbatim rather than silently switched to canonical
            "mask_aps_ids": {"help": "comma-separated list of APS_IDS to be masked"},
            "wlranges": {"help": "wavelenght range array for each elements of the setup"},
            "vacuum": {"default": False, "help": "transform wavelenght from air to vacuum"},
            "join_arms": {"help": "[Not in used] Stitch two arms"},
            "outpath": {"help": "Directory to keep WEAVE_RVS outputs"},
            "overwrite": {"default": True,
                          "help": "If enabled the code will overwrite the existing products, otherwise it will skip them"},
        },
        extra_args=[
            (("--alfa_exe",), dict(
                help='Full path of the alfa executable file', type=none_or_str, default=None, required=True)),
            (("--neat_exe",), dict(
                help='Full path of the neat executable file', type=none_or_str, default=None, required=True)),
            (("--columnnames",), dict(help='Futh path of the file, introducing the columns names (.JSON file)',
                type=none_or_str, default=None, required=True)),
            (("--neat_null",), dict(
                help='Full path of NEAT NULL fits file', type=none_or_str, default=None, required=True)),
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
    # join_arms<2 correction, arms_ratio length assert) is unchanged; the
    # duplicate join_arms<2 check further below (which repeated the exact
    # same, already-applied correction a second time) is dropped as
    # redundant — harmless, since the check is idempotent.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio

    ## For now, we assume this code only accept two input files
    assert (len(args.infiles)) == 2, 'The current version of this code needs two input L1 files [BLUE, RED]'


    if not os.path.exists(args.columnnames):
        sys.exit('No columnnames file found in %s' %(args.columnnames))
    columnnames = args.columnnames


    if not os.path.exists(args.neat_null):
        sys.exit('No NEAT NULL file found in %s' %(args.neat_null))
    neat_null = args.neat_null


    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ','')



    # Check alfa executable
    if not os.path.exists(args.alfa_exe):
        sys.exit('No alfa executable found in %s' %(args.alfa_exe))


    # Check neat executable
    if not os.path.exists(args.neat_exe):
        sys.exit('No neat executable found in %s' %(args.neat_exe))



    # Create a working directory to keep temporary and working files
    if not os.path.exists(outpath + '/alfa_neat_wd'):
        os.mkdir(outpath + '/alfa_neat_wd')
        print(outpath + '/alfa_neat_wd directory created')
    working_dir = outpath + '/alfa_neat_wd/'


    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
    area, mask_areas = resolved.area, resolved.mask_areas

    # print args and assigned/default values on the screen
    print_args(args,module='APS_ALFA_NEAT', version= __aps_alfa_neat_version__, path=outpath, headname=args.headname)



    alfa_neat_many(args.infiles, outpath, working_dir, neat_null, columnnames, args.headname, args.alfa_exe, args.neat_exe,  aps_ids = aps_ids, targsrvy = targsrvy, targclass = targclass,
        mask_aps_ids = mask_aps_ids, area=area, mask_areas=mask_areas, wlranges = wlranges, sens_corr=args.sens_corr, mask_gaps=args.mask_gaps, safe_mask_gaps=args.safe_mask_gaps, 
        tellurics=args.tellurics, vacuum=args.vacuum, fill_gap=args.fill_gap, 
        arms_ratio = arms_ratio, overwrite = args.overwrite, mproc =args.mproc )



#################################################################################################
if __name__ == '__main__':

# READ DATA and put them in the APSOBJ OBJECT

    debug_demo= ['--infiles', '/scratch/aps/PyAPS/PyAPS_data_dev/L1/AMY/4365/stacked_1003690.fit', '/scratch/aps/PyAPS/PyAPS_data_dev/L1/AMY/4365/stacked_1003689.fit',
    '--aps_ids', '995,989,984,998,987,978', # or 'None' to run for all available fibreids
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area', 'None',
    '--mask_areas', 'None',
    # '--wlranges',  '4000.0,6000.0','6200.0,9000.0',
    '--alfa_exe', '<PYAPS_DIR>/CS/ALFA-NEAT/ALFA/alfa',
    '--neat_exe', '<PYAPS_DIR>/CS/ALFA-NEAT/NEAT/neat',
    '--wlranges', 'None',
    '--sens_corr', 'True',
    '--safe_mask_gaps', 'True',
    '--mask_gaps', 'True',
    '--tellurics', 'False',
    '--vacuum', 'False',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0, 0.83',
    '--outpath', '<PYAPS_DATA>_dev/L2/20170911/4356/',
    '--neat_null', '<PYAPS_DIR>/CS/ALFA-NEAT/neat_null.fits',
    '--columnnames', '<PYAPS_DIR>/configs/alfa_neat_columnnames.json',
    '--headname', 'stacked_1003690__stacked_1003689',
    '--overwrite', 'True',
    '--mproc' ,'2']

    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    alfa_neat_runner(options=debug_demo)



