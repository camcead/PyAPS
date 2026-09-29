from   astropy.io import fits
import numpy      as np
import glob
import dill
import os
os.environ['OMP_NUM_THREADS'] = '1'
import logging
import PyAPS
from PyAPS import ExGalutil
from   scipy.interpolate import interp1d
import sys
import time
import pickle
import datetime
from pathlib import Path
from astropy.table import Table, vstack, join
APSVERS = PyAPS.__version__

# try:
#     # Try to use local version in sitePackages
# except:
# Then use system installed version instead
from ppxf.ppxf_util import log_rebin, gaussian_filter1d



"""
PURPOSE:
  This file contains a collection of functions necessary to prepare the input
  data for the following analysis steps.
"""

# PHYSICAL CONSTANTS
Clight = 299792.458  # km/s



def timer(start,end, message):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    ExGalutil.prettyOutput_Info("Running time [{}]: {:0>2}:{:0>2}:{:05.2f}".format(message,int(hours),int(minutes),seconds))




def rejectDefunctSpaxels_applySNRThreshold(exgal_targ, configs):
    """
    Select defunct spaxels, in particular those containing np.nan's or have a
    negative median. Further apply the minimum SNR threshold. Then mark those
    spaxels as outside of the analysis region.
    """
    # Select defunct spaxels
    idx_good = np.where( np.logical_and( np.all(np.isnan(exgal_targ['spec']) == False, axis=0), np.nanmedian(exgal_targ['spec'], axis=0) >  0.0 ))[0]
    idx_bad  = np.where( np.logical_or(  np.any(np.isnan(exgal_targ['spec']) == True,  axis=0), np.nanmedian(exgal_targ['spec'], axis=0) <= 0.0 ))[0]

    # Select spaxels with SNR above threshold
    idx_inside, idx_outside = applySNRThreshold(exgal_targ['snr'][idx_good], exgal_targ['signal'][idx_good], configs['MIN_SNR'])

    # Reject all selected spaxels
    idx_outside = np.unique( np.concatenate((idx_bad, idx_good[idx_outside])) )
    idx_inside  = idx_good[idx_inside]

    return( idx_inside, idx_outside )


def applySNRThreshold(snr, signal, min_snr):
    """
    Select those spaxels that are above the isophote level with a mean
    signal-to-noise ratio of MIN_SNR.
    """
    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "
    ExGalutil.prettyOutput_Running("Remove spaxels below the isophote with an average signal-to-noise of "+str(min_snr))

    idx_snr = np.where( np.abs(snr - min_snr) < 2. )[0]
    meanmin_signal = np.mean( signal[idx_snr] )
    idx_inside  = np.where( signal >= meanmin_signal )[0]
    idx_outside = np.where( signal < meanmin_signal )[0]

    if len(idx_inside) == 0 and len(idx_outside) == 0:
        idx_inside = np.arange( len(snr) )
        idx_outside = np.array([], dtype=np.int64)

    ExGalutil.prettyOutput_Done("Remove spaxels below the isophote with an average signal-to-noise of "+str(min_snr))
    logging.info("Remove spaxels below the isophote with an average signal-to-noise of "+str(min_snr)+"\n"\
            +loggingBlanks+"Selected "+str(len(idx_inside))+" spaxels inside and "+str(len(idx_outside))+" outside of the Voronoi region.")

    return(idx_inside, idx_outside)


def log_rebinning(exgal_targ, configs, rootname, outdir):
    """
    Logarithmically rebin spectra and error spectra. Save the resulting
    spectra to disk or, in case these are already available, load the spectra
    from disk and skip the rebinning process.
    """
    # Do log-rebinning and save spectra
    # if os.path.isfile(outdir+rootname+'_AllSpectra.fits') == False:

    # Do log-rebinning for spectra
    ExGalutil.prettyOutput_Running("Log-rebinning the spectra")
    log_spec, logLam = run_logrebinning\
            (exgal_targ['spec'], exgal_targ['velscale'], len(exgal_targ['x']), exgal_targ['wave'], configs)
    ExGalutil.prettyOutput_Done("Log-rebinning the spectra", progressbar=True)
    logging.info("Log-rebinned the spectra")

    # Do log-rebinning for error spectra
    ExGalutil.prettyOutput_Running("Log-rebinning the error spectra")

    # Following the comment made S. Zibetti
    # The IVAR spectrum cannot be rebinned as is, but must be inverted (VAR=1/IVAR) and rebinned using "flux" conservation
    # So we "correct" the resampled log_error array to account for variance conservation

    log_error, _ = run_logrebinning(exgal_targ['error'], exgal_targ['velscale'], len(exgal_targ['x']), exgal_targ['wave'], configs, ivar_cor= True )
    ExGalutil.prettyOutput_Done("Log-rebinning the error spectra", progressbar=True)
    logging.info("Log-rebinned the error spectra")

    # Save all spectra
    saveAllSpectra(rootname, exgal_targ['aps_id'], exgal_targ['targid'], exgal_targ['cname'], outdir, log_spec, log_error, exgal_targ['velscale'], logLam)
    # else:
    #     # Load all spectra
    #     log_spec, log_error, logLam = loadAllSpectra(rootname, outdir)

    return(log_spec, log_error, logLam)


def run_logrebinning(bin_data, velscale, nbins, wave, configs, ivar_cor=False ):
    """
    Calls the log-rebinning routine of pPXF (see Cappellari & Emsellem 2004;
    ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
    """

    ## Important Update (2 Feb 2022)
    ## Given that applying velscale may change the size of the output logLam and SSpNEW, we first run a dummy run to find the min lenght for the logLam among all fibres
    ## And then, create the empty arrays to be filled up by log_wave_i and log_bin_data_i for each fibre.

    logLam_len_list =[]
    for i in range(0, nbins):
        lamRange = np.array([np.amin(wave[:,i]),np.amax(wave[:,i])])
        sspNew, logLam, _ = log_rebin(lamRange, bin_data[:,i], velscale=velscale[i])
        logLam_len_list.append(len(logLam))

    ## here logLam_len refers to the minimum lenght of logLam among all fibres.
    logLam_len = np.min(logLam_len_list)

    # Setup arrays
    log_bin_data = np.zeros([logLam_len,nbins])
    log_wave = np.zeros([logLam_len,nbins])

    # Do log-rebinning
    for i in range(0, nbins):
        lamRange = np.array([np.amin(wave[:,i]),np.amax(wave[:,i])])
        log_bin_data_i, log_wave_i  =  corefunc_logrebin(lamRange, bin_data[:,i], velscale[i], len(logLam), i, nbins)
        log_bin_data[:,i] = log_bin_data_i[0:logLam_len]
        log_wave[:,i] = log_wave_i[0:logLam_len]


        # Following the comment made S. Zibetti
        # The IVAR spectrum cannot be rebinned as is, but must be inverted (VAR=1/IVAR) and rebinned using "flux" conservation
        # So we "correct" the resampled log_error array to account for variance conservation
        if ivar_cor:
            pixel_scale = wave[1,0] - wave[0,0]
            log_bin_data[:,i] *= np.sqrt(pixel_scale/(np.exp(log_wave[:,i])*velscale[i]/Clight))

    return (log_bin_data, log_wave)


def corefunc_logrebin(lamRange, bin_data, velscale, npix, iterate, nbins):
    """
    Calls the log-rebinning routine of pPXF (see Cappellari & Emsellem 2004;
    ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).

    TODO: Should probably be merged with run_logrebinning.
    """
    try:
        sspNew, logLam, _ = log_rebin(lamRange, bin_data, velscale=velscale)
        ExGalutil.printProgress(iterate+1, nbins, barLength = 50)
        return(sspNew,logLam)

    except:
        out = np.zeros(npix); out[:] = np.nan
        return(out,out)


def saveAllSpectra(rootname, aps_id, targid, cname,  outdir, log_spec, log_error, velscale, logLam):
    """ Save all logarithmically rebinned spectra to file. """
    outfits_spectra  = outdir+rootname+'_AllSpectra.fits'
    ExGalutil.prettyOutput_Running("Writing: "+rootname+'_AllSpectra.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append(fits.Column(name='APS_ID',  format='J',                    array=aps_id      ))
    cols.append(fits.Column(name='TARGID',  format='40A',                  array=targid      ))
    cols.append(fits.Column(name='CNAME',   format='40A',                  array=cname       ))
    cols.append( fits.Column(name='LOGLAM', format=str(len(log_spec))+'D', array=logLam.T    ))
    cols.append( fits.Column(name='SPEC',   format=str(len(log_spec))+'D', array=log_spec.T  ))
    cols.append( fits.Column(name='ESPEC',  format=str(len(log_spec))+'D', array=log_error.T ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'SPECTRA'

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header keywords
    fits.setval(outfits_spectra,'VELSCALE', value=velscale[0])
    fits.setval(outfits_spectra,'CRPIX1',   value=1.0)
    # fits.setval(outfits_spectra,'CRVAL1',   value=logLam[0])
    # fits.setval(outfits_spectra,'CDELT1',   value=logLam[1]-logLam[0])

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_AllSpectra.fits')
    logging.info("Wrote: "+outfits_spectra)


def loadAllSpectra(rootname, outdir):
    """ Loads all spectra from file. """
    hdu = fits.open(outdir+rootname+'_AllSpectra.fits')
    log_spec  = np.array( hdu[1].data.SPEC.T  )
    log_error = np.array( hdu[1].data.ESPEC.T )
    logLam    = np.array( hdu[1].data.LOGLAM.T  )
    return(log_spec, log_error, logLam)




def prepareSpectralTemplateLibrary(module, templates_dir, configs, lmin, lmax, velscale, velscale_ratio, LSF_Data, LSF_Templates, wl_offset=300.0):

    """
    WEAVE dedicated version
    BY: A. Molaeinezhad (CASU, Cambridge, UK) -- 9 Dec 2021

    Prepares the spectral template library. The templates are loaded from disk,
    shortened to meet the spectral range in consideration, convolved to meet the
    resolution of the observed spectra (according to the LSF), log-rebinned, and
    normalised. In addition, they are sorted in a three-dimensional array
    sampling the parameter space in age, metallicity and alpha-enhancement.
    """
    ExGalutil.prettyOutput_Running("Preparing the stellar population templates")
    cvel  = 299792.458



    ## check if PCA option has been properly set in the config file
    if configs["PCA"].upper().replace(' ','') in ['TRUE', 'T', '1', 'Y', 'YES']:
        pca_check = True
        ExGalutil.prettyOutput_Warning("Template Type: PCA Compressed")
    else:
        pca_check = False



    ## read the appropriate/requested templates file (depends if PCA is requested or not)
    template_file = Path(templates_dir).joinpath(configs['SSP_LIB']+('_PCA' if pca_check else '')+'.npy')
    assert template_file.is_file(), 'no template file named %s found in the %s directory' %(template_file.name, templates_dir)

    TempDict = np.load(template_file,allow_pickle=True)
    ## get it back to dict format
    TempDict = TempDict.item()


    ## only need/work with good templates [flag ==0]
    good_temps= np.ravel(np.where(np.array(TempDict['flag'])==0))

    ## Convert templates dict into an astropy Table, so easier to deal with rows and columns
    TempDict = Table(TempDict)
    assert len(good_temps) > 0, 'at least one decent template needed to start. Found: %d' %(len(good_temps))
    TempDict = TempDict[good_temps]
    ntemplates = len(TempDict)




    # Extract ages, metallicities and alpha from the templates
    try:
        # With WEAVE STANDARD SSP naming convention: Necessary for SFH-module
        logAge, metal, alpha, nAges, nMetal, nAlpha, ncomb = age_metal_alpha(TempDict)
        SSPNamingConvention = True
    except:
        # Without WEAVE STANDARD naming convention
        print('WARNING: The templates do not follow the WEAVE STANDARD SSP naming convention. SSP will be ignored!')
        SSPNamingConvention = False

    # Do SSP stuff only for SFH module
    if   module == "SFH"  and  SSPNamingConvention == False:
        message = "The templates do not follow the WEAVE STANDARD naming convention. "+\
                  "In order to execute the SFH module, SPPs following the WEAVE STANDARD naming convention must be supplied."
        ExGalutil.prettyOutput_Failed("Preparing the stellar population templates")
        print("             "+message)
        logging.critical(message)
        exit(1)
    elif module == "SFH"  and  SSPNamingConvention == True:
        SSPNamingConvention = True
    else:
        SSPNamingConvention = False




    # Shorten templates to size of data
    sample_wave = TempDict['wave'][0]
    wlrange_Ref = [lmin-wl_offset, lmax+wl_offset]
    good_wl  = (sample_wave >= wlrange_Ref[0]) & (sample_wave <= wlrange_Ref[1])

    ## check if good_wl is valid
    assert np.count_nonzero(good_wl) > 0, 'Error: No Wavelenght range-matched found'
    wmf = len(sample_wave) / np.count_nonzero(good_wl)
    ExGalutil.prettyOutput_Info(f'Wavelenght Overlap between Template and WEAVE-REF: {wmf * 100.0:.2f} Percent {"[Full Coverage]" if (wmf >=1) else "[Partial Coverage]"}')

    ## Update the new wlrange for templates
    lamRange_spmod = [np.min(sample_wave[good_wl]),np.max(sample_wave[good_wl])]


    ## select the first spectrum as test to extract some necessary parameters (size, etc) and estimate LSF
    flux_Ref = TempDict['flux'][0][good_wl]
    wave_Ref = sample_wave[good_wl]
    dwav_Ref = TempDict['dwav'][0]
    if hasattr(dwav_Ref, "__len__"): dwav_Ref = dwav_Ref[good_wl]

    ## Apply LSF of data into the first spectrum (wave_Ref), for which we have already applied the wlrange limit
    data_lsf   = LSF_Data(wave_Ref)
    temp_lsf   = LSF_Templates(wave_Ref)
    fwhm_diff  = np.sqrt(data_lsf**2 - temp_lsf**2)  # in angstroms
    bad_pix    = np.isnan(fwhm_diff)
    if np.sum(bad_pix) > 0:
       print("Some values of the data LSF are below the templates values")
    fwhm_diff[bad_pix] = 1E-2  # Fixing the FWHM_diff to a tiny value if there are NaNs
    ## convert fwhm to sigma and to pixel
    sigma_diff = fwhm_diff/2.355/dwav_Ref


    ## Run log_rebin on the reference flux just to estimate the size of templates array
    sspNew, _, _ = log_rebin(lamRange_spmod, flux_Ref, velscale=velscale/velscale_ratio)


    # Do NOT sort the templates in any way
    if SSPNamingConvention == False:

        # Load templates, convolve and log-rebin them
        templates = np.empty((sspNew.size, ntemplates))


        start_resample_ref = time.time()
        for TTmpl_i, Tmpl in enumerate(TempDict['id']):
            ssp_data = gaussian_filter1d(TempDict['flux'][TTmpl_i][good_wl] , sigma_diff)
            templates[:, TTmpl_i], logLam_spmod, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale/velscale_ratio)
        timer(start_resample_ref, time.time(), 'Resampled %d Template spectrum according to DATA LSF and Run Log_rebining ' % ntemplates)


        # Normalise templates in such a way to get mass-weighted results
        if configs['NORM_TEMP'] == 'MASS':
            templates = templates / np.mean( templates )

        # Normalise templates in such a way to get light-weighted results
        if configs['NORM_TEMP'] == 'LIGHT':
            for i in range( templates.shape[1] ):
                templates[:,i] = templates[:,i] / np.mean(templates[:,i], axis=0)

        ExGalutil.prettyOutput_Done("Preparing the stellar population templates [No SSP mode]")
        logging.info("Prepared the stellar population templates [No SSP mode]")

        return( templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam_spmod, ntemplates, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan )


    # Sort the templates in a exgal_targ of age, metal, alpha
    elif SSPNamingConvention == True:

        ## sort TempDict Table based on the age, metal and Alpha columns
        TempDict.sort(['Age','Metal','Alpha'])

        templates          = np.zeros((sspNew.size, nAges, nMetal, nAlpha))
        templates[:,:,:,:] = np.nan


        # Arrays to store properties of the models
        logAge_grid = np.empty((nAges, nMetal, nAlpha))
        metal_grid  = np.empty((nAges, nMetal, nAlpha))
        alpha_grid  = np.empty((nAges, nMetal, nAlpha))

        # Sort the templates in the exgal_targ of age, metal, alpha
        # This sorts for alpha
        for i, a in enumerate(alpha):
            # This sorts for metals
            for k, mh in enumerate(metal):
                TempDict_temp = TempDict[(TempDict['Alpha'] == a) & (TempDict['Metal']==mh)]

                # This sorts for ages
                for j, Tmpl in enumerate(TempDict_temp):
                    ssp_data = gaussian_filter1d(Tmpl['flux'][good_wl] , sigma_diff)
                    sspNew, logLam2, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale/velscale_ratio)
                    logAge_grid[j, k, i] = logAge[j]
                    metal_grid[j, k, i]  = metal[k]
                    alpha_grid[j, k, i]  = alpha[i]

                    # Normalise templates for light-weighted results
                    if configs['NORM_TEMP'] == 'LIGHT':
                        templates[:, j, k, i] = sspNew / np.mean(sspNew)
                    else:
                        templates[:, j, k, i] = sspNew

        # Normalise templates for mass-weighted results
        if configs['NORM_TEMP'] == 'MASS':
            templates = templates / np.mean( templates )


        ExGalutil.prettyOutput_Done("Preparing the stellar population templates [SSP mode]")
        logging.info("Prepared the stellar population templates [SSP mode]")

        return(templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam2, ntemplates, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha)





def age_metal_alpha(TempDict):

    ## create a unique list of not-NON values for Age, Metal and Alpha
    Age   = np.unique( TempDict['Age'][~np.isnan(TempDict['Age'])])
    Metal = np.unique( TempDict['Metal'][~np.isnan(TempDict['Metal'])])
    Alpha = np.unique( TempDict['Alpha'][~np.isnan(TempDict['Alpha'])])
    nAges  = len(Age)
    nMetal = len(Metal)
    nAlpha = len(Alpha)
    ncomb = nAges * nMetal * nAlpha
    return (np.log10(Age), Metal, Alpha, nAges, nMetal, nAlpha, ncomb)




# def spectralMasking(config_dir, configs, logLam, module, redshift):
#     """
#     Construct a spectral mask, according to the information provided in the file
#     spectralMasking_[module].config. Note that this is not considered in the
#     emission-line analysis with EMIPPXF, as emippxf uses its own, specific
#     emission-line setup file.
#     """

#     # Read file
#     mask        = np.genfromtxt( config_dir+"spectralMasking_"+module+".config", usecols=(0,1)          )
#     maskComment = np.genfromtxt( config_dir+"spectralMasking_"+module+".config", usecols=(2), dtype=str )
#     goodPixels  = np.arange( len(logLam) )

#     # In case there is only one mask
#     if len( mask.shape ) == 1  and  mask.shape[0] != 0:
#         mask        = mask.reshape(1,2)
#         maskComment = maskComment.reshape(1)

#     for i in range( mask.shape[0] ):

#         # Check for sky-lines
#         # if maskComment[i] == 'sky'  or  maskComment[i] == 'SKY'  or  maskComment[i] == 'Sky':
#         if 'SKY' in str(maskComment[i]).upper():
#             mask[i,0] = mask[i,0] / (1+redshift)

#         # Define masked pixel range
#         minimumPixel = int( np.round( ( np.log( mask[i,0] - mask[i,1]/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )
#         maximumPixel = int( np.round( ( np.log( mask[i,0] + mask[i,1]/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )

#         # Handle border of wavelength range
#         if minimumPixel < 0:            minimumPixel = 0
#         if maximumPixel < 0:            maximumPixel = 0
#         if minimumPixel == len(logLam): minimumPixel = len(logLam)-1
#         if maximumPixel == len(logLam): maximumPixel = len(logLam)-1
#         if minimumPixel > len(logLam):  minimumPixel = len(logLam)
#         if maximumPixel > len(logLam):  maximumPixel = len(logLam)

#         # Mark masked spectral pixels
#         goodPixels[minimumPixel:maximumPixel+1] = -1


#     ## Here we use masking method to limit the analysis to a wavelenght range, specified in the config file

#     # Mask condition
#     idx_lam_mask   = np.where( np.logical_or( np.exp(logLam) < configs['LMIN_PPXF'], np.exp(logLam) > configs['LMAX_PPXF'] ) )[0]

#     # unmask condition
#     idx_lam_pass   = np.where( np.logical_and( np.exp(logLam) >= configs['LMIN_PPXF'], np.exp(logLam) <= configs['LMAX_PPXF'] ) )[0]


#     # return a null array for goodPixels if after applying LMIN/LMAX_PPXF limits to the redshift-corrected spectrum nothing remains
#     if len(idx_lam_pass)  == 0:
#         return []


#     ## if len(idx_lam_pass)  > 0 then we add those pixels, belong to the regions out of the [LMIN, LMAX] to bad pixels
#     goodPixels[idx_lam_mask] = -1


#     ## NOTE BY APS (ALIREZA) on 27 Feb 2020:
#     ## We noticed, The original code always mask the last pixel, independent of what mask file sayes.
#     ## It turned out, this is just becuase Adrian has mixed two possible cases with > and ==
#     ## the original code was:
#     ##   if minimumPixel >= len(logLam):  minimumPixel = len(logLam)-1
#     ##   if maximumPixel >= len(logLam):  maximumPixel = len(logLam)-1
#     ## In the original case, even if mask regions were out of the logLam bondary, if masked the last pixel
#     ## So we break it into two conditions and it seems, it sole the problem. Most check are needed to confirm it.

#     goodPixels = goodPixels[ np.where( goodPixels != -1 )[0] ]
#     return(goodPixels)

def spectralMasking(config_dir, configs, logLam, module, redshift, LSF_Data=None, verbose=False):
    """
    Construct a spectral mask with LSF-dependent widths, according to the information
    provided in the file spectralMasking_[module].config. Note that this is not
    considered in the emission-line analysis with emippxf, as emippxf uses its own,
    specific emission-line setup file.

    Parameters:
    -----------
    LSF_Data : callable, optional
        Interpolation function giving data FWHM(λ) in Å
    verbose : bool, optional
        Print masking information
    """

    # Read file
    mask_file = config_dir + "spectralMasking_" + module + ".config"

    try:
        mask        = np.genfromtxt(mask_file, usecols=(0,1))
        maskComment = np.genfromtxt(mask_file, usecols=(2), dtype=str)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not read masking file {mask_file}: {e}")
        return np.arange(len(logLam))  # Return all pixels as good

    goodPixels  = np.arange( len(logLam) )

    # In case there is only one mask
    if len( mask.shape ) == 1  and  mask.shape[0] != 0:
        mask        = mask.reshape(1,2)
        if isinstance(maskComment, str):
            maskComment = np.array([maskComment])
        else:
            maskComment = maskComment.reshape(1)

    if verbose:
        print(f"Processing {len(mask)} mask regions from {mask_file}")
        if LSF_Data is not None:
            print(f"Using LSF-dependent masking widths")

    lsf_enhanced_regions = 0
    lsf_failed_regions = 0

    for i in range( mask.shape[0] ):

        # Extract mask parameters
        central_wavelength = mask[i, 0]
        base_width = mask[i, 1]  # Base width from config file
        comment = str(maskComment[i]).upper() if i < len(maskComment) else ""

        # Check for sky-lines
        # if maskComment[i] == 'sky'  or  maskComment[i] == 'SKY'  or  maskComment[i] == 'Sky':
        if 'SKY' in comment:
            corrected_wavelength = central_wavelength / (1+redshift)
        else:
            corrected_wavelength = central_wavelength

        # Calculate effective width using LSF if available
        if LSF_Data is not None:
            try:
                # Get instrumental FWHM at this wavelength
                lsf_fwhm = LSF_Data(corrected_wavelength)

                # Combine base width with instrumental broadening
                # Assume Gaussian profiles: total_width^2 = base^2 + instrumental^2
                total_width = np.sqrt(base_width**2 + lsf_fwhm**2)

                # Apply safety factor to ensure complete masking
                safety_factor = 2.5  # Could be made configurable
                effective_width = total_width * safety_factor

                lsf_enhanced_regions += 1

                if verbose and i < 3:  # Show first few enhancements
                    print(f"Region {i+1} ({comment}): base={base_width:.2f}Å + LSF={lsf_fwhm:.2f}Å "
                          f"→ total={total_width:.2f}Å × {safety_factor:.1f} = {effective_width:.2f}Å")

            except (ValueError, TypeError, AttributeError) as e:
                # Fallback to base width if LSF evaluation fails
                effective_width = base_width * 2.5  # Apply safety factor to base width
                lsf_failed_regions += 1

                if verbose and lsf_failed_regions <= 3:  # Only print first few failures
                    print(f"Warning: LSF evaluation failed for region {i+1} at {corrected_wavelength:.1f}Å: {e}")
                    print(f"  Using fallback width: {effective_width:.2f}Å")
        else:
            # No LSF available - use base width with safety factor
            effective_width = base_width * 2.0  # Conservative safety factor

        # Define masked pixel range using effective width
        minimumPixel = int( np.round( ( np.log( corrected_wavelength - effective_width/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )
        maximumPixel = int( np.round( ( np.log( corrected_wavelength + effective_width/2. ) - logLam[0] ) / (logLam[1] - logLam[0]) ) )

        # Handle border of wavelength range
        if minimumPixel < 0:           minimumPixel = 0
        if maximumPixel < 0:           maximumPixel = 0

        if minimumPixel == len(logLam): minimumPixel = len(logLam)-1
        if maximumPixel == len(logLam): maximumPixel = len(logLam)-1
        if minimumPixel > len(logLam):  minimumPixel = len(logLam)
        if maximumPixel > len(logLam):  maximumPixel = len(logLam)

        # Mark masked spectral pixels
        if minimumPixel <= maximumPixel:  # Ensure valid range
            goodPixels[minimumPixel:maximumPixel+1] = -1

            if verbose and i < 5:  # Show first few mask applications
                n_masked = maximumPixel - minimumPixel + 1
                print(f"Masked region {i+1}: pixels {minimumPixel}-{maximumPixel} ({n_masked} pixels) "
                      f"for {comment} at {corrected_wavelength:.1f}Å ±{effective_width/2:.1f}Å")

    # Print LSF usage summary
    if verbose and LSF_Data is not None:
        total_regions = len(mask)
        print(f"\nLSF masking summary:")
        print(f"  Total mask regions: {total_regions}")
        print(f"  LSF-enhanced: {lsf_enhanced_regions}")
        print(f"  LSF failed (fallback): {lsf_failed_regions}")
        print(f"  Success rate: {lsf_enhanced_regions/total_regions*100:.1f}%" if total_regions > 0 else "  No regions to process")


    ## Here we use masking method to limit the analysis to a wavelength range, specified in the config file

    # Mask condition
    idx_lam_mask   = np.where( np.logical_or( np.exp(logLam) < configs['LMIN_PPXF'], np.exp(logLam) > configs['LMAX_PPXF'] ) )[0]

    # unmask condition
    idx_lam_pass   = np.where( np.logical_and( np.exp(logLam) >= configs['LMIN_PPXF'], np.exp(logLam) <= configs['LMAX_PPXF'] ) )[0]

    # return a null array for goodPixels if after applying LMIN/LMAX_PPXF limits to the redshift-corrected spectrum nothing remains
    if len(idx_lam_pass)  == 0:
        return []

    ## if len(idx_lam_pass)  > 0 then we add those pixels, belong to the regions out of the [LMIN, LMAX] to bad pixels
    goodPixels[idx_lam_mask] = -1

    ## NOTE BY APS (ALIREZA) on 27 Feb 2020:
    ## We noticed, The original code always mask the last pixel, independent of what mask file says.
    ## It turned out, this is just because Adrian has mixed two possible cases with > and ==
    ## the original code was:
    ##   if minimumPixel >= len(logLam):  minimumPixel = len(logLam)-1
    ##   if maximumPixel >= len(logLam):  maximumPixel = len(logLam)-1
    ## In the original case, even if mask regions were out of the logLam boundary, it masked the last pixel
    ## So we break it into two conditions and it seems, it solved the problem. More checks are needed to confirm it.

    goodPixels = goodPixels[ np.where( goodPixels != -1 )[0] ]

    if verbose:
        n_masked_total = len(logLam) - len(goodPixels)
        print(f"Final masking result: {n_masked_total} pixels masked, {len(goodPixels)} pixels remaining")

    return(goodPixels)



def MaskGaps(error,goodPixels,error_limit):

    ###### Masking CCD GAPS ########
    # New approach, here for APS [to mask CCD GAPS]
    # we also masked those pixels with errors higher than a certain value to be sure they do not
    # contribute in the final fittings
    ## only select those goodpixels for which error is lower than error_limit (CCD gaps)
    ## Please note, as error is affected by Voronoi Binning
    ## and the error after voronio bining is av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
    ## we use 0.95 * np.sqrt(error_limit) as our error limit
    ## where 0.95 is 2 sigma around this value
    ## In MOS mode, in opsoite to IFU, we usually do not run VORONOI
    ## So, we do not expect the error to be affected.
    ## Just to make sure we are on the safe side, here we also use sqrt(error_limit) instead
    ## of error_limit, itself.

    try:
        goodpix_idx = np.ravel(np.where(error[goodPixels] < 0.95 * np.sqrt(error_limit)))
        badpix_idx = np.ravel(np.where(error > 0.95 * np.sqrt(error_limit)))


        if len(goodpix_idx) > 0:
            goodPixels = goodPixels[goodpix_idx]
    except:
        goodPixels=[]

    return goodPixels



def emippxf_wlimit(logLam, goodpixels,configs):

    ## Here we use masking method to limit the analysis to a wavelenght range, specified in the config file
    idx_lam   = np.where( np.logical_and( np.exp(logLam[goodpixels]) >= configs['LMIN_EMIPPXF'], np.exp(logLam[goodpixels]) <= configs['LMAX_EMIPPXF'] ) )[0]

    if len(idx_lam)  == 0:
        goodpixels = []
    else:
        ## Add those pixels, belong to the regions out of the [LMIN, LMAX] to bad pixels
        goodpixels = goodpixels[idx_lam]

    return goodpixels





def calculate_velscale_ratio(configs, logLam_data, velscale, LSF_Data, LSF_Templates, templates_dir, debug=False):
    """
    Calculate the proper velocity scale ratio between templates and data.

    Parameters:
    -----------
    configs : dict
        Configuration dictionary
    velscale : float or array
        Data velocity scale in km/s per pixel
    LSF_Data : callable
        Interpolation function for data LSF (FWHM in Angstrom)
    LSF_Templates : callable
        Interpolation function for template LSF (FWHM in Angstrom)
    outdir : str
        Output directory path
    templates_dir : str
        Templates directory path

    Returns:
    --------
    velscale_ratio : float
        Ratio of template velocity scale to data velocity scale
    template_velscale : float
        Template velocity scale in km/s per pixel
    data_velscale : float
        Data velocity scale in km/s per pixel
    """

    if debug:
        print("=== CALCULATING PROPER VELSCALE_RATIO ===")

    # Get data velocity scale
    if hasattr(velscale, '__len__'):
        data_velscale = velscale[0]  # Use first element if array
    else:
        data_velscale = velscale

    if debug:
        print(f"Input data velocity scale: {data_velscale:.3f} km/s/pixel")

    try:

        # Define wavelength range for template preparation
        lam_data = np.exp(logLam_data)
        lmin = np.nanmin(lam_data)
        lmax = np.nanmax(lam_data)

        if debug:
            print(f"Data wavelength range: {lmin:.1f} - {lmax:.1f} Å")

            # Prepare templates to get their velocity scale
            print("Preparing sample templates to determine velocity scale...")

        # Use dummy velscale_ratio = 1.0 for initial template preparation
        _, _, logLam_template, _ = prepareSpectralTemplateLibrary(
            "PPXF", templates_dir, configs, lmin, lmax, data_velscale, 1.0,
            LSF_Data, LSF_Templates)[:4]

        # Calculate template velocity scale
        template_velscale = (logLam_template[1] - logLam_template[0]) * Clight

        if debug:
            print(f"Template velocity scale: {template_velscale:.3f} km/s/pixel")

        # Calculate the velocity scale ratio
        velscale_ratio = template_velscale / data_velscale

        if debug:
            print(f"Calculated velscale_ratio: {velscale_ratio:.6f}")

            # Diagnostics and warnings
            print(f"\n=== VELOCITY SCALE DIAGNOSTICS ===")

            if abs(velscale_ratio - 1.0) < 0.01:
                print(f"✅ velscale_ratio ≈ 1.0 - velocity scales are well matched")
            elif abs(velscale_ratio - 1.0) < 0.1:
                print(f"⚠️  Small velscale_ratio deviation: {abs(velscale_ratio - 1.0):.3f}")
                print(f"   This may cause minor sigma scaling issues")
            else:
                print(f"🚨 SIGNIFICANT velscale_ratio deviation: {abs(velscale_ratio - 1.0):.3f}")
                print(f"   This explains sigma scaling issues in PPXF!")

                if velscale_ratio > 1.0:
                    print(f"   Templates have COARSER sampling → σ will be OVERESTIMATED")
                    print(f"   Expected sigma scaling factor: ~{velscale_ratio:.2f}×")
                else:
                    print(f"   Templates have FINER sampling → σ will be UNDERESTIMATED")
                    print(f"   Expected sigma scaling factor: ~{velscale_ratio:.2f}×")

            # Additional LSF diagnostics
            print(f"\n=== LSF COMPARISON AT 5000 Å ===")


        if debug:
            try:
                ref_wavelength = 5000.0
                if ref_wavelength < lmin or ref_wavelength > lmax:
                    ref_wavelength = (lmin + lmax) / 2
                    print(f"Using {ref_wavelength:.1f} Å as reference (5000 Å outside range)")

                template_lsf = LSF_Templates(ref_wavelength)
                data_lsf = LSF_Data(ref_wavelength)
                lsf_ratio = data_lsf / template_lsf

                print(f"Template LSF FWHM: {template_lsf:.2f} Å")
                print(f"Data LSF FWHM: {data_lsf:.2f} Å")
                print(f"LSF ratio (data/template): {lsf_ratio:.3f}")

                # Convert to velocity units for easier interpretation
                template_lsf_kms = template_lsf / ref_wavelength * Clight
                data_lsf_kms = data_lsf / ref_wavelength * Clight

                print(f"Template LSF: {template_lsf_kms:.1f} km/s")
                print(f"Data LSF: {data_lsf_kms:.1f} km/s")

                if lsf_ratio < 0.8:
                    print(f"⚠️  Data resolution significantly BETTER than templates")
                    print(f"   May cause sigma underestimation")
                elif lsf_ratio > 1.5:
                    print(f"⚠️  Data resolution significantly WORSE than templates")
                    print(f"   May cause sigma overestimation")
                else:
                    print(f"✅ LSF ratio is reasonable")

            except Exception as e:
                print(f"Could not compare LSF: {e}")

        # Template wavelength coverage check
        lam_template = np.exp(logLam_template)
        template_coverage = (np.nanmin(lam_template), np.nanmax(lam_template))
        data_coverage = (lmin, lmax)
        if debug:
            print(f"\n=== WAVELENGTH COVERAGE ===")
            print(f"Data range:     {data_coverage[0]:.1f} - {data_coverage[1]:.1f} Å")
            print(f"Template range: {template_coverage[0]:.1f} - {template_coverage[1]:.1f} Å")

            if template_coverage[0] > data_coverage[0] + 10:
                print(f"⚠️  Templates don't cover blue end of data")
            if template_coverage[1] < data_coverage[1] - 10:
                print(f"⚠️  Templates don't cover red end of data")
            if (template_coverage[0] <= data_coverage[0] and
                template_coverage[1] >= data_coverage[1]):
                print(f"✅ Templates fully cover data wavelength range")

            print("=" * 50)

        return int(round(velscale_ratio)), template_velscale, data_velscale

    except Exception as e:
        print(f"❌ ERROR calculating velscale_ratio: {e}")
        print("FALLBACK: Using velscale_ratio = 1.0")
        print("⚠️  This may cause sigma scaling issues!")
        print("=" * 50)

        return 1, data_velscale, data_velscale


def save_lsf_data(exgal_targ, headname, outpath):
    """
    Save LSF data using dill - can handle complex function objects.

    Dill is a more powerful version of pickle that can serialize almost anything.
    """
    try:
        # Validate inputs (same as before)
        if not outpath:
            raise ValueError("outpath is empty or None")
        if not headname:
            raise ValueError("headname is empty or None")
        if not isinstance(exgal_targ, dict):
            raise ValueError("exgal_targ must be a dictionary")

        # Create directory if needed
        if not os.path.exists(outpath):
            print(f"Creating directory: {outpath}")
            os.makedirs(outpath, exist_ok=True)

        # Create file path
        lsf_filepath = os.path.join(outpath, f"{headname}_lsf.dill")

        # Check required keys
        required_keys = ['aps_id', 'targid', 'z', 'zerr', 'lsf', 'glsf']
        missing_keys = [key for key in required_keys if key not in exgal_targ]
        if missing_keys:
            raise KeyError(f"Missing required keys in exgal_targ: {missing_keys}")

        # Create data dictionary
        lsf_dict = {
            'aps_id': exgal_targ['aps_id'],
            'targid': exgal_targ['targid'],
            'z': exgal_targ['z'],
            'zerr': exgal_targ['zerr'],
            'lsf': exgal_targ['lsf'],
            'glsf': exgal_targ['glsf']
        }

        # Save with dill (handles functions much better than pickle)
        with open(lsf_filepath, 'wb') as f:
            dill.dump(lsf_dict, f)

        # Verify file
        if not os.path.exists(lsf_filepath):
            raise IOError(f"Failed to create file: {lsf_filepath}")

        file_size = os.path.getsize(lsf_filepath)
        if file_size == 0:
            raise IOError(f"Created file is empty: {lsf_filepath}")

        print(f"LSF data saved with dill to: {lsf_filepath} ({file_size} bytes)")
        return lsf_filepath

    except Exception as e:
        print(f"Error saving LSF data with dill: {e}")
        raise

def load_lsf_data(configs):
    """Load LSF data saved with dill."""
    try:
        lsf_filepath = configs['lsfdir']
        with open(lsf_filepath, 'rb') as f:
            lsf_data = dill.load(f)
        print(f"LSF data loaded with dill from: {lsf_filepath}")
        return lsf_data
    except Exception as e:
        print(f"Error loading LSF data with dill: {e}")
        raise
