import numpy    as np
from astropy.io import fits
from multiprocessing import Queue, Process

import time
import logging
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys

import PyAPS
from PyAPS import ExGalutil
from PyAPS import ExGalPrepare
from PyAPS import IFUExGalPrepare    as IFUExGalPrepare

from PyAPS.apsPlot import kinematics_map as util_plot
from PyAPS.apsPlot import lambdar_map    as util_plot_lambdar
from PyAPS import aps_constants
from astropy import units


from ppxf.ppxf import ppxf
from ppxf import __version__ as PPXF_VERSION

import warnings
import traceback
APSVERS = PyAPS.__version__



# PHYSICAL CONSTANTS
C = 299792.458  # km/s
# PHYSICAL CONSTANTS
Clight = 299792.458  # km/s

"""
PURPOSE:
  This module executes the analysis of stellar kinematics in the pipeline.
  Basically, it acts as an interface between pipeline and the pPXF routine from
  Cappellari & Emsellem 2004 (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).
"""


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for templates, bin_data, noise, velscale, start, goodpixels_ppxf, nmoments,\
        adeg, mdeg, offset, velscale_ratio, error_limit, nsims, nbins, lam, lam_template, i\
        in iter(inQueue.get, 'STOP'):

        sol, bestfit, optimal_template, mc_results, formal_error, goodpixels_ppxf = \
          run_ppxf(templates, bin_data, noise, velscale, start, goodpixels_ppxf, error_limit, nmoments, adeg, mdeg, offset, velscale_ratio, nsims, nbins, lam, lam_template, i)

        outQueue.put(( i, sol, bestfit, optimal_template, mc_results, formal_error, goodpixels_ppxf ))


def run_ppxf( templates, log_bin_data, log_bin_error, velscale, start, goodpixels, error_limit, nmoments, adeg, mdeg,\
        offset, velscale_ratio, nsims, nbins, lam, lam_template, i):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    stellar kinematics.
    """
    ExGalutil.printProgress( i, nbins, barLength=50 )

    try:

        ## only select those goodpixels for which error is lower than error_limit (CCD gaps)
        ## Please note, as error is affected by Voronoi Binning
        ## and the error after voronio bining is av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))
        ## we use 0.95 * np.sqrt(error_limit) as our error limit
        ## where 0.95 is 2 sigma around this value
        goodpix_idx = np.ravel(np.where(log_bin_error[goodpixels] < 0.95 * np.sqrt(error_limit)))
        if len(goodpix_idx) > 0:
            goodpixels = goodpixels[goodpix_idx]

        # Call PPXF
        pp = ppxf(templates, log_bin_data, log_bin_error, velscale, start, goodpixels=goodpixels, plot=False, \
                    quiet=True, moments=nmoments, degree=adeg, mdegree=mdeg, velscale_ratio=velscale_ratio, lam=lam, lam_temp=lam_template)

        # Make the unconvolved optimal stellar template
        normalized_weights = pp.weights / np.sum( pp.weights )
        optimal_template   = np.zeros( templates.shape[0] )
        for j in range(0, templates.shape[1]):
            optimal_template = optimal_template + templates[:,j]*normalized_weights[j]

        # Correct the formal errors assuming that the fit is good
        formal_error = pp.error * np.sqrt(pp.chi2)

        # Do MC-Simulations
        sol_MC     = np.zeros((nsims,nmoments)); sol_MC[:,:] = np.nan
        mc_results = np.zeros(nmoments);         mc_results  = np.nan
        for o in range(0, nsims):
            # Add noise to bestfit:
            #   - Draw random numbers from normal distribution with mean of 0 and sigma of 1 (np.random.normal(0,1,npix)
            #   - standard deviation( (galaxy spectrum - bestfit)[goodpix] )
            noisy_bestfit = pp.bestfit  +  np.random.normal(0, 1, len(log_bin_data)) * np.std( log_bin_data[goodpixels] - pp.bestfit[goodpixels] )

            mc = ppxf(templates, noisy_bestfit, log_bin_error, velscale, start, goodpixels=goodpixels, plot=False, \
                    quiet=True, moments=nmoments, degree=adeg, mdegree=mdeg, velscale_ratio=velscale_ratio, lam=lam, lam_temp=lam_template, bias=0.0)
            sol_MC[o,:] = mc.sol[:]

        if nsims != 0:
            mc_results = np.nanstd( sol_MC, axis=0 )

        # set the bestfit to zero for all bad pixels
        pp.bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0

        return(pp.sol[:], pp.bestfit, optimal_template, mc_results, formal_error, goodpixels)

    except:
        return( np.nan, np.nan, np.nan, np.nan, np.nan, goodpixels)


def calc_LambdaR( ppxf_result, nbins, outdir, rootname ):
    """
    Calculate the lambda parameter as a proxy for the projected, specific
    angular momentum of the galaxy (see Emsellem et al. 2007;
    ui.adsabs.harvard.edu/#abs/2007MNRAS.379..401E). Note that this quantity is
    not calculated as integrated value per galaxy, but for every Voronoi-bin
    individually.
    """
    # Calculate lambda_r
    hdu  = fits.open(outdir+rootname+'_table.fits')
    BIN_ID = hdu[1].data.BIN_ID
    X      = hdu[1].data.X
    Y      = hdu[1].data.Y
    FLUX   = hdu[1].data.FLUX

    velocity = ppxf_result[:,0] - np.median(ppxf_result[:,0])
    sigma    = ppxf_result[:,1]

    lambda_r = np.zeros( nbins );  lambda_r[:] = np.nan
    for i in range(0, nbins):
        idx = np.where( BIN_ID == i )[0]
        numerator   = 0
        denominator = 0
        for o in range(0, len(idx)):
            radius = np.sqrt( X[idx[o]]**2 + Y[idx[o]]**2 )
            numerator   += FLUX[idx[o]] * radius * np.abs( velocity[i] )
            denominator += FLUX[idx[o]] * radius * np.sqrt( velocity[i]**2 + sigma[i]**2 )
        if denominator > 0:
            lambda_r[i] = numerator / denominator
        else:
            lambda_r[i] = np.nan

    return( lambda_r )


def save_ppxf(rootname,configs, outdir, ppxf_result, mc_results, formal_error, lambda_r,\
              ppxf_bestfit, logLam, spectra, error, goodpixels, optimal_template, logLam_template, npix, ubins):
    """ Saves all results to disk. """


    ## add Units to flux, model and ivar in the ppxf's spec table
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' %(configs['funits'])
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' %(configs['funits']**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'


    # ========================
    # SAVE RESULTS
    outfits_ppxf = outdir+rootname+'_ppxf.fits'
    ExGalutil.prettyOutput_Running("Writing: "+rootname+'_ppxf.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF output data
    cols = []
    cols.append( fits.Column(name='BIN_ID',         format='J', array=ubins             ))
    cols.append( fits.Column(name='V' ,             format='D', array=ppxf_result[:,0]  ))
    cols.append( fits.Column(name='SIGMA',          format='D', array=ppxf_result[:,1]  ))
    cols.append( fits.Column(name='H3',             format='D', array=ppxf_result[:,2]  ))
    cols.append( fits.Column(name='H4',             format='D', array=ppxf_result[:,3]  ))
    cols.append( fits.Column(name='H5',             format='D', array=ppxf_result[:,4]  ))
    cols.append( fits.Column(name='H6',             format='D', array=ppxf_result[:,5]  ))
    cols.append( fits.Column(name='LAMBDA_R',       format='D', array=lambda_r[:]       ))

    cols.append( fits.Column(name='ERR_V' ,         format='D', array=mc_results[:,0]   ))
    cols.append( fits.Column(name='ERR_SIGMA',      format='D', array=mc_results[:,1]   ))
    cols.append( fits.Column(name='ERR_H3',         format='D', array=mc_results[:,2]   ))
    cols.append( fits.Column(name='ERR_H4',         format='D', array=mc_results[:,3]   ))
    cols.append( fits.Column(name='ERR_H5',         format='D', array=mc_results[:,4]   ))
    cols.append( fits.Column(name='ERR_H6',         format='D', array=mc_results[:,5]   ))

    cols.append( fits.Column(name='FORM_ERR_V' ,    format='D', array=formal_error[:,0] ))
    cols.append( fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1] ))
    cols.append( fits.Column(name='FORM_ERR_H3',    format='D', array=formal_error[:,2] ))
    cols.append( fits.Column(name='FORM_ERR_H4',    format='D', array=formal_error[:,3] ))
    cols.append( fits.Column(name='FORM_ERR_H5',    format='D', array=formal_error[:,4] ))
    cols.append( fits.Column(name='FORM_ERR_H6',    format='D', array=formal_error[:,5] ))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'PPXF_DATA'


    # dataHDU.header['COMMENT'] = '---------- START OF PyAPS PPXF SETTING PARMAS ----------'
    dataHDU.header['CNF_PPXF'] = (configs['CONFIG_FILE'],'Configs. Filename used by PPXF')
    ExGalPrepare.write_ebmv_header(dataHDU.header, configs)
    dataHDU.header['PPXF_SIG'] = (configs['SIGMA'],'Initial guess for SIGMA in PPXF')
    dataHDU.header['MOM'] = (configs['MOM'],'Number of kinematic moments for PPXF')
    dataHDU.header['ADEG'] = (configs['ADEG'],'Degree of the add. Legendre polynomial for PPXF')
    dataHDU.header['MDEG'] = (configs['MDEG'],'Degree of the mult. Legendre polynomial for PPXF')
    dataHDU.header['MC_PPXF'] = (configs['MC_PPXF'],'N. of MC simulations to extract errors in PPXF')
    if 'LMIN_PPXF' in configs.keys():
        dataHDU.header['LW_PPXF'] = (configs['LMIN_PPXF'],'Min wavelength (OBS.) used by PPXF')
    if 'LMAX_PPXF' in configs.keys():
        dataHDU.header['HW_PPXF'] = (configs['LMAX_PPXF'],'Max wavelength (OBS.) used by PPXF')
    dataHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
    # dataHDU.header['COMMENT'] = '----------- END OF PyAPS PPXF SETTING PARMAS ----------'


    dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    dataHDU.header['APSVERS'] = (APSVERS,'APS version')
    dataHDU.header['APSPPXFV'] = (aps_constants.__aps_ifuppxf_version__,'PyAPS (IFU) PPXF wrapper version')
    dataHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')
    #keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')


    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_ppxf.fits')
    logging.info("Wrote: "+outfits_ppxf)


    # ========================
    # SAVE BESTFIT
    outfits_ppxf = outdir+rootname+'_ppxf_spec.fits'
    ExGalutil.prettyOutput_Running("Writing: "+rootname+'_ppxf_spec.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF bestfit
    cols = []

    ## Create a copy of loglam array to be place in each row of the EMIPPXF output
    loglam_tile = np.tile(logLam , (len(ubins), 1))


    cols.append( fits.Column(name='BIN_ID',                           format='J',           array=ubins                 ))
    cols.append( fits.Column(name='LOGLAM_PPXF',  unit=wave_unit_str, format=str(npix)+'D', array=loglam_tile  ))
    cols.append( fits.Column(name='FLUX_PPXF',    unit=flux_unit_str, format=str(npix)+'D', array=np.transpose(spectra) ))
    cols.append( fits.Column(name='ERROR_PPXF',   unit=flux_unit_str, format=str(npix)+'D', array=np.transpose(error)   ))
    cols.append( fits.Column(name='MODEL_PPXF',   unit=flux_unit_str, format=str(npix)+'D', array=ppxf_bestfit          ))
    cols.append( fits.Column(name='GOODPIX_PPXF',                     format=str(npix)+'J', array=goodpixels            ))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))

    dataHDU.name = 'BESTFIT'
    dataHDU.header['CNF_PPXF'] = (configs['CONFIG_FILE'],'Configs. Filename used by PPXF')
    ExGalPrepare.write_ebmv_header(dataHDU.header, configs)
    dataHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
    dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    dataHDU.header['APSVERS'] = (APSVERS,'APS version')
    dataHDU.header['APSPPXFV'] = (aps_constants.__aps_ifuppxf_version__,'PyAPS (IFU) PPXF wrapper version')
    dataHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')
    #keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_ppxf-spectra.fits')
    logging.info("Wrote: "+outfits_ppxf)

    # ============================
    # SAVE OPTIMAL TEMPLATE RESULT
    outfits = outdir+rootname+'_ppxf-optimalTemplates.fits'
    ExGalutil.prettyOutput_Running("Writing: "+rootname+'_ppxf-optimalTemplates.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Extension 1: Table HDU with optimal templates
    cols = []
    cols.append( fits.Column(name='OPTIMAL_TEMPLATES', format=str(optimal_template.shape[1])+'D', array=optimal_template ) )
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'OPTIMAL_TEMPLATES'

    dataHDU.header['CNF_PPXF'] = (configs['CONFIG_FILE'],'Configs. Filename used by PPXF')
    ExGalPrepare.write_ebmv_header(dataHDU.header, configs)
    dataHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
    dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    dataHDU.header['APSVERS'] = (APSVERS,'APS version')
    dataHDU.header['APSPPXFV'] = (aps_constants.__aps_ifuppxf_version__,'PyAPS (IFU) PPXF wrapper version')
    dataHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')

    #keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    # Extension 2: Table HDU with logLam_templates
    cols = []
    cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=logLam_template) )
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM_TEMPLATE'
    logLamHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
    logLamHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    logLamHDU.header['APSVERS'] = (APSVERS,'APS version')
    logLamHDU.header['APSPPXFV'] = (aps_constants.__aps_ifuppxf_version__,'PyAPS (IFU) PPXF wrapper version')
    logLamHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')
    #keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs['infiles']):
        logLamHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU])
    HDUList.writeto(outfits, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_ppxf-optimalTemplates.fits')
    logging.info("Wrote: "+outfits)

def runModule_PPXF(nthreads, configs, velscale, LSF_Data, LSF_Templates, outdir, config_dir, templates_dir, figdir, rootname, Z_IN, Z_IN_ERR,  error_limit, debug=False, output=False, bin_to_bucket=None, LSF_Data_by_bucket=None):
    """
    Starts the analysis of the stellar kinematics. Data is read in,
    emission-line contaminated regions are excluded, and pPXF is executed. The
    lambda parameter is computed, results are saved to disk, and the plotting
    routines called.

    `bin_to_bucket`/`LSF_Data_by_bucket` (both optional, default `None`):
    opt-in per-bin resolution (see `aps_ifu_spaxel_contrib.bucket_lsf_
    curves`) -- `bin_to_bucket` is an array of length `nbins` (positional,
    same 0..nbins-1 indexing this function's own `bin_data`/`ubins` already
    use) mapping each bin to a bucket id, `LSF_Data_by_bucket` is
    `{bucket_id: interpolate_function}`. When both are `None` (default),
    behaviour is *exactly* the pre-existing single-`LSF_Data`-for-every-bin
    path -- `LSF_Data` (the plain callable parameter) is used everywhere,
    unchanged. When given, one resolution-matched template set is built per
    *bucket* (not per bin) and the existing per-bin loop below (serial or
    parallel, unchanged in structure) picks the right one per bin.
    """

    # Read data from file
    hdu      = fits.open(outdir+rootname+'_BINSpectra.fits')
    bin_data = np.array( hdu[1].data.SPEC.T )
    noise    = np.array( hdu[1].data.ESPEC.T )

    if hdu[1].data.LOGLAM.ndim > 1:
        logLam   = np.array( hdu[1].data.LOGLAM[0,:])
    else:
        logLam   = np.array( hdu[1].data.LOGLAM)

    idx_lam  = np.where( np.logical_and( np.exp(logLam) > configs['LMIN_PPXF'], np.exp(logLam) < configs['LMAX_PPXF'] ) )[0]

    if len(idx_lam)  == 0:

        # update 5 Nov 2024
        # in case overlap between the deredshifted spectrum and the LMIN/MAX_PPXF is zero. we consider all to create a Null filled PPXF output
        idx_lam_nan_file  = np.where(  np.exp(logLam) > -1 )[0]

        bin_data = bin_data[idx_lam_nan_file,:]
        noise    = noise[idx_lam_nan_file,:]
        logLam   = logLam[idx_lam_nan_file]
        npix     = bin_data.shape[0]
        nbins    = bin_data.shape[1]
        ubins    = np.arange(0, nbins)

        # Array to store results of ppxf
        ppxf_result        = np.full((nbins, 6), np.nan)
        ppxf_bestfit       = np.zeros((nbins,npix))
        optimal_template   = np.zeros((npix,1))
        mc_results         = np.full((nbins, 6), np.nan)
        formal_error       = np.full((nbins, 6), np.nan)
        ppxf_goodpixels    = np.full((nbins, npix), -1)
        lambda_r           = np.full(nbins, np.nan)

        # Save stellar kinematics to file
        save_ppxf(rootname,configs, outdir, ppxf_result, mc_results, formal_error, lambda_r, ppxf_bestfit,\
        logLam, bin_data ,noise , ppxf_goodpixels, optimal_template, optimal_template, npix, ubins)

        ExGalutil.prettyOutput_Warning("LMIN/LMAX_PPXF [coming from config file] and Spectra wavelength are not consistent")
        logging.warning("LMIN/LMAX_PPXF [coming from config file] and Spectra wavelength are not consistent")
        ExGalutil.prettyOutput_Warning("A null PPXF output just generated. Skipping PPXF")
        logging.warning("A null PPXF output just generated. Skipping PPXF")

        return


    # in case of overlap found between the templates and spectra keep working from here
    bin_data = bin_data[idx_lam,:]
    noise    = noise[idx_lam,:]
    logLam   = logLam[idx_lam]
    npix     = bin_data.shape[0]
    nbins    = bin_data.shape[1]
    ubins    = np.arange(0, nbins)

    # Prepare templates. The raw SSP library (disk load + age/metal/alpha
    # sort -- resolution-independent) is loaded exactly once here regardless
    # of whether per-bucket resolution is active, and reused both for the
    # velscale-ratio probe call and every actual template-preparation call
    # below -- see IFUExGalPrepare.load_raw_template_library's own
    # docstring. Previously this same library was silently reloaded from
    # disk twice per patch (once for the probe, once for real); now it's
    # loaded once no matter how many resolution buckets there are.
    velscale_ratio = 1
    raw_templates = IFUExGalPrepare.load_raw_template_library(
        "PPXF", templates_dir, configs, configs['LMIN_PPXF'], configs['LMAX_PPXF'])
    velscale_ratio, template_velscale, data_velscale = IFUExGalPrepare.calculate_velscale_ratio(
        configs, logLam, velscale, LSF_Data, LSF_Templates, templates_dir, preloaded=raw_templates)

    if LSF_Data_by_bucket:
        logging.info(f"Using {len(LSF_Data_by_bucket)} resolution bucket(s) for PPXF template preparation")
        templates_by_bucket = {}
        for bucket, lsf_func in LSF_Data_by_bucket.items():
            tpl, lamRange_spmod, logLam_template, ntemplates = IFUExGalPrepare.prepareSpectralTemplateLibrary(
                "PPXF", templates_dir, configs, configs['LMIN_PPXF'], configs['LMAX_PPXF'],
                velscale, velscale_ratio, lsf_func, LSF_Templates, preloaded=raw_templates)[:4]
            templates_by_bucket[bucket] = tpl.reshape((tpl.shape[0], ntemplates))
        # lamRange_spmod/logLam_template/ntemplates are resolution-independent
        # (same grid every bucket -- only the convolved *values* differ), so
        # whichever bucket's own copy is used below is equally valid.
        default_bucket = next(iter(templates_by_bucket))
        templates_lookup = [templates_by_bucket[bin_to_bucket[i]] for i in range(nbins)]
        templates = templates_by_bucket[default_bucket]  # only used for its .shape below
    else:
        logging.info("Using full spectral library for PPXF")
        templates, lamRange_spmod, logLam_template, ntemplates = IFUExGalPrepare.prepareSpectralTemplateLibrary(
            "PPXF", templates_dir, configs, configs['LMIN_PPXF'], configs['LMAX_PPXF'],
            velscale, velscale_ratio, LSF_Data, LSF_Templates, preloaded=raw_templates)[:4]
        templates = templates.reshape( (templates.shape[0], ntemplates) )
        templates_lookup = [templates] * nbins

    # Last preparatory steps
    offset = (logLam_template[0] - logLam[0])*C
    # noise  = np.ones((npix,nbins))
    nsims  = configs['MC_PPXF']

    # Initial guesses
    start = np.zeros((nbins,2))
    if os.path.isfile(outdir+rootname+'_ppxf-guess.fits') == True:
        # Use a different initial guess for different bins, as provided in the *_ppxf-guess.fits file
        guess      = fits.open(outdir+rootname+'_ppxf-guess.fits')[1].data
        start[:,0] = guess.V
        start[:,1] = guess.SIGMA
    else:
        # Use the same initial guess for all bins, as stated in MasterConfig
        start[:,0] = 0.0
        start[:,1] = configs['SIGMA']

    # Define goodpixels
    # Sky/telluric mask wavelengths in the config file are in observed frame.
    # The spectrum is in rest frame (divided by 1+z in cube assembly).
    # spectralMasking divides SKY line wavelengths by (1+z) when redshift != 0.0,
    # so we must pass the actual target redshift here.
    # Later in the loop for each bin, we also mask CCD gaps
    goodpixels_ppxf = IFUExGalPrepare.spectralMasking(config_dir, logLam, 'PPXF', Z_IN, LSF_Data=LSF_Data, verbose=debug)

    # Array to store results of ppxf
    ppxf_result        = np.zeros((nbins,6))
    ppxf_bestfit       = np.zeros((nbins,npix))
    optimal_template   = np.zeros((nbins,templates.shape[0]))
    mc_results         = np.zeros((nbins,6))
    formal_error       = np.zeros((nbins,6))
    ppxf_goodpixels    = np.empty((nbins,npix))
    ppxf_goodpixels.fill(-1)

    # ====================
    # Run PPXF
    start_time = time.time()

    ###### IF RUNNING IN Parallel
    if nthreads > 1:
        ExGalutil.prettyOutput_Running("Running PPXF in parallel mode")
        logging.info("Running PPXF in parallel mode")

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [Process(target=workerPPXF, args=(inQueue, outQueue)) for _ in range(nthreads)]

        # Start worker processes
        for p in ps: p.start()

        # Fill the queue
        for i in range(nbins):
            inQueue.put( ( templates_lookup[i], bin_data[:,i], noise[:,i], velscale, start[i,:], goodpixels_ppxf,\
                            configs['MOM'], configs['ADEG'], configs['MDEG'], offset, velscale_ratio,\
                            error_limit, nsims, nbins, np.exp(logLam),np.exp(logLam_template), i) )

        # now get the results with indices
        ppxf_tmp = [outQueue.get() for _ in range(nbins)]

        # send stop signal to stop iteration
        for _ in range(nthreads): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()

        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]                        = ppxf_tmp[i][0]
            ppxf_result[i,:configs['MOM']]  = ppxf_tmp[i][1]
            ppxf_bestfit[i,:]               = ppxf_tmp[i][2]
            optimal_template[i,:]           = ppxf_tmp[i][3]
            mc_results[i,:configs['MOM']]   = ppxf_tmp[i][4]
            formal_error[i,:configs['MOM']] = ppxf_tmp[i][5]
            ppxf_goodpixels[i,0:len(ppxf_tmp[i][6])] = ppxf_tmp[i][6]

        # Sort output
        argidx = np.argsort( index )
        ppxf_result      = ppxf_result[argidx,:]
        ppxf_bestfit     = ppxf_bestfit[argidx,:]
        optimal_template = optimal_template[argidx,:]
        mc_results       = mc_results[argidx,:]
        formal_error     = formal_error[argidx,:]
        ppxf_goodpixels  = ppxf_goodpixels[argidx,:]

        ExGalutil.prettyOutput_Done("Running PPXF in parallel mode", progressbar=True)

    ###### IF RUNNING IN SERIAL
    if nthreads < 2:

        ExGalutil.prettyOutput_Running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")
        for i in range(0, nbins):
            ppxf_result[i,:configs['MOM']], ppxf_bestfit[i,:], optimal_template[i,:],\
              mc_results[i,:configs['MOM']], formal_error[i,:configs['MOM']],goodpixels_ppxf = run_ppxf\
                (templates_lookup[i], bin_data[:,i], noise[:,i], velscale, start[i,:], goodpixels_ppxf,\
                error_limit, configs['MOM'], configs['ADEG'], configs['MDEG'], offset, velscale_ratio,\
                nsims, nbins,np.exp(logLam), np.exp(logLam_template), i)

            # given that size of ppxf_goodpixels could be different than other arrays here
            ppxf_goodpixels[i,0:len(goodpixels_ppxf)] = goodpixels_ppxf
        ExGalutil.prettyOutput_Done("Running PPXF in serial mode", progressbar=True)

    print("             Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, nthreads))
    logging.info("Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, nthreads))

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]
    if len(idx_error) != 0:
        ExGalutil.prettyOutput_Warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             "+str(idx_error))
        logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
    else:
        print("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Calculate LAMBDA_R
    ExGalutil.prettyOutput_Running("Calculating Lambda_R")
    logging.info("Calculating Lambda_R")
    try:
        lambda_r = np.zeros( nbins );  lambda_r[:] = np.nan
        lambda_r = calc_LambdaR( ppxf_result, nbins, outdir, rootname )
        ExGalutil.prettyOutput_Done("Calculating Lambda_R")
    except Exception as e:
        ExGalutil.prettyOutput_Failed("Calculating Lambda_R")
        logging.warning("Failed to calculate Lambda_R: %s. Analysis continues!", e)


    # Save stellar kinematics to file
    save_ppxf(rootname,configs, outdir, ppxf_result, mc_results, formal_error, lambda_r, ppxf_bestfit,\
     logLam, bin_data ,noise , ppxf_goodpixels, optimal_template, logLam_template, npix, ubins)


    # Do plotting
    try:
        ExGalutil.prettyOutput_Running("Producing stellar kinematics maps")
        logging.info("Producing stellar kinematics maps")
        util_plot.plot_maps('PPXF', outdir, rootname)
        util_plot_lambdar.plot_maps(outdir, rootname)
        ExGalutil.prettyOutput_Done("Producing stellar kinematics maps")
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        print("".join(lines))
        sys.stdout.flush()
        ExGalutil.prettyOutput_Failed("Producing stellar kinematics maps")
        logging.warning("Failed to produce stellar kinematics maps. Analysis continues!")
        pass

    print("\033[0;37m"+" - - - - - PPXF done! - - - - -"+"\033[0;39m")
    print("")
    logging.info(" - - - PPXF Done - - - \n")

    if output:
        return ppxf_result, formal_error,  ppxf_bestfit, lambda_r, bin_data, noise, ppxf_goodpixels
    else:
        return
