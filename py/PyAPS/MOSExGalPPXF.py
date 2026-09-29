
from   astropy.io          import fits
import numpy               as np
import sys
import traceback
import warnings
import pickle


import os
os.environ['OMP_NUM_THREADS'] = '1'
import logging
import glob
from   scipy.interpolate import interp1d
import astropy.io.fits as pyfits
from multiprocessing import Queue, Process

import PyAPS
from PyAPS import ExGalutil
from PyAPS import ExGalPrepare
from PyAPS import MOSExGalPrepare    as MOSExGalPrepare
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class, apply_redshift_to_fwhm_corrected
from PyAPS import aps_constants
from PyAPS.apsPlot.ppxf import make_ppxf_plot
APSVERS = PyAPS.__version__


from ppxf.ppxf import ppxf
from ppxf import __version__ as PPXF_VERSION

# try:
#     # Try to use local version in sitePackages
# except:
#     # Then use system installed version instead
#     from ppxf.ppxf_util import log_rebin, gaussian_filter1d

from ppxf.ppxf_util import log_rebin, gaussian_filter1d

# PHYSICAL CONSTANTS
Clight = 299792.458  # km/s

############################################################################################################################




############################################################################################################################
# make_ppxf_plot now lives in PyAPS.apsPlot.ppxf, built on the shared
# spectrum_overlay_figure() (Plotly) used across the PyAPS.apsPlot platform.
# Imported at module top as `from PyAPS.apsPlot.ppxf import make_ppxf_plot`.
############################################################################################################################
def save_ppxf(rootname, configs, outdir, metalist, bin_data , bin_error, ppxf_result, mc_results, formal_error, lambda_r,\
              ppxf_bestfit, logLam, goodpixels, optimal_template, logLam_template, npix, ubins):
    """ Saves all results to disk. """
    # ========================

    #metalist is the basket to transfer basic info and parameters from the original files
    APS_ID = [item['APS_ID'] for item in metalist]
    TARGID = [item['TARGID'] for item in metalist]
    CNAME = [item['CNAME'] for item in metalist]
    Z = np.array([item['Z'] for item in metalist])
    ZERR = np.array([item['ZERR'] for item in metalist])
    BINID_meta = [item['BIN_ID'] for item in metalist]

    # Just to make sure data in our basket (metalist) are consistent with the PPXF output
    assert BINID_meta == list(ubins), 'input and output BIN_ID are not identical'


    ## add Units to flux, model and ivar in the ppxf's spec table
    if configs['sens_corr']:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' %(configs['funits'])
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' %(configs['funits']**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'



    # SAVE RESULTS
    outfits_ppxf = outdir+rootname+'_ppxf.fits'
    ExGalutil.prettyOutput_Running("Writing: "+rootname+'_ppxf.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF output data
    cols = []
    cols.append( fits.Column(name='BIN_ID',                     format='J', array=ubins             ))
    cols.append( fits.Column(name='APS_ID',                     format='J', array=APS_ID            ))
    cols.append( fits.Column(name='TARGID',                     format='40A', array=TARGID          ))
    cols.append( fits.Column(name='CNAME',                      format='40A', array=CNAME           ))

    # As the galaxy is at significant redshift z and as the wavelength has been
    # de-redshifted, the best-fitting redshift is now given by the following
    # formula (equation 2 of Cappellari et al. 2009, ApJ, 704, L34):
    #
    # Z_CORR =  (z_input + 1)*(1 + PPXF_V/Clight) - 1
    # and error in Z_corr is coming from
    # ZERR_CORR = (z_input_err)^2 + (1/Clight ^2) * [(z_input_err)^2)/(z_input)^2) + (PPXF_V_ERR)^2)/(PPXF_V)^2) + (PPXF_V_ERR)^2)]

    Z_CORR = (Z+1.0)*(1.0+(ppxf_result[:,0]/Clight))-1

    # Safely evaluate ZERR (corrected) using MCMC error in V
    with np.errstate(divide='ignore', invalid='ignore'):
        # Create safe versions that avoid division by zero
        Z_safe = np.where(Z != 0, Z, 1.0)
        V_safe = np.where(ppxf_result[:,0] != 0, ppxf_result[:,0], 1.0)

        # Calculate MCMC error
        ZERR_CORR_mc_p2 = ((ZERR)**2.0) + (1.0/Clight**2.0) * (
            (((ZERR)**2.0)/(Z_safe**2.0)) +
            (((mc_results[:,0])**2.0)/(V_safe**2.0)) +
            ((mc_results[:,0])**2.0)
        )

        # Calculate formal error
        ZERR_CORR_form_p2 = ((ZERR)**2.0) + (1.0/Clight**2.0) * (
            (((ZERR)**2.0)/(Z_safe**2.0)) +
            (((formal_error[:,0])**2.0)/(V_safe**2.0)) +
            ((formal_error[:,0])**2.0)
        )

        # Set to NaN where we had division by zero
        invalid_mask = (Z == 0) | (ppxf_result[:,0] == 0)
        ZERR_CORR_mc_p2[invalid_mask] = np.nan
        ZERR_CORR_form_p2[invalid_mask] = np.nan


    cols.append( fits.Column(name='ZCORR',                       format='D', array=Z_CORR           ))
    # cols.append( fits.Column(name='ZINPUT' ,        unit='km/s', format='D', array=Z                ))
    # cols.append( fits.Column(name='ERR_ZINPUT',     unit='km/s', format='D', array=ZERR             ))

    cols.append( fits.Column(name='V' ,             unit='km/s', format='D', array=ppxf_result[:,0] ))
    cols.append( fits.Column(name='SIGMA',          unit='km/s', format='D', array=ppxf_result[:,1] ))
    cols.append( fits.Column(name='H3',             unit='km/s', format='D', array=ppxf_result[:,2] ))
    cols.append( fits.Column(name='H4',             unit='km/s', format='D', array=ppxf_result[:,3] ))
    cols.append( fits.Column(name='H5',             unit='km/s', format='D', array=ppxf_result[:,4] ))
    cols.append( fits.Column(name='H6',             unit='km/s', format='D', array=ppxf_result[:,5] ))
    # cols.append( fits.Column(name='LAMBDA_R',       format='D', array=lambda_r[:]       ))

    cols.append( fits.Column(name='ERR_ZCORR',           format='D', array=np.sqrt(ZERR_CORR_mc_p2) ))
    cols.append( fits.Column(name='ERR_V' ,         unit='km/s', format='D', array=mc_results[:,0]  ))
    cols.append( fits.Column(name='ERR_SIGMA',      unit='km/s', format='D', array=mc_results[:,1]  ))
    cols.append( fits.Column(name='ERR_H3',         unit='km/s', format='D', array=mc_results[:,2]  ))
    cols.append( fits.Column(name='ERR_H4',         unit='km/s', format='D', array=mc_results[:,3]  ))
    cols.append( fits.Column(name='ERR_H5',         unit='km/s', format='D', array=mc_results[:,4]  ))
    cols.append( fits.Column(name='ERR_H6',         unit='km/s', format='D', array=mc_results[:,5]  ))

    cols.append( fits.Column(name='FORM_ERR_ZCORR',    format='D', array=np.sqrt(ZERR_CORR_form_p2) ))
    cols.append( fits.Column(name='FORM_ERR_V' ,    unit='km/s', format='D', array=formal_error[:,0]))
    cols.append( fits.Column(name='FORM_ERR_SIGMA', unit='km/s', format='D', array=formal_error[:,1]))
    cols.append( fits.Column(name='FORM_ERR_H3',    unit='km/s', format='D', array=formal_error[:,2]))
    cols.append( fits.Column(name='FORM_ERR_H4',    unit='km/s', format='D', array=formal_error[:,3]))
    cols.append( fits.Column(name='FORM_ERR_H5',    unit='km/s', format='D', array=formal_error[:,4]))
    cols.append( fits.Column(name='FORM_ERR_H6',    unit='km/s', format='D', array=formal_error[:,5]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'PPXF_TABLE'

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
    # dataHDU.header['COMMENT'] = '---------- END OF PyAPS PPXF SETTING PARMAS ----------'




    dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    dataHDU.header['APSVERS'] = (APSVERS,'APS version')
    dataHDU.header['APSPPXFV'] = (aps_constants.__aps_ppxf_version__,'PyAPS (MOS) PPXF wrapper version')
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
    cols.append( fits.Column(name='BIN_ID',                      format='J',           array=ubins       ))
    cols.append( fits.Column(name='APS_ID',                      format='J',            array=APS_ID     ))
    cols.append( fits.Column(name='TARGID',                      format='40A',          array=TARGID     ))
    cols.append( fits.Column(name='CNAME',                       format='40A',         array=CNAME       ))
    cols.append( fits.Column(name='LOGLAM_PPXF',  unit=wave_unit_str, format=str(npix)+'D', array=logLam      ))
    cols.append( fits.Column(name='FLUX_PPXF',    unit=flux_unit_str, format=str(npix)+'D', array=bin_data    ))
    cols.append( fits.Column(name='ERROR_PPXF',   unit=flux_unit_str, format=str(npix)+'D', array=bin_error   ))
    cols.append( fits.Column(name='MODEL_PPXF',   unit=flux_unit_str, format=str(npix)+'D', array=ppxf_bestfit))
    cols.append( fits.Column(name='GOODPIX_PPXF',                     format=str(npix)+'J', array=goodpixels  ))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'PPXF_SPEC'
    dataHDU.header['CNF_PPXF'] = (configs['CONFIG_FILE'],'Configs. Filename used by PPXF')
    ExGalPrepare.write_ebmv_header(dataHDU.header, configs)
    dataHDU.header['SAMPLING'] = (1,'Sampling mode (0: linear, 1: logarithmic')
    dataHDU.header['TEMPL'] = (configs['SSP_LIB'],'The library of spectral templates')
    dataHDU.header['PPXF_V'] = (PPXF_VERSION, 'PPXF version')
    dataHDU.header['APSVERS'] = (APSVERS,'APS version')
    dataHDU.header['APSPPXFV'] = (aps_constants.__aps_ppxf_version__,'PyAPS (MOS) PPXF wrapper version')
    dataHDU.header['CSB_PPXF'] = (configs['stitched'], 'Combines Spectral Bands Status for PPXF')


    #keep the basename of input file (infiles) to save as provinces later
    for n_province, province in enumerate(configs['infiles']):
        dataHDU.header['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')


    # Create HDU list and write to file

    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)

    ExGalutil.prettyOutput_Done("Writing: "+rootname+'_ppxf_spec.fits')
    logging.info("Wrote: "+outfits_ppxf)


############################################################################################################################
def preparePPXF(index, info_tab, configs, outdir, config_dir, templates_dir,
                bin_data, bin_error, logLam, velscale, LSF_Templates, error_limit, debug):

    # Assume bin_ID and index are aligned
    binid = np.ravel(np.where(info_tab['BIN_ID'] == index))[0]

    ppxf_meta = {
        'APS_ID': info_tab['APS_ID'][binid],
        'BIN_ID': info_tab['BIN_ID'][binid],
        'TARGID': info_tab['TARGID'][binid],
        'CNAME': info_tab['CNAME'][binid],
        'Z': info_tab['Z'][binid],
        'ZERR': info_tab['ZERR'][binid],
    }

    # old method
    # Load and correct LSF
    # LSF = np.genfromtxt(config_dir + 'LSF-Config_' + configs['SETMODE'], comments='#')
    # LSF[:, 0] /= (1 + ppxf_meta['Z'])
    # LSF[:, 1] /= (1 + ppxf_meta['Z'])
    # LSF_Data = interp1d(LSF[:, 0], LSF[:, 1], 'linear', fill_value='extrapolate')

    # new method with reading LSF from the L1 calibrations
    lsf_data = MOSExGalPrepare.load_lsf_data(configs)
    lsf_indx = np.where(lsf_data['aps_id'] == ppxf_meta['APS_ID'])[0]
    lsf_func = lsf_data['lsf'][lsf_indx][0]
    LSF_Data = apply_redshift_to_fwhm_corrected(lsf_func, ppxf_meta['Z'])

    try:

        # Velocity scale ratio calculation
        velscale_ratio, template_velscale, data_velscale = MOSExGalPrepare.calculate_velscale_ratio(
            configs, logLam, velscale, LSF_Data, LSF_Templates, templates_dir, debug=debug
        )

        # Good pixel selection
        goodpixels_ppxf = MOSExGalPrepare.spectralMasking(
            config_dir, configs, logLam, 'PPXF', ppxf_meta['Z'], LSF_Data=LSF_Data, verbose=debug
        )
        assert len(goodpixels_ppxf) > 0, 'ERROR: null spectrum after LMIN/LMAX masking!'

        goodpixels_ppxf = MOSExGalPrepare.MaskGaps(bin_error, goodpixels_ppxf, error_limit)
        assert len(goodpixels_ppxf) > 0, 'ERROR: null spectrum after masking CCD GAPS!'

        # Spectral template preparation
        lmin = np.min(np.exp(logLam))
        lmax = np.max(np.exp(logLam))
        templates, lamRange_spmod, logLam_template, ntemplates = MOSExGalPrepare.prepareSpectralTemplateLibrary(
            "PPXF", templates_dir, configs, lmin, lmax, velscale, velscale_ratio, LSF_Data, LSF_Templates
        )[:4]

        offset = (logLam_template[0] - logLam[0]) * Clight
        optimal_template = np.zeros((templates.shape[0],))

    except Exception as e:
        print(f"[WARN] preparePPXF failed for BIN_ID={index}: {e}")
        sys.stdout.flush()

        # Safe fallbacks
        templates = np.zeros((len(logLam), 1))
        lamRange_spmod = [np.nan, np.nan]
        logLam_template = np.zeros_like(logLam)
        ntemplates = 0
        offset = np.nan
        optimal_template = np.zeros(len(logLam))
        velscale_ratio = np.nan
        goodpixels_ppxf = np.array([], dtype=int)

    return ppxf_meta, LSF_Data, templates, offset, goodpixels_ppxf, optimal_template, logLam_template, velscale_ratio


############################################################################################################################
def workerPPXF_serial(outdir,config_dir, templates_dir, logLam, bin_data, bin_error, configs, info_tab, velscale, LSF_Templates, start, error_limit, nsims, nbins, debug, i):

    ppxf_meta, LSF_Data,  templates, offset, goodpixels_ppxf, optimal_template,logLam_template, velscale_ratio = \
    preparePPXF(i , info_tab, configs, outdir , config_dir, templates_dir, bin_data, bin_error, logLam, velscale ,LSF_Templates, error_limit, debug)

    sol, bestfit, optimal_template, mc_results, formal_error = \
    run_ppxf(templates, bin_data, bin_error, velscale, start, goodpixels_ppxf, configs['MOM'],\
        configs['ADEG'], configs['MDEG'], offset, velscale_ratio, nsims, nbins, logLam, logLam_template, debug, i)

    return i, sol, bestfit, optimal_template, mc_results, formal_error, goodpixels_ppxf, ppxf_meta
############################################################################################################################

def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process.
    """
    for outdir,config_dir,templates_dir, logLam, bin_data, bin_error, configs, info_tab, velscale, \
    LSF_Templates, start, error_limit, nsims, nbins, debug, i in iter(inQueue.get, 'STOP'):

        ## make sure outputs of the workerPPXF_serial are in the following order
        ## i, sol, bestfit, optimal_template, mc_results, formal_error, goodpixels_ppxf, ppxf_meta

        outQueue.put(workerPPXF_serial(outdir, config_dir, templates_dir, logLam, bin_data, bin_error, configs, info_tab, velscale, \
            LSF_Templates, start, error_limit, nsims, nbins, debug, i))

############################################################################################################################

def run_ppxf(templates, log_bin_data, log_bin_error, velscale, start, goodpixels, nmoments,
             adeg, mdeg, offset, velscale_ratio, nsims, nbins, logLam, logLam_template, debug, i):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004
    to determine the stellar kinematics.
    """
    ExGalutil.printProgress(i, nbins, barLength=50)

    try:
        # Convert log wavelengths
        lam_stellar = np.exp(logLam)
        lam_stellar_template = np.exp(logLam_template)

        # Run pPXF
        pp = ppxf(templates, log_bin_data, log_bin_error, velscale, start,
                  goodpixels=goodpixels, plot=False, quiet=True,
                  moments=nmoments, degree=adeg, mdegree=mdeg,
                  velscale_ratio=velscale_ratio,
                  lam=lam_stellar, lam_temp=lam_stellar_template)

        # Make the unconvolved optimal stellar template
        # normalized_weights = pp.weights / np.sum(pp.weights)

        weights_sum = np.sum(pp.weights)
        if weights_sum != 0:
            normalized_weights = pp.weights / weights_sum
        else:
            normalized_weights = np.zeros_like(pp.weights)

        optimal_template = np.zeros(templates.shape[0])
        for j in range(templates.shape[1]):
            optimal_template += templates[:, j] * normalized_weights[j]

        # Correct formal errors
        formal_error = pp.error * np.sqrt(pp.chi2)

        # Monte Carlo simulations
        sol_MC = np.full((nsims, nmoments), np.nan)
        mc_results = np.full(nmoments, np.nan)

        for o in range(nsims):
            noise = np.random.normal(0, 1, len(log_bin_data))
            stddev = np.std(log_bin_data[goodpixels] - pp.bestfit[goodpixels])
            noisy_bestfit = pp.bestfit + noise * stddev

            mc = ppxf(templates, noisy_bestfit, log_bin_error, velscale, start,
                      goodpixels=goodpixels, plot=False, quiet=True,
                      moments=nmoments, degree=adeg, mdegree=mdeg,
                      velscale_ratio=velscale_ratio, bias=0.0,
                      lam=lam_stellar, lam_temp=lam_stellar_template)

            sol_MC[o, :] = mc.sol[:]

        if nsims != 0:
            mc_results = np.nanstd(sol_MC, axis=0)

        # Set bestfit to zero outside goodpixels
        pp.bestfit[np.setdiff1d(np.arange(len(pp.bestfit)), goodpixels)] = 0.0

        return pp.sol[:], pp.bestfit, optimal_template, mc_results, formal_error

    except Exception as e:
        print(f"[WARN] pPXF failed for bin {i}: {e}")

        # Fallback: return arrays with correct shapes filled with np.nan or 0.0
        sol = np.full(nmoments, np.nan)
        bestfit = np.zeros_like(log_bin_data)
        optimal_template = np.zeros(templates.shape[0])
        mc_results = np.full(nmoments, np.nan)
        formal_error = np.full(nmoments, np.nan)

        return sol, bestfit, optimal_template, mc_results, formal_error




############################################################################################################################

def runModule_PPXF(nthreads, configs, velscale, LSF_Templates, outdir, config_dir, templates_dir, figdir,  rootname, error_limit, debug=False, output=False):

    # #################### Read data from file ##########################
    hdu_t = fits.open(outdir+rootname+'_table.fits')
    info_tab = hdu_t[1].data
    hdu_s      = fits.open(outdir+rootname+'_BINSpectra.fits')
    bin_data = np.array( hdu_s[1].data.SPEC )
    bin_error = np.array( hdu_s[1].data.ESPEC )
    logLam   = np.array( hdu_s[1].data.LOGLAM )


    nbins    = bin_data.shape[0]
    ubins    = np.arange(0, nbins)
    npix     = bin_data.shape[1]

    nsims  = configs['MC_PPXF']


    # Array to store results of ppxf
    ppxf_metalist      = [None for _ in range(nbins)]
    ppxf_result        = np.zeros((nbins,6))
    ppxf_bestfit       = np.zeros((nbins,npix))

    # ppxf_goodpixels    = np.zeros((nbins,npix))
    # optimal_template   = np.zeros((nbins,templates.shape[0]))
    mc_results         = np.zeros((nbins,6))
    formal_error       = np.zeros((nbins,6))
    ppxf_goodpixels    = np.empty((nbins,npix))
    ppxf_goodpixels.fill(-1)

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


    ## START RUNNING PPXF in serial or parallel mode
    if nthreads > 1:
        logging.info("Running PPXF in parallel mode")

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()

        # Create worker processes
        ps = [Process(target=workerPPXF, args=(inQueue, outQueue))
              for _ in range(nthreads)]

        # Start worker processes
        for p in ps: p.start()

        # Fill the queue
        for i in range(nbins):
            inQueue.put((outdir,config_dir,templates_dir,logLam[i,:], bin_data[i,:], bin_error[i,:], \
                configs, info_tab, velscale[i], LSF_Templates, start[i,:], error_limit, nsims, nbins, debug, i))

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
            # optimal_template[i,:]           = ppxf_tmp[i][3]
            mc_results[i,:configs['MOM']]   = ppxf_tmp[i][4]
            formal_error[i,:configs['MOM']] = ppxf_tmp[i][5]
            ppxf_goodpixels[i,0:len(ppxf_tmp[i][6])] = ppxf_tmp[i][6]
            ppxf_metalist[i]                = ppxf_tmp[i][7]

        # Sort output
        argidx = np.argsort( index )
        ppxf_result      = ppxf_result[argidx,:]
        ppxf_bestfit     = ppxf_bestfit[argidx,:]
        # optimal_template = optimal_template[argidx,:]
        mc_results       = mc_results[argidx,:]
        formal_error     = formal_error[argidx,:]
        ppxf_goodpixels  = ppxf_goodpixels[argidx,:]
        ppxf_metalist    = [ppxf_metalist[_lms] for _lms in argidx]

        ExGalutil.prettyOutput_Done("Running PPXF in parallel mode", progressbar=True)

    elif nthreads < 2:
        ExGalutil.prettyOutput_Running("Running PPXF in serial mode")
        logging.info("Running PPXF in serial mode")

        for i in range(0, nbins):
            _, ppxf_result[i,:configs['MOM']], ppxf_bestfit[i,:], optimal_template, mc_results[i,:configs['MOM']], \
            formal_error[i,:configs['MOM']], goodpixels_i, ppxf_metalist[i] = workerPPXF_serial(outdir, config_dir, \
            templates_dir, logLam[i,:], bin_data[i,:], bin_error[i,:], configs, info_tab, velscale[i], \
            LSF_Templates, start[i,:], error_limit, nsims, nbins, debug, i)

            # given that size of ppxf_goodpixels could be different than other arrays here
            ppxf_goodpixels[i,0:len(goodpixels_i)] = goodpixels_i

            ExGalutil.prettyOutput_Done("Running PPXF in serial mode", progressbar=True)

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( ppxf_result[:,0] ) == True )[0]
    if len(idx_error) != 0:
        aps_id_list_error=[]
        for idxerr in idx_error:
            aps_id_list_error.append(ppxf_metalist[idxerr]['APS_ID'])

        ExGalutil.prettyOutput_Warning("An error has occurred in processing of the spectra with the following APS_ID's: ")
        print("             "+str(aps_id_list_error))
        logging.warning("An error has occurred in processing of the spectra with the following APS_ID's: "+str(aps_id_list_error))
    else:
        print("             "+"No problems in the analysis.")
        logging.info("No problems in the analysis.")
    print("")

    # Save stellar kinematics to file
    ExGalutil.prettyOutput_Running("Saving PPXF result into the disk")
    save_ppxf(rootname, configs, outdir, ppxf_metalist ,bin_data, bin_error, ppxf_result, mc_results, formal_error, None,
        ppxf_bestfit, logLam, ppxf_goodpixels, None, None, npix, ubins)
    ExGalutil.prettyOutput_Done("Saving PPXF result into the disk")


    if figdir:
        try:
            ExGalutil.prettyOutput_Running("Plotting PPXF results")
            make_ppxf_plot(figdir, configs, ppxf_metalist, ppxf_result, logLam, bin_data, bin_error, ppxf_bestfit, ppxf_goodpixels)
            ExGalutil.prettyOutput_Done("Plotting PPXF results")
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print("".join(lines))
            sys.stdout.flush()
            ExGalutil.prettyOutput_Failed("Producing stellar kinematics maps")
            logging.warning("Failed to produce stellar kinematics maps for some APS_IDs. Analysis continues!")
            pass

    if output:
        return ppxf_result, formal_error,  ppxf_bestfit, logLam, bin_data, bin_error, ppxf_goodpixels, ppxf_metalist
    else:
        return
############################################################################################################################
