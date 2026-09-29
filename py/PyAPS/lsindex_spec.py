#!/usr/bin/env python
import optparse
import os
import sys
import warnings
import numpy
import matplotlib.pyplot as plt
from    astropy.io          import ascii
from    astropy.io          import fits
import warnings

#===============================================================================
# IMPROVED LSINDEX_SPEC with better NaN handling for different redshift bins
#===============================================================================

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    """
    filledLength     = int(round(barLength * iteration / float(total)))
    percents          = round(100.00 * (iteration / float(total)), decimals)
    bar                 = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")

#==============================================================================
def load_inputlist(inlist):
    # Reading inputlist
    data = ascii.read(inlist,comment='\s*#')
    names = data['col1']
    redshift = data['col2']
    err_redshift = data['col3']

    return names, redshift, err_redshift

#==============================================================================
def sum_counts(ll, c, b1, b2):
    """
    IMPROVED: Handle NaN values in wavelength and flux arrays
    """
    # Remove NaN values from both wavelength and flux
    valid_idx = ~(numpy.isnan(ll) | numpy.isnan(c))
    if numpy.sum(valid_idx) < 3:  # Need at least 3 points
        return numpy.nan
    
    ll_clean = ll[valid_idx]
    c_clean = c[valid_idx]
    
    # Check if we have coverage for this band
    if len(ll_clean) == 0 or ll_clean.min() > b2 or ll_clean.max() < b1:
        return numpy.nan
    
    # Central full pixel range
    dw = ll_clean[1]-ll_clean[0] if len(ll_clean) > 1 else numpy.median(numpy.diff(ll_clean))
    w  = ((ll_clean >= b1+dw/2.) & (ll_clean <= b2-dw/2.))
    
    if numpy.sum(w) == 0:  # No pixels in range
        return numpy.nan
        
    s  = numpy.sum(c_clean[w])

    # First fractional pixel
    pixb = ((ll_clean < b1+dw/2.) & (ll_clean > b1-dw/2.))
    if numpy.any(pixb):
        fracb = ((ll_clean[pixb]+dw/2.)-b1)/dw
        s      = s+c_clean[pixb]*fracb

    # Last fractional pixel
    pixr = ((ll_clean < b2+dw/2.) & (ll_clean > b2-dw/2.))
    if numpy.any(pixr):
        fracr = (b2-(ll_clean[pixr]-dw/2.))/dw
        s      = s+c_clean[pixr]*fracr

    return s

#==============================================================================
def calc_index(bands,name,ll,counts,plot):
    """
    IMPROVED: Better error handling for NaN values and insufficient data
    """
    # Calculate continuum and feature regions with NaN handling
    cb = sum_counts(ll, counts, bands[0], bands[1])  # Blue continuum
    cr = sum_counts(ll, counts, bands[4], bands[5])  # Red continuum  
    s  = sum_counts(ll, counts, bands[2], bands[3])  # Feature region
    
    # Check if we have valid measurements for all regions
    if numpy.isnan(cb) or numpy.isnan(cr) or numpy.isnan(s):
        return numpy.array([numpy.nan])
    
    # Check for zero-width bands (causes division by zero)
    if (bands[1] - bands[0]) == 0 or (bands[5] - bands[4]) == 0:
        return numpy.array([numpy.nan])

    lb = (bands[0]+bands[1])/2.0
    lr = (bands[4]+bands[5])/2.0
    cb = cb / (bands[1]-bands[0])
    cr = cr / (bands[5]-bands[4])
    
    # Check for valid continuum values
    if cb <= 0 or cr <= 0:
        return numpy.array([numpy.nan])
    
    m  = (cr-cb) / (lr-lb)
    c1 = (m*(bands[2]-lb))+cb
    c2 = (m*(bands[3]-lb))+cb
    cont = 0.5*(c1+c2)*(bands[3]-bands[2])
    
    # Check for valid continuum level
    if cont <= 0:
        return numpy.array([numpy.nan])

    if bands[6] == 1.:
        # atomic index
        ind = (1.0 - (s/cont))*(bands[3]-bands[2])
    elif bands[6] == 2.:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') 
            # molecular index
            if s <= 0 or cont <= 0:
                ind = numpy.nan
            else:
                ind = -2.5*numpy.log10(s/cont)
    else:
        ind = numpy.nan

    # IMPROVED: Handle plotting with NaN values
    if plot > 0 and not numpy.isnan(ind):
        try:
            # Remove NaN values for plotting
            valid_plot = ~(numpy.isnan(ll) | numpy.isnan(counts))
            if numpy.sum(valid_plot) > 10:  # Need enough points to plot
                ll_plot = ll[valid_plot]
                counts_plot = counts[valid_plot]
                
                minx = bands[0]-0.05*(bands[5]-bands[0])
                maxx = bands[5]+0.05*(bands[5]-bands[0])
                miny = numpy.amin(counts_plot)-0.05*(numpy.amax(counts_plot)-numpy.amin(counts_plot))
                maxy = numpy.amax(counts_plot)+0.05*(numpy.amax(counts_plot)-numpy.amin(counts_plot))
                
                plt.figure()
                plt.scatter(ll_plot,counts_plot,color='k')
                plt.xlabel("Wavelength ($\AA$)")
                plt.ylabel("Counts")
                plt.title(f"{name} = {ind:.3f}")
                plt.xlim([minx,maxx])
                plt.ylim([miny,maxy])
                
                # Add band markers
                for i in range(len(bands)-1):  # Skip last element (index type)
                    plt.axvline(bands[i], color='k', linestyle='--', alpha=0.7)
                plt.show()
        except Exception as e:
            print(f"Warning: Plotting failed for {name}: {str(e)}")

    return numpy.array([ind])

#==============================================================================
def lsindex(ll, flux_in, noise, z, lickfile, plot=0, sims=0, z_err=0):
    """
    IMPROVED: Measure line-strength indices with robust NaN handling
    
    This version can handle:
    - Partial NaN values in wavelength or flux arrays
    - Different wavelength coverage for different redshift bins
    - Missing data in specific wavelength regions
    """
    
    # IMPROVED: Input validation
    if len(ll) != len(flux_in) or len(ll) != len(noise):
        raise ValueError("Wavelength, flux, and noise arrays must have same length")
    
    # Deredshift spectrum to rest wavelength
    dll = ll/(z+1.)

    # IMPROVED: Handle NaN values in input
    total_pixels = len(flux_in)
    valid_pixels = numpy.sum(~(numpy.isnan(ll) | numpy.isnan(flux_in)))
    
    if valid_pixels < 0.1 * total_pixels:  # Less than 10% valid data
        print(f"Warning: Insufficient valid data ({valid_pixels}/{total_pixels} pixels)")
        # Return NaN for all indices
        tab = ascii.read(lickfile, comment='\s*#')
        names = tab['names']
        return names, numpy.full(len(names), numpy.nan), numpy.full(len(names), numpy.nan)
    
    if valid_pixels < total_pixels:
        print(f"Info: Using {valid_pixels}/{total_pixels} valid pixels ({100*valid_pixels/total_pixels:.1f}%)")

    # Use flux as-is (don't modify for NaN values - let individual index calculations handle them)
    flux = flux_in

    # Read index definition table
    tab    = ascii.read(lickfile, comment='\s*#')
    names = tab['names']
    bands = numpy.zeros((7,len(names)))
    bands[0,:] = tab['b1']
    bands[1,:] = tab['b2']
    bands[2,:] = tab['b3']
    bands[3,:] = tab['b4']
    bands[4,:] = tab['b5']
    bands[5,:] = tab['b6']
    bands[6,:] = tab['b7']

    # Measure line indices
    num_ind = len(bands[0,:])
    index    = numpy.zeros(num_ind)
    
    for k in range(num_ind):    # loop through all indices
        # IMPROVED: Check wavelength coverage with NaN handling
        valid_wave_idx = ~numpy.isnan(dll)
        if numpy.sum(valid_wave_idx) == 0:
            index[k] = numpy.nan
            continue
            
        dll_valid = dll[valid_wave_idx]
        
        # Check if we have any coverage for this index
        if len(dll_valid) == 0:
            index[k] = numpy.nan
            continue
            
        # Check wavelength coverage for this index
        if ((dll_valid.min() <= bands[0,k]) and (dll_valid.max() >= bands[5,k])):
            # We have coverage - attempt to calculate index value
            index0 = calc_index(bands[:,k], names[k], dll, flux, plot)
            index[k] = index0[0]
        else:
            # Index outside available wavelength range
            index[k] = numpy.nan

    # IMPROVED: Calculate errors with NaN handling
    index_error = numpy.zeros(num_ind,dtype='D')
    index_error[:] = numpy.nan
    index_noise = numpy.zeros([num_ind,sims], dtype='D')

    if sims > 0:
        # Create redshift and sigma errors
        dz = numpy.random.randn(sims)*z_err

        # Loop through the simulations
        for i in range(sims):
            # resample spectrum according to noise
            valid_noise_idx = ~numpy.isnan(noise)
            if numpy.sum(valid_noise_idx) > 0:
                ran = numpy.random.normal(0.0,1.0,len(dll))
                flux_n = flux + ran*noise
                
                # Handle NaN values in noisy spectrum
                flux_n[numpy.isnan(flux_n)] = flux[numpy.isnan(flux_n)]
            else:
                # No valid noise - use original flux
                flux_n = flux.copy()

            # loop through all indices
            for k in range(num_ind):
                # shift bands according to redshift error
                sz  = z + dz[i]
                dll_sim = ll/(sz+1.)
                bands2 = bands[:,k]
                
                # Check coverage for this simulation
                valid_sim_idx = ~numpy.isnan(dll_sim)
                if numpy.sum(valid_sim_idx) > 0:
                    dll_sim_valid = dll_sim[valid_sim_idx]
                    if ((dll_sim_valid.min() <= bands2[0]) and (dll_sim_valid.max() >= bands2[5])):
                        tmp = calc_index(bands2, names[k], dll_sim, flux_n, 0)
                        index_noise[k,i] = tmp[0]
                    else:
                        index_noise[k,i] = numpy.nan
                else:
                    index_noise[k,i] = numpy.nan

        # Get STD of distribution (index error) - handle NaN values
        for k in range(num_ind):
            valid_sims = ~numpy.isnan(index_noise[k,:])
            if numpy.sum(valid_sims) > 5:  # Need at least 5 valid simulations
                index_error[k] = numpy.std(index_noise[k,valid_sims])
            else:
                index_error[k] = numpy.nan

    return names, index, index_error

#==============================================================================
if __name__ == "__main__":
    # [Rest of main execution code remains the same]
    os.system('clear')
    warnings.filterwarnings("ignore")
    print("========================")
    print("= Running LSINDEX_SPEC =")
    print("========================")
    print("")

    # [Command line parsing code unchanged...]
    parser = optparse.OptionParser(usage="%prog -i inputlist -l lickfile -o outfits -n nsims")
    parser.add_option("-i", "--inputlist", dest="inputlist", type="string", default="../config_files/miles_ku.inputlist",  help="List of input spectra, redshift and err_redshift")
    parser.add_option("-l", "--lickfile",  dest="lickfile",  type="string", default="../config_files/lick_bands.conf",      help="Lick file with index definitions")
    parser.add_option("-o", "--outfits",    dest="outfits",    type="string", default="../results/lick_indices.fits",          help="Name of output FITS table with results")
    parser.add_option("-n", "--nsims",      dest="nsims",      type="int",     default="0",                                              help="Number of MC simulations for errors")
    parser.add_option("-p", "--plot",        dest="plot",        type="int",     default="0",                                              help="Plotting or not [0/1]")

    (options, args) = parser.parse_args()
    inputlist = options.inputlist
    lickfile  = options.lickfile
    outfits    = options.outfits
    nsims      = options.nsims
    plot_flag = options.plot

    # [Rest of main execution unchanged...]
    # Getting the list of FITS files to process
    print("# Loading inputlist: "+inputlist)
    inlist, redshift, err_redshift = load_inputlist(inputlist)
    nfiles = len(inlist)
    print("- "+str(nfiles)+" files found")
    print("")

    # Computing the magnitudes for each input FITS file
    print("# Computing indices...")
    root     = []
    for i in range(nfiles):

        # Opening the FITS file
        hdu      = fits.open(inlist[i])
        flux     = hdu[0].data
        npix     = len(flux)
        crpix    = hdu[0].header['CRPIX1']
        crval    = hdu[0].header['CRVAL1']
        cdelt    = hdu[0].header['CDELT1']
        wave     = ((numpy.arange(npix) + 1.0) - crpix) * cdelt + crval
        root     = numpy.append(root,os.path.basename(inlist[i]))

        # Computing the indices
        names, indices, errors = lsindex(wave, flux, flux*0.1, redshift[i], lickfile, plot=plot_flag, sims=nsims, z_err=err_redshift[i] )

        if i == 0:
            outls      = numpy.zeros((len(names),nfiles))
            outls_err = numpy.zeros((len(names),nfiles))

        outls[:,i] = indices
        outls_err[:,i] = errors

        printProgress(i+1, nfiles, prefix = ' ', suffix = 'Complete', barLength = 50)


    # Saving the results to a FITS table
    if os.path.exists(outfits):
        os.remove(outfits)
    print("# Results will be stored in the FITS table: "+outfits)
    print("")
    cols = []
    cols.append(fits.Column('Files', format='100A', array=root))
    ndim  = len(names)
    for i in range(ndim):
        cols.append(fits.Column(name=names[i],          format='D', array=outls[i,:]))
        cols.append(fits.Column(name="ERR_"+names[i], format='D', array=outls_err[i,:]))
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.writeto(outfits)

    print("# DONE!")
