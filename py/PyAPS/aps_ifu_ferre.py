"""
IFU-specific FERRE processing module for PyAPS

This module processes binned IFU spectra through FERRE for stellar parameter determination.
It reads the binned spectra files created by the Galactic mode preprocessing and runs
FERRE on each bin separately, handling the linear wavelength format used in Galactic mode.

Version History:
- 1.0: Original version
- 1.1: Fixed unique working directory issue for parallel processing (Oct 2025)
"""

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
import subprocess
import shutil
import tempfile
from pathlib import Path


def read_binned_spectra_for_ferre(binspec_file, configs_gal=None):
    """
    Read binned spectra file and prepare data for FERRE processing.
    
    Parameters:
    -----------
    binspec_file : str
        Path to the binned spectra FITS file
    configs_gal : dict
        Configuration dictionary containing setup information
        
    Returns:
    --------
    bin_data : dict
        Dictionary containing bin information and spectra
    """
    
    if not os.path.exists(binspec_file):
        raise FileNotFoundError(f"Binned spectra file not found: {binspec_file}")
    
    print(f"Reading binned spectra from: {binspec_file}")
    
    with fits.open(binspec_file) as hdul:
        bin_table = hdul['BIN_SPECTRA'].data
        bin_header = hdul['BIN_SPECTRA'].header
    
    n_bins = len(bin_table)
    print(f"Found {n_bins} bins to process")
    
    # Extract column names
    colnames = bin_table.columns.names
    
    # Prepare output structure
    bin_data = {
        'n_bins': n_bins,
        'bin_ids': [],
        'spectra': [],
        'errors': [],
        'wavelengths': [],
        'metadata': []
    }
    
    for i_bin in range(n_bins):
        row = bin_table[i_bin]
        
        # Get bin ID
        bin_id = row['BIN_ID'] if 'BIN_ID' in colnames else i_bin
        
        # Extract wavelength, flux, and error
        if 'SPEC_LIN_WAVE' in colnames:
            wave = row['SPEC_LIN_WAVE']
            flux = row['SPEC_LIN_FLUX']
            error = row['SPEC_LIN_ERR']
        elif 'SPEC_LIN_LAMBDA' in colnames:
            wave = row['SPEC_LIN_LAMBDA']
            flux = row['SPEC_LIN']
            error = row['ERR_LIN']
        else:
            wave = row['LAM']
            flux = row['SPEC']
            error = row['ESPEC']
        
        # Store metadata
        metadata = {'BIN_ID': bin_id}
        for key in ['APS_ID', 'TARGID', 'CNAME', 'X', 'Y']:
            if key in colnames:
                metadata[key] = row[key]
        
        bin_data['bin_ids'].append(bin_id)
        bin_data['spectra'].append(flux)
        bin_data['errors'].append(error)
        bin_data['wavelengths'].append(wave)
        bin_data['metadata'].append(metadata)
    
    return bin_data


def find_overlap_center(wlranges):
    """
    Find overlap between wavelength ranges and return center of overlap.
    
    Parameters:
    -----------
    wlranges : list of lists
        List of [wmin, wmax] pairs
        
    Returns:
    --------
    overlap_center : float or None
        Center wavelength of overlap region, or None if no overlap
    """
    if len(wlranges) < 2:
        return None
    
    # For simplicity, check overlap between first two ranges
    # (can be extended for multiple ranges if needed)
    wl1_min, wl1_max = wlranges[0]
    wl2_min, wl2_max = wlranges[1]
    
    # Find overlap region
    overlap_min = max(wl1_min, wl2_min)
    overlap_max = min(wl1_max, wl2_max)
    
    # Check if overlap exists
    if overlap_max > overlap_min:
        overlap_center = (overlap_min + overlap_max) / 2.0
        print(f"    Overlap detected: [{overlap_min:.1f}, {overlap_max:.1f}] Å")
        print(f"    Using split point at center: {overlap_center:.1f} Å")
        return overlap_center
    
    return None


def split_spectrum_by_setup(wave, flux, error, configs_gal):
    """
    Split a spectrum into separate arms based on original setups.
    
    If wavelength ranges overlap, splits at the center of the overlap.
    If no overlap, splits at original wavelength boundaries.
    
    Parameters:
    -----------
    wave : array
        Wavelength array
    flux : array
        Flux array
    error : array
        Error array
    configs_gal : dict
        Configuration containing orig_wlranges
        
    Returns:
    --------
    arm_spectra : list of dict
        List containing spectrum data for each arm
    """
    
    if configs_gal is None or 'orig_wlranges' not in configs_gal:
        # Return as single spectrum if no split info
        return [{
            'setup': 'full',
            'wave': wave,
            'flux': flux,
            'error': error
        }]
    
    wlranges = configs_gal['orig_wlranges']
    
    # Check if we should look for overlap
    if len(wlranges) < 2:
        # Single range, no split needed
        return [{
            'setup': configs_gal['orig_setups'][0] if 'orig_setups' in configs_gal else 'full',
            'wave': wave,
            'flux': flux,
            'error': error
        }]
    
    # Check for overlap
    overlap_center = find_overlap_center(wlranges)
    
    arm_spectra = []
    
    if overlap_center is not None:
        # USE OVERLAP CENTER AS SPLIT POINT
        # For 2-arm case: arm1 gets everything below center, arm2 gets everything above
        
        for ns, wlrange in enumerate(wlranges):
            setup_name = configs_gal['orig_setups'][ns] if 'orig_setups' in configs_gal else f'arm_{ns}'
            
            if ns == 0:
                # First arm: from original min to overlap center
                wmin = wlrange[0]
                wmax = overlap_center
                mask = (wave >= wmin) & (wave < wmax)
            else:
                # Second arm: from overlap center to original max
                wmin = overlap_center
                wmax = wlrange[1]
                mask = (wave >= wmin) & (wave <= wmax)
            
            if np.sum(mask) == 0:
                print(f"    Warning: No data in wavelength range [{wmin:.1f}, {wmax:.1f}] for setup {setup_name}")
                continue
            
            print(f"    Setup {setup_name}: [{wmin:.1f}, {wmax:.1f}] Å ({np.sum(mask)} pixels)")
            
            arm_spectra.append({
                'setup': setup_name,
                'wave': wave[mask],
                'flux': flux[mask],
                'error': error[mask],
                'mask': mask,
                'split_mode': 'overlap_center'
            })
    
    else:
        # NO OVERLAP - USE ORIGINAL WAVELENGTH BOUNDARIES (like RVS)
        print("    No overlap detected, using original wavelength boundaries")
        
        for ns, wlrange in enumerate(wlranges):
            wmin, wmax = wlrange
            setup_name = configs_gal['orig_setups'][ns] if 'orig_setups' in configs_gal else f'arm_{ns}'
            
            # Select wavelength range
            mask = (wave >= wmin) & (wave <= wmax)
            
            if np.sum(mask) == 0:
                print(f"    Warning: No data in wavelength range {wlrange} for setup {setup_name}")
                continue
            
            print(f"    Setup {setup_name}: [{wmin:.1f}, {wmax:.1f}] Å ({np.sum(mask)} pixels)")
            
            arm_spectra.append({
                'setup': setup_name,
                'wave': wave[mask],
                'flux': flux[mask],
                'error': error[mask],
                'mask': mask,
                'split_mode': 'original_boundaries'
            })
    
    return arm_spectra


def write_ferre_input(bin_id, arm_spectra, work_dir, configs_gal=None):
    """
    Write FERRE input files for a single bin.
    Creates .frd, .err, .wav, and .vrd files needed by FERRE.
    
    Parameters:
    -----------
    bin_id : int
        Bin identifier
    arm_spectra : list
        List of dictionaries containing spectrum data per arm
    work_dir : str
        Working directory for FERRE files
    configs_gal : dict
        Configuration dictionary
        
    Returns:
    --------
    success : bool
        True if files written successfully
    pixel : str
        Base name for FERRE files (e.g., 'bin0000')
    """
    
    import PyAPS.aps_constants as aps_constants
    large_error = aps_constants.large_error
    
    pixel = f'bin{bin_id:04d}'
    
    try:
        # Prepare data arrays
        wave_arrays = []
        flux_arrays = []
        error_arrays = []
        
        for arm_data in arm_spectra:
            setup = arm_data['setup']
            wave = arm_data['wave']
            flux = arm_data['flux']
            error = arm_data['error']
            
            # Create bad pixel mask
            badmask = (error <= 0) | np.isnan(flux) | np.isnan(error) | np.isinf(flux) | np.isinf(error)
            
            # Clean up data
            flux_clean = flux.copy()
            error_clean = error.copy()
            error_clean[badmask] = large_error
            
            if np.any(~badmask):
                good_median = np.nanmedian(flux[~badmask])
                flux_clean[badmask] = good_median
            
            wave_arrays.append(wave)
            flux_arrays.append(flux_clean)
            error_arrays.append(error_clean)
        
        # Concatenate all arms
        wave_full = np.concatenate(wave_arrays)
        flux_full = np.concatenate(flux_arrays)
        error_full = np.concatenate(error_arrays)
        
        # Write per-arm wavelength files (needed for later splitting)
        for i, arm_data in enumerate(arm_spectra):
            setup = arm_data['setup']
            wav_file = os.path.join(work_dir, f'{pixel}-{setup}.wav')
            np.savetxt(wav_file, wave_arrays[i][np.newaxis, :], fmt='%14.6e')
        
        # Write main FERRE input files
        vrd_file = os.path.join(work_dir, f'{pixel}.vrd')
        frd_file = os.path.join(work_dir, f'{pixel}.frd')
        err_file = os.path.join(work_dir, f'{pixel}.err')
        wav_file = os.path.join(work_dir, f'{pixel}.wav')
        
        # Write .vrd file (parameter file - initially zeros)
        with open(vrd_file, 'w') as f:
            # Format: ID, 8 parameters (all zeros for initial guess)
            f.write(f"{bin_id:30d} {0.0:6.2f} {0.0:10.2f} {0.0:6.2f} {0.0:6.2f} "
                   f"{0.0:12.9f} {0.0:12.9f} {0.0:12.9f} {0.0:12.9f}\n")
        
        # Write .frd file (flux data)
        flux_full.tofile(frd_file, sep=" ", format="%0.6e")
        with open(frd_file, 'a') as f:
            f.write("\n")
        
        # Write .err file (error data)
        error_full.tofile(err_file, sep=" ", format="%0.6e")
        with open(err_file, 'a') as f:
            f.write("\n")
        
        # Write .wav file (wavelength data)
        wave_full.tofile(wav_file, sep=" ", format="%14.6e")
        with open(wav_file, 'a') as f:
            f.write("\n")
        
        print(f"    BIN {bin_id}: Written FERRE input files ({len(wave_full)} pixels)")
        
        return True, pixel
        
    except Exception as e:
        print(f"    Error writing FERRE input files for bin {bin_id}: {e}")
        return False, None


def writenml(nml, nmlfile='input.nml', path=None):
    """
    Write FERRE control hash to an input.nml file.
    Adapted from original aps_ferre.py
    """
    if path is None:
        path = './'
    f = open(os.path.join(path, nmlfile), 'w')
    f.write('&LISTA\n')
    for item in nml.keys():
        f.write(str(item))
        f.write("=")
        f.write(str(nml[item]))
        f.write("\n")
    f.write(" /\n")
    f.close()
    return None


def mknml_ifu(synthfiles, path, pixel, grid_id, order, nthreads=1):
    """
    Create FERRE control hash for IFU binned spectra.
    Adapted from original aps_ferre.py mknml() function.
    
    Parameters:
    -----------
    synthfiles : list
        List of template file paths for each arm/setup
    path : str
        Working directory
    pixel : str
        Base name for files (e.g., 'bin0000')
    grid_id : str
        Grid identifier (e.g., '1', '2', etc.)
    order : int
        Interpolation order
    nthreads : int
        Number of threads
        
    Returns:
    --------
    nml : dict
        FERRE control parameters
    """
    from PyAPS.aps_ferre import head_synth
    
    root = os.path.join(path, pixel)
    header = head_synth(synthfiles[0])
    nml = {}
    
    ndim = int(header['N_OF_DIM'])
    nml['NDIM'] = ndim
    nml['NOV'] = ndim
    nml['INDV'] = ' '.join(map(str, np.arange(ndim) + 1))
    
    # Add all synthfiles - these should be full paths to template files, not joined with path
    for i in range(len(synthfiles)):
        nml['SYNTHFILE(' + str(i + 1) + ')'] = "'" + synthfiles[i] + "'"
    
    # Input/output files - use pixel name for all files
    nml['PFILE'] = "'" + root + ".vrd" + "'"
    nml['FFILE'] = "'" + root + ".frd" + "'"
    nml['ERFILE'] = "'" + root + ".err" + "'"
    nml['OPFILE'] = "'" + root + ".opf" + str(grid_id) + "'"
    nml['OFFILE'] = "'" + root + ".mdl" + str(grid_id) + "'"
    nml['SFFILE'] = "'" + root + ".nrd" + str(grid_id) + "'"
    nml['WFILE'] = "'" + root + ".wav" + "'"
    
    # FERRE parameters
    nml['ERRBAR'] = 1
    nml['COVPRINT'] = 1
    nml['WINTER'] = 2
    nml['INTER'] = order
    nml['ALGOR'] = 3
    nml['NTHREADS'] = 1
    nml['F_FORMAT'] = 1
    nml['F_ACCESS'] = 0
    nml['CONT'] = 3
    nml['NCONT'] = 50
    
    return nml


def run_ferre_on_bin(bin_id, pixel, work_dir, ferre_exe, ferre_templates, 
                     grid_ids, grid_prefix, configs_gal, nthreads=1, lsf=None):
    """
    Run FERRE on a single bin using the proper workflow from aps_ferre.py
    
    Parameters:
    -----------
    bin_id : int
        Bin identifier
    pixel : str
        Base name for files (e.g., 'bin0000')
    work_dir : str
        Working directory
    ferre_exe : str
        Path to FERRE executable
    ferre_templates : str
        Path to FERRE template directory
    grid_ids : list
        List of grid IDs to use
    grid_prefix : str
        Grid prefix ('n', 'm', or 'p')
    configs_gal : dict
        Configuration dictionary containing:
        - orig_setups: Original setup names
        - targs_mode: Observing mode
    nthreads : int
        Number of threads for FERRE
        
    Returns:
    --------
    results : dict
        Dictionary containing FERRE results
    """
    
    # Validate inputs
    if not isinstance(configs_gal, dict):
        print(f"    ERROR: configs_gal must be a dict, got {type(configs_gal)}")
        print(f"    bin_id={bin_id}, pixel={pixel}")
        return None
    
    # Import required functions from original aps_ferre
    from PyAPS.aps_ferre import ferre_exe_worker, opfmerge
    
    # Extract info from configs_gal
    orig_setups = configs_gal.get('orig_setups', [''])
    targs_mode = configs_gal.get('targs_mode', 'IFU')
    
    # Set maxorder based on grid prefix (from original code)
    if grid_prefix == 'n':
        maxorder = {'1':3, '2':3, '3':3, '4':2, '5':1, '6':3, '7':3, '8':3, '9':3}
    elif grid_prefix in ['m','p']:
        maxorder = {'1':3, '2':3, '3':3, '4':3, '5':2, '6':3, '7':3, '8':3, '9':3}
    else:
        print(f"    Unknown grid_prefix: {grid_prefix}")
        return None
    
    # Verify all grid_ids have maxorder defined
    for grid_id in grid_ids:
        if grid_id not in maxorder:
            print(f"    Grid {grid_id} not found in maxorder dictionary")
            return None
    
    # Prepare FERRE control files for each grid
    cmdstr_list = []
    
    for grid_id in grid_ids:
        # Build synthfile list for this grid
        # For IFU binned spectra, we concatenate all arms into single files,
        # so we need one template file per setup/arm
        synthfiles = []
        for setup in orig_setups:
            if setup == '':
                # No setup specified - use base grid
                gridfile = os.path.join(ferre_templates, 
                                       f'{grid_prefix}_rweave{grid_id}.hdr')
            else:
                # Setup specified - use setup-specific grid
                # Format: n_rweave1-IFU_BLUEL11.hdr
                gridfile = os.path.join(ferre_templates,
                                       f'{grid_prefix}_rweave{grid_id}-{targs_mode.replace(" ","")}_{setup}.hdr')
            
            # Check if this specific template exists
            if not os.path.exists(gridfile):
                # Try without the mode prefix (e.g., n_rweave1-BLUEL11.hdr)
                gridfile_alt = os.path.join(ferre_templates,
                                           f'{grid_prefix}_rweave{grid_id}-{setup}.hdr')
                if os.path.exists(gridfile_alt):
                    gridfile = gridfile_alt
                else:
                    # Fall back to base grid without setup
                    gridfile_base = os.path.join(ferre_templates, 
                                                f'{grid_prefix}_rweave{grid_id}.hdr')
                    if os.path.exists(gridfile_base):
                        print(f"    Warning: Setup-specific template not found, using base grid: {os.path.basename(gridfile_base)}")
                        gridfile = gridfile_base
            
            synthfiles.append(gridfile)
        
        # Check if template files exist
        missing_templates = [f for f in synthfiles if not os.path.exists(f)]
        if missing_templates:
            print(f"    Grid {grid_id}: Missing template files: {missing_templates[:2]}")
            continue
        
        # Create FERRE control file using mknml
        nml = mknml_ifu(synthfiles, work_dir, pixel, grid_id, 
                       maxorder[grid_id], nthreads=1)
        
        # Write control file
        nmlfile = f'input.nml_{grid_id}'
        writenml(nml, nmlfile=nmlfile, path=work_dir)
        
        # Add to command list
        cmdstr_list.append([os.path.join(work_dir, nmlfile), ferre_exe, False])
    
    if len(cmdstr_list) == 0:
        print(f"    Bin {bin_id}: No valid grids to process")
        return None
    
    print(f"    Running FERRE on Bin {bin_id} with {len(cmdstr_list)} grids")
    
    # Execute FERRE for each grid
    successful_grids = []
    failed_grids = []
    
    for i, cmdstr in enumerate(cmdstr_list):
        grid_id = grid_ids[i] if i < len(grid_ids) else str(i)
        
        try:
            success, return_code, error_msg = ferre_exe_worker(cmdstr)
            
            # Check if output files exist even if FERRE returned exit code 1
            # (IEEE underflow warnings cause exit code 1 but results are valid)
            opf_file = os.path.join(work_dir, f'{pixel}.opf{grid_id}')
            
            if success or os.path.exists(opf_file):
                # Either clean success or output file exists despite warnings
                if not success and os.path.exists(opf_file):
                    print(f"    Grid {grid_id}: FERRE completed with warnings (IEEE underflow/denormal), but output valid")
                successful_grids.append(grid_id)
            else:
                failed_grids.append(grid_id)
                print(f"    Grid {grid_id} failed: {error_msg}")
                
        except Exception as e:
            failed_grids.append(grid_id)
            print(f"    Grid {grid_id} exception: {e}")
    
    if len(successful_grids) == 0:
        print(f"    Bin {bin_id}: All FERRE runs failed")
        return None
    
    # Merge results from multiple grids using opfmerge
    if len(successful_grids) > 1:
        try:
            merge_result = opfmerge(
                pixel=pixel,
                grid_ids=successful_grids,
                grid_prefix=grid_prefix,
                path=work_dir,
                min_grids_required=min(3, len(successful_grids)),  # Need at least 3 or all available
                wait_time=2,
                max_wait=30,
                cooldown=1
            )
            
            if not merge_result['success']:
                print(f"    Bin {bin_id}: OPFMERGE failed")
                return None
                
        except Exception as e:
            print(f"    Bin {bin_id}: OPFMERGE error: {e}")
            return None
    else:
        # Only one grid - rename output files
        grid_id = successful_grids[0]
        for ext in ['opf', 'mdl', 'nrd']:
            src = os.path.join(work_dir, f'{pixel}.{ext}{grid_id}')
            dst = os.path.join(work_dir, f'{pixel}.{ext}')
            if os.path.exists(src):
                shutil.copy(src, dst)
    
    # Parse FERRE output
    try:
        result = parse_ferre_output_ifu(work_dir, pixel, bin_id, grid_prefix)
        return result
    except Exception as e:
        print(f"    Bin {bin_id}: Error parsing results: {e}")
        return None


def parse_ferre_output_ifu(work_dir, pixel, bin_id, grid_prefix):
    """
    Parse FERRE output files for IFU bins.
    Adapted from ferre_outdict() in aps_ferre.py
    
    Parameters:
    -----------
    work_dir : str
        Working directory containing FERRE outputs
    pixel : str
        Base filename (e.g., 'bin0000')
    bin_id : int
        Bin identifier
    grid_prefix : str
        Grid prefix ('n', 'm', or 'p')
        
    Returns:
    --------
    result : dict
        Parsed FERRE parameters
    """
    
    root = os.path.join(work_dir, pixel)
    
    # Read FERRE output files
    try:
        opf_file = f'{root}.opf'
        
        if not os.path.exists(opf_file):
            print(f"    FERRE parameter file not found: {opf_file}")
            return None
        
        # Read parameter file
        with open(opf_file, 'r') as f:
            line = f.readline()
        
        cells = line.split()
        m_cells = len(cells)
        
        # Calculate dimensionality
        if m_cells < 7:
            print(f"    Invalid FERRE output format (too few columns: {m_cells})")
            return None
        
        ndim = int(np.sqrt(m_cells - 3) - 1)
        
        # Initialize result dictionary
        result = {
            'BIN_ID': bin_id,
            'TEFF_FERRE': np.nan,
            'LOGG_FERRE': np.nan,
            'FEH_FERRE': np.nan,
            'ALPHA_FERRE': np.nan,
            'MICRO_FERRE': np.nan,
            'TEFF_ERR_FERRE': np.nan,
            'LOGG_ERR_FERRE': np.nan,
            'FEH_ERR_FERRE': np.nan,
            'ALPHA_ERR_FERRE': np.nan,
            'MICRO_ERR_FERRE': np.nan,
            'SNR_FERRE': np.nan,
            'CHISQ_FERRE': np.nan,
            'FLAG_FERRE': 0,
            'COVAR_FERRE': None
        }
        
        # Parse based on grid dimensionality (from original code)
        if m_cells == 19:
            # 3D Kurucz grids (Teff, logg, [Fe/H])
            result['FEH_FERRE'] = float(cells[1])
            result['TEFF_FERRE'] = float(cells[2])
            result['LOGG_FERRE'] = float(cells[3])
            result['FEH_ERR_FERRE'] = float(cells[4])
            result['TEFF_ERR_FERRE'] = float(cells[5])
            result['LOGG_ERR_FERRE'] = float(cells[6])
            result['SNR_FERRE'] = float(cells[8])
            result['CHISQ_FERRE'] = 10.**float(cells[9])
            
            if grid_prefix == 'n':
                cov = np.reshape(np.array(cells[10:], dtype=float), (3, 3))
                result['COVAR_FERRE'] = cov
            
        elif m_cells == 39:
            # 5D Kurucz grids (Teff, logg, [Fe/H], [alpha/Fe], micro)
            result['FEH_FERRE'] = float(cells[1])
            result['ALPHA_FERRE'] = float(cells[2])
            result['MICRO_FERRE'] = float(cells[3])
            result['TEFF_FERRE'] = float(cells[4])
            result['LOGG_FERRE'] = float(cells[5])
            result['FEH_ERR_FERRE'] = float(cells[6])
            result['ALPHA_ERR_FERRE'] = float(cells[7])
            result['MICRO_ERR_FERRE'] = float(cells[8])
            result['TEFF_ERR_FERRE'] = float(cells[9])
            result['LOGG_ERR_FERRE'] = float(cells[10])
            result['SNR_FERRE'] = float(cells[12])
            result['CHISQ_FERRE'] = 10.**float(cells[13])
            
            if grid_prefix in ['m', 'p']:
                cov = np.reshape(np.array(cells[14:], dtype=float), (5, 5))
                result['COVAR_FERRE'] = cov
                
        elif m_cells == 12:
            # 2D white dwarfs (Teff, logg)
            result['FEH_FERRE'] = -10.0  # WD marker
            result['TEFF_FERRE'] = float(cells[1])
            result['LOGG_FERRE'] = float(cells[2])
            result['TEFF_ERR_FERRE'] = float(cells[3])
            result['LOGG_ERR_FERRE'] = float(cells[4])
            result['SNR_FERRE'] = float(cells[6])
            result['CHISQ_FERRE'] = 10.**float(cells[7])
            
            if grid_prefix == 'n':
                cov = np.zeros((3, 3))
                cov[1:, 1:] = np.reshape(np.array(cells[8:], dtype=float), (2, 2))
                result['COVAR_FERRE'] = cov
            elif grid_prefix in ['m', 'p']:
                cov = np.zeros((5, 5))
                cov[3:, 3:] = np.reshape(np.array(cells[8:], dtype=float), (2, 2))
                result['COVAR_FERRE'] = cov
        
        # Set quality flag (from original code)
        if result['CHISQ_FERRE'] < 1.0 and result['SNR_FERRE'] > 5.0:
            result['FLAG_FERRE'] = 1
        else:
            result['FLAG_FERRE'] = 0
        
        return result
        
    except Exception as e:
        print(f"    Error parsing FERRE output: {e}")
        return None

def proc_ferre_ifu(headname, outpath, ferre_exe, ferre_templates, ferre_grid_ids,
                   ferre_grid_prefix='n', configs_gal=None, nthreads=1, overwrite=False,
                   outspec=True, lsf=None, bin_to_bucket=None, lsf_by_bucket=None):
    """
    Main function to process IFU binned spectra through FERRE.

    `bin_to_bucket`/`lsf_by_bucket` (both optional, default `None`):
    opt-in per-bin resolution -- same contract as `aps_ifu_rvs.
    process_single_bin`'s own matching parameters (bin position in this
    function's own `for i_bin in range(n_bins)` loop, not `bin_id` itself,
    is what indexes `bin_to_bucket` -- matches every other runModule_*/
    process_* function's own "loop position = bucket lookup key"
    convention in this codebase).
    
    Parameters:
    -----------
    headname : str
        Header name for output files
    outpath : str
        Output directory path
    ferre_exe : str
        Path to FERRE executable
    ferre_templates : str
        Path to FERRE template directory
    ferre_grid_ids : list
        List of grid IDs to use
    ferre_grid_prefix : str
        Grid prefix ('n', 'm', or 'p')
    configs_gal : dict
        Configuration dictionary
    nthreads : int
        Number of threads
    overwrite : bool
        Whether to overwrite existing outputs
    outspec : bool
        Whether to generate spectral output file
        
    Returns:
    --------
    success : bool
        True if processing completed successfully
    """
    
    # Construct input file path
    binspec_file = os.path.join(outpath, f'{headname}_BINSpectra_linear.fits')
    
    if not os.path.exists(binspec_file):
        print(f"ERROR: Input file not found: {binspec_file}")
        return False
    
    # Define output paths
    ferre_param_out = os.path.join(outpath, f'ferre_{headname}.fits')
    ferre_spec_out = os.path.join(outpath, f'ferre_spec_{headname}.fits') if outspec else None
    
    # CRITICAL FIX: Make workpath unique per headname to avoid collisions in parallel processing
    safe_headname = str(headname).replace(' ', '').replace('/', '_').replace('\\', '_')
    ferre_workpath = os.path.join(outpath, f'fr_wd_{safe_headname}')
    
    if not overwrite and os.path.exists(ferre_param_out):
        print(f"Output exists and overwrite=False: {ferre_param_out}")
        return True
    
    # Check FERRE executable
    if not os.path.exists(ferre_exe):
        print(f"ERROR: FERRE executable not found: {ferre_exe}")
        return False
    
    if not os.path.exists(ferre_templates):
        print(f"ERROR: FERRE templates directory not found: {ferre_templates}")
        return False
    
    # Create working directory
    if os.path.exists(ferre_workpath):
        if overwrite:
            print(f"WORKPATH: {ferre_workpath} exists. Removing and recreating (overwrite=True)...")
            shutil.rmtree(ferre_workpath)
            os.makedirs(ferre_workpath)
    else:
        os.makedirs(ferre_workpath)
        print(f"WORKPATH: {ferre_workpath} Created!")
    
    print(f"Processing binned spectra: {binspec_file}")
    print(f"FERRE working directory: {ferre_workpath}")
    
    # Read binned spectra
    try:
        bin_data = read_binned_spectra_for_ferre(binspec_file, configs_gal)
    except Exception as e:
        print(f"ERROR reading binned spectra: {e}")
        return False
    
    n_bins = bin_data['n_bins']
    results = []
    
    # Track merge metadata for all bins
    all_merge_metadata = []
    
    # Process each bin
    for i_bin in range(n_bins):
        bin_id = bin_data['bin_ids'][i_bin]
        wave = bin_data['wavelengths'][i_bin]
        flux = bin_data['spectra'][i_bin]
        error = bin_data['errors'][i_bin]
        metadata = bin_data['metadata'][i_bin]
        
        print(f"\nProcessing BIN_ID {bin_id} ({i_bin+1}/{n_bins})")
        
        try:
            # Split spectrum by setup/arm if needed
            arm_spectra = split_spectrum_by_setup(wave, flux, error, configs_gal)
            
            # Write FERRE input files
            success, pixel = write_ferre_input(bin_id, arm_spectra, ferre_workpath, configs_gal)
            
            if not success:
                print(f"    Failed to write FERRE input files for bin {bin_id}")
                continue
            
            # Run FERRE
            lsf_this_bin = (lsf_by_bucket[bin_to_bucket[i_bin]]
                             if bin_to_bucket is not None and lsf_by_bucket is not None
                             else lsf)
            ferre_result = run_ferre_on_bin(
                bin_id, pixel, ferre_workpath, ferre_exe, ferre_templates,
                ferre_grid_ids, ferre_grid_prefix, configs_gal, nthreads, lsf_this_bin
            )
            
            if ferre_result is not None:
                # Add metadata to results
                for key, value in metadata.items():
                    if key not in ferre_result:
                        ferre_result[key] = value
                
                # Read FERRE model and normalized flux for spec output
                if outspec:
                    try:
                        root = os.path.join(ferre_workpath, pixel)
                        
                        # Read model spectrum
                        mdl_file = f'{root}.mdl'
                        if os.path.exists(mdl_file):
                            model = np.genfromtxt(mdl_file, dtype=float)
                            if model.ndim == 1:
                                model = model[np.newaxis, ...]
                            ferre_result['MODEL_FERRE'] = model[0]
                        
                        # Read normalized flux
                        nrd_file = f'{root}.nrd'
                        if os.path.exists(nrd_file):
                            norm_flux = np.genfromtxt(nrd_file, dtype=float)
                            if norm_flux.ndim == 1:
                                norm_flux = norm_flux[np.newaxis, ...]
                            ferre_result['FLUX_NORM_FERRE'] = norm_flux[0]
                        
                        # Read error
                        err_file = f'{root}.err'
                        if os.path.exists(err_file):
                            err = np.genfromtxt(err_file, dtype=float)
                            if err.ndim == 1:
                                err = err[np.newaxis, ...]
                            ferre_result['ERROR_NORM_FERRE'] = err[0]
                        
                        # Read wavelength
                        wav_file = f'{root}.wav'
                        if os.path.exists(wav_file):
                            wav = np.genfromtxt(wav_file, dtype=float)
                            if wav.ndim == 1:
                                wav = wav[np.newaxis, ...]
                            ferre_result['LAMBDA_FERRE'] = wav[0]
                            
                    except Exception as e:
                        print(f"    Warning: Could not read spectral data for bin {bin_id}: {e}")
                
                results.append(ferre_result)
            
        except Exception as e:
            print(f"    Error processing bin {bin_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("ERROR: No valid results obtained from FERRE processing")
        return False
    
    print(f"\nSuccessfully processed {len(results)}/{n_bins} bins")
    
    # Create output table for parameters
    outtab = Table(results)
    
    # Set units
    unit_map = {
        'TEFF_FERRE': 'K',
        'TEFF_ERR_FERRE': 'K',
        'LOGG_FERRE': 'dex',
        'LOGG_ERR_FERRE': 'dex',
        'FEH_FERRE': 'dex',
        'FEH_ERR_FERRE': 'dex',
        'ALPHA_FERRE': 'dex',
        'ALPHA_ERR_FERRE': 'dex',
        'MICRO_FERRE': 'dex',
        'MICRO_ERR_FERRE': 'dex'
    }
    
    for col, unit in unit_map.items():
        if col in outtab.colnames:
            outtab[col].unit = unit
    
    # Add metadata to parameter table
    import PyAPS
    outtab.meta['EXTNAME'] = 'FERRE_IFU_TABLE'
    outtab.meta['APSVERS'] = (PyAPS.__version__, 'APS version')
    outtab.meta['BINSPEC'] = (os.path.basename(binspec_file), 'Input binned spectra file')
    outtab.meta['NBINS'] = (n_bins, 'Total number of bins in input')
    outtab.meta['NPROC'] = (len(results), 'Number of successfully processed bins')
    outtab.meta['FERRE_EXE'] = (ferre_exe, 'FERRE executable path')
    outtab.meta['FERRE_TPL'] = (ferre_templates, 'FERRE templates directory')
    outtab.meta['WORKPATH'] = (ferre_workpath, 'FERRE working directory')
    
    # Sort by BIN_ID
    if 'BIN_ID' in outtab.colnames:
        outtab.sort('BIN_ID')
    
    # Create parameter-only table
    param_cols = ['BIN_ID', 'TEFF_FERRE', 'TEFF_ERR_FERRE', 'LOGG_FERRE', 'LOGG_ERR_FERRE',
                  'FEH_FERRE', 'FEH_ERR_FERRE', 'ALPHA_FERRE', 'ALPHA_ERR_FERRE',
                  'MICRO_FERRE', 'MICRO_ERR_FERRE', 'SNR_FERRE', 'CHISQ_FERRE', 'FLAG_FERRE']
    
    # Add optional columns if they exist
    optional_cols = ['APS_ID', 'TARGID', 'CNAME', 'X', 'Y', 'COVAR_FERRE']
    for col in optional_cols:
        if col in outtab.colnames and col not in param_cols:
            param_cols.insert(1 if col in ['APS_ID', 'TARGID', 'CNAME'] else -3, col)
    
    param_tab = outtab[[col for col in param_cols if col in outtab.colnames]]
    param_tab.meta = outtab.meta.copy()
    
    # Write parameter file
    try:
        hx = fits.HDUList()
        hx.append(fits.PrimaryHDU())
        hx.append(fits.convenience.table_to_hdu(param_tab))
        hx.writeto(ferre_param_out, overwrite=True)
        print(f"✓ FERRE parameters written to: {ferre_param_out}")
    except Exception as e:
        print(f"ERROR writing parameter file: {e}")
        return False
    
    # Write spectral output file if requested
    if outspec and ferre_spec_out is not None:
        try:
            # Create spectral table - remove parameter columns but keep spectral data
            spec_cols = ['BIN_ID']
            
            # Add metadata columns
            for col in ['APS_ID', 'TARGID', 'CNAME', 'X', 'Y']:
                if col in outtab.colnames:
                    spec_cols.append(col)
            
            # Add spectral columns
            for col in ['LAMBDA_FERRE', 'FLUX_NORM_FERRE', 'ERROR_NORM_FERRE', 'MODEL_FERRE']:
                if col in outtab.colnames:
                    spec_cols.append(col)
            
            spec_tab = outtab[[col for col in spec_cols if col in outtab.colnames]]
            
            # Set up metadata
            spec_tab.meta['EXTNAME'] = 'FERRE_IFU_SPEC'
            spec_tab.meta['APSVERS'] = (PyAPS.__version__, 'APS version')
            spec_tab.meta['BINSPEC'] = (os.path.basename(binspec_file), 'Input binned spectra file')
            spec_tab.meta['NBINS'] = (n_bins, 'Total number of bins in input')
            spec_tab.meta['NPROC'] = (len(results), 'Number of successfully processed bins')
            spec_tab.meta['FERRE_EXE'] = (ferre_exe, 'FERRE executable path')
            spec_tab.meta['FERRE_TPL'] = (ferre_templates, 'FERRE templates directory')
            spec_tab.meta['VACUUM'] = (False, 'Wavelengths are in air')
            spec_tab.meta['SAMPLING'] = (0, 'Sampling mode (0: linear, 1: logarithmic)')
            spec_tab.meta['WORKPATH'] = (ferre_workpath, 'FERRE working directory')
            
            # Set units for spectral columns
            if 'LAMBDA_FERRE' in spec_tab.colnames:
                spec_tab['LAMBDA_FERRE'].unit = 'Angstrom'
            # Flux is normalized, so dimensionless
            if 'FLUX_NORM_FERRE' in spec_tab.colnames:
                spec_tab['FLUX_NORM_FERRE'].unit = ''
            if 'ERROR_NORM_FERRE' in spec_tab.colnames:
                spec_tab['ERROR_NORM_FERRE'].unit = ''
            if 'MODEL_FERRE' in spec_tab.colnames:
                spec_tab['MODEL_FERRE'].unit = ''
            
            # Write spectral file
            hx = fits.HDUList()
            hx.append(fits.PrimaryHDU())
            hx.append(fits.convenience.table_to_hdu(spec_tab))
            hx.writeto(ferre_spec_out, overwrite=True)
            print(f"✓ FERRE spectra written to: {ferre_spec_out}")
            
        except Exception as e:
            print(f"ERROR writing spectral file: {e}")
            import traceback
            traceback.print_exc()
            # Don't return False here - parameter file is already written successfully
    
    return True


def run_ferre_for_ifu_gal(headname, outpath, ferre_exe, ferre_templates,
                          ferre_grid_ids, ferre_grid_prefix='n',
                          configs_gal=None, nthreads=1, overwrite=True, outspec=True, lsf=None,
                          bin_to_bucket=None, lsf_by_bucket=None):
    """
    Wrapper function to run FERRE for IFU Galactic sources.

    This function is called from ifu_Gal in aps_ifu.py.

    Parameters:
    -----------
    outspec : bool
        Whether to generate spectral output file (default: True)
    bin_to_bucket, lsf_by_bucket : optional
        Opt-in per-bin resolution, see `proc_ferre_ifu`'s own docstring.
    """

    success = proc_ferre_ifu(
        headname=headname,
        outpath=outpath,
        ferre_exe=ferre_exe,
        ferre_templates=ferre_templates,
        ferre_grid_ids=ferre_grid_ids,
        ferre_grid_prefix=ferre_grid_prefix,
        configs_gal=configs_gal,
        nthreads=nthreads,
        overwrite=overwrite,
        outspec=outspec,
        lsf=lsf,
        bin_to_bucket=bin_to_bucket,
        lsf_by_bucket=lsf_by_bucket,
    )
    
    if success:
        print("FERRE processing completed successfully")
    else:
        print("FERRE processing failed")
    
    return success