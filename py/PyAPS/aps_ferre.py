from __future__ import absolute_import, division, print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pdb
import sys
import glob
import re
import subprocess
import time
from astropy.io import fits
from astropy.table import Table, Column
import astropy.wcs as pywcs

import numpy as np
from collections import OrderedDict
import argparse
import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist, overlap_finder, add_extra_columns, index_in_class, check_and_fix_overlap
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import aps_constants
from PyAPS.apsPlot.ferre import make_fr_plot
import shutil
from multiprocessing.pool import ThreadPool
import subprocess
from pathlib import Path
import warnings
import traceback
import time
from datetime import datetime
APSVERS = PyAPS.__version__

#################################################
"""
aps_ferre
Python wrapper to run FERRE code on WEAVE DATA and generate output tables

versions:
 1.0 Original version by C. Allende (IAC, July 2019)
 1.2 By A. Molaeinezhad and C. Allende (IAC, August 2019)- Customized for WEAVE
 2.0 By A. Molaeinezhad and C. Allende (IAC, November 2019)
 3.0 By A. Molaeinezhad and C. Allende (IAC, October 2020)
 3.1 By A. Molaeinezhad (CASU, May 2021)
 3.2 By A. Molaeinezhad (CASU, Nov 2021)
 3.3 By A. Molaeinezhad (CASU, Jul 2022)
 3.4 By A. Molaeinezhad (CASU, Oct 2023)
 3.5 By A. Molaeinezhad (CASU, Sept 2025)
 3.6 By A. Molaeinezhad (CASU, Oct 2025) - Fixed unique working directory issue
 3.7 By Thomas Hajnik (CASU, Dec 2025) - Fixing a bug in opfmerge function and adding robust error handling
 3.8 By A. Ardern-Arentsen (CASU, Dec 2025) - Main update: LSF implementation. Also: new option to mask Balmer lines, use higher NCONT for HR
 3.9 By A. Molaeinezhad (CASU. Feb 2026) - Use fmode_unit to generate templates names (like MOS, MIFU, LIFU) instead of mode
 4.0 By Thomas Hajnik (CASU, Apr 2026) - Added option to use linemasks for stellar parameter fitting
 4.1 By Thomas Hajnik (CASU, Jul 2026) - Full spectrum and model information is now retained in the ferre_spec* output files when using linemasks
 4.2 (CASU, Aug 2026) - make_fr_plot moved to PyAPS.apsPlot.ferre (Plotly), built on the same spectrum_overlay_figure shared with aps_rr/aps_rvs - part of the new cross-module PyAPS.apsPlot platform. Same output convention and layout (2 arms x 2 halves).
"""

#################################################

def find_template_file(synthfile: str) -> str:
    """
    Find template file with fallback to alternative resolution versions.

    If the specified file doesn't exist, tries replacing the resolution
    suffix (e.g., L21, L41) with L11 as a fallback.

    Args:
        synthfile: Path to the template file

    Returns:
        str: Path to the existing file (original or fallback)

    Raises:
        FileNotFoundError: If neither the original nor fallback file exists

    Examples:
        >>> find_template_file('/path/to/template-LIFU_BLUEL21.hdr')
        '/path/to/template-LIFU_BLUEL21.hdr'  # if exists

        >>> find_template_file('/path/to/template-LIFU_BLUEL21.hdr')
        '/path/to/template-LIFU_BLUEL11.hdr'  # if original doesn't exist
    """

    # Check if original file exists (follows symlinks by default)
    if os.path.exists(synthfile):
        return synthfile

    # Try fallback: replace L21 or L41 with L11
    fallback_file = re.sub(r'([LH])(21|41)', r'\g<1>11', synthfile)

    # Check if fallback is different from original and if it exists
    if fallback_file != synthfile and os.path.exists(fallback_file):
        return fallback_file

    # Neither file exists - raise original error
    raise FileNotFoundError(
        f"Template file not found: {synthfile}\n"
        f"Also tried fallback: {fallback_file}"
    )

#################################################
# make_fr_plot now lives in PyAPS.apsPlot.ferre, built on the shared
# spectrum_overlay_figure() (Plotly) used across the PyAPS.apsPlot platform.
# Imported at module top as `from PyAPS.apsPlot.ferre import make_fr_plot`.
#################################################################################################


#get names of extensions from a FITS file
#################################################################################################

def extnames(hdu):
    #hdu must have been open as follows hdu=fits.open(filename)
    x=hdu.info(output=False)
    names=[]
    for entry in x: names.append(entry[1])
    return(names)
#################################################################################################


#extract the header of a synthfile
#################################################################################################

def head_synth(synthfile):
    file=open(synthfile,'r')
    line=file.readline()
    header={}
    while (1):
        line=file.readline()
        part=line.split('=')
        if (len(part) < 2): break
        k=part[0].strip()
        v=part[1].strip()
        header[k]=v
    return header
#################################################################################################


#extract the wavelength array for a FERRE synth file
#################################################################################################

def lambda_synth(synthfile):
    header=head_synth(synthfile)
    tmp=header['WAVE'].split()
    npix=int(header['NPIX'])
    step=float(tmp[1])
    x0=float(tmp[0])
    x=np.arange(npix)*step+x0
    if header['LOGW']:
        if int(header['LOGW']) == 1: x=10.**x
        if int(header['LOGW']) == 2: x=np.exp(x)
    return x
#################################################################################################


#write ferre files
#################################################################################################

def write_ferre_input(root,ids,par,x,y,ey,path=None):

    if path is None: path="./"

    #open ferre input files
    vrd=open(os.path.join(path,root)+'.vrd','w')
    frd=open(os.path.join(path,root)+'.frd','w')
    err=open(os.path.join(path,root)+'.err','w')
    wav=open(os.path.join(path,root)+'.wav','w')

    nspec, freq = y.shape

    #loop to write data files
    i=0
    while (i < nspec):

        ppar=[ids[i]]
        for item in par[ids[i]]: ppar.append(item)
        vrd.write("%30s %6.2f %10.2f %6.2f %6.2f %12.9f %12.9f %12.9f %12.9f\n" % tuple(ppar) )

        xx=x[i,:]
        xx.tofile(wav,sep=" ",format="%14.6e")
        wav.write("\n")

        yy=y[i,:]
        yy.tofile(frd,sep=" ",format="%0.6e")
        frd.write("\n")

        eyy=ey[i,:]
        eyy.tofile(err,sep=" ",format="%0.6e")
        err.write("\n")

        i+=1
    #close files
    vrd.close()
    frd.close()
    err.close()
    wav.close()
#################################################################################################


def _prepare_ferre_error_arrays(yivar, yflux):
    """Separate original invalid pixels from errors used only for fitting."""
    yivar = np.asarray(yivar, dtype=float).copy()
    yflux = np.asarray(yflux)

    large_error = aps_constants.large_error
    mask_ivar = 1.0 / (large_error**2)
    invalid_ivar_limit = 1.0 / ((0.95 * large_error)**2)

    invalid_mask = (
        ~np.isfinite(yivar)
        | ~np.isfinite(yflux)
        | (yivar <= invalid_ivar_limit)
    )

    output_error = np.full(yivar.shape, np.nan, dtype=float)
    valid_mask = ~invalid_mask
    output_error[valid_mask] = np.sqrt(1.0 / yivar[valid_mask])

    ferre_ivar = yivar.copy()
    ferre_ivar[invalid_mask] = mask_ivar

    return ferre_ivar, output_error, invalid_mask


#################################################################################################


#write grid-specific LSF files
#################################################################################################

def write_ferre_lsf(root, ids, lsf_data, k, path=None):

    if path is None: path="./"

    lsf_filename = f'{root}.lsf{k}'
    lsf = open(os.path.join(path, lsf_filename), 'w')

    nspec = len(ids)
    for i in range(nspec):
        ll = lsf_data[i, :]
        ll.tofile(lsf, sep=" ", format="%0.6e")
        lsf.write("\n")

    lsf.close()
#################################################################################################


#create a SLURM script for a given pixel
#################################################################################################

def writeslurm_weave(pixel,nthreads=1, path=None,grid_ids=None, pydir=None, ferre=None, hpc_mode=None, grid_prefix='n'):
    if pydir is None:
        pydir=sys.executable
    if ferre is None:
        ferre='<FERRE_DIR>/bin/ferre.x'
    if hpc_mode is None:
        hpc_mode=0

    host='deimos'

    now=time.strftime("%c")
    sdir=pixel[:-2]
    if path is None: path='.'
    if grid_ids is None: grid_ids=['1']

    f=open(os.path.join(path,pixel+'.slurm'),'w')
    f.write("#!/bin/bash \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("#This script was written by aps_ferre.py on "+now+" \n")
    if host[:4] == 'cori':
        f.write("#SBATCH --qos=regular" + "\n")
        f.write("#SBATCH --constraint=haswell" + "\n")
        f.write("#SBATCH --time=180"+"\n") #minutes
        f.write("#SBATCH --cpus-per-task="+str(nthreads*2)+"\n")
    else:
        f.write("#SBATCH    -J "+str(pixel)+" \n")
        f.write("#SBATCH    -p batch"+" \n")
        f.write("#SBATCH    -o "+str(pixel)+"_%j.out"+" \n")
        f.write("#SBATCH    -e "+str(pixel)+"_%j.err"+" \n")
        f.write("#SBATCH    -n "+str(nthreads)+" \n")
        f.write("#SBATCH    -t 04:00:00"+" \n") #hh:mm:ss
        f.write("#SBATCH    -D "+os.path.abspath(path)+" \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("export OMP_NUM_THREADS="+str(nthreads)+"\n")
    for grid_i , grid  in enumerate(grid_ids):
        f.write("cp input.nml_"+str(grid)+" input.nml \n")
        f.write("time "+ferre+" \n")
    if len(grid_ids) > 1:
        this_file_path=os.path.dirname(os.path.realpath(__file__))
        if hpc_mode > 0:
            f.write("wait \n")
            f.write("python3 -c \"import sys; sys.path.insert(0, '"+pydir+"'); sys.path.append('"+this_file_path+"'); from aps_ferre import opfmerge; opfmerge(\'"+str(pixel)+"\' , \'"+str(grid_ids)+"\'   ,grid_prefix='"+str(grid_prefix)+"', path='"+str(path)+"')\"\n")
    f.close()
    os.chmod(os.path.join(path,pixel+'.slurm'),0o755)

    return None
#################################################################################################


#write out a FERRE control hash to an input.nml file
#################################################################################################
def writenml(nml,nmlfile='input.nml',path=None):
    if path is None: path='./'
    f=open(os.path.join(path,nmlfile),'w')
    f.write('&LISTA\n')
    for item in nml.keys():
        f.write(str(item))
        f.write("=")
        f.write(str(nml[item]))
        f.write("\n")
    f.write(" /\n")
    f.close()
    return None
#################################################################################################


#create a FERRE control hash (content for a ferre input.nml file)
#################################################################################################
def mknml(synthfiles, path, pixel, k, order, nthreads=1, setups=None):


    root = os.path.join(path,pixel)
    header=head_synth(synthfiles[0])
    nml={}

    # Set FERRE nml parameters
    ndim=int(header['N_OF_DIM'])
    nml['NDIM']=ndim
    nml['NOV']=ndim
    nml['INDV']=' '.join(map(str,np.arange(ndim)+1))
    for i in range(len(synthfiles)): nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(path,synthfiles[i])+"'"
    nml['PFILE'] = "'"+root+".vrd"+"'"
    nml['FFILE'] = "'"+root+".frd"+"'"
    nml['ERFILE'] = "'"+root+".err"+"'"
    nml['OPFILE'] = "'"+root+".opf"+str(k)+"'"
    nml['OFFILE'] = "'"+root+".mdl"+str(k)+"'"
    nml['SFFILE'] = "'"+root+".nrd"+str(k)+"'"
    nml['WFILE'] = "'"+root+".wav"+"'"
    nml['ERRBAR']=1
    nml['COVPRINT']=1
    nml['WINTER']=2
    nml['INTER']=order
    nml['ALGOR']=3
    nml['NTHREADS']=nthreads
    nml['F_FORMAT']=1
    nml['F_ACCESS']=0
    nml['CONT']=3

    # Check OB setup to provide best NCONT value for HR OBs (LR NCONT is always 50)
    if setups is None:
        warnings.warn("Warning: OB setup could not be determined (setups=None). FERRE NCONT will be set to 50!")
        ncont = 50

    elif any(setup.startswith(("GREENH", "BLUEH", "REDH")) for setup in setups):
        ncont = 600  # corresponds to 30 Angstrom --> Tests show significant improvement

    else:
        ncont = 50

    # Set NCONT
    nml["NCONT"] = ncont

    # Set LSF and provide filepath
    nml['LSFFILE'] = "'"+root+".lsf"+str(k)+"'"
    nml['LSF'] = 14  # LSF = 14 option in FERRE is for a wavelength-dependent LSF per spectrum

    return nml

#################################################################################################

def opfmerge(pixel, grid_ids, grid_prefix='n', wait_on_sorted=False, path=None,
             min_grids_required=5, wait_time=10, max_wait=300, cooldown=2):
    """
    Merge FERRE output files from multiple grids with robust error handling.

    Parameters:
    -----------
    pixel : str
        Base name for the output files
    grid_ids : list
        List of grid IDs to process
    grid_prefix : str
        Grid prefix ('n' for 3D, 'm' or 'p' for 5D)
    wait_on_sorted : bool
        Whether to wait for sorted files
    path : str
        Working directory path
    min_grids_required : int
        Minimum number of successful grids required to continue (default: 5)
    wait_time : int
        Seconds to wait between file checks (default: 10)
    max_wait : int
        Maximum seconds to wait for files (default: 300 = 5 minutes)
    cooldown : int
        Seconds to wait after files are found before processing (default: 2)

    Returns:
    --------
    dict: Dictionary containing:
        - 'success': bool
        - 'grids_used': list of grid IDs that were successfully merged
        - 'grids_failed': list of grid IDs that failed
        - 'num_grids_used': number of grids used
        - 'output_files': list of created output files
        - 'wait_time_elapsed': seconds waited for files
    """

    if path is None:
        path = "./"
    root = Path(path) / pixel

    print("\n" + "="*80)
    print(f"WAITING FOR FERRE OUTPUT FILES")
    print(f"Wait interval: {wait_time}s | Max wait: {max_wait}s | Cooldown: {cooldown}s")
    print("="*80)

    # Wait for files to be created by FERRE processes
    start_time = time.time()
    wait_count = 0
    previous_found = set()
    stable_count = 0

    while (time.time() - start_time) < max_wait:
        current_found = set()
        temp_available = []

        # Check which files exist right now
        for item in grid_ids:
            o_file = Path(path) / f"{pixel}.opf{item}"
            m_file = Path(path) / f"{pixel}.mdl{item}"
            n_file = Path(path) / f"{pixel}.nrd{item}"

            if o_file.is_file() and m_file.is_file() and n_file.is_file():
                current_found.add(item)
                temp_available.append(item)

        # Check if we have new files since last check
        new_files = current_found - previous_found
        if new_files:
            print(f"[{int(time.time()-start_time)}s] Found new output files for grids: {', '.join(sorted(new_files))}")
            previous_found = current_found
            stable_count = 0
        else:
            stable_count += 1

        # If we have enough grids, wait for cooldown and break
        if len(temp_available) >= min_grids_required:
            print(f"[{int(time.time()-start_time)}s] Found {len(temp_available)} grids (minimum: {min_grids_required})")
            print(f"Waiting {cooldown}s for file system cooldown...")
            time.sleep(cooldown)
            break

        # If files are stable for 3 checks, stop waiting
        if stable_count >= 3:
            print(f"[{int(time.time()-start_time)}s] File count stable at {len(temp_available)} grids")
            if len(temp_available) >= min_grids_required:
                print(f"Waiting {cooldown}s for file system cooldown...")
                time.sleep(cooldown)
            break

        wait_count += 1
        if wait_count == 1:
            print(f"Waiting for FERRE output files... (checking every {wait_time}s)")
        elif wait_count % 6 == 0:  # Report every minute if wait_time=10
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Still waiting... Found {len(temp_available)}/{len(grid_ids)} grids so far")

        time.sleep(wait_time)

    # Final check after waiting
    elapsed_total = int(time.time() - start_time)
    if elapsed_total >= max_wait:
        print(f"[{elapsed_total}s] Maximum wait time reached ({max_wait}s)")

    # Now do the final file check with detailed reporting
    available_grids = []
    missing_files = {}
    o = {}
    m = {}
    n = {}

    # Create return dictionary
    return_info = {
        'success': False,
        'grids_used': [],
        'grids_failed': [],
        'num_grids_used': 0,
        'output_files': [],
        'wait_time_elapsed': elapsed_total
    }

    print("\n" + "="*80)
    print("FERRE OUTPUT FILE STATUS CHECK")
    print("="*80)

    for item in grid_ids:
        o_file = Path(path) / f"{pixel}.opf{item}"
        m_file = Path(path) / f"{pixel}.mdl{item}"
        n_file = Path(path) / f"{pixel}.nrd{item}"

        missing = []
        if not o_file.is_file():
            missing.append(f".opf{item}")
        if not m_file.is_file():
            missing.append(f".mdl{item}")
        if not n_file.is_file():
            missing.append(f".nrd{item}")

        if len(missing) == 0:
            available_grids.append(item)
            o[item] = str(o_file)
            m[item] = str(m_file)
            n[item] = str(n_file)
            print(f"✓ Grid {item}: All output files found")
        else:
            missing_files[item] = missing
            return_info['grids_failed'].append(item)
            print(f"✗ Grid {item}: Missing files: {', '.join(missing)}")

    # Update return info
    return_info['grids_used'] = available_grids
    return_info['num_grids_used'] = len(available_grids)

    print("-"*80)
    print(f"Summary: {len(available_grids)}/{len(grid_ids)} grids completed successfully")

    # Write metadata file
    metadata_file = Path(path) / f"{pixel}_opfmerge_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"OPFMERGE METADATA\n")
        f.write(f"="*50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Pixel: {pixel}\n")
        f.write(f"Grid prefix: {grid_prefix}\n")
        f.write(f"Grids requested: {','.join(grid_ids)}\n")
        f.write(f"Grids successful: {','.join(available_grids)}\n")
        f.write(f"Grids failed: {','.join(return_info['grids_failed'])}\n")
        f.write(f"Number of grids used: {len(available_grids)}\n")
        f.write(f"Wait time elapsed: {elapsed_total}s\n")
        f.write(f"Minimum grids required: {min_grids_required}\n")
        f.write(f"Status: {'SUCCESS' if len(available_grids) >= min_grids_required else 'FAILED'}\n")

    # Check if we meet minimum requirements
    if len(available_grids) < min_grids_required:
        error_msg = []
        error_msg.append("\n" + "="*80)
        error_msg.append("CRITICAL ERROR: Insufficient FERRE grids completed successfully")
        error_msg.append("="*80)
        error_msg.append(f"Minimum grids required: {min_grids_required}")
        error_msg.append(f"Grids completed: {len(available_grids)}")
        error_msg.append(f"Grids requested: {len(grid_ids)}")
        error_msg.append("")
        error_msg.append("SUCCESSFUL GRIDS:")
        if available_grids:
            for g in available_grids:
                error_msg.append(f"  ✓ Grid {g}")
        else:
            error_msg.append("  None")
        error_msg.append("")
        error_msg.append("FAILED GRIDS:")
        for grid_id, missing in missing_files.items():
            error_msg.append(f"  ✗ Grid {grid_id}: Missing {', '.join(missing)}")
        error_msg.append("")
        error_msg.append("TROUBLESHOOTING SUGGESTIONS:")
        error_msg.append("  1. Check FERRE error files (*.nml_*.error) for details")
        error_msg.append("  2. Verify template files exist for failed grids")
        error_msg.append("  3. Check input spectra for NaN or infinite values")
        error_msg.append("  4. Consider excluding problematic grids from grid_ids")
        error_msg.append("  5. Review wavelength range compatibility with grid templates")
        error_msg.append("="*80)

        error_message = "\n".join(error_msg)
        print(error_message)

        # Save error report
        error_file = Path(path) / f"{pixel}_opfmerge_error.txt"
        with open(error_file, 'w') as f:
            f.write(error_message)

        raise RuntimeError(error_message)

    # If we have enough grids but not all, print informative message
    if len(available_grids) < len(grid_ids):
        print("\n" + "="*80)
        print("PROCEEDING WITH REDUCED GRID SET")
        print("="*80)
        print(f"Using {len(available_grids)} out of {len(grid_ids)} requested grids")
        print(f"Active grids: {', '.join(available_grids)}")
        print(f"Skipped grids: {', '.join([g for g in grid_ids if g not in available_grids])}")
        print("="*80 + "\n")

    # Set up limits for available grids
    llimit = {'1':3500.,'2':5500.,'3':7000.,'4':10000.,'5':20000.,'6':6000.,'7':10000.,'8':10000.,'9':15000.}

    if grid_prefix == 'n':
        iteff = {'1':2, '2':2, '3':2, '4':2, '5':2, '6':1, '7':1, '8':1, '9':1}
        ilchi = {'1':9, '2':9, '3':9, '4':9, '5':9, '6':7, '7':7, '8':7, '9':7}
    elif grid_prefix in ['m', 'p']:
        iteff = {'1':4, '2':4, '3':4, '4':4, '5':4, '6':1, '7':1, '8':1, '9':1}
        ilchi = {'1':13, '2':13, '3':13, '4':13, '5':13, '6':7, '7':7, '8':7, '9':7}
    else:
        raise ValueError(f'Error: grid_prefix "{grid_prefix}" is not supported. Use "n", "m", or "p"')

    # Validate parameters for available grids
    for item in available_grids:
        if item not in llimit:
            raise KeyError(f'llimit dict missing entry for grid_id = {item}')
        if item not in iteff:
            raise KeyError(f'iteff dict missing entry for grid_id = {item}')
        if item not in ilchi:
            raise KeyError(f'ilchi dict missing entry for grid_id = {item}')

    # Open input files
    of = {}
    mf = {}
    nf = {}

    try:
        for item in available_grids:
            of[item] = open(o[item], 'r')
            mf[item] = open(m[item], 'r')
            if len(n) > 0:
                nf[item] = open(n[item], 'r')
    except IOError as e:
        # Clean up any opened files
        for f in list(of.values()) + list(mf.values()) + list(nf.values()):
            if hasattr(f, 'close'):
                f.close()
        raise IOError(f"Failed to open input files: {e}")

    # Open output files
    try:
        oo = open(str(root)+'.opf', 'w')
        mo = open(str(root)+'.mdl', 'w')
        no = None
        if len(n) > 0:
            no = open(str(root)+'.nrd', 'w')
    except IOError as e:
        # Clean up input files
        for f in list(of.values()) + list(mf.values()) + list(nf.values()):
            f.close()
        raise IOError(f"Failed to create output files: {e}")

    # Process each line
    try:
        first_grid = available_grids[0]
        line_count = 0

        for line in of[first_grid]:
            line_count += 1
            array = line.split()

            # Validate line format
            if len(array) <= max(ilchi[first_grid], iteff[first_grid]):
                print(f"Warning: Line {line_count} has insufficient columns, skipping")

                mf[first_grid].readline()
                if len(n) > 0:
                    nf[first_grid].readline()
                for grid_id in available_grids[1:]:
                    of[grid_id].readline()
                    mf[grid_id].readline()
                    if len(n) > 0:
                        nf[grid_id].readline()
                continue

            # Initialise best-solution variables
            min_chi   = None
            min_oline = None
            min_mline = None
            min_nline = None

            # Process first grid
            try:
                chi_first = float(array[ilchi[first_grid]])
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse chi-squared value on line {line_count} in grid {first_grid}: {e}")
                chi_first = None

            mline_first = mf[first_grid].readline()
            nline_first = nf[first_grid].readline() if len(n) > 0 else None

            # Only consider first grid if chi is valid and not -1000
            if chi_first is not None and chi_first != -1000.0:
                min_chi   = chi_first
                min_oline = line
                min_mline = mline_first
                if len(n) > 0:
                    min_nline = nline_first

            # Process remaining grids
            for grid_id in available_grids[1:]:
                oline = of[grid_id].readline()
                mline = mf[grid_id].readline()
                nline = nf[grid_id].readline() if len(n) > 0 else None

                # Handle EOF: if a grid runs out of lines, skip it
                if not oline:
                    continue

                array = oline.split()
                if len(array) <= max(ilchi[grid_id], iteff[grid_id]):
                    print(f"Warning: Line {line_count} has insufficient columns in grid {grid_id}, skipping this grid for this object")
                    continue

                try:
                    chi  = float(array[ilchi[grid_id]])
                    teff = float(array[iteff[grid_id]])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing grid {grid_id} line {line_count}: {e}")
                    continue

                # Exclude this grid if chi2 == -1000
                if chi == -1000.0:
                    continue

                # Apply Teff cutoff for this grid
                if teff <= llimit[grid_id] * 1.01:
                    continue

                # Update best solution if this grid is better
                if (min_chi is None) or (chi < min_chi):
                    min_chi   = chi
                    min_oline = oline
                    min_mline = mline
                    if len(n) > 0:
                        min_nline = nline

            # If no valid solution found, skip this object
            if min_oline is None:
                print(f"Warning: No valid solution for object on line {line_count}, skipping")
                continue

            # Write the best fit to output
            oo.write(min_oline)
            mo.write(min_mline)
            if len(n) > 0 and no is not None:
                no.write(min_nline)

        print(f"Successfully processed {line_count} objects")

    except Exception as e:
        print(f"Error during merging: {e}")
        raise

    finally:
        # Close all files
        for grid_id in available_grids:
            if grid_id in of and hasattr(of[grid_id], 'close'):
                of[grid_id].close()
            if grid_id in mf and hasattr(mf[grid_id], 'close'):
                mf[grid_id].close()
            if grid_id in nf and hasattr(nf[grid_id], 'close'):
                nf[grid_id].close()

        if hasattr(oo, 'close'):
            oo.close()
        if hasattr(mo, 'close'):
            mo.close()
        if no is not None and hasattr(no, 'close'):
            no.close()

    print("\n" + "="*80)
    print(f"OPFMERGE COMPLETED SUCCESSFULLY")
    print(f"Merged results from {len(available_grids)} grids into:")
    print(f"  - {root}.opf")
    print(f"  - {root}.mdl")
    if len(n) > 0:
        print(f"  - {root}.nrd")
    print(f"Grids used: {', '.join(sorted(available_grids))}")
    print(f"Metadata saved to: {metadata_file}")
    print("="*80 + "\n")

    # Update return info with output files
    return_info['success'] = True
    return_info['output_files'] = [
        str(root) + '.opf',
        str(root) + '.mdl'
    ]
    if len(n) > 0:
        return_info['output_files'].append(str(root) + '.nrd')

    return return_info


#################################################################################################

def ferrerun(path=None, ferre=None):
    if path is None: path="./"
    pwd=os.path.abspath(os.curdir)
    os.chdir(path)
    if ferre is None:
        ferre="<FERRE_DIR>/bin/ferre.x"
    code = subprocess.call(ferre)
    os.chdir(pwd)
    return code

#################################################################################################

def ferre_exe_worker(cmdstr):
    """
    Execute FERRE with proper error handling and output capture.

    Parameters:
    -----------
    cmdstr : list
        List containing:
        [0] nmlfile: FERRE input filename (*.nml) with full path
        [1] ferre: Path to FERRE executable
        [2] return_err: Boolean - whether to raise exception on error

    Returns:
    --------
    tuple: (success, return_code, error_msg)
        - success: Boolean indicating if FERRE ran successfully
        - return_code: Integer exit code from FERRE
        - error_msg: String with error details if failed, None otherwise
    """
    nmlfile = cmdstr[0]
    ferre_exe = cmdstr[1]
    return_err = cmdstr[2]

    # Extract working directory and filename
    work_dir = os.path.dirname(nmlfile)
    nml_filename = os.path.basename(nmlfile)

    # Validate inputs
    if not os.path.exists(ferre_exe):
        error_msg = f"FERRE executable not found: {ferre_exe}"
        print(f"ERROR: {error_msg}")
        if return_err:
            raise FileNotFoundError(error_msg)
        return (False, -1, error_msg)

    if not os.path.exists(nmlfile):
        error_msg = f"Input file not found: {nmlfile}"
        print(f"ERROR: {error_msg}")
        if return_err:
            raise FileNotFoundError(error_msg)
        return (False, -1, error_msg)

    # Build command
    cmd = [ferre_exe, nml_filename]

    # Store output for error diagnosis
    stdout_lines = []
    stderr_lines = []

    try:
        # Run FERRE in its working directory
        popen = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=work_dir
        )

        # Capture stdout in real-time
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line, end='')
            stdout_lines.append(stdout_line)

        # Wait for process to complete
        popen.stdout.close()
        return_code = popen.wait()

        # Capture any stderr
        stderr = popen.stderr.read()
        if stderr:
            stderr_lines = stderr.split('\n')
            for line in stderr_lines:
                if line.strip():
                    print(f"STDERR: {line}")

        # Extract grid ID from filename for output file checking
        grid_id = None
        pixel_name = None
        if '.nml_' in nml_filename:
            try:
                grid_id = nml_filename.split('.nml_')[1]
                pixel_name = nml_filename.split('.nml_')[0].replace('input.nml', '').strip('_')
                if not pixel_name:
                    pass
            except:
                pass

        # Check for output file existence
        output_file_exists = False
        if grid_id and work_dir:
            import glob
            opf_pattern = os.path.join(work_dir, f'*.opf{grid_id}')
            opf_files = glob.glob(opf_pattern)
            output_file_exists = len(opf_files) > 0

        # Check for FERRE-specific error indicators
        error_indicators = [
            "IEEE_OVERFLOW_FLAG",
            "IEEE_UNDERFLOW_FLAG",
            "IEEE_DENORMAL",
            "STOP 1",
            "quit: Run ended"
        ]

        has_ferre_error = any(
            indicator in line
            for line in stdout_lines + stderr_lines
            for indicator in error_indicators
        )

        # Determine success based on:
        # 1. Clean exit (return_code == 0 and no error indicators), OR
        # 2. Output file exists (even if return_code != 0 or has IEEE warnings)
        if return_code == 0 and not has_ferre_error:
            # Clean success
            return (True, return_code, None)
        elif output_file_exists:
            # FERRE completed with warnings but produced valid output
            if has_ferre_error:
                print(f"    FERRE completed with warnings (IEEE underflow/denormal) but output file exists - treating as success")
            return (True, return_code, None)
        else:
            # Real failure - no output file
            error_msg = f"FERRE failed for {nml_filename}"
            if grid_id:
                error_msg = f"FERRE failed for grid {grid_id}"

            if has_ferre_error:
                error_msg += " (numerical errors detected)"
            error_msg += f" - exit code: {return_code}"

            print(f"\nWARNING: {error_msg}")

            # Store error details to file for debugging
            error_file = os.path.join(work_dir, f"{nml_filename}.error")
            with open(error_file, 'w') as f:
                f.write(f"Error: {error_msg}\n")
                f.write(f"Return code: {return_code}\n")
                f.write("\nSTDOUT:\n")
                f.writelines(stdout_lines)
                f.write("\nSTDERR:\n")
                f.writelines(stderr_lines)

            # Only raise exception if requested
            if return_err and return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

            return (False, return_code, error_msg)

    except subprocess.CalledProcessError as e:
        error_msg = f"FERRE process failed: {e}"
        print(f"ERROR: {error_msg}")
        if return_err:
            raise
        return (False, e.returncode, error_msg)

    except Exception as e:
        error_msg = f"Unexpected error running FERRE: {e}"
        print(f"ERROR: {error_msg}")
        if return_err:
            raise
        return (False, -1, error_msg)

    finally:
        # Ensure pipes are closed
        if 'popen' in locals():
            if popen.stdout and not popen.stdout.closed:
                popen.stdout.close()
            if popen.stderr and not popen.stderr.closed:
                popen.stderr.close()

#################################################################################################

def ferre_execute(cmd, path=None, return_err=False):
    """Execute cmd and return output line by line"""
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        universal_newlines=True,
        shell=True,
        cwd=path
    )

    for line in iter(popen.stdout.readline, ""):
        print(line, end='')

    popen.stdout.close()
    return_code = popen.wait()

    if return_err and return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


## return a dictionary, contain all FERRE outputs and a few empty columns to be filled out later
#################################################################################################
def ferre_outdict(pixel, targs_info, targs_setups, grid_prefix='n', path=None,
                  output_error_by_id=None, invalid_mask_by_id=None):

    if path is None:
        path = "./"

    root = os.path.join(path, pixel)

    # FERRE does not create separate wave/flux/error arrays for each arm by default
    # Here we use the extra wav files, created for each arm
    # to measure the length of wavelength in each arm. Then, based on the length of WV
    # for each arm, we create those.
    # FERRE does not create separate wave/flux/error arrays for each arm by default.
    # Use the per-arm wavelength files to determine the arm lengths.
    npix_s = []
    for i, s in enumerate(targs_setups):
        x = np.genfromtxt(root + '-' + s + '.wav', dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, ...]
        npix_s.append(len(x[0]))

    x = np.genfromtxt(root + '.wav', dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, ...]

    # If no per-setup wavelength files exist, use the total length
    if len(npix_s) == 0:
        npix_s.append(len(x[0]))

    out_param_f = glob.glob(root + ".opf")
    out_model_f = glob.glob(root + ".mdl")
    norm_flux_f = glob.glob(root + ".nrd")
    raw_flux_f = glob.glob(root + ".frd")
    raw_error_f = glob.glob(root + ".err")
    in_lambda_f = glob.glob(root + ".wav")

    ### GENERATE OUTPUT TABLE STRUCTURE
    columns = [
        'APS_ID', 'TARGID', 'CNAME', 'TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR',
        'FEH', 'FEH_ERR', 'ALPHA', 'ALPHA_ERR', 'MICRO', 'MICRO_ERR',
        'COVAR', 'ELEM', 'ELEM_ERR', 'SNR_FR', 'CHISQ_TOT', 'FLAG_FR'
    ]

    ## here we use original targs_setups as it is used to create the columns in the output fits/tables
    ## and it is independent of the libraries used by FERRE
    for s in targs_setups:
        columns.append('LAMBDA_FR_%s' % s[0])
        columns.append('FLUX_FR_%s' % s[0])
        columns.append('ERROR_FR_%s' % s[0])
        columns.append('MODEL_FR_%s' % s[0])

    outdict = OrderedDict()
    for c in columns:
        outdict[c] = []

    # Output parameter file
    out_param = open(out_param_f[0], 'r')

    # Best fitted model file
    out_model = np.genfromtxt(out_model_f[0], dtype=float)
    if out_model.ndim == 1:
        out_model = out_model[np.newaxis, ...]

    # Normalized input spectra model
    # FERRE-normalized observed spectrum
    in_flux = np.genfromtxt(norm_flux_f[0], dtype=float)
    if in_flux.ndim == 1:
        in_flux = in_flux[np.newaxis, ...]

    # Original unnormalized spectrum supplied to FERRE
    raw_flux = np.genfromtxt(raw_flux_f[0], dtype=float)
    if raw_flux.ndim == 1:
        raw_flux = raw_flux[np.newaxis, ...]

    # Non-normalized input error array
    # Error array actually supplied to FERRE, including any inflated
    # uncertainties used to downweight/mask pixels.
    raw_error = np.genfromtxt(raw_error_f[0], dtype=float)
    if raw_error.ndim == 1:
        raw_error = raw_error[np.newaxis, ...]

    # input (and output) wavelength array
    in_labmda = np.genfromtxt(in_lambda_f[0], dtype=float)
    if in_labmda.ndim == 1:
        in_labmda = in_labmda[np.newaxis, ...]

    # FERRE continuum relation:
    #
    #     normalized_flux = raw_flux / continuum
    #
    # therefore:
    #
    #     normalized_error = raw_error / continuum
    #                      = raw_error * normalized_flux / raw_flux
    #
    in_error = np.full_like(raw_error, np.nan, dtype=float)

    good = (
        np.isfinite(raw_error)
        & np.isfinite(raw_flux)
        & np.isfinite(in_flux)
        & (raw_flux != 0.0)
    )

    in_error[good] = (
        raw_error[good]
        * in_flux[good]
        / raw_flux[good]
    )

    for nline, line in enumerate(out_param):

        cells = line.split()

        #for N dim (since COVPRINT=1 in FERRE), there are m= 4 + N*(2+N) cells
        #and likewise we can calculate N = sqrt(m-3)-1
        m_cells = len(cells)
        assert m_cells > 6, (
            'Error, the output param file has less than 7 columns, '
            'which would correspond to ndim=2'
        )

        ndim = int(np.sqrt(m_cells - 3) - 1)

        # First fill the parameters, independent of the grid types
        aps_id = np.int32(int(cells[0]))

        outdict['APS_ID'].append(aps_id)
        outdict['CNAME'].append(str(targs_info[aps_id]['CNAME']))
        outdict['TARGID'].append(str(targs_info[aps_id]['TARGID']))

        ## Only originally invalid pixels are hidden from the output. Pixels excluded
        ## from the fit retain their measured flux/error and the complete FERRE model.
        # Only genuinely invalid input pixels are hidden from the final output.
        # Linemask/Balmer pixels remain present, together with their inflated
        # FERRE fitting uncertainties.
        if invalid_mask_by_id is not None:
            output_key = int(aps_id)

            if output_key not in invalid_mask_by_id:
                raise KeyError(
                    f'Missing preserved invalid-pixel mask for APS_ID {output_key}'
                )

            badmask = np.asarray(
                invalid_mask_by_id[output_key],
                dtype=bool
            )

            expected_shape = in_error[nline].shape

            if badmask.shape != expected_shape:
                raise ValueError(
                    f'Preserved invalid mask for APS_ID {output_key} has shape '
                    f'{badmask.shape}; expected {expected_shape}'
                )

            in_flux[nline][badmask] = np.nan
            in_error[nline][badmask] = np.nan
            out_model[nline][badmask] = np.nan

        # Extract spectra for each setup
        # Check if we have multiple setups (arms)
        # Split the complete arrays back into the individual arms.
        if len(targs_setups) > 1 and len(npix_s) == len(targs_setups):

            # Multiple arms - split the data according to npix_s
            pix_s_i = 0

            for i, s in enumerate(targs_setups):

                pix_s_f = pix_s_i + npix_s[i]

                outdict['FLUX_FR_%s' % s[0]].append(
                    in_flux[nline, pix_s_i:pix_s_f]
                )
                outdict['ERROR_FR_%s' % s[0]].append(
                    in_error[nline, pix_s_i:pix_s_f]
                )
                outdict['LAMBDA_FR_%s' % s[0]].append(
                    in_labmda[nline, pix_s_i:pix_s_f]
                )
                outdict['MODEL_FR_%s' % s[0]].append(
                    out_model[nline, pix_s_i:pix_s_f]
                )

                pix_s_i = pix_s_f

            # Single setup or joined arms - use all data for each setup
            # This handles the case when arms are joined but we still need to provide
            # data for each original setup column
        else:

            for s in targs_setups:

                # Single setup - use all data
                if len(targs_setups) == 1:

                    outdict['FLUX_FR_%s' % s[0]].append(in_flux[nline])
                    outdict['ERROR_FR_%s' % s[0]].append(in_error[nline])
                    outdict['LAMBDA_FR_%s' % s[0]].append(in_labmda[nline])
                    outdict['MODEL_FR_%s' % s[0]].append(out_model[nline])

                    # Multiple setups but data might be joined
                    # Check if we have per-setup wavelength files
                else:

                    wav_file = root + '-' + s + '.wav'

                    if os.path.exists(wav_file):

                        # We have per-setup wavelength, extract corresponding data
                        wav_s = np.genfromtxt(wav_file, dtype=float)

                        if wav_s.ndim == 1:
                            wav_s = wav_s[np.newaxis, ...]

                        npix = len(
                            wav_s[nline]
                            if len(wav_s.shape) > 1
                            else wav_s
                        )

                        # Find the corresponding indices in the full wavelength array
                        # This is needed when arms are split after joining
                        # First arm
                        if s == targs_setups[0]:
                            idx_start = 0
                            idx_end = npix
                            # Second arm
                        else:
                            idx_start = len(in_labmda[nline]) - npix
                            idx_end = len(in_labmda[nline])

                        outdict['FLUX_FR_%s' % s[0]].append(
                            in_flux[nline, idx_start:idx_end]
                        )
                        outdict['ERROR_FR_%s' % s[0]].append(
                            in_error[nline, idx_start:idx_end]
                        )
                        outdict['LAMBDA_FR_%s' % s[0]].append(
                            in_labmda[nline, idx_start:idx_end]
                        )
                        outdict['MODEL_FR_%s' % s[0]].append(
                            out_model[nline, idx_start:idx_end]
                        )

                    else:

                        # No per-setup files, probably shouldn't happen but handle gracefully
                        # Use empty arrays
                        outdict['FLUX_FR_%s' % s[0]].append(np.array([]))
                        outdict['ERROR_FR_%s' % s[0]].append(np.array([]))
                        outdict['LAMBDA_FR_%s' % s[0]].append(np.array([]))
                        outdict['MODEL_FR_%s' % s[0]].append(np.array([]))

        #Kurucz grids with 3 dimensions
        if m_cells == 19:

            outdict['FEH'].append(np.float64(cells[1]))
            outdict['TEFF'].append(np.float64(cells[2]))
            outdict['LOGG'].append(np.float64(cells[3]))
            outdict['ALPHA'].append(np.nan)
            outdict['MICRO'].append(np.nan)

            outdict['FEH_ERR'].append(np.float64(cells[4]))
            outdict['TEFF_ERR'].append(np.float64(cells[5]))
            outdict['LOGG_ERR'].append(np.float64(cells[6]))
            outdict['ALPHA_ERR'].append(np.nan)
            outdict['MICRO_ERR'].append(np.nan)

            outdict['ELEM'].append([np.nan, np.nan])
            outdict['ELEM_ERR'].append([np.nan, np.nan])
            outdict['CHISQ_TOT'].append(10.**np.float64(cells[9]))
            outdict['SNR_FR'].append(np.float64(cells[8]))

            if grid_prefix == 'n':
                cov = np.reshape(
                    np.array(cells[10:], dtype=float),
                    (3, 3)
                )
                outdict['COVAR'].append(cov)
            else:
                sys.exit(
                    'Error: a 3 parameter grid was unexpectedly '
                    'included among the *m* grids'
                )

            #Kurucz grids with 5 dimensions
        elif m_cells == 39:

            outdict['FEH'].append(np.float64(cells[1]))
            outdict['TEFF'].append(np.float64(cells[4]))
            outdict['LOGG'].append(np.float64(cells[5]))
            outdict['ALPHA'].append(np.float64(cells[2]))
            outdict['MICRO'].append(np.float64(cells[3]))

            outdict['FEH_ERR'].append(np.float64(cells[6]))
            outdict['TEFF_ERR'].append(np.float64(cells[9]))
            outdict['LOGG_ERR'].append(np.float64(cells[10]))
            outdict['ALPHA_ERR'].append(np.float64(cells[7]))
            outdict['MICRO_ERR'].append(np.float64(cells[8]))

            outdict['ELEM'].append([np.nan, np.nan])
            outdict['ELEM_ERR'].append([np.nan, np.nan])
            outdict['CHISQ_TOT'].append(10.**np.float64(cells[13]))
            outdict['SNR_FR'].append(np.float64(cells[12]))

            if grid_prefix in ['m', 'p']:
                cov = np.reshape(
                    np.array(cells[14:], dtype=float),
                    (5, 5)
                )
                outdict['COVAR'].append(cov)
            else:
                sys.exit(
                    'Error: a 5 parameter grid was unexpectedly '
                    'included among the *n* grids'
                )

            #white dwarfs 2 dimensions
        elif m_cells == 12:

            outdict['FEH'].append(-10.)
            outdict['TEFF'].append(np.float64(cells[1]))
            outdict['LOGG'].append(np.float64(cells[2]))
            outdict['ALPHA'].append(np.nan)
            outdict['MICRO'].append(np.nan)

            outdict['FEH_ERR'].append(np.nan)
            outdict['TEFF_ERR'].append(np.float64(cells[3]))
            outdict['LOGG_ERR'].append(np.float64(cells[4]))
            outdict['ALPHA_ERR'].append(np.nan)
            outdict['MICRO_ERR'].append(np.nan)

            outdict['ELEM'].append([np.nan, np.nan])
            outdict['ELEM_ERR'].append([np.nan, np.nan])
            outdict['CHISQ_TOT'].append(10.**np.float64(cells[7]))
            outdict['SNR_FR'].append(np.float64(cells[6]))

            if grid_prefix == 'n':
                cov = np.zeros((3, 3))
                cov[1:, 1:] = np.reshape(
                    np.array(cells[8:], dtype=float),
                    (2, 2)
                )
                outdict['COVAR'].append(cov)

            elif grid_prefix in ['m', 'p']:
                cov = np.zeros((5, 5))
                cov[3:, 3:] = np.reshape(
                    np.array(cells[8:], dtype=float),
                    (2, 2)
                )
                outdict['COVAR'].append(cov)

        # chi**2<10 and S/N>5
        if (
            outdict['CHISQ_TOT'][-1] < 1.
            and outdict['SNR_FR'][-1] > 5.
        ):
            outdict['FLAG_FR'].append(1)
        else:
            outdict['FLAG_FR'].append(0)

    out_param.close()

    # Now start with do some housekeeping on this dictionary
    outdict = Table(outdict)

    outdict['FEH'].unit = 'dex'
    outdict['TEFF'].unit = 'K'
    outdict['LOGG'].unit = 'dex'
    outdict['ALPHA'].unit = 'dex'
    outdict['MICRO'].unit = 'dex'
    outdict['ELEM'].unit = 'dex'
    outdict['FEH_ERR'].unit = 'dex'
    outdict['TEFF_ERR'].unit = 'K'
    outdict['LOGG_ERR'].unit = 'dex'
    outdict['ALPHA_ERR'].unit = 'dex'
    outdict['MICRO_ERR'].unit = 'dex'
    outdict['ELEM_ERR'].unit = 'dex'

    return outdict
#################################################################################################


def ferre_write_fits(outdict, infiles, setups, join_arms, units_str, param_file,
                     spec_file=None, match_table=None, merge_metadata=None,
                     linemask=None, uselinemasks=None):
    """
    Write FERRE outputs to FITS files with enhanced metadata tracking.
    """

    ## GENERATE FERRE OUTPUT TABLES (params only)
    outdict.sort('APS_ID')

    FR_param = Table(outdict['APS_ID', 'TARGID', 'CNAME', 'TEFF', 'TEFF_ERR', 'LOGG',
                             'LOGG_ERR', 'FEH', 'FEH_ERR', 'ALPHA', 'ALPHA_ERR', 'MICRO',
                             'MICRO_ERR', 'COVAR', 'ELEM', 'ELEM_ERR', 'SNR_FR', 'CHISQ_TOT',
                             'FLAG_FR'])

    ## Add any extra columns if necessary from the match_table
    if match_table is not None:
        FR_param = add_extra_columns(FR_param, match_table=match_table)

    FR_param.meta['EXTNAME'] = 'FR_PARAM'

    # Keep the basename of input files as provinces
    for n_province, province in enumerate(infiles):
        FR_param.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

    FR_param.meta['EXTNAME'] = 'FR_TABLE'
    FR_param.meta['APSVERS'] = (APSVERS, 'PyAPS version')
    FR_param.meta['APS_FR_V'] = (aps_constants.__aps_ferre_version__, 'PYAPS FERRE wrapper version')
    FR_param.meta['CSB_FR'] = (join_arms, 'Combines Spectral Bands Status for FERRE')

    linemask_used = int(bool(uselinemasks)) and linemask is not None
    linemask_name = Path(linemask).name if linemask_used else "None"

    FR_param.meta['LMASKUSE'] = (linemask_used, "1 if linemask was used 0 otherwise")
    FR_param.meta['LMASK'] = (linemask_name, 'Path to FERRE linemask file')

    # Add grid tracking metadata if available
    if merge_metadata is not None:
        FR_param.meta['NGRIDS'] = (merge_metadata['num_grids_used'],
                                   'Number of FERRE grids used in merging')
        FR_param.meta['GRIDSUSED'] = (','.join(merge_metadata['grids_used']),
                                      'Grid IDs successfully used')
        if merge_metadata.get('grids_failed'):
            FR_param.meta['GRIDSFAIL'] = (','.join(merge_metadata['grids_failed']),
                                          'Grid IDs that failed')
        FR_param.meta['WAITTIME'] = (merge_metadata['wait_time_elapsed'],
                                     'Seconds waited for FERRE outputs')
        FR_param.meta['OPFMERGE'] = (merge_metadata['success'],
                                     'OPFMERGE success status')

    # Write parameter file
    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU())
    hx.append(fits.convenience.table_to_hdu(FR_param))
    hx.writeto(os.path.expandvars(param_file), overwrite=True)

    # If outspec is requested, also return the flux, error, lambda and best fitted model
    if spec_file is not None:

        # Remove parameter columns for spec file
        outdict.remove_columns(['TEFF','TEFF_ERR','LOGG','LOGG_ERR','FEH',
                              'FEH_ERR','ALPHA','ALPHA_ERR','MICRO','MICRO_ERR',
                              'COVAR','ELEM','ELEM_ERR','SNR_FR','CHISQ_TOT','FLAG_FR'])

        # Add any extra columns if necessary
        if match_table is not None:
            outdict = add_extra_columns(outdict, match_table=match_table)

        outdict.meta['EXTNAME'] = 'FR_SPEC'

        # Add units to the spectra
        for s in setups:
            outdict['LAMBDA_FR_%s' % s[0]].unit = units_str['wave']
            outdict['FLUX_FR_%s' % s[0]].unit = units_str['flux']
            outdict['ERROR_FR_%s' % s[0]].unit = units_str['flux']
            outdict['MODEL_FR_%s' % s[0]].unit = units_str['flux']

        # Keep the basename of input files as provinces
        for n_province, province in enumerate(infiles):
            outdict.meta['APSREF_%d' %(n_province)] = (os.path.basename(province), 'L1 reference file')

        outdict.meta['APSVERS'] = (APSVERS, 'APS version')
        outdict.meta['APS_FR_V'] = (aps_constants.__aps_ferre_version__, 'PYAPS FERRE wrapper version')
        outdict.meta['VACUUM'] = (False, 'Wavelengths are in VACUUM')
        outdict.meta['SAMPLING'] = (0, 'Sampling mode (0: linear, 1: logarithmic)')
        outdict.meta['CSB_FR'] = (join_arms, 'Combines Spectral Bands Status for FERRE')

        outdict.meta['LMASKUSE'] = (linemask_used, '1 if linemask used, 0 otherwise')
        outdict.meta['LMASK'] = (linemask_name, 'Path to FERRE linemask file')

        # Add grid tracking metadata if available
        if merge_metadata is not None:
            outdict.meta['NGRIDS'] = (merge_metadata['num_grids_used'],
                                      'Number of FERRE grids used in merging')
            outdict.meta['GRIDSUSED'] = (','.join(merge_metadata['grids_used']),
                                         'Grid IDs successfully used')
            if merge_metadata.get('grids_failed'):
                outdict.meta['GRIDSFAIL'] = (','.join(merge_metadata['grids_failed']),
                                             'Grid IDs that failed')
            outdict.meta['WAITTIME'] = (merge_metadata['wait_time_elapsed'],
                                        'Seconds waited for FERRE outputs')
            outdict.meta['OPFMERGE'] = (merge_metadata['success'],
                                        'OPFMERGE success status')

        # Write spectra file
        hx = fits.HDUList()
        hx.append(fits.PrimaryHDU())
        hx.append(fits.convenience.table_to_hdu(outdict))
        hx.writeto(os.path.expandvars(spec_file), overwrite=True)

#################################################################################################


def check_rvs_input(rvsfile):
    """Exit with a clear message when the RVS table that FERRE depends on does not exist."""
    if rvsfile is None or not os.path.isfile(rvsfile):
        sys.exit(f"ERROR: the RVS output required by FERRE does not exist: {rvsfile}. "
                 f"The RVS step of this OB did not produce it (check the RVS_L2_* job log); "
                 f"rerun RVS before FERRE.")


def proc_ferre(infiles, classfile, param_fits, aps_ids=None, targsrvy=None, targclass=None,
               mask_aps_ids=None, area=None, mask_areas=None, figdir=None, wlranges=None,
               use_rvs=True, rvsfile=None, path=None, outpath=None, spec_fits=None,
               pixel=None, templates=None, grid_prefix='n', grid_ids=None, outspec=None,
               sens_corr=None, mask_gaps=None, safe_mask_gaps=None, tellurics=None,
               vacuum=None, fill_gap=None, arms_ratio=None, join_arms=None, split_arms=None,
               nthreads=None, ferre=None, match_table=None, catdir=None, caldir=None,
               configdir=None, min_grids_required=2, maskbalmer=None, linemask=None,
               uselinemasks=None):
    """
    Main FERRE processing function with enhanced error handling and grid tracking.

    Additional parameters:
    min_grids_required : int
        Minimum number of successful grids required to continue (default: 1)
    """

    ## For MOS mode, we need the APSID of the interested sources, coming from the classfile
    # read classifier output
    class_tab = fits.getdata(classfile, 1)
    working_classlist = ['STAR', 'WD']
    # this return the aps_ids of the targets, satisfied the class condition
    aps_ids_in_class = aps_ids_class(classfile, working_classlist, aps_ids=aps_ids,
                                        nthreads=1, threadid=0, rank=3, ncchar=2)

    # for i in range(len(class_tab)):
    #     print(f"APS_ID: {class_tab['APS_ID'][i]},targclass: {class_tab['TARGCLASS'][i]},targsrvy: {class_tab['TARGSRVY'][i]}, CLASS: {class_tab['CLASS'][i]}, SRVY_CLASS: {class_tab['SRVY_CLASS'][i]}")

    ## Check if ids_in_class is OK or not
    if aps_ids_in_class is None or not aps_ids_in_class.size:
        return

    # FERRE starts from the RVS result. The SLURM chain submits FR with --dependency=afterany
    # on the RVS job (so L2merge still runs), which means FR also starts when RVS failed:
    # stop here with a readable message instead of a traceback from deep inside astropy.
    if use_rvs:
        check_rvs_input(rvsfile)


    if grid_ids is None:
        print("WARNING: NO grid_ids param has been set. We try with the default WEAVE grid IDS")
        grid_ids = ['1','2','3','4','5','6','7','8','9']
        print("grid_ids updated to ['1','2','3','4','5','6','7','8','9']")

    ## check all elements of the grid_ids are str and check grid_ids elements and format
    assert isinstance(grid_ids, list), 'grid_ids must be a list'
    assert all(isinstance(item, str) for item in grid_ids), 'all grid_ids elements must be str'
    grid_ids = [item.replace(" ","") for item in grid_ids]

    # Check for overlapped regions and handle join_arms
    init_check = l1_fileinfo(infiles, wlranges=wlranges, arms_ratio=arms_ratio)


    # IMPORTANT: FIX the overlap as FERRE DOES not like overlap.
    if (len(init_check['wlranges']) > 1) and (overlap_finder(init_check['wlranges']) > 0):
        wlranges = check_and_fix_overlap(wlranges)
        print(f'Overlap detected in wavelenght ranges of arms. We split them and continue')

    if use_rvs:
        # read RVS output
        rvs_tab = fits.getdata(rvsfile, 1)

    # READ DATA and put them in the APSOBJ OBJECT
    APSOBJ = APSOB(infiles, targsrvy=targsrvy, targclass=targclass, mask_aps_ids=mask_aps_ids,
                   area=area, mask_areas=mask_areas, wlranges=wlranges, aps_ids=aps_ids_in_class,
                   sens_corr=sens_corr, mask_gaps=mask_gaps, safe_mask_gaps=safe_mask_gaps,
                   vacuum=vacuum, tellurics=tellurics, fill_gap=fill_gap, arms_ratio=arms_ratio,
                   join_arms=join_arms, split_arms=split_arms, catdir=catdir, caldir=caldir,
                   configdir=configdir)

    targs = APSOBJ.data()
    targs_infiles = APSOBJ.infiles()
    targs_id = APSOBJ.id()
    targs_idfx = APSOBJ.idfx()
    targs_idxf = APSOBJ.idxf()
    targs_nbands = APSOBJ.nbands()
    targs_funits = APSOBJ.funits()
    targs_wavelist = APSOBJ.wavelist()
    targs_mode = APSOBJ.mode()
    # Return the normalized spectrograph unit name like MOS, MIFU or LIFU
    targs_fmode_unit = APSOBJ.fmode_unit()
    targs_join_arms = APSOBJ.join_arms()
    targs_setups = APSOBJ.setups()
    setups_original = APSOBJ.setups_original()

    # Check if uselinemask is applicable (only needed for HR OBs!)
    is_hr_ob = any(setup.startswith(("GREENH", "BLUEH", "REDH")) for setup in setups_original)

    if uselinemasks and not is_hr_ob:
        warnings.warn(
            f"Linemasks are only applicable to HR observations. "
            f"Found setups_original={setups_original}. Full spectrum fit will be performed.")
        uselinemasks = False

    if uselinemasks and not Path(linemask).exists():
        warnings.warn("No linemask csv with has been detected. Full spectrum fit will be performed.")
        uselinemasks = False

    if uselinemasks:
        mask_windows = np.genfromtxt(linemask,
                                        delimiter=',',
                                        names=True,
                                        dtype=None,
                                        encoding=None)

    # Handle setup names for joined arms
    if targs_join_arms and (len(setups_original) > 1):
        orig_setups = ['_'.join(setups_original)]
    else:
        orig_setups = targs_setups

    ### Create units dictionary
    if sens_corr:
        flux_unit_str = '%2e erg/(s cm**2 Angstrom)' %(targs_funits)
        ivar_unit_str = '%2e cm**4 Angstrom**2 /(s**2 erg**2)' %(targs_funits**-2)
    else:
        flux_unit_str = 'count'
        ivar_unit_str = '1/count**2'
    wave_unit_str = 'Angstrom'
    units_str = {'flux':flux_unit_str, 'ivar':ivar_unit_str, 'wave':wave_unit_str}

    # Create targs_info dictionary
    targs_info = {}

    class_index_dict = index_in_class(classfile, working_classlist, aps_ids=list(targs_id), ncchar=2)

    for tgs in targs_id:
        assert targs[targs_idfx[tgs]].id == tgs, 'It should never happen. Something wrong in APSOB'

        z_class = class_index_dict[tgs]['Z']
        zerr_class = class_index_dict[tgs]['ZERR']

        if use_rvs:
            id_in_rvs = np.ravel(np.where(rvs_tab['APS_ID'] == tgs))
            if len(id_in_rvs) > 0:
                id_in_rvs = id_in_rvs[0]
            else:
                sys.exit('ERROR: No RVS INFO for [APS_ID: %s] found' %(tgs))

            vrad_rvs = rvs_tab['VRAD'][id_in_rvs]
            vrad_err_rvs = rvs_tab['VRAD_ERR'][id_in_rvs]
        else:
            vrad_rvs = np.nan
            vrad_err_rvs = np.nan

        ## Filling the targs_info
        targs_info[tgs] = {'APS_ID':targs[targs_idfx[tgs]].aps_id,
                          'TARGID':targs[targs_idfx[tgs]].targid,
                          'CNAME':targs[targs_idfx[tgs]].cname,
                          'TARGSRVY':targs[targs_idfx[tgs]].targsrvy,
                          'Z_CLASS':z_class,
                          'ZERR_CLASS':zerr_class,
                          'VRAD':vrad_rvs,
                          'VRAD_ERR':vrad_err_rvs}

    # Set up maxorder based on grid prefix
    if grid_prefix == 'n':
        maxorder = {'1':3, '2':3, '3':3, '4':2, '5':1, '6':3, '7':3, '8':3, '9':3}
    elif grid_prefix in ['m','p']:
        maxorder = {'1':3, '2':3, '3':3, '4':3, '5':2, '6':3, '7':3, '8':3, '9':3}
    else:
        sys.exit('Error: grid_prefix is neither n, m, nor p -- unknown grid family')

    # Set the grids to be used
    grids = []
    for item_i, item in enumerate(grid_ids):
        assert item in maxorder.keys(), 'maxorder dict has not been set for grid_ids: %s' %(item)
        grid_filename = templates + grid_prefix + '_rweave' + item
        grids.append(grid_filename)

    # Set ids array and par dictionary
    ids = []
    par = {}
    for tgid in targs_id:
        id = str(tgid)
        ids.append(id)
        par[id] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    ### -----------------
    ## Extract grid parameters for LSF computation (wavelengths, pixel step, fwhm)
    ## NOTE: TO DO: replace this with lookup table (aps_constants?)

    print('--> Reading grid headers to extract wavelength parameters for LSF computation...')

    # Initialize parameter dictionaries
    grid_wave_map = {}
    pixel_step_map = {}
    fwhm_mod_map = {}

    # Loop over all grids to extract their specific wavelength parameters
    for grid_idx, grid in enumerate(grids):
        grid_id = grid_ids[grid_idx]
        grid_wave_map[grid_id] = {}
        pixel_step_map[grid_id] = {}
        fwhm_mod_map[grid_id] = {}

        print(f'    Processing grid {grid_id}: {grid}')

        for setup in orig_setups:
            if setup == '':
                synthfile = find_template_file(grid + '.hdr')
            else:
                synthfile = find_template_file(grid + '-' + targs_fmode_unit.replace(" ", "") + '_' + setup + '.hdr')

            # Parse header and extract wavelength array
            header = head_synth(synthfile)
            wave_arr = lambda_synth(synthfile)
            pixel_step = float(header['WAVE'].split()[1])

            # Search through COMMENT headers to find FWHM value
            # Expected format in header: COMMENTS6 = 'FWHM:1.5'
            fwhm_mod = None
            for i in range(1, 11):  # Check up to COMMENT10
                comment_key = f'COMMENTS{i}'
                if comment_key in header:
                    comment_value = header[comment_key]
                    if 'FWHM' in comment_value:
                        try:
                            # Extract value after colon
                            fwhm_mod = float(comment_value.split(':')[1].split('\'')[0])
                            break
                        except (ValueError, IndexError):
                            continue
                else:
                    break  # No more COMMENT headers

            # FWHM is required for LSF mode - raise error if not found
            if fwhm_mod is None:
                ## NOTE: This should be fixed. Maybe with lookup file.
                sys.exit(f'ERROR: FWHM value not found in grid header {synthfile}. '
                            f'Cannot run with LSF mode without FWHM_mod value.')

            # Store parameters for this grid_id and setup
            grid_wave_map[grid_id][setup] = wave_arr
            pixel_step_map[grid_id][setup] = pixel_step
            fwhm_mod_map[grid_id][setup] = fwhm_mod

            print(f'      Setup {setup}: npix={len(wave_arr)}, step={pixel_step:.4f}, FWHM={fwhm_mod}')

    ### -----------------

    ## Loop over targets in APSOBJ in each setup
    for ns, s in enumerate(orig_setups):
        print('--> Analyzing %2d of %2d input (spectra) files | SETUP: %10s | FileName: %s'
              %(ns+1, len(orig_setups), s, targs_infiles[ns]))

        npass = len(targs_id)
        x_in = np.array(targs_wavelist[ns])

        x2 = np.zeros((npass, len(x_in)))
        y2 = np.zeros((npass, len(x_in)))
        ey2 = np.zeros((npass, len(x_in)))
        output_ey2 = np.zeros((npass, len(x_in)))
        invalid2 = np.zeros((npass, len(x_in)), dtype=bool)

        # Loop over targets
        for n_tgs, tgs in enumerate(targs_id):
            targs_indx = targs_idfx[tgs]

            if use_rvs:
                id_in_rvs = np.ravel(np.where(rvs_tab['APS_ID'] == tgs))
                id_in_rvs = id_in_rvs[0]
                clight = 299792.458  # km/s
                zz = rvs_tab['VRAD'][id_in_rvs] / clight
            else:
                zz = targs_info[tgs]['Z_CLASS']

            # Apply redshift correction
            x = x_in * (1. - zz)

            source_ivar = targs[targs_indx].spectra[ns].ivar
            yflux = targs[targs_indx].spectra[ns].flux

            large_error = aps_constants.large_error
            yivar, output_yerr, invalid_mask = _prepare_ferre_error_arrays(
                source_ivar, yflux
            )

            # Mask Balmer regions if requested
            if maskbalmer:
                balmer = [[4334,4349],[4853,4868],[6555,6570]]
                # lithium = [6705,6715]
                for balmer_range in balmer:
                    balmer_mask = (x >= balmer_range[0]) & (x <= balmer_range[1])
                    yivar[balmer_mask] = 1.0 / (large_error**2)

            if uselinemasks:
                mask_list = np.column_stack((mask_windows['start'], mask_windows['end'])).tolist()
                
                for mask_range in mask_list:
                    line_mask = (x >= mask_range[0]) & (x<= mask_range[1])
                    yivar[line_mask] = 1.0 / (large_error**2)

            yerr = np.sqrt(1.0 / yivar)

            x2[n_tgs,:] = x
            y2[n_tgs,:] = yflux
            ey2[n_tgs,:] = yerr
            output_ey2[n_tgs,:] = output_yerr
            invalid2[n_tgs,:] = invalid_mask

        if ns == 0:
            xx = x2
            yy = y2
            eyy = ey2
            output_eyy = output_ey2
            invalid_pixels = invalid2
        else:
            xx = np.concatenate((xx, x2), axis=1)
            yy = np.concatenate((yy, y2), axis=1)
            eyy = np.concatenate((eyy, ey2), axis=1)
            output_eyy = np.concatenate((output_eyy, output_ey2), axis=1)
            invalid_pixels = np.concatenate((invalid_pixels, invalid2), axis=1)

        # Save wavelength for each band
        np.savetxt(os.path.join(path, pixel)+'-'+targs_setups[ns]+'.wav', x2, fmt='%14.6e')

    ## Write Ferre inputs
    print('--> Preparing FERRE input files [.frd, .err, .vrd, .wav]...')
    write_ferre_input(pixel, ids, par, xx, yy, eyy, path=path)

    output_error_by_id = {
        int(tgs): output_eyy[index] for index, tgs in enumerate(targs_id)
    }
    invalid_mask_by_id = {
        int(tgs): invalid_pixels[index] for index, tgs in enumerate(targs_id)
    }

    ## Prepare FERRE control files and grids
    print('--> Preparing FERRE control files (input.nml_[seq]) and grids...')

    cmdstr_list = []
    for grid_k, grid in enumerate(grids):
        grid_id_k = grid_ids[grid_k]

        # res is used to set the NCONT in mknml, and to set backup resolution. HACK - to be improved.
        res=None
        if is_hr_ob:
            res = 'HR'
        else:
            res = 'LR'

        ### -----------------
        ### LSF

        print(f'--> Creating LSF file for grid {grid_id_k}')

        npass = len(targs_id)
        fwhms_grid = None

        for ns, s in enumerate(orig_setups):
            current_setup = targs_setups[ns]

            # Get grid-specific wavelength parameters
            grid_wave = grid_wave_map[grid_id_k][current_setup]
            pixel_step = pixel_step_map[grid_id_k][current_setup]
            fwhm_mod = fwhm_mod_map[grid_id_k][current_setup]

            grid_wave_len = len(grid_wave)
            fwhm2 = np.zeros((npass, grid_wave_len))

            for n_tgs, tgs in enumerate(targs_id):
                targs_indx = targs_idfx[tgs]

                try:
                    fwhm_funcs = APSOBJ.get_fwhm(aps_id=tgs, fwhm_key='fwhm')
                    fwhm_func = fwhm_funcs[ns]
                    fwhm_obs_grid = fwhm_func(grid_wave)
                except Exception as e:
                    print(e.message)
                    ## Set to fixed values ~typical of each setup
                    if res == 'LR':
                        fixfwhm = 1.5
                        fwhm_obs_grid = np.ones(grid_wave)*fixfwhm
                    elif res == 'HR':
                        if 'BLUE' in s:
                            fixfwhm = 0.35
                        elif 'GREEN' in s:
                            fixfwhm = 0.5
                        elif 'RED' in s:
                            fixfwhm = 0.35
                        fwhm_obs_grid = np.ones(grid_wave)*fixfwhm
                    print(f"WARNING:  APS_ID {tgs} setup {s}: FWHM failed ({e}), using constant FWHM ({fixfwhm} A)")

                # Compute relative FWHM in pixels (cannot be zero so set to a small minimum)
                fwhm_diff_sq = np.maximum(0.01**2, fwhm_obs_grid**2 - fwhm_mod**2)
                fwhm_pixels = np.sqrt(fwhm_diff_sq) / pixel_step

                fwhm2[n_tgs, :] = fwhm_pixels

            # Concatenate across setups
            if fwhms_grid is None:
                fwhms_grid = fwhm2
            else:
                fwhms_grid = np.concatenate((fwhms_grid, fwhm2), axis=1)

        # Write grid-specific LSF file
        write_ferre_lsf(pixel, ids, fwhms_grid, k=grid_id_k, path=path)

        ### -----------------

        ## Make array with names of synthfiles
        synthfiles = []
        for j in range(len(orig_setups)):
            if orig_setups[j] == '':
                gridfile = find_template_file(grid + '.hdr')
            else:
                gridfile = find_template_file(grid + '-' + (targs_fmode_unit).replace(" ",'') + '_' + orig_setups[j] + '.hdr')
            synthfiles.append(gridfile)

        # Prepare ferre control file
        nml = mknml(synthfiles, path, pixel, grid_id_k, maxorder[grid_id_k], nthreads=nthreads, setups=setups_original)

        nmlfile = 'input.nml_' + str(grid_id_k)
        writenml(nml, nmlfile=nmlfile, path=path)
        writenml(nml, path=path)

        # Make list of commands for parallel worker
        cmdstr_list.append([os.path.join(path, nmlfile), ferre, True])

    ## Multiprocessing to run FERRE
    mpprocs = ThreadPool(nthreads)
    print("Running with {} processes".format(nthreads))

    if "OMP_NUM_THREADS" in os.environ:
        nthread = int(os.environ["OMP_NUM_THREADS"])
        if nthread != 1:
            print("WARNING: {} multi-processes running, each with "
                  "{} threads ({} total)".format(nthreads, nthread, nthreads*nthread))
            print("WARNING: Please ensure this is <= the number of physical cores on the system")
    else:
        print("WARNING: using multiprocessing, but the OMP_NUM_THREADS")
        print("WARNING: environment variable is not set- your system may be oversubscribed.")
        sys.stdout.flush()

    # Track FERRE execution results
    ferre_results = {}
    successful_grids = []
    failed_grids = []

    print("\n" + "="*80)
    print("STARTING FERRE EXECUTION")
    print("="*80)

    for i, cmdstr in enumerate(cmdstr_list):
        grid_id = grid_ids[i]
        print(f"Submitting FERRE job for grid {grid_id}")
        result = mpprocs.apply_async(ferre_exe_worker, (cmdstr,))
        ferre_results[grid_id] = result

    print('--- Waiting for all jobs to be finished. Please be patient ;)')
    mpprocs.close()
    mpprocs.join()

    # Check results
    print("\n" + "="*80)
    print("FERRE EXECUTION RESULTS")
    print("="*80)

    for grid_id, result in ferre_results.items():
        try:
            result_value = result.get()
            # Handle both old format (no return) and new format (tuple)
            if result_value is None:
                # Old format, assume success if no exception
                successful_grids.append(grid_id)
                print(f"✓ Grid {grid_id}: Completed")
            elif isinstance(result_value, tuple):
                success, return_code, error_msg = result_value
                if success:
                    successful_grids.append(grid_id)
                    print(f"✓ Grid {grid_id}: Success")
                else:
                    failed_grids.append(grid_id)
                    print(f"✗ Grid {grid_id}: Failed - {error_msg}")
            else:
                # Unexpected format, assume success
                successful_grids.append(grid_id)
                print(f"✓ Grid {grid_id}: Completed")
        except Exception as e:
            failed_grids.append(grid_id)
            print(f"✗ Grid {grid_id}: Exception - {e}")

    print(f"\nSummary: {len(successful_grids)}/{len(grid_ids)} grids completed")

    ## Extract the best set of parameters from all results
    print('\n--- Running OPFMERGE with file waiting and monitoring')

    merge_result = None
    try:
        # Call enhanced opfmerge
        merge_result = opfmerge(
            pixel=pixel,
            grid_ids=grid_ids,
            grid_prefix=grid_prefix,
            path=path,
            min_grids_required=min_grids_required,
            wait_time=10,      # Check every 10 seconds
            max_wait=300,      # Wait up to 5 minutes
            cooldown=2         # 2 second cooldown
        )

        if merge_result['success']:
            print(f"✓ OPFMERGE successful using {merge_result['num_grids_used']} grids")
            print(f"  Grids used: {', '.join(merge_result['grids_used'])}")

    except RuntimeError as e:
        print(f"OPFMERGE failed with error:\n{e}")

        # Save failure report
        failure_log = os.path.join(path, f"{pixel}_FAILED.log")
        with open(failure_log, 'w') as f:
            f.write(f"FERRE Processing Failed\n")
            f.write(f"="*50 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {e}\n")
            f.write(f"Successful grids: {successful_grids}\n")
            f.write(f"Failed grids: {failed_grids}\n")

        raise  # Re-raise to stop processing

    ## Put the best params into a dictionary
    outdict = ferre_outdict(
        pixel, targs_info, targs_setups, grid_prefix=grid_prefix, path=path,
        output_error_by_id=output_error_by_id,
        invalid_mask_by_id=invalid_mask_by_id,
    )

    ## GENERATE PLOTS
    if figdir is not None:
        try:
            make_fr_plot(outdict, figdir, targs_setups, units_str=units_str)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            print("".join(lines))
            sys.stdout.flush()
            print('Failed to generate plots. 1- Check your X11 configuration. 2- Check spectra in output fits file.')
            pass

    ## Write outputs into fits file with merge metadata
    ferre_write_fits(outdict, targs_infiles, targs_setups, targs_join_arms, units_str,
                    param_fits, spec_file=spec_fits, match_table=match_table,
                    merge_metadata=merge_result, linemask=linemask, uselinemasks=uselinemasks)



#################################################################################################

def ferre_weave(options=None):

    # Shared registry -- see aps_common_args.py's own module docstring.
    # `overrides` reproduces this script's own current --help wording
    # verbatim wherever it genuinely differs from the shared default
    # (including the "APS_IDS"/"wavelenght" typos, left as-is here).
    parser = build_common_parser(
        description="RUN FERRE for WEAVE target spectra.",
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing", "caldirs", "output"],
        overrides={
            "mask_aps_ids": {"help": "comma-separated list of APS_IDS to be masked"},
            "wlranges": {"help": "wavelenght range array for each elements of the setup"},
            "vacuum": {"default": False},
            "outpath": {"help": "Directory to keep PyFERRE outputs"},
            "headname": {"help": "Output headname. The output filenames will be generated based on this headname"},
        },
        extra_args=[
            (("--classfile",), dict(type=none_or_str, default=None, required=True,
                                     help="The input fits file, contains the classification table")),
            (("--grid_prefix",), dict(type=none_or_str, default='n', required=False,
                                       help="Gird prefix, n:3D grids, m: 5D grids, p: PCA compressed 5D grids")),
            (("--rvsfile",), dict(type=none_or_str, default=None, required=False,
                                   help="The input fits file, contains the Radial velocities table")),
            (("--templates",), dict(type=none_or_str, default=None, required=True,
                                     help="The full path of templates for PyFERRE")),
            (("--grid_ids",), dict(type=none_or_str, default=None, required=False,
                                    help="comma-separated list of grid IDs")),
            (("--outspec",), dict(type=str2bool, default=False, required=False,
                                   help="if True, the code return the spectra and best fitted model [FITS file]")),
            (("--split_arms",), dict(type=str2bool, default=False, required=False,
                                      help="cut joined arms based on the mean wavelength at the overlap region")),
            (("--mp",), dict(type=int, default=1, required=False,
                              help="The number of threads to run original FERRE CODE in Fortran")),
            (("--ferre_exe",), dict(type=none_or_str, default=None, required=True,
                                     help="The full path of the FERRE executable file")),
            (("--fig",), dict(type=str2bool, default=False, required=False,
                               help="if True, the code also produces plots of the best fitted model")),
            (("--linemask",), dict(type=none_or_str, default=None, required=False,
                                    help="Path to FERRE linemask file")),
            (("--maskbalmer",), dict(type=str2bool, default=None, required=False,
                                      help="Mask wavelength regions inside Balmer bins (H-alpha, H-beta, H-gamma)")),
            (("--uselinemasks",), dict(type=str2bool, default=None, required=False,
                                        help="Use linemask when determining stellar parameters")),
        ],
    )

    ## Check if any command-line argument has been passed to the module
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')

        args = parser.parse_args(options)

    ### Now we have both infiles and wlranges array. We use resolve_common_args
    ### (aps_common_args.py) to update infiles/wlranges/arms_ratio/join_arms
    ### and put them in the right order, if needed -- see that function's
    ### own docstring. However, we had similar test done by APSOB.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio

    # split_arms has no equivalent in any other script, so it stays local
    # rather than living in the shared resolve function -- same <2-infiles
    # correction join_arms itself already got inside resolve_common_args.
    if len(args.infiles) < 2:
        args.split_arms = False

    if args.classfile is None:
        raise Exception('You need to specify the CLASS/Z file to proceed')
    classfile = args.classfile

    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
    area, mask_areas = resolved.area, resolved.mask_areas


    grid_ids = None
    if args.grid_ids is not None:
        grid_ids = [ str(x).replace(" ","") for x in args.grid_ids.split(",") ]
        args.grid_ids=grid_ids

    use_rvs=True
    if args.rvsfile is None:
        use_rvs=False


    if args.grid_prefix not in ['m','n', 'p']:
        grid_prefix = 'n'
        print('No valid grid_prefix keyword found. We automatically set it to n: 3D grids')
    else:
        grid_prefix = args.grid_prefix


    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath = args.outpath.replace(' ', '')

    ## workpath is where the code keeps all intermediate files and does all housekeeping.
    ## CRITICAL FIX: Make it unique per run using headname to avoid collisions when running parallel jobs
    safe_headname = str(args.headname).replace(' ', '').replace('/', '_').replace('\\', '_')
    workpath = os.path.join(outpath, 'fr_wd_' + safe_headname) + '/'

    if not os.path.exists(workpath):
        os.makedirs(workpath)
        print("WORKPATH: %s Created!" %(workpath))
    elif args.overwrite:
        print("WORKPATH: %s exists. Removing and recreating (overwrite=True)..." %(workpath))
        shutil.rmtree(workpath)
        os.makedirs(workpath)
    else:
        print("WORKPATH: %s exists. Using existing directory (overwrite=False)..." %(workpath))


    param_fits=os.path.join(args.outpath,'ferre_'+str(args.headname)+'.fits').replace(' ', '')


    spec_fits=None
    if args.outspec:
        spec_fits=os.path.join(args.outpath,'ferre_spec_'+str(args.headname)+'.fits').replace(' ', '')



    if not os.path.isfile(args.ferre_exe):
        sys.exit('No FERRE executable file found')


    if not os.path.exists(args.templates):
        sys.exit('TEMPLATES directory: %s does not exist'%(args.templates))

    figdir=None
    if args.fig:
        figdir=args.outpath + '/figs/'
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            print("FIGDIR: %s Created!" %(figdir))
        figdir=figdir+'/'
        figdir=figdir.replace(' ', '')

    # print args and assigned/default values on the screen
    print_args(args,module='FR', version= aps_constants.__aps_ferre_version__ , path=outpath, headname=args.headname)

    ## RUN/TRIGGER THE MAIN MODULE

    proc_ferre(args.infiles, classfile, param_fits, aps_ids=aps_ids, targsrvy= targsrvy, targclass = targclass,
        mask_aps_ids = mask_aps_ids, area = area, mask_areas= mask_areas, figdir=figdir, wlranges=wlranges, use_rvs=use_rvs, rvsfile=args.rvsfile,
        path=workpath, outpath=outpath, spec_fits =spec_fits, pixel=args.headname, templates=args.templates, grid_prefix= grid_prefix,
        grid_ids=grid_ids, outspec=args.outspec, sens_corr=args.sens_corr, mask_gaps = args.mask_gaps,safe_mask_gaps = args.safe_mask_gaps,
        tellurics=args.tellurics, vacuum=args.vacuum, fill_gap=args.fill_gap, arms_ratio=arms_ratio,
        join_arms=args.join_arms, split_arms=args.split_arms, nthreads=args.mp, ferre=args.ferre_exe, match_table=None,
        catdir=args.catdir, caldir=args.caldir, configdir=args.configdir, maskbalmer=args.maskbalmer,
        linemask=args.linemask, uselinemasks=args.uselinemasks)


#################################################################################################

if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    MOS_demo = ['--infiles', '<PYAPS_DATA>/<night>/<obid>/stack_<runid>.fit', '<PYAPS_DATA>/<night>/<obid>/stack_<runid>.fit',
    '--classfile' , '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/zbest_stack_<runid>__stack_<runid>.fits',
    '--rvsfile' , '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/rvs_stack_<runid>__stack_<runid>.fits',
    '--templates' , '<PYAPS_DIR>/PyAPS_templates/templates_FR/',
    '--outpath', '<PYAPS_DIR>/PyAPS_results/<night>/3133_2/',
    '--aps_ids', '1006,1007', # or 'None' to run for all available fibreids
    '--grid_prefix' , 'n', # other options: 'm': 5D grids, 'p':PCA compressed 5D grids
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area', 'None',
    '--mask_areas', 'None',
    '--headname', 'stack_<runid>__stack_<runid>',
    '--wlranges', 'None',
    '--outspec' , 'True',
    '--overwrite', 'True',
    '--sens_corr' , 'True',
    '--mp' ,'3',
    '--ferre_exe' , '<PYAPS_DIR>/externals/ferre/bin/ferre.x',
    '--grid_ids' , '1,2,3,4,5,6,7,8,9',
    '--safe_mask_gaps', 'True',
    '--mask_gaps', 'True',
    '--tellurics', 'False',
    '--vacuum', 'False',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0, 0.83',
    '--join_arms', 'False',
    '--split_arms', 'False',
    '--fig', 'True',
    '--caldir', '<PYAPS_DATA>/CAL',
    '--catdir', '<PYAPS_DATA>/CAT',
    '--configdir', '<PYAPS_DIR>/configs/ExGal_configs',
    '--maskbalmer', 'None'
    ]



    ## If no command-line argument has been passed to this module, use the debug list as input and runs in the DEMO/DEBUG mode!
    ferre_weave(options=MOS_demo)

#################################################################################################
