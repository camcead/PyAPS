
import os
from collections import OrderedDict
from astropy.table import Table, Column, vstack
from astropy.io import fits
import numpy as np

###################################################################################
def emifile(infits, extnum):
    """
    Return emission line config table from the fits data table.
    UPDATED: Reads from data table columns instead of truncated GLNAM headers.

    This function now extracts emission line information directly from the data
    table columns (FLUX_*, V_*, SIGMA_*, etc.) rather than relying on the
    GLNAM header keywords which are limited to 40 lines.

    History: By A. Molaeinezhad (4 March 2020)
    Updated: For IFU emi kinematics - reads from data table (2025)

    Parameters:
    -----------
    infits : str
        Path to FITS file
    extnum : int
        Extension number containing the data table

    Returns:
    --------
    emmi_table : astropy.table.Table
        Table with emission line setup information
    """

    print("=" * 70)
    print("EXTRACTING EMISSION LINE INFO FROM DATA TABLE (not header)")
    print("=" * 70)

    # Read header for summary statistics
    hdr_gand = fits.getheader(infits, extnum)

    # Get summary info from header (these are still available)
    ngllines = int(hdr_gand.get('N_LINES_ALL', 0))
    n_fitted = int(hdr_gand.get('N_FITTED', 0))
    n_masked = int(hdr_gand.get('N_MASKED', 0))
    n_out_range = int(hdr_gand.get('N_OUT_RANGE', 0))

    print(f"Header summary: {ngllines} total lines")
    print(f"  - Fitted: {n_fitted}, Masked: {n_masked}, Out-of-range: {n_out_range}")

    # Read the actual data table
    try:
        data_table = Table.read(infits, extnum)
    except:
        # Try different extension
        try:
            data_table = Table.read(infits, extnum-1)
        except:
            data_table = Table.read(infits, extnum+1)

    print(f"Reading emission line info from data table with {len(data_table.colnames)} columns...")

    # Extract emission line names from FLUX_* columns
    line_names = []
    for col_name in data_table.colnames:
        if col_name.startswith('FLUX_'):
            line_name = col_name[5:]  # Remove 'FLUX_' prefix
            line_names.append(line_name)

    print(f"Found {len(line_names)} emission lines in data table")

    # Enhanced structure for emission table
    columns = ['LINE_ID', 'i', 'name', '_lambda', 'action', 'kind', 'a', 'v', 's', 'fit', 'aon', 'status']

    ### GENERATE OUTPUT TABLE STRUCTURE
    emmi_table = OrderedDict()
    for c in columns:
        emmi_table[c] = []

    # Process each emission line found in the data
    for i_emi, line_name in enumerate(line_names):
        try:
            # Extract wavelength from line name (format: LineID_wavelength)
            # e.g., "Halpha_6562.80" -> wavelength = 6562.80
            if '_' in line_name:
                parts = line_name.rsplit('_', 1)  # Split from right, max 1 split
                try:
                    line_lambda = float(parts[-1])
                    line_id_str = parts[0]
                except:
                    # If wavelength parsing fails, use index
                    line_lambda = 0.0
                    line_id_str = line_name
            else:
                line_lambda = 0.0
                line_id_str = line_name

            # Check if line has valid flux data in at least one spaxel/bin
            flux_col_name = f'FLUX_{line_name}'
            has_valid_flux = False

            if flux_col_name in data_table.colnames:
                flux_data = np.array(data_table[flux_col_name])
                # Check if any spaxel has non-zero, finite flux
                has_valid_flux = np.any(np.isfinite(flux_data) & (flux_data != 0))

            # Determine action and status based on data availability
            if has_valid_flux:
                line_action = 'f'  # fitted
                line_fit = 'True'
                line_status = 'FITTED'
            else:
                line_action = 'o'  # out of range or not fitted
                line_fit = 'False'
                line_status = 'UNKNOWN'

            # Try to read from header if available (for first 40 lines)
            # Otherwise use defaults
            line_kind = 'emission'  # Default
            line_a = 0.0
            line_v = 0.0
            line_s = 0.0
            line_aon = 0.0

            # Try to get from header if line index < 40
            if i_emi < 40:
                try:
                    if f'GLKIN_{i_emi}' in hdr_gand:
                        line_kind = str(hdr_gand[f'GLKIN_{i_emi}']).strip()
                    if f'GLA_{i_emi}' in hdr_gand:
                        line_a = float(hdr_gand[f'GLA_{i_emi}'])
                    if f'GLV_{i_emi}' in hdr_gand:
                        line_v = float(hdr_gand[f'GLV_{i_emi}'])
                    if f'GLS_{i_emi}' in hdr_gand:
                        line_s = float(hdr_gand[f'GLS_{i_emi}'])
                    if f'GLAON_{i_emi}' in hdr_gand:
                        line_aon = float(hdr_gand[f'GLAON_{i_emi}'])
                except:
                    pass  # Use defaults if header reading fails

            # Append all fields
            emmi_table['LINE_ID'].append(i_emi)
            emmi_table['i'].append(i_emi)
            emmi_table['name'].append(line_name)
            emmi_table['_lambda'].append(line_lambda)
            emmi_table['action'].append(line_action)
            emmi_table['kind'].append(line_kind)
            emmi_table['a'].append(line_a)
            emmi_table['v'].append(line_v)
            emmi_table['s'].append(line_s)
            emmi_table['fit'].append(line_fit)
            emmi_table['aon'].append(line_aon)
            emmi_table['status'].append(line_status)

        except Exception as e:
            print(f"Warning: Error processing line {i_emi} ({line_name}): {e}")
            # Fill with default values
            emmi_table['LINE_ID'].append(i_emi)
            emmi_table['i'].append(i_emi)
            emmi_table['name'].append(line_name)
            emmi_table['_lambda'].append(0.0)
            emmi_table['action'].append('o')
            emmi_table['kind'].append('emission')
            emmi_table['a'].append(0.0)
            emmi_table['v'].append(0.0)
            emmi_table['s'].append(0.0)
            emmi_table['fit'].append('False')
            emmi_table['aon'].append(0.0)
            emmi_table['status'].append('UNKNOWN')

    result_table = Table(emmi_table)

    print(f"Successfully extracted info for {len(result_table)} emission lines from data")
    print(f"  - Lines with valid flux data: {sum([s == 'FITTED' for s in emmi_table['status']])}")
    print(f"  - Lines without data: {sum([s == 'UNKNOWN' for s in emmi_table['status']])}")
    print("=" * 70)

    return result_table


###################################################################################
def lsnames(infits, extnum):
    """
    Return line strength index names from the fits header
    UPDATED: Enhanced error handling for missing indices

    History: By A. Molaeinezhad (5 March 2020)
    Updated: Enhanced error handling (2025)
    """
    hdr_ls = fits.getheader(infits, extnum)
    nlslines = int(hdr_ls['NLS']) # Number of indices, used by LS

    ### GENERATE OUTPUT TABLE STRUCTURE
    columns = ['i','name']
    ls_table = OrderedDict()

    for c in columns:
        ls_table[c] = []

    for i_ls in range(nlslines):
        try:
            ls_table['i'].append(int(i_ls))
            ls_table['name'].append(str(hdr_ls['LS_%s' %(i_ls)]).strip())
        except KeyError:
            print(f"Warning: Missing LS header key for index {i_ls}")
            ls_table['i'].append(i_ls)
            ls_table['name'].append(f'UNKNOWN_{i_ls}')

    return Table(ls_table)

######################################################################################
def gand_array(key, infits, extnum):
    """
    Return the results from the gandalf, based on a key from the emission lines table
    UPDATED: Enhanced to handle new IFU emi kinematics format with per-line kinematics

    NEW: Supports both legacy format and new per-line kinematics format
    NEW: Handles fitted, masked, and out-of-range lines properly

    e.g. gandalf_Flux = gand_array('FLUX', 'galaxy_emippxf.fits', 1)
         gandalf_V_per_line = gand_array('V', 'galaxy_emippxf.fits', 1)  # Per-line velocities

    History: By A. Molaeinezhad (4 March 2020)
    Updated: For IFU per-line kinematics (2025)
    """
    # Prepare key
    assert isinstance(key, str), 'key must be str'
    key = key.upper().replace(" ","")

    # Get emission line setup and data table
    emission_table = emifile(infits, extnum)

    # Try to read the data table (could be in different extensions)
    try:
        gandalf_table = Table.read(infits, extnum)
    except:
        # Try different extension if main one fails
        try:
            gandalf_table = Table.read(infits, extnum-1)
        except:
            gandalf_table = Table.read(infits, extnum+1)

    n_lines = len(emission_table)
    n_bins = len(gandalf_table['BIN_ID'])

    # Initialize output array with NaN (for non-fitted lines)
    out_table = np.full((n_bins, n_lines), np.nan)

    # NEW: Track which lines were successfully loaded
    successful_lines = 0
    failed_lines = []

    for i_line in range(n_lines):
        line_name = emission_table['name'][i_line]
        line_wavelength = emission_table['_lambda'][i_line]
        line_action = emission_table['action'][i_line]

        # FIX: Handle Table vs dict access properly
        if hasattr(emission_table, 'get'):
            line_status = emission_table.get('status', ['UNKNOWN'] * n_lines)[i_line]
        else:
            # For Astropy Table, check if column exists
            if 'status' in emission_table.colnames:
                line_status = emission_table['status'][i_line]
            else:
                line_status = 'UNKNOWN'

        # NEW: Handle different column naming conventions
        possible_column_names = [
            f'{key}_{line_name}',                                    # New IFU format
            f'{key}_{line_name}_{line_wavelength:.2f}',             # Legacy format with wavelength
            f'{key}_{line_name}_{line_wavelength:.1f}',             # Legacy format alternative
            f'{key}_{line_name.replace("[", "").replace("]", "")}', # Bracket-free version
        ]

        column_found = False
        for col_name in possible_column_names:
            if col_name in gandalf_table.colnames:
                try:
                    out_table[:, i_line] = np.array(gandalf_table[col_name])
                    successful_lines += 1
                    column_found = True
                    break
                except Exception as e:
                    print(f"Warning: Error reading column {col_name}: {e}")
                    continue

        if not column_found:
            # Only report as failed if line was expected to be fitted
            if line_action == 'f' or line_status == 'FITTED':
                failed_lines.append(f"{line_name} ({line_status})")
            # For masked/out-of-range lines, NaN is expected and correct

    # Report loading results
    fitted_lines = len([l for l in emission_table if l['action'] == 'f'])
    print(f"Loading {key} data: {successful_lines}/{fitted_lines} fitted lines loaded successfully")

    if failed_lines:
        print(f"  Failed to load {len(failed_lines)} fitted lines: {failed_lines[:5]}{'...' if len(failed_lines) > 5 else ''}")

    return out_table

######################################################################################

def load_emippxf_ifu(infits, extnum=1):
    """
    NEW FUNCTION: Load IFU emi kinematics results with per-line kinematics support.

    This function loads the enhanced emi kinematics output from the new IFU pipeline,
    including per-line velocities and velocity dispersions for each emission line.

    Parameters:
    -----------
    infits : str
        Path to the emi kinematics FITS file (e.g., 'galaxy_emippxf.fits')
    extnum : int, optional
        Extension number containing the main data table (default: 1)

    Returns:
    --------
    emippxf_result : dict
        Dictionary containing all emi kinematics data:
        {
            'emission_table': Table with line information,
            'velocities_per_line': array[n_bins, n_lines] - V for each line,
            'sigmas_per_line': array[n_bins, n_lines] - σ for each line,
            'fluxes': array[n_bins, n_lines] - Flux for each line,
            'amplitudes': array[n_bins, n_lines] - Amplitude for each line,
            'aon': array[n_bins, n_lines] - AON for each line,
            'velocity_errors': array[n_bins, n_lines] - V errors,
            'sigma_errors': array[n_bins, n_lines] - σ errors,
            'flux_errors': array[n_bins, n_lines] - Flux errors,
            'bin_ids': array[n_bins] - Bin IDs,
            'line_categories': dict - Categorization of lines,
            'redshift_info': dict - Redshift information if available
        }
    """

    print(f"Loading IFU emi kinematics from: {infits}")

    # Load emission line setup
    emission_table = emifile(infits, extnum)

    # Load main data table
    try:
        data_table = Table.read(infits, extnum)
    except:
        # Try reading FITS table directly if astropy Table fails
        data_table = fits.open(infits)[extnum].data

    n_lines = len(emission_table)
    n_bins = len(data_table['BIN_ID']) if 'BIN_ID' in data_table.colnames else len(data_table)

    print(f"  Lines: {n_lines}, Bins: {n_bins}")

    # Initialize result dictionary
    emippxf_result = {
        'emission_table': emission_table,
        'bin_ids': np.array(data_table['BIN_ID']) if 'BIN_ID' in data_table.colnames else np.arange(n_bins),
        'velocities_per_line': np.full((n_bins, n_lines), np.nan),
        'sigmas_per_line': np.full((n_bins, n_lines), np.nan),
        'fluxes': np.full((n_bins, n_lines), np.nan),
        'amplitudes': np.full((n_bins, n_lines), np.nan),
        'aon': np.full((n_bins, n_lines), np.nan),
        'velocity_errors': np.full((n_bins, n_lines), np.nan),
        'sigma_errors': np.full((n_bins, n_lines), np.nan),
        'flux_errors': np.full((n_bins, n_lines), np.nan),
        'amplitude_errors': np.full((n_bins, n_lines), np.nan),
        'line_categories': {},
        'redshift_info': {}
    }

    # Load per-line kinematics (NEW IFU FEATURE)
    print("  Loading per-line kinematics...")
    emippxf_result['velocities_per_line'] = gand_array('V', infits, extnum)
    emippxf_result['sigmas_per_line'] = gand_array('SIGMA', infits, extnum)

    # Load emission line properties
    print("  Loading emission line properties...")
    emippxf_result['fluxes'] = gand_array('FLUX', infits, extnum)
    emippxf_result['amplitudes'] = gand_array('AMPL', infits, extnum)
    emippxf_result['aon'] = gand_array('AON', infits, extnum)

    # Load errors
    print("  Loading error estimates...")
    emippxf_result['velocity_errors'] = gand_array('ERR_V', infits, extnum)
    emippxf_result['sigma_errors'] = gand_array('ERR_SIGMA', infits, extnum)
    emippxf_result['flux_errors'] = gand_array('ERR_FLUX', infits, extnum)
    emippxf_result['amplitude_errors'] = gand_array('ERR_AMPL', infits, extnum)

    # Categorize lines by status
    line_categories = {
        'fitted': [],
        'masked': [],
        'out_of_range': [],
        'no_wavelength': [],
        'all': []
    }

    for i, line in enumerate(emission_table):
        line_name = line['name']
        line_action = line['action']

        # FIX: Handle Table vs dict access properly
        if 'status' in emission_table.colnames:
            line_status = line['status']
        else:
            line_status = 'UNKNOWN'

        line_categories['all'].append(line_name)

        if line_action == 'f' or line_status == 'FITTED':
            line_categories['fitted'].append(line_name)
        elif line_action == 'm' or line_status == 'MASKED':
            line_categories['masked'].append(line_name)
        elif line_action == 'o' or line_status == 'OUT_OF_RANGE':
            line_categories['out_of_range'].append(line_name)
        elif line_action == 'n' or line_status == 'NO_WAVELENGTH':
            line_categories['no_wavelength'].append(line_name)

    emippxf_result['line_categories'] = line_categories

    # Load redshift information if available
    if 'Z_INPUT' in data_table.colnames:
        emippxf_result['redshift_info']['z'] = np.array(data_table['Z_INPUT'])
        if 'ERR_Z_INPUT' in data_table.colnames:
            emippxf_result['redshift_info']['z_err'] = np.array(data_table['ERR_Z_INPUT'])

    # Summary statistics
    valid_kinematics = np.sum(np.isfinite(emippxf_result['velocities_per_line']), axis=0)
    print(f"  Per-line kinematics: {np.sum(valid_kinematics > 0)}/{n_lines} lines have valid measurements")
    print(f"  Line categories: Fitted={len(line_categories['fitted'])}, "
          f"Masked={len(line_categories['masked'])}, "
          f"Out-of-range={len(line_categories['out_of_range'])}")

    return emippxf_result

######################################################################################

def get_emission_line_names_ifu(infits, extnum=1, category='fitted'):
    """
    NEW FUNCTION: Get emission line names by category from IFU emi kinematics results.

    Parameters:
    -----------
    infits : str
        Path to the emi kinematics FITS file
    extnum : int, optional
        Extension number (default: 1)
    category : str, optional
        Which lines to return: 'fitted', 'masked', 'out_of_range', 'all' (default: 'fitted')

    Returns:
    --------
    line_names : list
        List of emission line names in the specified category
    """
    emippxf_result = load_emippxf_ifu(infits, extnum)
    return emippxf_result['line_categories'].get(category, [])

######################################################################################

def detect_file_format(dirprefix):
    """
    NEW FUNCTION: Detect whether we're dealing with legacy or new IFU format files.

    Parameters:
    -----------
    dirprefix : str
        Directory prefix for the analysis files

    Returns:
    --------
    format_info : dict
        Dictionary containing format detection results
    """
    format_info = {
        'is_ifu': False,
        'has_emippxf': False,
        'has_per_line_kinematics': False,
        'gandalf_format': 'legacy',  # 'legacy' or 'ifu'
        'available_files': []
    }

    # Check for new IFU emi kinematics files
    emi_files = [
        dirprefix + '_emippxf.fits',
        dirprefix + '_emippxfSpec.fits',
        dirprefix + '_emippxf_mapping.fits'
    ]

    ifu_emi_files_found = [f for f in emi_files if os.path.isfile(f)]

    if ifu_emi_files_found:
        format_info['is_ifu'] = True
        format_info['has_emippxf'] = True
        format_info['available_files'].extend(ifu_emi_files_found)

        # Check if per-line kinematics are available
        try:
            hdr = fits.getheader(emi_files[0], 1)
            if 'PER_LINE_KIN' in hdr and hdr['PER_LINE_KIN']:
                format_info['has_per_line_kinematics'] = True
                format_info['gandalf_format'] = 'ifu'
        except:
            pass

    # Check for GANDALF files
    gandalf_files = [
        dirprefix + '_gandalf_BIN.fits',
        dirprefix + '_gandalf_SPAXEL.fits',
        dirprefix + '_gandalf_spec_BIN.fits',
        dirprefix + '_gandalf_spec_SPAXEL.fits'
    ]

    gandalf_files_found = [f for f in gandalf_files if os.path.isfile(f)]
    format_info['available_files'].extend(gandalf_files_found)

    # Check for enhanced IFU GANDALF format
    for gfile in gandalf_files_found:
        try:
            hdr = fits.getheader(gfile, 1)
            if 'N_LINES_ALL' in hdr or 'PER_LINE_KIN' in hdr:
                format_info['gandalf_format'] = 'ifu'
                break
        except:
            continue

    return format_info

######################################################################################
def loadData(self):
    """
    Load all available data from _APS.fits file.
    NEW: Only supports PPXF format with per-line kinematics.
    """

    try:
        # Clean all figures
        for i in range(len(self.axes)):
            self.axes[i].cla()
        self.cax.cla()
        try:
            self.cax3.cla()
            self.cax4.cla()
        except:
            pass
        self.canvas.draw()
    except:
        pass

    # Check if APS file exists
    APS_FILE = self.dirprefix + '_APS.fits'
    if not os.path.isfile(APS_FILE):
        self.dialogNotAvailable("directory")
        return None

    print('MAPVIEWER WORKING MODE: USING FINAL _APS.fits file')

    # ==============================
    # Open APS file and read main tables
    # ==============================
    APS_FILE_HDU = fits.open(APS_FILE)

    # Read GALAXY_SPEC (spectral data)
    APS_GAL_SPEC, APS_GAL_SPEC_HEAD = fits.getdata(APS_FILE, extname='GALAXY_SPEC', header=True)

    # Read GALAXY_TABLE (results table with emission line info in header)
    APS_GAL_TABLE, APS_GAL_TABLE_HEAD = fits.getdata(APS_FILE, extname='GALAXY_TABLE', header=True)

    # ==============================
    # Check for PPXF stellar kinematics
    # ==============================
    if 'FLUX_PPXF' in APS_GAL_SPEC.columns.names:
        self.PPXF = True
        print("  ✓ PPXF stellar kinematics found")
    else:
        self.PPXF = False
        print("  ✗ No PPXF stellar kinematics")

    # ==============================
    # Check for emission line data (PPXF format only)
    # ==============================
    if 'N_LINES_ALL' in APS_GAL_TABLE_HEAD:
        self.EMIPPXF = True
        print("  ✓ PPXF emission line data with per-line kinematics found")
        print(f"    Total lines: {APS_GAL_TABLE_HEAD['N_LINES_ALL']}")
        print(f"    Fitted lines: {APS_GAL_TABLE_HEAD.get('N_FITTED', 'unknown')}")
    else:
        self.EMIPPXF = False
        print("  ✗ No PPXF emission line data found")

    # ==============================
    # Check for Line Strength indices
    # ==============================
    if 'LS_RES' in APS_GAL_TABLE_HEAD:
        self.LINE_STRENGTH = True
        self.LsLevel = APS_GAL_TABLE_HEAD['LS_RES']
        print(f"  ✓ Line Strength indices found ({self.LsLevel})")
    else:
        self.LINE_STRENGTH = False
        print("  ✗ No Line Strength indices")

    # SFH not supported in new format
    self.SFH = False

    # ==============================
    # Read spatial table (PATCH_TABLE)
    # ==============================
    self.table = fits.open(APS_FILE)['PATCH_TABLE'].data
    self.table.X = self.table.X * -1 * np.cos(np.deg2rad(self.table.Y_0))
    self.table.XBIN = self.table.XBIN * -1 * np.cos(np.deg2rad(self.table.Y_0))

    try:
        self.pixelsize = fits.open(APS_FILE)['PATCH_TABLE'].header['PIXSIZE']
    except:
        print("  Warning: 'PIXSIZE' not found. Using default 0.20")
        self.pixelsize = 0.20

    _, idxConvertShortToLong = np.unique(np.abs(self.table.BIN_ID), return_inverse=True)

    # ==============================
    # Read binned spectra (PATCH_BINSPEC)
    # ==============================
    APS_PATCH_BINSPEC, APS_PATCH_BINSPEC_HEAD = fits.getdata(APS_FILE, extname='PATCH_BINSPEC', header=True)
    self.Spectra = APS_PATCH_BINSPEC.SPEC
    nbins = self.Spectra.shape[0]
    npix = self.Spectra.shape[1]
    self.Lambda = np.array(APS_PATCH_BINSPEC.LOGLAM)[0,:]

    print(f"  Loaded {nbins} binned spectra with {npix} pixels")

    # ==============================
    # Read PPXF stellar kinematics results
    # ==============================
    if self.PPXF:
        print("Loading PPXF stellar kinematics...")
        ppxf = np.array([
            APS_GAL_TABLE.V,
            APS_GAL_TABLE.SIGMA,
            APS_GAL_TABLE.H3,
            APS_GAL_TABLE.H4,
            APS_GAL_TABLE.LAMBDA_R
        ]).T

        self.ppxf_results = ppxf[idxConvertShortToLong, :]
        self.ppxfBestfit = APS_GAL_SPEC.MODEL_PPXF
        self.ppxfLambda = APS_GAL_SPEC.LOGLAM_PPXF[0, :]
        self.ppxfGoodpix = APS_GAL_SPEC.GOODPIX_PPXF

        median_V_stellar = np.nanmedian(self.ppxf_results[:, 0])
        self.ppxf_results[:, 0] = self.ppxf_results[:, 0] - median_V_stellar

        print(f"  ✓ Loaded stellar kinematics (median V = {median_V_stellar:.1f} km/s)")
    else:
        median_V_stellar = 0.0

    # ==============================
    # Read PPXF emission line results (per-line kinematics)
    # ==============================
    if self.EMIPPXF:
        print("Loading PPXF emission line data...")

        # Load emission line setup from header
        emi_table = emifile(APS_FILE, APS_FILE_HDU.index_of('GALAXY_TABLE'))
        self.gandalf_setup = emi_table  # Keep name for compatibility

        # Extract line names
        line_names = [line['name'] for line in emi_table]
        n_lines = len(line_names)
        n_table_bins = len(APS_GAL_TABLE)

        print(f"  Processing {n_lines} emission lines for {n_table_bins} bins...")

        # Initialize arrays for per-line data
        vel_per_line = np.full((n_table_bins, n_lines), np.nan)
        sigma_per_line = np.full((n_table_bins, n_lines), np.nan)
        flux_per_line = np.full((n_table_bins, n_lines), np.nan)
        ampl_per_line = np.full((n_table_bins, n_lines), np.nan)
        aon_per_line = np.full((n_table_bins, n_lines), np.nan)

        # Load per-line data from table columns
        successful_lines = 0
        failed_lines = []

        for i, line_name in enumerate(line_names):
            line_action = emi_table['action'][i]
            line_lambda = emi_table['_lambda'][i]

            # Only load data for fitted lines
            if line_action == 'f':
                found_any = False

                # Try different wavelength precisions for matching
                # e.g., both "Halpha_6562.80" and "Halpha_6562.8"
                possible_names = [
                    line_name,  # Exact match
                    f"{line_name.split('_')[0]}_{line_lambda:.2f}",  # 2 decimals
                    f"{line_name.split('_')[0]}_{line_lambda:.1f}",  # 1 decimal
                    f"{line_name.split('_')[0]}_{line_lambda:.0f}",  # 0 decimals
                ]

                # Remove duplicates while preserving order
                possible_names = list(dict.fromkeys(possible_names))

                # Try each possible name
                for try_name in possible_names:
                    # Try to load velocity
                    v_col = f'V_{try_name}'
                    if v_col in APS_GAL_TABLE.columns.names:
                        vel_per_line[:, i] = APS_GAL_TABLE[v_col]
                        found_any = True

                    # Try to load sigma
                    s_col = f'SIGMA_{try_name}'
                    if s_col in APS_GAL_TABLE.columns.names:
                        sigma_per_line[:, i] = APS_GAL_TABLE[s_col]
                        found_any = True

                    # Try to load flux
                    f_col = f'FLUX_{try_name}'
                    if f_col in APS_GAL_TABLE.columns.names:
                        flux_per_line[:, i] = APS_GAL_TABLE[f_col]
                        found_any = True

                    # Try to load amplitude
                    a_col = f'AMPL_{try_name}'
                    if a_col in APS_GAL_TABLE.columns.names:
                        ampl_per_line[:, i] = APS_GAL_TABLE[a_col]
                        found_any = True

                    # Try to load AON
                    aon_col = f'AON_{try_name}'
                    if aon_col in APS_GAL_TABLE.columns.names:
                        aon_per_line[:, i] = APS_GAL_TABLE[aon_col]
                        found_any = True

                    # If we found data with this name variant, stop trying
                    if found_any:
                        break

                if found_any:
                    successful_lines += 1
                else:
                    failed_lines.append(line_name)

        print(f"  ✓ Loaded {successful_lines} emission lines successfully")
        if failed_lines:
            print(f"  ⚠ Failed to load {len(failed_lines)} fitted lines:")
            for failed_line in failed_lines[:5]:
                print(f"    - {failed_line}")
            if len(failed_lines) > 5:
                print(f"    ... and {len(failed_lines) - 5} more")


        print(f"  ✓ Loaded {successful_lines} emission lines successfully")
        if failed_lines:
            print(f"  ⚠ Failed to load {len(failed_lines)} fitted lines (marked as fitted but no data in table):")
            for failed_line in failed_lines[:5]:
                print(f"    - {failed_line}")
            if len(failed_lines) > 5:
                print(f"    ... and {len(failed_lines) - 5} more")



        # Build results array: shape (n_bins, 5, n_lines)
        # Stack: [velocities, sigmas, fluxes, amplitudes, aon]
        emi_results = np.stack([
            vel_per_line,
            sigma_per_line,
            flux_per_line,
            ampl_per_line,
            aon_per_line
        ], axis=1)

        # Reorder to match spatial table
        self.gandalf_results = emi_results[idxConvertShortToLong, :, :]

        # Subtract median stellar velocity from emi velocities
        if self.PPXF:
            self.gandalf_results[:, 0, :] = self.gandalf_results[:, 0, :] - median_V_stellar

        # Load spectral fits
        if 'FLUX_CLEAN_GAND' in APS_GAL_SPEC.columns.names:
            self.EmissionSubtractedSpectraBIN = np.array(APS_GAL_SPEC.FLUX_CLEAN_GAND)
        if 'MODEL_GAND' in APS_GAL_SPEC.columns.names:
            self.gandalfBestfit = APS_GAL_SPEC.MODEL_GAND
        if 'LOGLAM_GAND' in APS_GAL_SPEC.columns.names:
            self.gandalfLambda = APS_GAL_SPEC.LOGLAM_GAND[0, :]
        if 'GOODPIX_GAND' in APS_GAL_SPEC.columns.names:
            self.gandalfGoodpix = APS_GAL_SPEC.GOODPIX_GAND

        # Create line name list for compatibility
        self.listOfLineNames = np.array(line_names)

        print(f"  Final emi data shape: {self.gandalf_results.shape}")

    else:
        # No emi kinematics available
        self.gandalf_results = None
        self.listOfLineNames = np.array([])

    # ==============================
    # Read Line Strength indices
    # ==============================
    if self.LINE_STRENGTH:
        print("Loading Line Strength indices...")

        # Find LS columns in the table
        ls_cols = [col for col in APS_GAL_TABLE.columns.names
                   if not col.startswith(('V_', 'SIGMA_', 'FLUX_', 'AMPL_', 'AON_', 'ERR_'))
                   and col not in ['BIN_ID', 'X', 'Y', 'SNR', 'H3', 'H4', 'LAMBDA_R']]

        if ls_cols:
            ls_data = np.array([APS_GAL_TABLE[col] for col in ls_cols]).T
            self.line_strength = ls_data[idxConvertShortToLong, :]
            self.lsList = np.array(ls_cols)
            print(f"  ✓ Loaded {len(ls_cols)} line strength indices")
        else:
            self.LINE_STRENGTH = False
            print("  ✗ No line strength columns found")

    # ==============================
    # Final setup
    # ==============================
    if not hasattr(self, 'gandalf_results') or self.gandalf_results is None:
        self.gandalf_results = None
    if not hasattr(self, 'listOfLineNames'):
        self.listOfLineNames = np.array([])

    print("\n✓ Data loading complete!")
    print(f"  Bins: {nbins}")
    print(f"  PPXF: {self.PPXF}")
    print(f"  emi lines: {len(self.listOfLineNames)}")
    print(f"  Line Strength: {self.LINE_STRENGTH}")
######################################################################################

def get_line_by_name_ifu(self, line_name, data_type='velocity'):
    """
    NEW FUNCTION: Get data for a specific emission line by name from IFU emi kinematics.

    Parameters:
    -----------
    line_name : str
        Name of the emission line (e.g., 'Halpha', '[OIII]_5006.77')
    data_type : str, optional
        Type of data to retrieve: 'velocity', 'sigma', 'flux', 'amplitude', 'aon',
        'velocity_error', 'sigma_error', 'flux_error' (default: 'velocity')

    Returns:
    --------
    data : array
        Data array for the specified line and data type, or None if not found
    line_info : dict
        Information about the line (wavelength, status, etc.)
    """

    if not hasattr(self, 'emippxf_result') and not hasattr(self, 'gandalf_results'):
        print("Error: No emi kinematics data loaded")
        return None, None

    # Find line index
    line_idx = None
    line_info = {}

    if hasattr(self, 'listOfLineNames'):
        try:
            line_idx = list(self.listOfLineNames).index(line_name)
        except ValueError:
            print(f"Line '{line_name}' not found in line list")
            return None, None

    if line_idx is None:
        return None, None

    # Get line information
    if hasattr(self, 'gandalf_setup'):
        line_info = {
            'name': self.gandalf_setup['name'][line_idx],
            'wavelength': self.gandalf_setup['_lambda'][line_idx],
            'action': self.gandalf_setup['action'][line_idx],
            'status': self.gandalf_setup['status'][line_idx] if 'status' in self.gandalf_setup.colnames else 'UNKNOWN'
        }

    # Get data based on type
    data = None

    if hasattr(self, 'emippxf_result'):
        # NEW IFU format with dedicated arrays
        data_map = {
            'velocity': self.emippxf_result['velocities_per_line'][:, line_idx],
            'sigma': self.emippxf_result['sigmas_per_line'][:, line_idx],
            'flux': self.emippxf_result['fluxes'][:, line_idx],
            'amplitude': self.emippxf_result['amplitudes'][:, line_idx],
            'aon': self.emippxf_result['aon'][:, line_idx],
            'velocity_error': self.emippxf_result['velocity_errors'][:, line_idx],
            'sigma_error': self.emippxf_result['sigma_errors'][:, line_idx],
            'flux_error': self.emippxf_result['flux_errors'][:, line_idx]
        }
        data = data_map.get(data_type)

    elif hasattr(self, 'gandalf_results') and self.gandalf_results is not None:
        # Legacy format or converted data
        data_map = {
            'velocity': self.gandalf_results[:, 0, line_idx],
            'sigma': self.gandalf_results[:, 1, line_idx],
            'flux': self.gandalf_results[:, 2, line_idx],
            'amplitude': self.gandalf_results[:, 3, line_idx],
            'aon': self.gandalf_results[:, 4, line_idx]
        }
        data = data_map.get(data_type)

        # Errors not available in legacy format
        if data_type.endswith('_error'):
            print(f"Warning: {data_type} not available in legacy format")
            return None, line_info

    if data is None:
        print(f"Error: Could not retrieve {data_type} for line {line_name}")
        return None, line_info

    return data, line_info

######################################################################################

def get_line_summary_ifu(self):
    """
    NEW FUNCTION: Get summary statistics for all emission lines from IFU emi kinematics.

    Returns:
    --------
    summary : dict
        Dictionary containing summary statistics for all lines
    """

    if not (hasattr(self, 'emippxf_result') or hasattr(self, 'gandalf_results')):
        print("Error: No emi kinematics data loaded")
        return None

    summary = {
        'line_names': [],
        'line_wavelengths': [],
        'line_status': [],
        'n_valid_velocity': [],
        'n_valid_flux': [],
        'median_velocity': [],
        'median_sigma': [],
        'velocity_range': [],
        'sigma_range': []
    }

    if hasattr(self, 'listOfLineNames'):
        for i, line_name in enumerate(self.listOfLineNames):
            summary['line_names'].append(line_name)

            # Get line info
            if hasattr(self, 'gandalf_setup'):
                summary['line_wavelengths'].append(self.gandalf_setup['_lambda'][i])
                summary['line_status'].append(self.gandalf_setup['action'][i])
            else:
                summary['line_wavelengths'].append(0.0)
                summary['line_status'].append('unknown')

            # Get velocity statistics
            if hasattr(self, 'emippxf_result'):
                velocities = self.emippxf_result['velocities_per_line'][:, i]
                sigmas = self.emippxf_result['sigmas_per_line'][:, i]
                fluxes = self.emippxf_result['fluxes'][:, i]
            elif hasattr(self, 'gandalf_results') and self.gandalf_results is not None:
                velocities = self.gandalf_results[:, 0, i]
                sigmas = self.gandalf_results[:, 1, i]
                fluxes = self.gandalf_results[:, 2, i]
            else:
                velocities = np.full(10, np.nan)
                sigmas = np.full(10, np.nan)
                fluxes = np.full(10, np.nan)

            # Calculate statistics
            valid_v = velocities[np.isfinite(velocities)]
            valid_s = sigmas[np.isfinite(sigmas)]
            valid_f = fluxes[np.isfinite(fluxes)]

            summary['n_valid_velocity'].append(len(valid_v))
            summary['n_valid_flux'].append(len(valid_f))

            if len(valid_v) > 0:
                summary['median_velocity'].append(np.median(valid_v))
                summary['velocity_range'].append(np.max(valid_v) - np.min(valid_v))
            else:
                summary['median_velocity'].append(np.nan)
                summary['velocity_range'].append(np.nan)

            if len(valid_s) > 0:
                summary['median_sigma'].append(np.median(valid_s))
                summary['sigma_range'].append(np.max(valid_s) - np.min(valid_s))
            else:
                summary['median_sigma'].append(np.nan)
                summary['sigma_range'].append(np.nan)

    return summary


######################################################################################

def migrate_plotting_data_to_ifu(self):
    """
    NEW FUNCTION: Helper function to migrate existing plotting code to use IFU format.
    Sets up backward-compatible data structures while enabling access to new features.
    """

    if not hasattr(self, 'emippxf_result'):
        print("No IFU emi data available for migration")
        return False

    print("Migrating plotting data structures to IFU format...")

    # Create backward-compatible gandalf_results if not already done
    if not hasattr(self, 'gandalf_results') or self.gandalf_results is None:
        n_bins = len(self.emippxf_result['bin_ids'])
        n_lines = len(self.emippxf_result['emission_table'])

        self.gandalf_results = np.zeros((n_bins, 5, n_lines))
        self.gandalf_results[:, 0, :] = self.emippxf_result['velocities_per_line']  # V
        self.gandalf_results[:, 1, :] = self.emippxf_result['sigmas_per_line']      # Sigma
        self.gandalf_results[:, 2, :] = self.emippxf_result['fluxes']               # Flux
        self.gandalf_results[:, 3, :] = self.emippxf_result['amplitudes']           # Amplitude
        self.gandalf_results[:, 4, :] = self.emippxf_result['aon']                  # AON

    # Set up emission line names for compatibility
    if not hasattr(self, 'listOfLineNames'):
        self.listOfLineNames = np.array([line['name'] for line in self.emippxf_result['emission_table']])

    # Set up emission setup table for compatibility
    if not hasattr(self, 'gandalf_setup'):
        self.gandalf_setup = self.emippxf_result['emission_table']

    # Add new IFU-specific attributes for advanced plotting
    self.ifu_line_categories = self.emippxf_result['line_categories']
    self.ifu_velocity_errors = self.emippxf_result['velocity_errors']
    self.ifu_sigma_errors = self.emippxf_result['sigma_errors']
    self.ifu_flux_errors = self.emippxf_result['flux_errors']

    print("  Migration completed - legacy plotting code should work")
    print("  New IFU features available via self.emippxf_result and self.ifu_* attributes")

    return True

######################################################################################

def validate_data_consistency_ifu(self):
    """
    NEW FUNCTION: Validate consistency between different data products in IFU format.

    Returns:
    --------
    validation_report : dict
        Dictionary containing validation results and any issues found
    """

    validation_report = {
        'overall_status': 'UNKNOWN',
        'issues': [],
        'warnings': [],
        'recommendations': [],
        'data_shapes': {},
        'missing_data': []
    }

    try:
        # Check basic data availability
        if hasattr(self, 'emippxf_result'):
            emippxf_result = self.emippxf_result
            n_bins = len(emippxf_result['bin_ids'])
            n_lines = len(emippxf_result['emission_table'])

            validation_report['data_shapes']['emi_bins'] = n_bins
            validation_report['data_shapes']['emi_lines'] = n_lines

            # Check array shapes
            expected_shape = (n_bins, n_lines)

            arrays_to_check = {
                'velocities_per_line': emippxf_result['velocities_per_line'],
                'sigmas_per_line': emippxf_result['sigmas_per_line'],
                'fluxes': emippxf_result['fluxes'],
                'amplitudes': emippxf_result['amplitudes']
            }

            for name, array in arrays_to_check.items():
                if array.shape != expected_shape:
                    validation_report['issues'].append(
                        f"Shape mismatch in {name}: expected {expected_shape}, got {array.shape}"
                    )

        # Check PPXF consistency
        if hasattr(self, 'ppxf_results') and hasattr(self, 'emippxf_result'):
            ppxf_bins = self.ppxf_results.shape[0]
            emi_bins = len(self.emippxf_result['bin_ids'])

            if ppxf_bins != emi_bins:
                validation_report['warnings'].append(
                    f"Bin count mismatch: PPXF has {ppxf_bins} bins, emi has {emi_bins} bins"
                )

        # Check for missing critical data
        if hasattr(self, 'emippxf_result'):
            # Check how many lines have valid measurements
            fitted_lines = self.emippxf_result['line_categories']['fitted']
            masked_lines = self.emippxf_result['line_categories']['masked']

            if len(fitted_lines) == 0:
                validation_report['issues'].append("No fitted emission lines found")
            elif len(fitted_lines) < 5:
                validation_report['warnings'].append(f"Only {len(fitted_lines)} fitted lines - limited analysis possible")

            if len(masked_lines) > len(fitted_lines):
                validation_report['warnings'].append(f"More masked ({len(masked_lines)}) than fitted ({len(fitted_lines)}) lines")

            # Check velocity consistency across lines
            if len(fitted_lines) > 1:
                velocities = self.emippxf_result['velocities_per_line']
                valid_velocities = velocities[np.isfinite(velocities)]

                if len(valid_velocities) > 0:
                    v_range = np.max(valid_velocities) - np.min(valid_velocities)
                    if v_range > 500:  # Very large velocity range
                        validation_report['warnings'].append(
                            f"Large velocity range across lines: {v_range:.1f} km/s - check for systematic issues"
                        )

        # Generate recommendations
        if len(validation_report['issues']) == 0 and len(validation_report['warnings']) == 0:
            validation_report['overall_status'] = 'GOOD'
            validation_report['recommendations'].append("Data appears consistent - ready for analysis")
        elif len(validation_report['issues']) == 0:
            validation_report['overall_status'] = 'WARNING'
            validation_report['recommendations'].append("Minor issues detected - proceed with caution")
        else:
            validation_report['overall_status'] = 'ERROR'
            validation_report['recommendations'].append("Serious issues detected - investigate before analysis")

        # Add format-specific recommendations
        if hasattr(self, 'format_info'):
            if self.format_info['has_per_line_kinematics']:
                validation_report['recommendations'].append("Per-line kinematics available - consider multi-component analysis")
            if self.format_info['gandalf_format'] == 'ifu':
                validation_report['recommendations'].append("Enhanced IFU format detected - full feature set available")

    except Exception as e:
        validation_report['overall_status'] = 'ERROR'
        validation_report['issues'].append(f"Validation failed: {str(e)}")

    return validation_report

######################################################################################

def print_data_summary_ifu(self):
    """
    NEW FUNCTION: Print a comprehensive summary of loaded IFU data for debugging.
    """

    print("\n" + "="*60)
    print("IFU DATA SUMMARY")
    print("="*60)

    # Format information
    if hasattr(self, 'format_info'):
        print(f"Format: {'IFU' if self.format_info['is_ifu'] else 'Legacy'}")
        print(f"Per-line kinematics: {self.format_info['has_per_line_kinematics']}")
        print(f"GANDALF format: {self.format_info['gandalf_format']}")

    # Basic data availability
    print(f"\nData modules available:")
    print(f"  PPXF (stellar): {getattr(self, 'PPXF', False)}")
    print(f"  emi kinematics: {getattr(self, 'EMIPPXF', False)}")
    print(f"  GANDALF (legacy): {getattr(self, 'GANDALF', False)}")
    print(f"  Line strength: {getattr(self, 'LINE_STRENGTH', False)}")
    print(f"  SFH: {getattr(self, 'SFH', False)}")

    # Spectral data
    if hasattr(self, 'Spectra'):
        print(f"\nSpectral data:")
        print(f"  Shape: {self.Spectra.shape} (bins × pixels)")
        print(f"  Wavelength range: {np.exp(self.Lambda[0]):.1f} - {np.exp(self.Lambda[-1]):.1f} Å")

    # emi kinematics details
    if hasattr(self, 'emippxf_result'):
        print(f"\nIFU emi kinematics:")
        print(f"  Total lines in config: {len(self.emippxf_result['emission_table'])}")
        print(f"  Fitted lines: {len(self.emippxf_result['line_categories']['fitted'])}")
        print(f"  Masked lines: {len(self.emippxf_result['line_categories']['masked'])}")
        print(f"  Out-of-range lines: {len(self.emippxf_result['line_categories']['out_of_range'])}")

        # Show some example lines
        fitted_lines = self.emippxf_result['line_categories']['fitted']
        if len(fitted_lines) > 0:
            print(f"  Example fitted lines: {fitted_lines[:5]}{'...' if len(fitted_lines) > 5 else ''}")

        masked_lines = self.emippxf_result['line_categories']['masked']
        if len(masked_lines) > 0:
            print(f"  Example masked lines: {masked_lines[:3]}{'...' if len(masked_lines) > 3 else ''}")

    elif hasattr(self, 'gandalf_results') and self.gandalf_results is not None:
        print(f"\nLegacy GANDALF:")
        print(f"  Shape: {self.gandalf_results.shape} (bins × properties × lines)")
        if hasattr(self, 'listOfLineNames'):
            print(f"  Lines: {len(self.listOfLineNames)}")
            print(f"  Example lines: {self.listOfLineNames[:5]}{'...' if len(self.listOfLineNames) > 5 else ''}")

    # PPXF details
    if hasattr(self, 'ppxf_results'):
        print(f"\nPPXF stellar kinematics:")
        print(f"  Shape: {self.ppxf_results.shape} (bins × moments)")
        v_range = np.nanmax(self.ppxf_results[:, 0]) - np.nanmin(self.ppxf_results[:, 0])
        s_range = np.nanmax(self.ppxf_results[:, 1]) - np.nanmin(self.ppxf_results[:, 1])
        print(f"  Velocity range: {v_range:.1f} km/s")
        print(f"  Sigma range: {s_range:.1f} km/s")

    # Line strength details
    if hasattr(self, 'line_strength'):
        print(f"\nLine strength:")
        print(f"  Shape: {self.line_strength.shape} (bins × indices)")
        if hasattr(self, 'lsList'):
            print(f"  Indices: {len(self.lsList)}")
            print(f"  Example indices: {self.lsList[:5]}{'...' if len(self.lsList) > 5 else ''}")

    print("="*60)

    # Run validation
    validation = validate_data_consistency_ifu(self)
    print(f"\nValidation status: {validation['overall_status']}")
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  ❌ {issue}")
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    if validation['recommendations']:
        print("Recommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")

    print("="*60 + "\n")
