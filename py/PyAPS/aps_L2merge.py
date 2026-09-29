import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import warnings
import argparse
import numpy as np
import re

from astropy.io import fits
from astropy.table import Table, hstack,join
import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class, l1_fileinfo, stack_level
from PyAPS import aps_constants
from functools import reduce
from datetime import datetime
import math
APSVERS = PyAPS.__version__

# the L2 datamodel we are using for this process
DATAMVER= '0.9'


#################################################
"""
aps_outs
A simple module to merge all PyAPS products into multi-ext fits files

versions:
 2.0 By A. Molaeinezhad (IAC, Jan 2020) - Original version
 3.0 By A. Molaeinezhad (IAC, Oct 2020)
 3.5 By A. Molaeinezhad (CASU, June 2021)
 4.0 By A. Molaeinezhad (CASU, Feb 2022)
 5.0 By A. Molaeinezhad (CASU, Oct 2022)
 5.1 By A. Molaeinezhad (CASU, Feb 2024)
 5.2 By A. Molaeinezhad (CASU, Feb 2024)
 5.3 By A. Molaeinezhad (CASU, Feb 2024)
 5.4 By A. Molaeinezhad (CASU, Apr 2024)
 5.5 By A. Molaeinezhad (CASU, Apr 2024)
 5.6 By A. Molaeinezhad (CASU, Apr 2024)
 5.7 By A. Molaeinezhad (CASU, June 2024)
 5.8 By A. Molaeinezhad (CASU, Nov 2024) - APS_ID enforced to int32

TODO LIST:
1- Complete headers based on the info coming from different files


########### BAISC APS PARAM ###################################################################################################
|    param                |    APS default  |  MOSL2MERGE EQUIVALENT       | MOSL2MERGE DEFAULT  |   available options/notes    |
|                         |                 |                              |                   |                               |
######### L2merge INITAIL PARAM ###############################################################################################
                          |                 |                              |                   |                               |
######### DEDICATED L2merge PARAM #############################################################################################
                          |                 |  --infiles   (Required)      |          -        |                               |
                          |                 |  --outpath   (Required)      |          -        |                               |
                          |                 |  --headname  (Optional)      |     'headname'    |                               |
                          |                 |  --wlranges  (Optional)      |                   |                               |
                          |                 |  --outfile_suffix (Optional) |      '_APS'       |                               |
                          |                 |  --EMIPPXF_LEVEL (Optional)  |        'BIN'      |                               |
                          |                 |  --LS_RES  (Optional)        |      'ADAPTED'    |                               |
################################################################################################################################
FOR IFU mode only
                          |                 |  --patch area  (Optional)     | [RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec], ANGLE [CCW in deg]]   |
                          |                 |  --patch_id (Optional)        |      '1'    |                                                             |




example:

MOS MODE:
python3 <PYAPS_DIR>/py/PyAPS/aps_L2merge.py --infiles <PYAPS_DATA>/gal_test/superstack_100001.fits <PYAPS_DATA>/gal_test/superstack_100000.fits --outpath <PYAPS_DIR>/PyAPS_results/20160903/3294/ --headname stacked_1002046__stacked_1002045 --wlranges None --outfile_suffix _APS --EMIPPXF_LEVEL BIN --LS_RES ADAPTED --mode MOS

IFU MODE:
python3 <PYAPS_DIR>/py/PyAPS/aps_L2merge.py --infiles <PYAPS_DATA>/gal_test/superstack_100001.fits <PYAPS_DATA>/gal_test/superstack_100000.fits --outpath <PYAPS_DIR>/PyAPS_results/20160903/3294/ --headname stacked_1002046__stacked_1002045 --wlranges None --outfile_suffix _APS --EMIPPXF_LEVEL BIN --LS_RES ADAPTED --mode IFU --patch_area 123.2,32.3,0.012,0.010,44.0 --patch_id 1


History:
14 Jan 2020: 1st version (By A. Molaeinezhad)
01 March 2020: Updated, based on the new L2 structure (By A. Molaeinezhad)
02 March 2020: Merge MOS and IFU functions (By A. Molaeinezhad)
20 Oct 2020: in-line documentation updated
20 June 2021: History, code header and examples updated
20 June 2021: new parameters added to the command-line arguments (infiles, wlranges, patch_area, patch_id)
20 June 2021: upon WAS request, new keywords added to the primary header (obsdate, obs_mode) and also patch parameters for the ifu mode
04 Feb 2022: A reserved 3-element list for reference L1 files (L1_ref_1, L1_ref_2, L1_ref_3). Following the comment by D Murphy on 1st of Feb 2022
04 Feb 2022: gen_hdu function now checks for all missing values in the match_keys and replace them by np.nan to avoid issues, occurred during opr4_jan2022 (missing TARGID caused error in this module)
25 March 2022: add a check to gen_hdu to assert if number of input columns (after removing those marked as excluded) and outputs (considering removed duplicated due to match_keys) are identical
24 Oct 2022: CCNAME definition fixed (following JIRA: WEAVESPA-661).
24 Oct 2022: BUNDLEID added to the primary header of the IFU products (following JIRA: WEAVESPA-661).
01 Feb 2024: Updating the way L2 inheritate header keywords from L1. An new function generate_PrimaryHDU added.
02 Feb 2024: UAPSID added to this to be reflected in the primary header
16 Feb 2024: now each of mos/ifuL2merge functions can return sys.exit if specific HUDs are empty. Good for catching errors in slurm
20 Feb 2024: update_filename function added to make sure for supertargets in mos and all IFU mode, we generate the updated APS filename, based on ICD27
02 Apr 2024: fixed a bug in ifuGalL2merge where the hdu0 where inserted twice.
02 Apr 2024: wave/warc files (per arms) are now referenced in primary header
02 Apr 2024: DATAMVER from CPS is now in primary hedear as L1_DMVER
02 Apr 2024: DATAMVER used by APS is now in primary hedear (for now its hard-codeed and defined at the top of this file)
02 Apr 2024: CALDATE also added to the primary hedear
09 Apr 2024: _apsqflag is now an output of the gen_hdu and it is -1 if an hdu is empty (its a placeholder and later can be filled with other values for other cases)
09 Apr 2024: APSQGLAG added as a new keyword in the primary header to reflect the quality of L2 files
10 Apr 2024: following WEAVESPA-690, CCNAME added to the primary header
19 Apr 2024: following WEAVEQAG-35, CAT-NAME added to the primary header
03 May 2024: LS_RES value updated to ADAPTED as its original value
06 June 2024: CATOVER keyword added to the primary header reflecting APS has overridden the user-defined APS-specific columns (including APS_FLAG) during processing of a file.
13 June 2024: Given that now the headname is generated at DB level, we deprecated the update_filename function. Still kept the function as reference. It is now called headname_updater in the orchestration layer
25 June 2024: Handling nan like values in generate_PrimaryHDU . It might happen while dealing with non-standard L1 data
12 Nov 2024: operational_outpath_updater function added to Updates the output path to end at the last part containing a YYYYMMDD date format. It is the format requried by the dataflow
14 Nov 2024: APS_ID column now enforced to be int32 (FITS format 'J') throughout all operations
"""
##################################################################################

def operational_outpath_updater(outpath):
    """
    Updates the output path to end at the last part containing a YYYYMMDD date format.

    Parameters:
    outpath (str): The original file or directory path.

    Returns:
    str: The updated path truncated at the last part containing a YYYYMMDD date format.
         If no part contains a date, returns the original path.

    Example:
    operational_outpath_updater("<PYAPS_DATA>/L2/20240731/12085")
    '<PYAPS_DATA>/L2/20240731/'

    operational_outpath_updater("<PYAPS_DATA>/L2/20240731_new/12085")
    '<PYAPS_DATA>/L2/20240731_new/'

    # and it does nothing if the output does not contain any YYYYMMDD structure
    operational_outpath_updater("<PYAPS_DATA>/L2/other_folder/12085")
    '<PYAPS_DATA>/L2/other_folder/12085'
    """
    # Split the path into individual parts by '/'
    parts = outpath.rstrip('/').split('/')

    # Iterate through the parts in reverse order to find the last segment containing a YYYYMMDD format
    for i in range(len(parts) - 1, -1, -1):
        # Check if the current part contains exactly 8 digits in a row (YYYYMMDD)
        if re.search(r"\b\d{8}\b", parts[i]):
            # If found, return the path up to and including this part
            return "/" + os.path.join(*parts[:i+1]) + "/"

    # If no part contains a date in YYYYMMDD format, return the original path as is
    return outpath

##################################################################################
def extract_P_index(headname):
    # extract the P index from the headname

    # Regular expression pattern to match 'P' followed by exactly 4 digits
    pattern = r'P(\d{4})'

    # Search for the pattern in the input string
    match = re.search(pattern, headname)

    # Check if a match is found
    if match:
        # Extract the matched structure
        structure = match.group(1)
        return structure
    else:
        print("No structure matching 'Pxxxx' found in the input string")
        return 'P9999'

##################################################################################

def find_available_filename(base_filename, directory):
    # Iterate over suffixes '00' to '99'
    for i in range(100):
        # Generate the filename with the current suffix
        filename = f"{base_filename}_{i:02}"

        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            return filename  # Return the available filename

    # If loop completes without finding an available filename, return an error
    raise ValueError("No available filename found in the range '00' to '99'")

##################################################################################

def update_filename(infiles, outpath, headname, wlranges=None, outfile_suffix='_APS',super_product=False):

    l1_info = l1_fileinfo(infiles, wlranges=wlranges)

    h0_infile = fits.getheader(infiles[0], 0)
    WMODE=l1_info['mode'].upper()

    # Generate CNAME/CCNAME for different modes
    if WMODE in ['MOS','MOSLIFU', 'MOSMIFU']:
        try:
            of_ccname = h0_infile['CNAME']
        except:
            of_ccname = '00000000+0000000'
    else:
        try:
            # For LIFU mode (DM v8.0 Oct 2022), BUNDLEIUD is 0, while the corresponding CCNAME keyword is CCNAME1
            # For mIFU mode, BUNDLEIUD starts from 1, and the corresponding CCNAME keyword would be CCNAME1,...
            if int(h0_infile["BUNDLEID"]) == 0:
                BNDID = "1"
            else:
                BNDID = str(int(h0_infile["BUNDLEID"])).replace(" ", "")
            CCNAME_X = "CCNAME" + BNDID

            of_ccname = h0_infile[CCNAME_X]
            of_bundleid = h0_infile["BUNDLEID"]

        except:
            of_bundleid = -1
            of_ccname = '00000000+0000000'

    # Generate first_char for different modes

    if WMODE == 'LIFU':
        first_char = 'L'
    elif WMODE == 'MIFU':
        first_char = 'm'
    elif WMODE in ['MOS','MOSLIFU', 'MOSMIFU']:
        first_char = ''
    else:
        first_char = 'X'

    # Generate arm_name
    arm_name =''.join(cam_i[0] for cam_i in l1_info['camera'])

    # Generate res_name
    res_name = l1_info['res_mode'][0][0]
    if res_name not in ['L','H']:
        print("Unsupported resolution mode deteced")
        res_name  = 'X'

    # Generate bin_name
    bin_name = l1_info['xbin'][0]
    if bin_name not in [1,2,4]:
        print("Unsupported binning mode deteced")
        bin_name = '0'
    bin_name = str(bin_name)

    if WMODE not in ['MOS','MOSLIFU', 'MOSMIFU']:
        p_index = f"_P{extract_P_index(headname)}"
    else:
        p_index = ''

    if not super_product:

        # Iterate over suffixes '00' to '99'
        for ii in range(1,100):
            # Generate the filename with the current suffix
            new_filename = f"{first_char}{of_ccname}_{ii:02}_{arm_name}_{res_name}{bin_name}{p_index}{outfile_suffix}.fits"

            # Construct the full file path
            full_file_path = os.path.join(outpath, new_filename)

            # Check if the file exists
            if not os.path.exists(full_file_path):
                return new_filename  # Return the available filename

        # If loop completes without finding an available filename, return an error
        raise ValueError("No available filename found in the range '00' to '99'")

    else:
        return f"{first_char}{of_ccname}_00_{arm_name}_{res_name}{bin_name}{p_index}{outfile_suffix}.fits"


##################################################################################
def ensure_aps_id_int32(table):
    """
    Ensure APS_ID column is int32 format if it exists in the table.
    Converts from int16 or other integer types to int32.

    Parameters:
    -----------
    table : astropy.table.Table
        Input table that may contain APS_ID column

    Returns:
    --------
    table : astropy.table.Table
        Table with APS_ID converted to int32 if present
    """
    if 'APS_ID' in table.colnames:
        aps_id_col = table['APS_ID']

        # Check current dtype
        if aps_id_col.dtype != np.int32:
            print(f"  INFO: Converting APS_ID from {aps_id_col.dtype} to int32")

            # Convert to int32
            if hasattr(aps_id_col, 'filled'):
                # Masked column - preserve mask but convert data type
                table['APS_ID'] = aps_id_col.filled(-1).astype(np.int32)
            else:
                # Regular column
                table['APS_ID'] = aps_id_col.astype(np.int32)

    return table

##################################################################################
def gen_hdu(in_fits, hdu_nums, extname, match_keys= None, exclude=None, exclude_prefix=None, sort_index=False):

    # default value for apsqflag
    _apsqflag = 0
    assert isinstance(in_fits, list), 'in_fits must be a list'
    assert isinstance(hdu_nums, list), 'hdu_nums must be a list'
    if match_keys is not None:
        assert isinstance(match_keys, list), 'keys must be a list'

    if exclude is not None:
        assert isinstance(exclude, list), 'exclude must be a list'
    if exclude_prefix is not None:
        assert isinstance(exclude_prefix, list), 'exclude_prefix must be a list'

    if len(in_fits) > 1 and match_keys is None:
        print('WARNING BY APS: No key has been set to match different tables. It might return ambiguous results')

    tables = []
    ncolns = []
    headers = []
    assert len(in_fits) == len(hdu_nums),  'input fits_names and hdu_nums must have identical length'

    for i_fits, fits_file in enumerate(in_fits):
        if os.path.isfile(fits_file):
            table_i = Table.read(fits_file, format='fits', hdu=hdu_nums[i_fits])
            header_i = fits.getheader(fits_file, hdu_nums[i_fits])

            ### More housekeeping by excluding the potentially duplicated colnames or useless columns
            if exclude is not None:
                for excl in exclude:
                    if excl in table_i.colnames: table_i.remove_columns(excl)

            # exclude_prefix: drop every column whose name starts with any
            # of these prefixes, regardless of exact name (unlike
            # `exclude` above, which needs the literal column name).
            # Added 1 Sep 2026 for EW_<line>/ERR_EW_<line> specifically --
            # see the merge call sites for why: a real per-line column
            # family (62 lines x N fields each) can add up fast, and FITS
            # binary tables have a hard 999-column TFIELDS limit (3-digit
            # keyword suffix, e.g. TTYPE999 is the last valid one --
            # TTYPE1000 would be a 9-character keyword, invalid). Found
            # live during an end-to-end SLURM test run: adding
            # EW_/ERR_EW_ to EMIPPXF_TABLE (812 columns total, of which
            # 124 are EW/ERR_EW) pushed the merged GALAXY_TABLE
            # (PPXF+EMIPPXF+LS, already 25+688+209=922 columns before EW)
            # to 1037 -- past the limit, breaking L2merge outright with an
            # astropy VerifyError. The standalone `_emippxf_BIN.fits` file
            # (not merged) is unaffected and still carries EW/ERR_EW in
            # full -- only the combined `_APS.fits` GALAXY_TABLE drops
            # them, to stay inside a hard format limit that has no
            # workaround. See doc/aps_rr.md for the full account.
            if exclude_prefix is not None:
                drop_cols = [c for c in table_i.colnames
                             if any(c.startswith(p) for p in exclude_prefix)]
                if drop_cols:
                    table_i.remove_columns(drop_cols)

            ## Make sure match_keys columns do not contain missing values (so that the join would be failed)
            ## We find missing values and replace them with np.nan
            mask_fill_value_dict ={'i':-999, 'u':-999, 'f': np.nan , 'S':'nan', 'U':'nan', }
            if match_keys is not None:
                for mkeys in match_keys:
                    if hasattr(table_i[mkeys], 'mask'):
                        mask_fill_value = mask_fill_value_dict[table_i[mkeys].dtype.kind]
                        table_i[mkeys] = table_i[mkeys].filled(mask_fill_value)

            ncolns.append(len(table_i.colnames))
            tables.append(table_i)
            headers.append(header_i)

        else:
            print(f'Warning: No {fits_file} file found. This file will be excluded from the extensions-join process')



    if len(tables) > 0:
        df_final = reduce(lambda left,right : join(left, right, join_type='outer', keys=match_keys , metadata_conflicts='warn'), tables)

        # this is to make sure all modification, especially the one that converts missing values into 'np.nan' for match_keys will not remove the mask attribute from the table
        # so we create a copy of the final table and convert it to masked table
        df_final = Table(df_final, masked=True, copy=False)

        # *** ADDED: Ensure APS_ID is int32 ***
        df_final = ensure_aps_id_int32(df_final)

        ## Replace missing value, based on dtype
        ## Note: No way to replace integer missing data by np.nan
        for col in df_final.itercols():
            if col.dtype.kind in ['c', 'f']:
                col.fill_value = np.nan
            elif col.dtype.kind in ['i', 'u']:
                col.fill_value = -1
            elif col.dtype.kind in ['U', 'O', 'S']:
                col.fill_value = 'nan'
            else:
                sys.exit(f'Unsupported column format. We cannot guaranty we can properly handle missing data for this column type {str(col.dtype.kind)}!')


        ## check if number of input columns (after removing those marked as excluded) and outputs (considering removed duplicated due to match_keys) are identical

        if match_keys is not None:
            len_match_keys = len(match_keys)
        else:
            len_match_keys = 0
        assert len(df_final.colnames) == np.sum(ncolns) - len_match_keys * (len(tables) -1), 'number of input and output columns are not matched'


        # if sort_index is requested, we update the index of key columns to be on the head of the table
        if (match_keys is not None) and (sort_index):
            for mkey_i, mkey in enumerate(match_keys):
                df_final.rename_column(mkey, mkey+"_dummy")
                df_final.add_column(df_final[mkey+"_dummy"], name=mkey, index=mkey_i)
                df_final.remove_column(mkey+"_dummy")


    if len(tables) == 0:
        df_final = Table()
        print(f'Warning: No input for the final {extname} found. An empty extension is generated for that.')
        # also update the value of _apsqflag to -1 to indicate something is wrong here
        _apsqflag = -1

    # Handle variable-length array columns before converting to FITS
    # First, collect column names to avoid mutation during iteration
    column_names = list(df_final.colnames)

    for col_name in column_names:
        col = df_final[col_name]

        if col.dtype.kind == 'O':  # Object dtype columns
            # Check if it's a column of arrays
            first_valid = None
            for val in col:
                if val is not None and val is not np.ma.masked:
                    first_valid = val
                    break

            if first_valid is not None and isinstance(first_valid, (np.ndarray, np.ma.MaskedArray)):
                # Determine the underlying dtype from valid entries
                sample_dtype = None
                for val in col:
                    if isinstance(val, (np.ndarray, np.ma.MaskedArray)) and len(val) > 0:
                        if isinstance(val, np.ma.MaskedArray):
                            sample_dtype = val.data.dtype
                        else:
                            sample_dtype = val.dtype
                        break

                if sample_dtype is None:
                    sample_dtype = np.float64  # default

                # Determine fill value based on dtype
                if sample_dtype.kind in ['f', 'c']:  # float or complex
                    fill_value = np.nan
                elif sample_dtype.kind in ['i', 'u']:  # integer
                    fill_value = -999
                elif sample_dtype.kind in ['U', 'S']:  # string
                    fill_value = 'nan'
                else:
                    fill_value = np.nan  # default

                # Convert all entries
                converted = []
                for val in col:
                    if val is np.ma.masked or val is None:
                        # Create array of fill values with appropriate shape
                        if first_valid is not None and hasattr(first_valid, 'shape'):
                            filled_array = np.full(first_valid.shape, fill_value, dtype=sample_dtype)
                        else:
                            filled_array = np.array([fill_value], dtype=sample_dtype)
                        converted.append(filled_array)
                    elif isinstance(val, np.ma.MaskedArray):
                        # Replace masked values with fill_value
                        filled_array = val.filled(fill_value).astype(sample_dtype)
                        converted.append(filled_array)
                    elif isinstance(val, np.ndarray):
                        # Keep as is but ensure consistent dtype
                        converted.append(val.astype(sample_dtype))
                    else:
                        # Unexpected type, convert to fill value
                        converted.append(np.array([fill_value], dtype=sample_dtype))

                # Store the unit before removing column
                col_unit = col.unit

                # Replace the column
                df_final.remove_column(col_name)
                df_final[col_name] = converted
                if col_unit is not None:
                    df_final[col_name].unit = col_unit

    out_hdul = fits.convenience.table_to_hdu(df_final)
    out_hdul.header['EXTNAME']=extname
    out_hdul.header['APSVERS']= APSVERS

    # *** ADDED: Explicitly ensure APS_ID column uses 'J' format (int32) in FITS ***
    if 'APS_ID' in df_final.colnames:
        aps_id_idx = df_final.colnames.index('APS_ID')
        # Get the column format from the HDU
        if hasattr(out_hdul, 'columns') and aps_id_idx < len(out_hdul.columns):
            col_format = out_hdul.columns[aps_id_idx].format
            # Check if format needs to be changed to 'J' (int32)
            if col_format not in ['J', '1J']:
                print(f"  INFO: Updating APS_ID FITS format from '{col_format}' to 'J' (int32) in {extname}")
                out_hdul.columns[aps_id_idx].format = 'J'

    ## Add header's keyword comments from the original fits files to the final product
    ## By default, ASTROPY, remove headers keyword comments while transforming fits (header) to table (meta)
    ## To overcome this issue, I keep the original headers in a list called (headers)
    ## Then, I search for each item of the final header in the provinces' headers
    ## As soon as it found the first answer, it break the inner loop and go through the next keyword
    if len(headers) > 0:
        for key_item in list(out_hdul.header.keys()):
            for hdr_i in headers:
                if key_item in list(hdr_i.keys()):
                    out_hdul.header.comments[key_item] = hdr_i.comments[key_item]
                    break

    return out_hdul, _apsqflag
##################################################################################

def mosL2merge(infiles, outpath, headname, wlranges = None, outfile_suffix= '_APS' , EMIPPXF_LEVEL='BIN', LS_RES ='ADAPTED', UAPSID=None):

    # create a placeholder to store all apsqflags
    _apsqflag_list = []
    # Check if we are dealing with super products or not
    stacking_level = stack_level(infiles[0])
    super_product = True if stacking_level == 4 else False

    hdulist = fits.HDUList()
    hdu0= generate_PrimaryHDU(infiles, IFU_MODE=False, wlranges = wlranges,  patch_id=None, patch_area=None, UAPSID=UAPSID)
    hdulist.append(hdu0)

    ## Note (4Feb2022): Even for EXT1 amd EXT4, which only have 1 table to merge, we add match_keys parameter to deal with missing values within this columns (though we do not apply join for these EXTs)

    ## EXT 1: CLASS TABLE
    in_fits = [os.path.join(outpath,'zbest_'+headname+'.fits')]
    hdux_1, _apsqflag = gen_hdu(in_fits, [1], 'CLASS_TABLE', match_keys=['APS_ID','TARGID','CNAME'])
    hdulist.append(hdux_1)
    _apsqflag_list.append(1 if _apsqflag == -1 else 0)

    ## EXT 2: STAR TABLE
    in_fits = [os.path.join(outpath,'rvs_'+headname+'.fits'), os.path.join(outpath,'ferre_'+headname+'.fits')]
    hdux_2, _apsqflag = gen_hdu(in_fits, [1,1], 'STAR_TABLE', match_keys=['APS_ID','TARGID','CNAME'] )
    hdulist.append(hdux_2)
    _apsqflag_list.append(2 if _apsqflag == -1 else 0)

    ## EXT 3: GALAXY TABLE
    in_fits = [os.path.join(outpath, headname+'_ppxf'+'.fits'), os.path.join(outpath, headname+'_emippxf_'+EMIPPXF_LEVEL+'.fits'), os.path.join(outpath, headname+'_ls_'+LS_RES+'.fits')]
    # exclude_prefix drops EW_/ERR_EW_ from the merge -- see gen_hdu's own
    # docstring/comment for why (FITS 999-column TFIELDS limit). Still
    # present in full in the standalone _emippxf_<LEVEL>.fits file.
    hdux_3, _apsqflag = gen_hdu(in_fits, [1,2,1], 'GALAXY_TABLE', exclude=['BIN_ID'],
                                 exclude_prefix=['EW_', 'ERR_EW_'], match_keys=['APS_ID','TARGID','CNAME'] )
    hdulist.append(hdux_3)
    _apsqflag_list.append(4 if _apsqflag == -1 else 0)

    ## EXT 4: CLASS SPEC
    in_fits = [os.path.join(outpath,'zspec_'+headname+'.fits')]
    hdux_4, _apsqflag = gen_hdu(in_fits, [1], 'CLASS_SPEC', match_keys=['APS_ID','TARGID','CNAME'])
    hdulist.append(hdux_4)
    _apsqflag_list.append(8 if _apsqflag == -1 else 0)


    ## EXT 5: STAR SPEC
    in_fits = [os.path.join(outpath,'rvsspec_'+headname+'.fits'), os.path.join(outpath,'ferre_spec_'+headname+'.fits')]
    hdux_5, _apsqflag = gen_hdu(in_fits, [1,1], 'STAR_SPEC', match_keys=['APS_ID','TARGID','CNAME'])
    hdulist.append(hdux_5)
    _apsqflag_list.append(16 if _apsqflag == -1 else 0)


    ## EXT 6: GALAXY SPEC
    in_fits = [os.path.join(outpath, headname+'_ppxf_spec'+'.fits'), os.path.join(outpath, headname+'_emippxf_spec_'+EMIPPXF_LEVEL+'.fits')]
    hdux_6, _apsqflag = gen_hdu(in_fits, [1,1], 'GALAXY_SPEC', exclude=['BIN_ID'], match_keys=['APS_ID','TARGID','CNAME'])
    hdulist.append(hdux_6)
    _apsqflag_list.append(32 if _apsqflag == -1 else 0)


    # Update the APSQFLAG value in the primary header based on the sum of the _apsqflag_list
    hdulist[0].header['APSQFLAG'] = np.sum(_apsqflag_list)


    if len(hdux_1.data) == 0 or (len(hdux_2.data) + len(hdux_3.data)) == 0:
        hdulist.close()
        return -1
    else:
        # if super_product:
        #     updated_filename = update_filename(infiles, outpath, headname, wlranges=wlranges, outfile_suffix=outfile_suffix,super_product=super_product)
        #     print(f"DEBUG: Supertarget detected. Filename for the final product would be {updated_filename}")
        # else:
        updated_filename = headname+outfile_suffix+'.fits'
        print(f"DEBUG: No Supertarget detected. Filename would be {updated_filename}")

        # convert the outpath to the operational outpath for the final file
        operational_outpath = operational_outpath_updater(outpath)

        hdulist.writeto(os.path.expandvars(os.path.join(operational_outpath,updated_filename)), overwrite=True, checksum=True)
        hdulist.close()
        return 0

########################################################

def ifuExGalL2merge(infiles, outpath, headname, wlranges = None, outfile_suffix= '_APS' , EMIPPXF_LEVEL='BIN', LS_RES='ADAPTED', patch_area = None, patch_id = None, UAPSID=None, no_spec_ext=False):

    # create a placeholder to store all apsqflags
    _apsqflag_list = []

    # Check if we are dealing with super products or not
    stacking_level = stack_level(infiles[0])
    super_product = True if stacking_level ==4 else False

    hdulist = fits.HDUList()
    hdu0= generate_PrimaryHDU(infiles, IFU_MODE=True,wlranges = wlranges, patch_id=patch_id, patch_area=patch_area, UAPSID=UAPSID)

    hdulist.append(hdu0)

    ## EXT 1: PATCH_ALLSPEC
    # in_fits = [os.path.join(outpath,headname+'_AllSpectra'+'.fits')]
    # hdux_1, _apsqflag = gen_hdu(in_fits, [1] , 'PATCH_ALLSPEC')
    # hdulist.append(hdux_1)
    # _apsqflag_list.append(1 if _apsqflag == -1 else 0)


    ## EXT 1: PATCH_TABLE
    in_fits = [os.path.join(outpath,headname+'_table'+'.fits')]
    hdux_1, _apsqflag = gen_hdu(in_fits, [1], 'PATCH_TABLE')
    hdulist.append(hdux_1)
    _apsqflag_list.append(2 if _apsqflag == -1 else 0)


    ## EXT 2: PATCH_BINSPEC
    in_fits = [os.path.join(outpath,headname+'_BINSpectra'+'.fits')]
    hdux_2, _apsqflag = gen_hdu(in_fits, [1], 'PATCH_BINSPEC')
    hdulist.append(hdux_2)
    _apsqflag_list.append(4 if _apsqflag == -1 else 0)


    ## EXT 3: GALAXY_TABLE
    in_fits = [os.path.join(outpath, headname+'_ppxf'+'.fits'), os.path.join(outpath, headname+'_emippxf_'+EMIPPXF_LEVEL+'.fits'), os.path.join(outpath, headname+'_ls_'+LS_RES+'.fits')]
    # exclude_prefix drops EW_/ERR_EW_ from the merge -- see gen_hdu's own
    # docstring/comment for why (FITS 999-column TFIELDS limit). Still
    # present in full in the standalone _emippxf_<LEVEL>.fits file.
    hdux_3, _apsqflag = gen_hdu(in_fits, [1,2,1], 'GALAXY_TABLE',
                                 exclude_prefix=['EW_', 'ERR_EW_'], match_keys= ['BIN_ID'] )
    hdulist.append(hdux_3)
    _apsqflag_list.append(8 if _apsqflag == -1 else 0)


    if not no_spec_ext:
        ## EXT 4: GALAXY SPEC
        in_fits = [os.path.join(outpath, headname+'_ppxf_spec'+'.fits'), os.path.join(outpath, headname+'_emippxf_spec_'+EMIPPXF_LEVEL+'.fits')]
        hdux_4, _apsqflag = gen_hdu(in_fits, [1,1], 'GALAXY_SPEC', match_keys= ['BIN_ID'])
        hdulist.append(hdux_4)
        _apsqflag_list.append(16 if _apsqflag == -1 else 0)
    else:
        print(f'WARNING: no_spec_ext option has been requested by user. The spectra extensions will be discarded from the final APS file')
        hdulist[0].header['NO_SPEC'] = (no_spec_ext, 'Spectra extensions omitted by user request')


    # Update the APSQFLAG value in the primary header based on the sum of the _apsqflag_list
    hdulist[0].header['APSQFLAG'] = np.sum(_apsqflag_list)

    if len(hdux_1.data) == 0 or len(hdux_2.data) == 0:
        hdulist.close()
        return -1
    else:
        # updated_filename = update_filename(infiles, outpath, headname, wlranges=wlranges, outfile_suffix=outfile_suffix,super_product=super_product)
        updated_filename = headname+outfile_suffix+'.fits'
        print(f"DEBUG: Filename for the final product would be {updated_filename}")

        # convert the outpath to the operational outpath for the final file
        operational_outpath = operational_outpath_updater(outpath)

        hdulist.writeto(os.path.expandvars(os.path.join(operational_outpath,updated_filename)), overwrite=True, checksum=True)
        hdulist.close()
        return 0


########################################################

def ifuGalL2merge(infiles, outpath, headname, wlranges = None, outfile_suffix= '_APS', patch_area = None, patch_id = None, UAPSID=None, no_spec_ext=False):

    # create a placeholder to store all apsqflags
    _apsqflag_list = []

    # Check if we are dealing with super products or not
    stacking_level = stack_level(infiles[0])
    super_product = True if stacking_level ==4 else False


    hdulist = fits.HDUList()
    hdu0= generate_PrimaryHDU(infiles, IFU_MODE=True, wlranges = wlranges, patch_id=patch_id, patch_area=patch_area, UAPSID=UAPSID)

    hdulist.append(hdu0)

    ## EXT 1: PATCH_TABLE
    in_fits = [os.path.join(outpath,headname+'_table'+'.fits')]
    hdux_1, _apsqflag = gen_hdu(in_fits, [1], 'PATCH_TABLE')
    hdulist.append(hdux_1)
    _apsqflag_list.append(1 if _apsqflag == -1 else 0)


    ## EXT 2: STAR TABLE
    in_fits = [os.path.join(outpath,'rvs_'+headname+'.fits'), os.path.join(outpath,'ferre_'+headname+'.fits')]
    hdux_2, _apsqflag = gen_hdu(in_fits, [1,1], 'STAR_TABLE', match_keys=['BIN_ID'], exclude=['APS_ID','TARGID','CNAME'] , sort_index=True)
    hdulist.append(hdux_2)
    _apsqflag_list.append(2 if _apsqflag == -1 else 0)

    if not no_spec_ext:
        ## EXT 3: STAR SPEC
        in_fits = [os.path.join(outpath,'rvsspec_'+headname+'.fits'), os.path.join(outpath,'ferre_spec_'+headname+'.fits')]
        hdux_3, _apsqflag = gen_hdu(in_fits, [1,1], 'STAR_SPEC', match_keys=['BIN_ID'], exclude=['APS_ID','TARGID','CNAME'], sort_index=True)
        hdulist.append(hdux_3)
        _apsqflag_list.append(4 if _apsqflag == -1 else 0)
    else:
        print(f'WARNING: no_spec_ext option has been requested by user. The spectra extensions will be discarded from the final APS file')
        hdulist[0].header['NO_SPEC'] = (no_spec_ext, 'Spectra extensions omitted by user request')


    # Update the APSQFLAG value in the primary header based on the sum of the _apsqflag_list
    hdulist[0].header['APSQFLAG'] = np.sum(_apsqflag_list)

    if len(hdux_1.data) == 0:
        hdulist.close()
        return -1
    else:
        # updated_filename = update_filename(infiles, outpath, headname, wlranges=wlranges, outfile_suffix=outfile_suffix,super_product=super_product)
        updated_filename = headname+outfile_suffix+'.fits'
        print(f"DEBUG: Filename for the final product would be {updated_filename}")

        # convert the outpath to the operational outpath for the final file
        operational_outpath = operational_outpath_updater(outpath)

        hdulist.writeto(os.path.expandvars(os.path.join(operational_outpath,updated_filename)), overwrite=True, checksum=True)
        hdulist.close()
        return 0

########################################################
def get_header_value(header, keyword):
    value = header.get(keyword, '')
    if value == '' or (isinstance(value, str) and value.strip().lower() in ['nan', 'n/a', 'none', '']):
        return ''
    if isinstance(value, float) and (math.isnan(value) or value in [float('nan'), float('NaN'), float('NAN')]):
        return ''
    return value





def generate_PrimaryHDU(infiles, IFU_MODE=False, wlranges=None, patch_id=None, patch_area=None, UAPSID=None):
    '''
    Generate primary header for the final APS product, contains CPS and APS params
    '''
    # read the primary header of the first infiles
    h0_infile = fits.getheader(infiles[0], 0)

    ## Create an hdu list
    prmHDR = fits.Header()

    # read the basic info from the corrected version of infiles
    l1_info = l1_fileinfo(infiles, wlranges=wlranges)
    # generate the L1 reference files to be added to the header in L1_Ref_B/G/R format
    # create template of L1_Ref_B/G/R to be filled in the next step
    for cam_id in ['BLUE','GREEN','RED']:
        prmHDR[f'L1_Ref_{cam_id[0]}'] = ('', f'L1 reference file in {cam_id} arm')

    for i_of_infl , of_infl in enumerate(l1_info['infiles']):
        camera_i = l1_info['camera'][i_of_infl]
        if not (camera_i.upper() in ['BLUE', 'GREEN', 'RED']):
            print(f'Warning: CANNOT detect cammera mode for {of_infl}. Filling L1_Ref_B/G/R wih default value')
            continue
        prmHDR[f'L1_Ref_{camera_i[0]}'] = (os.path.basename(of_infl), f'L1 reference file for {camera_i} arm')

    # direcyly fetch some basic header parameters from the L1 primary header (they are not arm dependent)
    # we also introduced l1_keywords_trunc in case APS wants to change the keyword name in
    # the primary header (e.g.for DATAMVER which will be updated to L1_DMVER in APS primary header)
    l1_keywords       = ['DATE-OBS' ,'OBSMODE' ,'CASUDATE' ,'CASUVERS' ,'SOFTAUTH' ,'SOFTINST' ,'SOFTVERS', 'CALDATE', 'DATAMVER','CAT-NAME']
    l1_keywords_trunc = ['DATE-OBS' ,'OBSMODE' ,'CASUDATE' ,'CASUVERS' ,'SOFTAUTH' ,'SOFTINST' ,'SOFTVERS', 'CALDATE', 'L1_DMVER','CAT-NAME']

    for i_keyword, keyword in enumerate(l1_keywords):
        # Case-insensitive check
        matching_keywords = [k for k in h0_infile if k.lower() == keyword.lower()]

        if matching_keywords:
            # Get the first matching keyword (case-insensitive)
            matched_keyword = matching_keywords[0]
            # value = h0_infile[matched_keyword]
            value = get_header_value(h0_infile, matched_keyword)
            comment = h0_infile.comments[matched_keyword]
        else:
            # If keyword is missing, fill with NaN
            value = ''
            comment = f"CPS: Keyword '{keyword}' not found in original header"

        # Add the cloned keyword to the new header
        new_comment = f"CPS: {comment}"
        prmHDR[l1_keywords_trunc[i_keyword]] = (value, new_comment)


    # Now for a specific set of L1 keywords, we fetch some basic header parameters in each arm
    # and label them with arms indicators
    l1h_keywords = ['CASUDATE','CHECKSUM', 'CASUID','OBID','WAVEFILE','WARCFILE']
    # shorter version of names to be used for our longer  *_B/G/R format
    l1h_keywords_trunc = ['CASUDT','CASCHK', 'CASID', 'OBID','WAVEF','WARCF']

    # create template of {l1h_keywords_trunc}_B/G/R to be filled in the next step
    # we also creates the comment parts based on the original l1h_keyboards
    for cam_id in ['BLUE','GREEN','RED']:
        for i_keyword, keyword in enumerate(l1h_keywords):
            prmHDR[f'{l1h_keywords_trunc[i_keyword]}_{cam_id[0]}'] = (None, f'CPS: {keyword} for {cam_id} arm')

    for i_of_infl , of_infl in enumerate(l1_info['infiles']):
        camera_i = l1_info['camera'][i_of_infl]

        if not (camera_i.upper() in ['BLUE', 'GREEN', 'RED']):
            print(f'Warning: CANNOT detect cammera mode for {of_infl}. Filling l1h_keywords wih default value')
            continue
        h_infile = fits.getheader(of_infl, 0)
        for i_keyword, keyword in enumerate(l1h_keywords):
            # Case-insensitive check
            matching_keywords = [k for k in h_infile if k.lower() == keyword.lower()]

            if matching_keywords:
                # Get the first matching keyword (case-insensitive)
                new_keword = f'{l1h_keywords_trunc[i_keyword]}_{camera_i[0]}'
                matched_keyword = matching_keywords[0]
                # value = h_infile[matched_keyword]
                value = get_header_value(h_infile, matched_keyword)
                prmHDR[new_keword] = value
            else:
                print(f"Warning: CPS: Keyword '{keyword}' not found in original header")


    if IFU_MODE:
        try:
            # For LIFU mode (DM v8.0 Oct 2022), BUNDLEIUD is 0, while the corresponding CCNAME keyword is CCNAME1
            # For mIFU mode, BUNDLEIUD starts from 1, and the corresponding CCNAME keyword would be CCNAME1,...
            if int(h0_infile["BUNDLEID"]) == 0:
                BNDID = "1"
            else:
                BNDID = str(int(h0_infile["BUNDLEID"])).replace(" ", "")
            CCNAME_X = "CCNAME" + BNDID

            of_ccname = h0_infile[CCNAME_X]
            of_bundleid = h0_infile["BUNDLEID"]

        except:
            of_bundleid = -1
            of_ccname = '00000000+0000000'

        prmHDR['BUNDLEID'] = (of_bundleid, 'CPS: BUNDLEID')
        prmHDR['CCNAME'] = (of_ccname, 'CPS: CCNAME')

        ## ADD patch parameters to the h0 header
        if patch_id is not None:
            prmHDR['P_ID'] = (int(patch_id), 'Patch param: ID')

        if patch_area is not None:
            assert len(patch_area) ==5 , 'patch area must contain 5 elements [RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec], ANGLE [CCW in deg]]'
            prmHDR['P_RA'] = (patch_area[0], 'Patch param: central RA [deg]')
            prmHDR['P_DEC'] = (patch_area[1], 'Patch param: central DEC [deg]')
            prmHDR['P_A'] = (patch_area[2], 'Patch param: Total length (diameter) of semi-major axis [arcsec]')
            prmHDR['P_B'] = (patch_area[3], 'Patch param: Total length (diameter) of semi-minor axis [arcsec]')
            prmHDR['P_THETA'] = (patch_area[4], 'Patch param: Rotation in degrees anti-clockwise')

    # also add APS related header keywords

    # Get the current date and time
    crtime = datetime.now()
    APSDATE = crtime.strftime('%Y-%m-%dT%H:%M:%S')
    UAPSID_str = UAPSID if UAPSID is not None else ''
    prmHDR['CATOVER'] = (1, 'APS has overridden the user-defined APS-specific columns')
    prmHDR['APSVERS'] = (APSVERS, 'APS Version')
    prmHDR['APSDATE'] = (APSDATE, 'APS data processing date')
    prmHDR['APSUID'] = (UAPSID_str, 'APS Repository Identifier (UPASID)')
    prmHDR['APSQFLAG'] = (0, 'APS data quality flag')
    prmHDR['DATAMVER'] = (DATAMVER, 'WEAVE Data Model Version for APS')

    hdu0= fits.PrimaryHDU(header=prmHDR)

    return hdu0

    ################################################

def aps_L2merge(options=None):

    parser = argparse.ArgumentParser(description="RUN MERGE_L2 on the PyAPS products")

    parser.add_argument("--infiles", type=none_or_str, default=None,
        required=True, help="input fits files (L1)", nargs='*')


    parser.add_argument('--outpath',
        help='Directory contains PyAPS products',
        type=none_or_str, default=None, required=True)

    parser.add_argument('--headname',
        help='Output headname. The output will be generated based on this headname',
        type=none_or_str, default='headname', required=True)


    parser.add_argument("--wlranges", type=none_or_str, default=None,
        required=False, help="wavelength range array for each elements of the setup", nargs='*')


    parser.add_argument('--outfile_suffix',
        help='Output suffix. The suffix to be added to the PyAPS products',
        type=none_or_str, default='_APS', required=False)

    parser.add_argument('--EMIPPXF_LEVEL',
        help='EMIPPXF RUNNING LEVEL [BIN, SPAXEL], needs to find the appropriate EMIPPXF outputs',
        type=none_or_str, default='BIN', required=False)

    parser.add_argument('--LS_RES',
        help='LS RESOLUTION MODE [ORIGINAL, ADAPTED], needs to find the appropriate LS outputs',
        type=none_or_str, default='ADAPTED', required=False)

    parser.add_argument('--mode',
        help='WORKING MODE [MOS, IFU]',
        type=none_or_str, default='MOS', required=False)


    parser.add_argument("--patch_area", type=none_or_str, default=None,
        required=False, help="The Patch info in [RA_CENT[deg], DEC_CENT[deg], A[arcsec], B[arcsec], ANGLE [CCW in deg]]")

    parser.add_argument("--patch_id", type=int, default=0,
        required=False, help="Patch id [INT]")

    parser.add_argument('--uapsid',
        help='Directory contains PyAPS products',
        type=none_or_str, default=None, required=False)


    ## Check if any command-line argument has been passed to the module. It counts the number of system arguments to check this.
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')

        args = parser.parse_args(options)

    wlranges = None
    if args.wlranges[0] is not None:
        wlranges=[]
        for i in range(len(args.infiles)):
            wlranges.append([float(x) for x in args.wlranges[i].split(",")])

    patch_area = None
    if args.patch_area is not None:
        patch_area = [float(x) for x in args.patch_area.split(",")]
        ## also update the args.patch_area to be printed in the final format in the report file
        args.patch_area=patch_area



    # print args and assigned/default values on the screen
    print_args(args,module='APS_L2_MERGE', version= aps_constants.__aps_L2_merge_version__ , path=args.outpath, headname=args.headname)


    if (args.mode).replace(" ", "").upper() in ['MOS', 'MOSLIFU', 'MOSMIFU']:
        ## RUN THE MOSL2MERGE function
        return_code = mosL2merge(args.infiles, args.outpath, args.headname,wlranges= wlranges, outfile_suffix=args.outfile_suffix, EMIPPXF_LEVEL=args.EMIPPXF_LEVEL, LS_RES =args.LS_RES, UAPSID=args.uapsid)
        # also check and issue a sys.exit if some mandatory conditions have not passed
        if return_code == -1:
            sys.exit('mosL2merge module failed due to issue in some of HUDs')



    if (args.mode).replace(" ", "").upper() in ['LIFU', 'MIFU', 'IFU']:
        ## RUN THE IFUL2MERGE function

        if os.path.exists(os.path.join(args.outpath,'rvs_'+args.headname+'.fits')):
            return_code = ifuGalL2merge(args.infiles, args.outpath, args.headname, wlranges = wlranges, outfile_suffix=args.outfile_suffix, patch_area = args.patch_area, patch_id = args.patch_id, UAPSID=args.uapsid)
        else:
            return_code = ifuExGalL2merge(args.infiles, args.outpath, args.headname, wlranges = wlranges, outfile_suffix=args.outfile_suffix, EMIPPXF_LEVEL=args.EMIPPXF_LEVEL, LS_RES =args.LS_RES, patch_area=args.patch_area, patch_id=args.patch_id, UAPSID=args.uapsid)

        # also check and issue a sys.exit if some mandatory conditions have not passed
        if return_code == -1:
            sys.exit('ifuL2merge module failed due to issue in some of HUDs')

#####################################################################
if __name__ == '__main__':

    #MOS example1

    # #MOS example2
    # debug_demo1= [
    # '--infiles', '<PYAPS_DATA>/gal_test/superstack_100001.fits', '<PYAPS_DATA>/gal_test/superstack_100000.fits',
    # '--outpath', '<PYAPS_DIR>/PyAPS_results/20170223/3800/',
    # '--headname' , 'superstack_100001__superstack_100000',
    # '--wlranges', '4500.0,6200.0', '5800.0,7000.0', # or 'None' to use the whole available wlrange
    # '--outfile_suffix' , '_APS',
    # '--EMIPPXF_LEVEL', 'BIN',
    # '--LS_RES' ,    'ADAPTED',
    # '--uapsid' ,    'None',
    # '--mode', 'MOS']

    #MOS example2
    debug_demo2= [
    '--infiles', '<PYAPS_DATA>/gal_test/superstack_100001.fits', '<PYAPS_DATA>/gal_test/superstack_100000.fits',
    '--outpath', '$HOME//PyAPS/PyAPS_results/20170223/3800/',
    '--headname' , 'superstack_100001__superstack_100000',
    '--wlranges', 'None',
    '--outfile_suffix' , '_APS',
    '--EMIPPXF_LEVEL', 'BIN',
    '--LS_RES' ,    'ADAPTED',
    '--uapsid' ,    'None',
    '--mode', 'MOS']



    # #IFU mode
    # debug_demo3= [
    # '--infiles', '<PYAPS_DATA>/gal_test/superstack_100001.fits', '<PYAPS_DATA>/gal_test/superstack_100000.fits',
    # '--outpath', '<PYAPS_DIR>/PyAPS_results/20170223/3800/',
    # '--headname' , 'superstack_100001__superstack_100000',
    # '--wlranges', '4500.0,6200.0', '5800.0,7000.0', # or 'None' to use the whole available wlrange
    # '--outfile_suffix' , '_APS',
    # '--EMIPPXF_LEVEL', 'BIN',
    # '--LS_RES' ,    'ADAPTED',
    # '--mode', 'IFU',
    # '--patch_area' , '123.2,32.3,0.012,0.010,44.0',
    # '--uapsid' ,    'None',
    # '--patch_id', '1'
    # ]


    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    aps_L2merge(options=debug_demo2)
