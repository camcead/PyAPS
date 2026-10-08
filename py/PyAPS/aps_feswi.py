import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import logging
import shutil
from copy import deepcopy

import PyAPS
import numpy as np
from PyAPS import aps_constants
from PyAPS.aps_utils import none_or_str, str2bool, print_args, l1_fileinfo
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS.aps_ferre import proc_ferre
APSVERS = PyAPS.__version__


################################################################################


def get_ferre_parser(description="RUN FERRE for WEAVE target spectra."):
    # Shared registry -- see aps_common_args.py's own module docstring.
    # This mirrors aps_ferre.py's own parser (this module wraps/calls
    # into aps_ferre.proc_ferre directly) but is deliberately its own,
    # slightly smaller flag set -- no --catdir/--caldir/--configdir and
    # no --split_arms/--linemask/--maskbalmer/--uselinemasks, none of
    # which this script has ever taken -- reproduced exactly as-is
    # rather than "caught up" to aps_ferre.py's own current set, since
    # that's not this migration's job.
    return build_common_parser(
        description=description,
        groups=["target_selection", "spatial_selection", "wavelength", "l1_processing", "output"],
        overrides={
            "mask_aps_ids": {"help": "comma-separated list of APS_IDS to be masked"},
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
            (("--mp",), dict(type=int, default=1, required=False,
                              help="The number of threads to run original FERRE CODE in Fortran")),
            (("--ferre_exe",), dict(type=none_or_str, default=None, required=True,
                                     help="The full path of the FERRE executable file")),
            (("--fig",), dict(type=str2bool, default=False, required=False,
                               help="if True, the code also produces plots of the best fitted model")),
        ],
    )


################################################################################


def get_args(parser, options=None):

    ## Check if any command-line argument has been passed to the module. It counts the number of system arguments to check this.
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')
        
        args = parser.parse_args(options)
    
    return args

        
################################################################################


def get_ferre_proc_args(args):


    ### Now we have both infiles and wlranges array. We use resolve_common_args
    ### (aps_common_args.py) to update infiles/wlranges/arms_ratio/join_arms
    ### and put them in the right order, if needed -- see that function's
    ### own docstring. However, we had similar test done by APSOB.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio

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
    outpath=args.outpath #+os.path.sep
    outpath=outpath.replace(' ', '')
    
    ## workpath is wehre the code keep all intermediate files and do all housekeeping.
    workpath = outpath + 'fr_wd/'
    
    if not os.path.exists(workpath):
        os.makedirs(workpath)
    elif args.overwrite:
        shutil.rmtree(workpath)
        os.makedirs(workpath)


    param_fits=os.path.join(args.outpath,'ferre_'+str(args.headname)+'.fits')
    param_fits=param_fits.replace(' ', '')


    spec_fits=None
    if args.outspec:
        spec_fits=os.path.join(args.outpath,'ferre_spec_'+str(args.headname)+'.fits')
        spec_fits.replace(' ', '')



    if not os.path.isfile(args.ferre_exe):
        sys.exit('No FERRE executable file found')


    if not os.path.exists(args.templates):
        sys.exit('TEMPLATES directory: %s does not exist'%(args.griddir))

    figdir=None
    if args.fig:
        figdir=args.outpath + '/figs/'
        if not os.path.exists(figdir):
            os.makedirs(figdir)
            print("FIGDIR: %s Created!" %(figdir))
        figdir=figdir+'/'#+str(args.headname)
        figdir=figdir.replace(' ', '')

    # Make a deep copy of args
    
    proc_args = deepcopy(args)

    # Overwrite the attributes which have been dereferenced in this function

    proc_args.wlranges = wlranges
    proc_args.arms_ratio = arms_ratio
    proc_args.classfile = classfile
    proc_args.aps_ids = aps_ids
    proc_args.targsrvy = targsrvy
    proc_args.targclass = targclass
    proc_args.mask_aps_ids = mask_aps_ids
    proc_args.area = area
    proc_args.mask_areas = mask_areas
    proc_args.grid_ids = grid_ids
    proc_args.use_rvs = use_rvs
    proc_args.grid_prefix = grid_prefix
    proc_args.outpath = outpath
    proc_args.workpath = workpath

    # Add the new attributes which did not exist in args

    proc_args.param_fits = param_fits
    proc_args.spec_fits = spec_fits
    proc_args.figdir = figdir

    return proc_args


################################################################################


def _get_feswi_parser():
    
    description = 'RUN FEESI for WEAVE target spectra.'
    
    parser = get_ferre_parser(description=description)
    
    parser.add_argument('--feswi_run_ferre', default=False, type=str2bool,
        help='run FERRE before running FESWI')
    
    parser.add_argument(
        '--feswi_path', required=True, type=none_or_str,
        help='path to the directory which contains the main FESWI code')
    
    parser.add_argument(
        '--feswi_spectral_windows', required=True, type=none_or_str,
        help='JSON file with the spectral windows for each element')
    
    parser.add_argument(
        '--feswi_cold_blue_grid', required=True, type=none_or_str,
        help='file with cold grid in the blue range')
    
    parser.add_argument(
        '--feswi_cold_red_grid', required=True, type=none_or_str,
        help='file with cold grid in the red range')
    
    parser.add_argument(
        '--feswi_hot_blue_grid', required=True, type=none_or_str,
        help='file with hot grid in the blue range')
    
    parser.add_argument(
        '--feswi_hot_red_grid', required=True, type=none_or_str,
        help='file with hot grid in the red range')
    
    parser.add_argument(
        '--feswi_ferre_ndim', default=0, type=int, choices=[0, 3, 5],
        help='number of dimensions of the model grids (0 for guessing it)')
    
    parser.add_argument(
        '--feswi_ferre_f_format', default=0, type=int, choices=[0, 1],
        help='format of the model grids: 0 for text or 1 for binary')
    
    parser.add_argument(
        '--feswi_ferre_f_access', default=0, type=int, choices=[0, 1],
        help='mode of accessing grid data: 0 for RAM or 1 for direct-access')
    
    parser.add_argument(
        '--feswi_ferre_nthreads', default=1, type=int,
        help='number of threads for parallel processing (0 for all the cores)')
    
    return parser


################################################################################


def _get_feswi_proc_args(args):

    proc_args = get_ferre_proc_args(args)
    
    # This function is ready to do things like those in get_ferre_proc_args
    # but for the additional feswi attributes
    
    return proc_args


################################################################################


def _check_arguments(proc_args):
    
    if proc_args.outspec == False:
        logging.warning('FESWI needs that the output spectra from ferre')
    
    if proc_args.join_arms == True:
        logging.warning('FESWI expects to receive non-joined arms from ferre')


################################################################################


def _check_setup(filename):

    l1_dict = l1_fileinfo([filename])

    setups = l1_dict['setups'][0]

    feswi_setup_flag = (setups == 'MOSLR11')
    
    if feswi_setup_flag == False:
        logging.warning('unsupported setups for FESWI: {}'.format(setups))
    
    return feswi_setup_flag


################################################################################


def _assert_paths_and_files_exist(proc_args):
    
    assert os.path.exists(proc_args.feswi_path)
    assert os.path.isdir(proc_args.feswi_path)
    
    assert os.path.exists(proc_args.outpath)
    assert os.path.isdir(proc_args.outpath)
    
    assert os.path.exists(proc_args.param_fits)
    assert os.path.isfile(proc_args.param_fits)
    assert os.path.exists(proc_args.spec_fits)
    assert os.path.isfile(proc_args.spec_fits)
    
    assert os.path.exists(proc_args.feswi_spectral_windows)
    assert os.path.isfile(proc_args.feswi_spectral_windows)
    assert os.path.exists(proc_args.feswi_cold_blue_grid)
    assert os.path.isfile(proc_args.feswi_cold_blue_grid)
    assert os.path.exists(proc_args.feswi_cold_red_grid)
    assert os.path.isfile(proc_args.feswi_cold_red_grid)
    assert os.path.exists(proc_args.feswi_hot_blue_grid)
    assert os.path.isfile(proc_args.feswi_hot_blue_grid)
    assert os.path.exists(proc_args.feswi_hot_red_grid)
    assert os.path.isfile(proc_args.feswi_hot_red_grid)


################################################################################


def feswi_weave(options=None):
    
    parser = _get_feswi_parser()
    
    # Get args and prog_args like in ferre
    # (extra attributes from FESWI will not be a problem)
    
    args = get_args(parser, options=options)
    proc_args = _get_feswi_proc_args(args)
    
    # Print the args in the same way than ferre module does it
    
    print_args(args, module='FW',
               #*** Discuss with Alireza if he likes FW
               #*** Discuss with Alireza what should write for version
#               version=aps_constants.__aps_ferre_version__,
               path=proc_args.outpath, headname=proc_args.headname)
    
    # Check if the provided args match FESWI requirements
    
    if _check_arguments(proc_args) == False:
        logging.warning('FESWI will not do anything due to the provided args')
        return None
    
    # If requested, we run ferre before calling FESWI
    
    if proc_args.feswi_run_ferre == True:
        
        logging.info('running FERRE for FESWI')
        
        if _check_setup(proc_args.infiles[0]):
            logging.warning(
                'FESWI will not do anything due to the setup of the input')
            return None
        
        try:
            proc_ferre(
                proc_args.infiles, proc_args.classfile, proc_args.param_fits,
                aps_ids=proc_args.aps_ids, targsrvy=proc_args.targsrvy,
                targclass=proc_args.targclass, mask_aps_ids=proc_args.mask_aps_ids,
                area=proc_args.area, mask_areas=proc_args.mask_areas,
                figdir=proc_args.figdir, wlranges=proc_args.wlranges,
                use_rvs=proc_args.use_rvs, rvsfile=proc_args.rvsfile,
                path=proc_args.workpath, outpath=proc_args.outpath,
                spec_fits=proc_args.spec_fits, pixel=proc_args.headname,
                templates=proc_args.templates, grid_prefix=proc_args.grid_prefix,
                grid_ids=proc_args.grid_ids, outspec=proc_args.outspec,
                sens_corr=proc_args.sens_corr, mask_gaps=proc_args.mask_gaps,
                safe_mask_gaps=proc_args.safe_mask_gaps,
                tellurics=proc_args.tellurics, vacuum=proc_args.vacuum,
                fill_gap=proc_args.fill_gap, arms_ratio=proc_args.arms_ratio,
                join_arms=proc_args.join_arms, nthreads=proc_args.mp,
                ferre=proc_args.ferre_exe)
        except:
            logging.error('FERRE failed when called from FESWI')
            raise Exception
    
    else:
        
        logging.info('skipping running FERRE for FESWI')
    
            
    # Save the current working directory and move to the output path
    #*** feswi will include an outpath parameter in a near future to avoid this
    
    orig_dir = os.path.abspath(os.curdir)
    os.chdir(proc_args.outpath)
    
    # Assert that the paths and files to be provided to FESWI exist
    
    _assert_paths_and_files_exist(proc_args)
    
    # Add feswi_path to the Python path and import feswi
    
    sys.path.insert(1, proc_args.feswi_path)
    from feswi import feswi
    
    # Call FESWI
    
    logging.info('running FESWI')
    
    feswi_filename = feswi(proc_args.param_fits, proc_args.spec_fits,
          spectral_windows_file=proc_args.feswi_spectral_windows,
          cold_blue_grid_file=proc_args.feswi_cold_blue_grid,
          cold_red_grid_file=proc_args.feswi_cold_red_grid,
          hot_blue_grid_file=proc_args.feswi_hot_blue_grid,
          hot_red_grid_file=proc_args.feswi_hot_red_grid,
          ferre_cmd=proc_args.ferre_exe,
          ferre_ndim=proc_args.feswi_ferre_ndim,
          ferre_f_format=proc_args.feswi_ferre_f_format,
          ferre_f_access=proc_args.feswi_ferre_f_access,
          ferre_nthreads=proc_args.feswi_ferre_nthreads,
          overwrite=proc_args.overwrite)
    
    # Restore the current working directory to the original path
    
    os.chdir(orig_dir)
    
    return feswi_filename


################################################################################


if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    debug_demo = [
        # Options to be inherited in a potential call to ferre
        '--infiles', '<PYAPS_DATA>/<night>/<obid>/stack_<runid>.fit', '<PYAPS_DATA>/<night>/<obid>/stack_<runid>.fit',
        '--classfile' , '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/zbest_stack_<runid>__stack_<runid>.fits',
        '--rvsfile' , '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/rvs_stack_<runid>__stack_<runid>.fits',
        '--templates' , '<PYAPS_DIR>/PyAPS_templates/templates_FR/',
        '--outpath', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/',
        '--aps_ids', '1006,1007', # or 'None' to run for all available fibreids
        '--grid_prefix' , 'n', # other options: 'm': 5D grids, 'p':PCA compressed 5D grids
        '--targsrvy', 'None',
        '--targclass', 'None',
        '--mask_aps_ids', 'None',
        '--area', 'None',
        '--mask_areas', 'None',
        '--headname', 'stack_<runid>__stack_<runid>',
        # '--wlranges', '4200.0,6000', '6000.0,8000',
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
        '--join_arms', 'True',
        '--fig', 'True',
        # Options to be only used by FESWI
        '--feswi_run_ferre', 'False',
        '--feswi_path', '<PYAPS_DIR>/CS/FESWI/',
        '--feswi_spectral_windows', '<PYAPS_DIR>/CS/FESWI/spectral_windows/spectral_windows-dev.json',
        '--feswi_cold_blue_grid', '<PYAPS_DIR>/PyAPS_templates/templates_FESWI/grid_cold-blue.dat',
        '--feswi_cold_red_grid', '<PYAPS_DIR>/PyAPS_templates/templates_FESWI/grid_cold-red.dat',
        '--feswi_hot_blue_grid', '<PYAPS_DIR>/PyAPS_templates/templates_FESWI/grid_hot-blue.dat',
        '--feswi_hot_red_grid', '<PYAPS_DIR>/PyAPS_templates/templates_FESWI/grid_hot-red.dat',
        '--feswi_ferre_ndim', '0',
        '--feswi_ferre_f_format', '0',
        '--feswi_ferre_f_access', '0',
        '--feswi_ferre_nthreads', '1']
    
    feswi_weave(options=debug_demo)
