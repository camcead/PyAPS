
from __future__ import absolute_import, division, print_function
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QGridLayout, QWidget, QVBoxLayout,QHBoxLayout,  QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import sys
import time
from astropy.io import fits
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QPlainTextEdit
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import re
import warnings
import traceback
import argparse
import copy
import json
import numpy as np
import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, l1_fileinfo, gen_targlist, filter_targetlist, read_infiles_list
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import aps_constants
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
import pyqtgraph as pg
from astropy.io import fits
import time
from PyAPS.samp import Aladin
from pyqtgraph import mkPen
APSVERS = PyAPS.__version__

"""
aps_l1_multi_plot
A collection of functions to visualize l1 data

versions:

 1.0 By A. Molaeinezhad (CASU/IOA, September 2022)
 2.0 By A. Molaeinezhad (CASU/IOA, October 2023)


Example:
./aps_l1_preview.py --help


History:
12 Sept 2022: Initial version
24 Oct 2023: Aladin support added. Plots are now in seperate windows.

"""
class L1_preview_flux(QtWidgets.QWidget):
    def __init__(self, targs):
        super().__init__()  # Change 'super(L1_preview_flux, self).__init' to 'super().__init'
        self.targs = targs
        self.targs_apstoid = {value.aps_id: index for index, value in enumerate(self.targs)}
        self.initUI()


        self.init_zoom = '120arcmin'
        self.RA_cent = 0.0
        self.DEC_cent = 0.0
        

    def initUI(self):

        # Create the second window for flux with p2
        self.win2 = pg.GraphicsLayoutWidget(show=True)
        self.win2.setBackground('white')
        self.win2.resize(800, 400)  # Adjust the size as needed
        self.win2.setWindowTitle('Window 2')

        self.p2 = self.win2.addPlot(row=1, col=1)
        self.p2.setLabel('left', 'FLUX (1e-18 erg/s/cm2/A)')
        self.p2.setLabel('bottom', 'Wavelength (Angstrom)')
        self.p2.showGrid(x=True, y=True)

        self.win2.show()

     

    def run(self):

        self.plot_spectra()
        pg.setConfigOptions(antialias=True)
        pg.exec()




    def plot_spectra(self):
        self.p2.clear()

        aps_ids = [value.aps_id for value in self.targs]

        for plt_id in aps_ids:
            target_id = self.targs_apstoid[int(plt_id)]
            if (target_id is None):
                return

            # self.p2.setLabel('top', "FIBREID: %s  CNAME: %s TARGID: %s" % (self.targs[target_id].meta[0]['APS_ID'],
            #                                                             self.targs[target_id].meta[0]['CNAME'],
            #                                                             self.targs[target_id].meta[0]['TARGID']))

            # self.p4.setLabel('top', "FIBREID: %s  CNAME: %s TARGID: %s" % (self.targs[target_id].meta[0]['APS_ID'],
            #                                                             self.targs[target_id].meta[0]['CNAME'],
            #                                                             self.targs[target_id].meta[0]['TARGID']))
            custom_pen_b = mkPen(color='b', width=0.5)
            custom_pen_r = mkPen(color='r', width=0.5)
            self.p2.plot(self.targs[target_id].spectra[0].wave, self.targs[target_id].spectra[0].flux, pen=custom_pen_b)
            self.p2.plot(self.targs[target_id].spectra[1].wave, self.targs[target_id].spectra[1].flux, pen=custom_pen_r)



def l1_preview(options=None):

    parser = build_common_parser(
        description=None,
        groups=["target_selection", "spatial_selection", "wavelength",
                "l1_processing"],
        # this tool has no --catdir/--caldir/--configdir and no
        # --outpath/--headname/--overwrite at all (it pops up an
        # interactive Qt preview, not a pipeline product) — neither
        # "caldirs" nor "output" is included above.
        overrides={
            # --infiles_list is an alternative to --infiles here, so
            # --infiles itself is optional (unlike every other script)
            "infiles": {"required": False},
            # this script's own wording/default differs from canonical,
            # preserved verbatim rather than silently switched
            "mask_aps_ids": {"help": "comma-separated list of APS_IDS to be masked"},
            "mask_gaps": {"default": False},
            "vacuum": {"default": False},
        },
        extra_args=[
            (("--infiles_list",), dict(type=none_or_str, default=None,
                required=False, help="An ASCII file holds input files, with each pair on a row separated by a comma.")),
            (("--l2_reference",), dict(type=none_or_str, default=None,
                required=False, help="A fits file serves as reference of l2 info for targets")),
            (("--crr",), dict(type=str2bool, default=False,
                required=False, help="if True, APS runs a fast and simple cosmic ray rejection algorithm")),
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

    if args.infiles is None and args.infiles_list is None:
        sys.exit('Either --infiles or --infiles_list must be specified')

    if args.infiles_list is not None:
        infiles_list = read_infiles_list(args.infiles_list)
        args.infiles = None
    else:
        infiles_list = [args.infiles]

    # generate the laster target_list from all set of infiles
    targ_list_master = []
    for c_infiles, i_infiles in enumerate(infiles_list):
        print(f"[{c_infiles+1}/{len(infiles_list)}] Preparing targets for {i_infiles}")

        # resolve_common_args needs a real args.infiles to run its
        # l1_fileinfo-based normalization against for THIS iteration's file
        # list (which can differ in length/order between --infiles_list
        # rows) — a shallow per-iteration copy keeps that isolated without
        # touching the outer `args` object used elsewhere in this function.
        # This also fixes a real bug: the previous code computed a
        # corrected join_arms into a local variable here but then never
        # used it — the downstream APSOB() call always used the raw,
        # uncorrected args.join_arms. It now correctly uses the
        # per-iteration corrected value.
        iter_args = copy.copy(args)
        iter_args.infiles = i_infiles
        resolved = resolve_common_args(iter_args)
        i_infiles, wlranges, arms_ratio = resolved.infiles, resolved.wlranges, resolved.arms_ratio
        aps_ids, targsrvy, targclass, mask_aps_ids = (
            resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)
        area, mask_areas = resolved.area, resolved.mask_areas

        # print args and assigned/default values on the screen — this tool
        # has no --headname/--outpath, so there's no per-run log file to
        # write to; print to screen only. Previously this tool never
        # logged its resolved arguments at all.
        print_args(iter_args, module="L1MultiPlot", version=aps_constants.__aps_version__,
                   screen_only=True)

        # Generate/read APS targets
        targs_i =APSOB(i_infiles, aps_ids=aps_ids, targsrvy= targsrvy, targclass = targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
            wlranges=wlranges, sens_corr=args.sens_corr, mask_gaps=args.mask_gaps,safe_mask_gaps=args.safe_mask_gaps, vacuum=args.vacuum, tellurics=args.tellurics,
            fill_gap=args.fill_gap, arms_ratio=arms_ratio, join_arms=iter_args.join_arms, funit= 1.0e18, offset_gap_pix = 10, collapse=False, crr=args.crr, aps_id_sum=int(100007*c_infiles))

        targ_list_master.extend(targs_i._targetlist)


    # Generate a master list of CNAMEs
    if args.l2_reference is not None:
        l2_ref_table = Table.read(args.l2_reference, format='fits')
    else:
        l2_ref_table = None

    selector = 'FEH'

    if selector =='FEH':
        # Step 2: Filter rows with SNR_RVS > 20
        filtered_table = l2_ref_table[l2_ref_table['SNR_RVS'] > 10]


        # Step 3: Select 20 samples with uniformly distributed FEH values
        unique_feh_values = np.unique(filtered_table['FEH'])
        n_unique_feh_values = len(unique_feh_values)

        if n_unique_feh_values >= 20:
            # If there are 20 or more unique FEH values, select 20 of them uniformly
            selected_feh_values = np.linspace(unique_feh_values.min(), unique_feh_values.max(), 20)
        else:
            # If there are fewer than 20 unique FEH values, just use all unique values
            selected_feh_values = unique_feh_values

        tolerance = 0.003  # Set your desired tolerance
        feh_mask = np.zeros(len(filtered_table), dtype=bool)

        for i, feh_value in enumerate(selected_feh_values):
            feh_mask |= np.isclose(filtered_table['FEH'], feh_value, atol=tolerance)

        # Apply the mask to get the final subset of 20 rows
        final_subset = filtered_table[feh_mask]

        # Step 4: generate list of CNAMEs
        cname_list = list(final_subset['CNAME'])
        mod_targ_list_master = filter_targetlist(targ_list_master, min_snr=None, cname_list = cname_list, TARGUSE=None)




    if selector =='FLUX':
    

        high_SNR_targs = []
        for targ in targ_list_master:
            if np.nanmean([targ.meta[0]['SNR'],targ.meta[1]['SNR']]) > 8 :
                high_SNR_targs.append(targ)

        all_flux_values = []
        for targ in high_SNR_targs:
            flux_targ = np.nanmean([targ.spectra[0].flux.mean(), targ.spectra[1].flux.mean()])
            all_flux_values.append(flux_targ)

        selected_flux_values = np.linspace(np.nanmin(all_flux_values), np.nanmax(all_flux_values), 20)
        # selected_flux_values = np.logspace(np.log10(np.nanmin(all_flux_values)), np.log10(np.nanmax(all_flux_values)), 20)
        tolerance = 0.3  # Set your desired tolerance
        flux_mask = np.zeros(len(high_SNR_targs), dtype=bool)

        for i, flux_value in enumerate(selected_flux_values):
            flux_mask |= np.isclose(all_flux_values, flux_value, atol=tolerance)

        true_indices = [ii for ii, value in enumerate(flux_mask) if value]

        # Apply the mask to get the final subset of 20 rows
        if len(true_indices) > 0:
            final_subset = []
            for ii in true_indices:
                final_subset.append(high_SNR_targs[ii])
        else:
            sys.exit('No targets found with the specified flux range.')

        # Step 4: generate list of CNAMEs
        cname_list = [targ.cname for targ in final_subset]

        mod_targ_list_master = filter_targetlist(targ_list_master, min_snr=None, cname_list = cname_list, TARGUSE=['T', 'C'])

    app = QtWidgets.QApplication([])
    L1_plot = L1_preview_flux(mod_targ_list_master)
    L1_plot.run()
    sys.exit(app.exec_())


if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).
    M15_demo= [
    '--infiles','<PYAPS_DATA>/L1/<night>/stack_<runid>.fit','<PYAPS_DATA>/L1/<night>/stack_<runid>.fit',
    '--infiles_list', '<PYAPS_DATA>/L1/<night>/filelist_20230924.cat',
    # '--infiles_list', 'None',
    '--l2_reference', '<PYAPS_DATA>/L1/<night>/l2_reference.fits',
    '--aps_ids', 'None',
    '--targsrvy', 'None',
    '--targclass', 'None',
    '--mask_aps_ids', 'None',
    '--area','None', # or '352.91089757874164,25.953070510899504,3.0,3.0,0.0',
    '--mask_areas', 'None',
    '--wlranges', '4000.0,5920.0', '5900.0, 9270.0',
    # '--wlranges', 'None',
    '--sens_corr', 'True',
    '--safe_mask_gaps', 'False',
    '--mask_gaps', 'False',
    '--tellurics', 'False',
    '--vacuum', 'False',
    '--fill_gap', 'False',
    '--arms_ratio', '1.0, 1.0',
    '--join_arms', 'False',
    '--crr' ,'False' ]
    l1_preview(options=M15_demo)