"""
This file contains all constant or pre-defined/default values for running PyAPS
20 June 2021: Updated version
27 Jan 2022: Versions and CDX and DEF_RES updated
18 Nov 2022: Versions updated
18 Nov 2022: CosmicRay rejection parameters added
15 feb 2024: Following a note from Mike, from May 2023, for all LIFU data in binning 2, the sampling is like the unbinned data
12 Sept 2024: gap_bands updated using the date up to July 2024
18 Aug 2026: Versions bumped to 1.9 -- per-bin spaxel-weighted LSF now the
  default for all IFU pipeline code (aps_ifu_Gal.py/aps_ifu_ExGal.py/
  aps_ifu_prepare.py's classification stage), see SPAXEL_WEIGHTED_LSF in
  every bundled LIFU*/MIFU* IFU_params JSON and doc/aps_ifu_ExGal.md's
  own "Spaxel-Weighted LSF" section for the full mechanism
"""

## Default APSOB parameters
offset_gap_pix=10
funit=1.0e18
large_error=1.0e18

# WEAVE fibre core diameter on sky, in arcsec — MOS and mIFU fibres are
# the same physical size (1.3"); LIFU's own fibres are larger (2.6").
# Sourced from the WEAVE survey-design paper (Jin et al., "The wide-field,
# multiplexed, spectroscopic facility WEAVE", arXiv:2212.03981) rather
# than derived/guessed from a FITS header keyword — checked directly,
# no such keyword exists in a real L1 FIBTABLE/primary header. Keyed by
# the same working-mode strings aps_utils.l1_fileinfo/gen_targlist already
# use (MOS/MOSLIFU/MOSMIFU) for the fibre-level modes, used for the
# optional "true fibre size" Aladin overlay mode in aps_explorer.py
# (explicit user request: "a circle with the size of the fibre width, so
# I can see exactly how the mapping is [each] fibre on the sky").
#
# "LIFU"/"MIFU" (bare, no MOS prefix) were deliberately left out of this
# dict until now, since those key a *stacked cube* (no single fibre per
# spatial position) rather than a per-fibre product -- see is_fibre_level
# elsewhere. Added (same physical values as their MOS-prefixed twins,
# same source) for aps_ifu_spaxel_contrib.py's opt-in spaxel-weighted
# LSF/FWHM feature, which needs the real individual-fibre footprint size
# for a cube's *underlying* single-exposure fibres, not the cube itself.
# Confirmed safe for every pre-existing caller (aps_explorer.py's
# _true_size_radius_deg/_CONTRIB_MARKER_RADIUS_DEG): none of them ever
# look up a bare "LIFU"/"MIFU" key today, so this is purely additive.
WEAVE_FIBRE_DIAMETER_ARCSEC = {
    "MOS": 1.3,
    "MOSMIFU": 1.3,
    "MOSLIFU": 2.6,
    "MIFU": 1.3,
    "LIFU": 2.6,
}

# Experimental (for Cosmic ray rejection)
cr_cut_level = 9.0 # Median cut level for CR rejection
cr_med_length = 3 # length of median filter for CR rejection
cr_wing_length = 1 # length of wing in each side of CR to be dealt with [following the discussion with Mile Irwin on 17 Nov 2022]


## VERSION CONTROL
##########################################################
__aps_version__            = '%10s' %(2.0) # 29 September 2026
__aps_rr_version__         = '%10s' %(2.0) # 29 September 2026
__aps_rvs_version__        = '%10s' %(2.0) # 29 September 2026
__aps_ferre_version__      = '%10s' %(2.0) # 29 September 2026
__aps_mosgal_version__     = '%10s' %(2.0) # 29 September 2026
__aps_mosExGal_version__   = '%10s' %(2.0) # 29 September 2026
__aps_L2_merge_version__   = '%10s' %(2.0) # 29 September 2026
__aps_ppxf_version__       = '%10s' %(2.0) # 29 September 2026
__aps_emi_version__       = '%10s' %(2.0) # 29 September 2026
__aps_ls_version__         = '%10s' %(2.0) # 29 September 2026
__aps_ifuls_version__      = '%10s' %(2.0) # 29 September 2026
__emippxf_version__        = '%10s' %(2.0) # 29 September 2026
__ls_version__             = '%10s' %(2.0) # 29 September 2026
__aps_ifuppxf_version__    = '%10s' %(2.0) # 29 September 2026
__aps_ifuemi_version__    = '%10s' %(2.0) # 29 September 2026
__aps_ifuls_version__      = '%10s' %(2.0) # 29 September 2026
__aps_OAT_version__        = '%10s' %(2.0) # 29 September 2026
__aps_ifugal_version__     = '%10s' %(2.0) # 29 September 2026
__aps_script_gen_version__ = '%10s' %(2.0) # 29 September 2026
__aps_ifu_version__        = '%10s' %(2.0) # 29 September 2026
__aps_l1_preview_version__ = '%10s' %(2.0) # 29 September 2026
##########################################################


## Define the default CD1_1 (MOS) or CD3_3 (IFU) for different configurations
## The main reason behind defining this dictionary is that, in OPR3b we have some binned data, prepared by different survey (NOT CASU)
## in which XBIN and YBIN in the header have not been updated.
##
## Later, instead of this dirty dictionary, we will evaluate the resolution by means of LSF file
## Update 15 Feburary 2024: Following a note from Mike, from May 2023, for all LIFU data in binning 2, the sampling is like the unbinned data
## Still n ot sure if that's the case for HR as well
##########################################################
cdx = {
        'MOSLR11'    :0.25,'MOSLR21'    :0.50, 'MOSLR41'    :1.0, 'MOSHR11'    :0.05, 'MOSHR21'    :0.10, 'MOSHR41'    :0.20,
        'LIFULR11'   :0.50,'LIFULR21'   :0.50, 'LIFULR41'   :1.0, 'LIFUHR11'   :0.10, 'LIFUHR21'   :0.20, 'LIFUHR41'   :0.40,
        'MIFULR11'   :0.25,'MIFULR21'   :0.50, 'MIFULR41'   :1.0, 'MIFUHR11'   :0.05, 'MIFUHR21'   :0.10, 'MIFUHR41'   :0.20,
        'MOSLIFULR11':0.50,'MOSLIFULR21':0.50, 'MOSLIFULR41':1.0, 'MOSLIFUHR11':0.10, 'MOSLIFUHR21':0.20, 'MOSLIFUHR41':0.40,
        'MOSMIFULR11':0.25,'MOSMIFULR21':0.50, 'MOSMIFULR41':1.0, 'MOSMIFUHR11':0.05, 'MOSMIFUHR21':0.10, 'MOSMIFUHR41':0.20 }


## DEFINE ALL AVAIBLE WORKING MODES FOR APS
## define default resolution in various configurations. (Later, it will be exported to an external config file, generated based on LSFs)
## Later, instead of this dirty dictionary, we will evaluate the resolution by means of LSF file
##########################################################

def_res = {
        'MOSLR11_RED'      :5000.00, 'MOSLR21_RED'      :2500.00, 'MOSLR41_RED'      :1250.00, 'MOSHR11_RED'     :20000.0, 'MOSHR21_RED'     :10000.0, 'MOSHR41_RED'     :5000.0,
        'MOSLR11_BLUE'     :5000.00, 'MOSLR21_BLUE'     :2500.00, 'MOSLR41_BLUE'     :1250.00, 'MOSHR11_BLUE'    :20000.0, 'MOSHR21_BLUE'    :10000.0, 'MOSHR41_BLUE'    :5000.0,
        'MOSHR11_GREEN'    :20000.0, 'MOSHR21_GREEN'    :10000.0, 'MOSHR41_GREEN'    :5000.0,
        'LIFULR11_RED'     :2500.00, 'LIFULR21_RED'     :1250.00, 'LIFULR41_RED'     :625.00,  'LIFUHR11_RED'    :10000.0, 'LIFUHR21_RED'    :5000.0,  'LIFUHR41_RED'    :2500.0,
        'LIFULR11_BLUE'    :2500.00, 'LIFULR21_BLUE'    :1250.00, 'LIFULR41_BLUE'    :625.00,  'LIFUHR11_BLUE'   :10000.0, 'LIFUHR21_BLUE'   :5000.0, 'LIFUHR41_BLUE'    :2500.0,
        'LIFUHR11_GREEN'   :10000.0, 'LIFUHR21_GREEN'   :5000.0, 'LIFUHR41_GREEN'    :2500.0,
        'MIFULR11_RED'     :5000.00, 'MIFULR21_RED'     :2500.00, 'MIFULR41_RED'     :1250.00, 'MIFUHR11_RED'    :20000.0, 'MIFUHR21_RED'    :10000.0, 'MIFUHR41_RED'    :5000.0,
        'MIFULR11_BLUE'    :5000.00, 'MIFULR21_BLUE'    :2500.00, 'MIFULR41_BLUE'    :1250.00, 'MIFUHR11_BLUE'   :20000.0, 'MIFUHR21_BLUE'   :10000.0, 'MIFUHR41_BLUE'   :5000.0,
        'MIFUHR11_GREEN'   :20000.0, 'MIFUHR21_GREEN'   :10000.0, 'MIFUHR41_GREEN'   :5000.0,
        'MOSLIFULR11_RED'  :2500.00, 'MOSLIFULR21_RED'  :1250.00, 'MOSLIFULR41_RED'  :625.00,  'MOSLIFUHR11_RED' :10000.0, 'MOSLIFUHR21_RED' :5000.0, 'MOSLIFUHR41_RED'  :2500.0,
        'MOSLIFULR11_BLUE' :2500.00, 'MOSLIFULR21_BLUE' :1250.00, 'MOSLIFULR41_BLUE' :625.00,  'MOSLIFUHR11_BLUE':10000.0, 'MOSLIFUHR21_BLUE':5000.0, 'MOSLIFUHR41_BLUE' :2500.0,
        'MOSLIFUHR11_GREEN':10000.0, 'MOSLIFUHR21_GREEN':5000.0,  'MOSLIFUHR41_GREEN':2500.0,
        'MOSMIFULR11_RED'  :5000.00, 'MOSMIFULR21_RED'  :2500.00, 'MOSMIFULR41_RED'  :1250.00, 'MOSMIFUHR11_RED' :20000.0, 'MOSMIFUHR21_RED' :10000.0, 'MOSMIFUHR41_RED' :5000.0 ,
        'MOSMIFULR11_BLUE' :5000.00, 'MOSMIFULR21_BLUE' :5000.00, 'MOSMIFULR41_BLUE' :5000.00, 'MOSMIFUHR11_BLUE':20000.0, 'MOSMIFUHR21_BLUE':10000.0, 'MOSMIFUHR41_BLUE':5000.0,
        'MOSMIFUHR11_GREEN':20000.0, 'MOSMIFUHR21_GREEN':10000.0, 'MOSMIFUHR41_GREEN':5000.0}



###define default gaps bands (wavelength ranges)
## to-do : Update the table for all modes

gap_bands = {
        'MOSLR_RED'       :[[5700.0,5850.0],[7540.0,7680.0],[9300.0,9800.0]],
        'MOSLR_BLUE'      :[[3400.0,3650.0],[5450.0,5565.0],[5920.0,6000.0]],
        'MOSHR_RED'       :[[5900.0,6080.0],[6360.0,6460.0],[6750.0,6880.0]],
        'MOSHR_GREEN'     :[[4600.0,4770.0],[5310.0,5370.0],[5460.0,5500.0]],
        'MOSHR_BLUE'      :[[3950.0,4020.0],[4510.0,4550.0],[4650.0,4700.0]],
        'LIFULR_RED'      :[[5700.0,5850.0],[7540.0,7680.0],[9300.0,9800.0]],
        'LIFULR_BLUE'     :[[3400.0,3650.0],[5450.0,5565.0],[5920.0,6000.0]],
        'LIFUHR_RED'      :[[5900.0,6080.0],[6360.0,6460.0],[6750.0,6880.0]],
        'LIFUHR_GREEN'    :[[4600.0,4770.0],[5310.0,5370.0],[5460.0,5500.0]],
        'LIFUHR_BLUE'     :[[3950.0,4030.0],[4500.0,4560.0],[4650.0,4700.0]],
        'MIFULR_RED'      :[[5700.0,5850.0],[7540.0,7680.0],[9300.0,9800.0]],
        'MIFULR_BLUE'     :[[3400.0,3650.0],[5450.0,5565.0],[5920.0,6000.0]],
        'MIFUHR_RED'      :[[5900.0,6080.0],[6360.0,6460.0],[6750.0,6880.0]],
        'MIFUHR_GREEN'    :[[4600.0,4770.0],[5310.0,5370.0],[5460.0,5500.0]],
        'MIFUHR_BLUE'     :[[3950.0,4020.0],[4510.0,4550.0],[4650.0,4700.0]],
        'MOSLIFULR_RED'   :[[5700.0,5850.0],[7540.0,7680.0],[9300.0,9800.0]],
        'MOSLIFULR_BLUE'  :[[3400.0,3650.0],[5450.0,5565.0],[5920.0,6000.0]],
        'MOSLIFUHR_RED'   :[[5900.0,6080.0],[6360.0,6460.0],[6750.0,6880.0]],
        'MOSLIFUHR_GREEN' :[[4600.0,4770.0],[5310.0,5370.0],[5460.0,5500.0]],
        'MOSLIFUHR_BLUE'  :[[3950.0,4020.0],[4510.0,4550.0],[4650.0,4700.0]],
        'MOSMIFULR_RED'   :[[5700.0,5850.0],[7540.0,7680.0],[9300.0,9800.0]],
        'MOSMIFULR_BLUE'  :[[3400.0,3650.0],[5450.0,5565.0],[5920.0,6000.0]],
        'MOSMIFUHR_RED'   :[[5900.0,6080.0],[6360.0,6460.0],[6750.0,6880.0]],
        'MOSMIFUHR_GREEN' :[[4600.0,4770.0],[5310.0,5370.0],[5460.0,5500.0]],
        'MOSMIFUHR_BLUE'  :[[3950.0,4020.0],[4510.0,4550.0],[4650.0,4700.0]],
        }

###define default tellurics bands (wavelength ranges)
# tellurics = [[8130.0,8350.0],[6850.0,7000.0],[8940.0,9240.0],[9250.0,9545.0]]
###AA update Dec 2025: added [7160,7340]
tellurics = [[8130.0,8350.0],[7160,7340],[6850.0,7000.0],[8940.0,9545.0]]

## Define the default bad overlap epsilon
bad_overlap_epsilon = 0.2
## Define the default bad overlap mask region in wavelenght
bad_overlap_mask = [5800.0,6800.0]
