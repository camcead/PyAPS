"""
!/usr/bin/env python3
Nikolay Britavskiy, 28/06/2021 (original version)
RR Lyrae analysis of WEAVE-GA survey. Part I. CS Gamma-velocities module.

Python wrapper to run CS RRLGV
versions:
 1.0 First version (Feb 2021)
 2.0 Second version (May 2021)
 3.0 Third version (June 2021)
 3.5 Modified to be ingested by APS (Alireza Molaeinezhad, IOA, Aug 2021)
 4.0 Modified to be ingested by APS (Nikolay Britavskiy and Alireza Molaeinezhad, Feb 2022)



# targclass or Targsurvey

# data L1 (coordinates) + L2 (stellar table); coordinates FK5 (EPOCH 2015) gaia dr2

# Command line run:
# python3 aps_rrlgv.py --apsfile <PYAPS_DIR>/PyAPS_results/<night>/<obid>/stack_<runid>__stack_<runid>_APS.fits --infiles <PYAPS_DATA>/opr4_jan2022/<night>/stack_<runid>.fit <PYAPS_DATA>/opr4_jan2022/<night>/stack_<runid>.fit --outpath <PYAPS_DIR>/PyAPS_results/<night>/<obid>/ --headname stack_<runid>__stack_<runid> --targclass STAR_RRL --aps_ids None --ephemfile <PYAPS_DIR>/CS/RRLGV/ephemerids_files/ephemerids.txt --calibrate_metal <PYAPS_DIR>/CS/RRLGV/metallic.txt  --calibrate_liu <PYAPS_DIR>/CS/RRLGV/liu.txt




History:
28/06/2021: Original version (version 3.0) by Nikolay Britavskiy
09/08/2021: (Alireza Molaeinezhad) find_edges_true_regions, find_edges and between functions imported directly to the code (form the original utilities code)
09/08/2021: (Alireza Molaeinezhad) outpath and headname keywords added to the code to redirect the output files
09/08/2021: (Alireza Molaeinezhad) code description updated
16/02/2022: (Alireza Molaeinezhad) ephemerids.txt file updated based on the email from Giuseppina Battaglia on 11 Feb 2022
21/02/2022: (Nikolay Britavskiy and Alireza Molaeinezhad) Use of gen_targlist from PyAPS to search for suitable aps_ids and targclasses
21/05/2022: (Nikolay Britavskiy) Fits output header fixed according to the bug description
"""


import numpy as np
import sys, re, os
os.environ['OMP_NUM_THREADS'] = '1'
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import argparse

import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist, islandinfo
from PyAPS import aps_constants

__aps_rrlgv_version__=5.0
APSVERS = PyAPS.__version__



def find_nearest(array, value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return array[idx], idx


def main_RRLGV(rrlgv_args):

    name_file_aps, name_file_ephem, mark_targ_class, infiles, calibrate_liu, calibrate_metal, aps_ids, outpath, headname = rrlgv_args

    assert os.path.exists(name_file_aps), 'No APS file found'


    # Reading the L1_file folder
    hdulist0 = pyfits.open(infiles[0])
    # hdulist0.info()

    data0 = hdulist0['FIBTABLE'].data
    header_aps = hdulist0[0].header

    JD_L1 = header_aps['JD']
    MJD_OBS_L1 = 2400000+header_aps['MJD-OBS']

    fiber_id0 = data0['FIBREID']
    ra_aps = data0['TARGRA']
    dec_aps = data0['TARGDEC']
    pm_ra_L1 = data0['TARGPMRA']
    pm_dec_L1 = data0['TARGPMDEC']

    try:
        targ_class_L1 = data0['TARGCLASS']
    except:
        targ_class_L1 = data0['TARGPROG']

    # Reading the APS ouptut file from the L2_files folder
    hdulist1 = pyfits.open(name_file_aps, ignore_missing_end=True)
    # hdulist1.info()

    # Reading the APS file header and other fits tables in it.
    data = hdulist1['CLASS_TABLE'].data
    data_rvs = hdulist1['STAR_TABLE'].data

    #Reading the specific INPUT parameters from the headers and the tables of given L2 file.

    targ_name = data['TARGID']
    targ_group = data['TARGSRVY']
    cname_L2 = data['CNAME']
    fiber_id = data['APS_ID']
    fiber_id = list(fiber_id)

    fiberid_rvs = data_rvs['APS_ID']
    vrad_rvs_all = data_rvs['VRAD']
    err_vrad_rvs_err = data_rvs['VRAD_ERR']
    snr_rvs_all = data_rvs['SNR_RVS']
    fiberid_rvs = list(fiberid_rvs)
    fiber_id0 = list(fiber_id0)

    targ_class = []
    name = []
    cname = []
    snr = []
    jd = []
    rvs = []
    err_rvs = []
    ra = []
    dec = []
    c2 = []
    pm_ra = []
    pm_dec = []
    MJD_OBS = []
    targsrvy = []
    FIBERID = []

    # Start to make a final array which combine RVS table with a main header of OB per each target.


    ## Data preparation

    infiles_check = l1_fileinfo(infiles)
    mod_aps_ids, mod_idt, mod_info, mod_la, mod_wcs = gen_targlist(infiles_check['infiles'][0],
        infiles_check['mode'], aps_ids = aps_ids, targsrvy= None, targclass = targ_class_L1,
        mask_aps_ids = None, area=None, mask_areas=None, la_out=False)

    # ## Make sure at least one valid aps_id exist after applying all filters
    assert len(mod_aps_ids) > 0, 'No valid APS_ID(s) found'


    ## Loop over all fibreids that pass both aps_ids and targclass conditions
    for i in mod_aps_ids:

        ind_fiber = fiber_id0.index(i)

        jd.append(JD_L1)
        MJD_OBS.append(MJD_OBS_L1)
        targ_class.append(targ_class_L1[ind_fiber])
        ra.append(float(ra_aps[ind_fiber]))
        dec.append(float(dec_aps[ind_fiber]))
        pm_ra.append(float(pm_ra_L1[ind_fiber]))
        pm_dec.append(float(pm_dec_L1[ind_fiber]))
        c2.append(SkyCoord(ra_aps[ind_fiber] * u.deg, dec_aps[ind_fiber] * u.deg))
        FIBERID.append(fiber_id0[ind_fiber])

        try:
            ind_targid_L2_rvs = fiberid_rvs.index(fiber_id0[ind_fiber])
            rvs.append(vrad_rvs_all[ind_targid_L2_rvs])
            err_rvs.append(err_vrad_rvs_err[ind_targid_L2_rvs])
            snr.append(snr_rvs_all[ind_targid_L2_rvs])

        except:
            rvs.append(-999)
            err_rvs.append(-999)
            snr.append(-999)

        try:
            ind_targid_L2 = fiber_id.index(fiber_id0[ind_fiber])
            name.append(targ_name[ind_targid_L2])
            cname.append(cname_L2[ind_targid_L2])
            targsrvy.append(targ_group[ind_targid_L2])

        except:
            name.append('000')
            cname.append('000')
            targsrvy.append('000')
    

    del hdulist1, hdulist0, data, data_rvs, data0

    ra = np.asarray(ra)
    dec = np.asarray(dec)
    pm_ra = np.asarray(pm_ra)
    pm_dec = np.asarray(pm_dec)

    snr = np.asarray(snr)
    jd = np.asarray(jd)
    MJD_OBS = np.asarray(MJD_OBS)

    targsrvy = np.asarray(targsrvy)
    rvs = np.asarray(rvs)
    err_rvs = np.asarray(err_rvs)
    FIBERID = np.asarray(FIBERID)

    # Reading the synthetic radial velocity curves
    # Reading the synthetic radial velocity сuvrve from Sesar at al. (2012)
    fmod_curve = ascii.read(calibrate_metal)

    phase = fmod_curve['col1']
    rv_curve = fmod_curve['col2']-(max(fmod_curve['col2'])+min(fmod_curve['col2']))/2  #Normalising the radial velocity curves

    # Reading the synthetic radial velocity сuvrve from Liu (1991)
    fmod_curve = ascii.read(calibrate_liu)
    phase_l = fmod_curve['col1']
    rv_curve_l = fmod_curve['col2']

    epoch_coord = []
    arr = []
    name_star = []
    epoch = []
    period = []
    d_period = []
    amplitude = []
    ra_ephem = []
    dec_ephem = []
    c1 = []
    # Reading the file with the ephemerids (catalog):

    try:
        inp = open(name_file_ephem, "r")  # Reading the catalog


        lines = inp.readlines()[1:]
        for line in lines:
            line_t = re.sub(r' +', '',line)
            numbers = line_t.split(',')

            arr.append(numbers)


        for i in range(0,len(arr)):
            ra_ephem.append(float(arr[i][0]))
            dec_ephem.append(float(arr[i][1]))
            epoch_coord.append(float(arr[i][2]))
            name_star.append(arr[i][3])
            epoch.append(float(arr[i][4]))
            period.append(float(arr[i][5]))
            d_period.append(float(arr[i][6]))
            amplitude.append(float(arr[i][7]))
            c1.append(SkyCoord(ra_ephem[i]*u.deg, dec_ephem[i]*u.deg))
    except:
        sys.exit('The ephemerids (external catalog) file is missing or in the incorrect format !!!')

    Vcm=-999

    # Creating a variables which will need to build the final table.
    ra_to_plot=[]
    dec_to_plot=[]
    c_name_to_plot=[]
    name_to_plot=[]
    name_star_to_plot=[]
    err_phase_to_plot=[]
    sep_to_plot=[]

    phase_obs_fin_plot=[]
    Vcm_plot=[]
    Vcm_d_m_plot=[]
    rvs_plot=[]
    Vcm_l_plot=[]
    jd_to_plot=[]


    targ_class_plot=[]
    pm_ra_to_plot=[]
    pm_dec_to_plot=[]
    err_rvs_plot=[]
    snr_plot=[]

    targsrvy_plot=[]
    mjd_plot=[]
    name_ephem_plot=[]
    fiberid_to_plot=[]
    epoch_coord_to_plot=[]
    d_period_to_plot=[]
    amplitude_to_plot=[]
    period_to_plot=[]

    k = 1.16
    ft = 0.03
    sigma_fit = 2.4 #km/s




    # Starting to crossmatch the L2 file for a given OB with the ephemerids file

    for i in range(0, len(name_star)):

        for jj in range(0, len(ra)):

            sep = c1[i].separation(c2[jj])  # separation between observed RAD DEC and ephemerids RA DEC

            if sep.arcsecond <= 1.0:      # The separation factor is 1 arcsecond.

                Vcm_d=[]
                phase_obs_d_d=[]
                phase_obs_fin_test_d=[]

                #####
                # Phase calculation
                phase_obs=((jd[jj]-epoch[i])/period[i])
                phase_obs_fin=(phase_obs-int(phase_obs))

                # Calulation of the phase uncertainties
                s_d = np.random.normal(period[i], d_period[i], 100)  # Starting the Monte-Carlo simulations.
                for j in range(0,100):
                    phase_obs_d_d.append((jd[jj]-epoch[i])/s_d[j])
                    phase_obs_fin_test_d.append(phase_obs_d_d[j]-int(phase_obs_d_d[j]))
                phase_obs_d=np.std(phase_obs_d_d)
                phase_error=np.std(phase_obs_fin_test_d)


                #####################
                ### Vcm determination
                Arv=amplitude[i]*25.6+35
                Arv_l=(amplitude[i]*40.5+42.7)/1.37

                '''
                Arv_ha=(amplitude[i]*35.6+78.2)
                Arv_hb=(amplitude[i]*42.1+51.1)
                Arv_hg=(amplitude[i]*46.1+38.5)
                '''
                [y_dx, indxy_phase]=find_nearest(phase,phase_obs_fin)
                Tf=rv_curve[indxy_phase]
                Vcm=(rvs[jj]-Arv*Tf) #Actual Gamma-Velocity

                # Vcm_sigma.append(((Arv+sigma_fit)**2)*(ft**2+(0.1*k)**2))
                # Calulation of the Vcm uncertainties

                s = np.random.normal(phase_obs_fin, phase_error, 100)
                for j in range(0,100):
                    [y_dx,indxy_phase] = find_nearest(phase,s[j])
                    Tf = rv_curve[indxy_phase]
                    Vcm_d.append(rvs[jj]-Arv*Tf)
                Vcm_d_m = np.sqrt(3**2+np.std(Vcm_d)**2)


                [y_dx, indxy_phase_l]=find_nearest(phase_l,phase_obs_fin)

                Tf_l = rv_curve_l[indxy_phase_l]
                Vcm_l = (rvs[jj]-Arv_l*Tf_l+Arv_l/2)
                Vcm_sigma = (((Arv+sigma_fit)**2)*(ft**2+(0.1*k)**2))
                #####

                # Converting the values to the arrays which will go to the final FITS table
                jd_to_plot.append(jd[jj])
                ra_to_plot.append(ra[jj])
                dec_to_plot.append(dec[jj])
                c_name_to_plot.append(cname[jj])
                name_to_plot.append(name[jj])

                name_star_to_plot.append(name_star[i])
                epoch_coord_to_plot.append(round(epoch_coord[i],1))
                period_to_plot.append(round(period[i],8))
                d_period_to_plot.append(round(d_period[i],8))
                amplitude_to_plot.append(round(amplitude[i],2))


                err_phase_to_plot.append(round(phase_error,2))
                sep_to_plot.append(round(sep.arcsecond,2))
                phase_obs_fin_plot.append(round(phase_obs_fin,2))
                Vcm_plot.append(round(Vcm,2))
                Vcm_d_m_plot.append(round(Vcm_d_m,2))
                rvs_plot.append(round(rvs[jj],2))
                Vcm_l_plot.append(round(Vcm_l,2))


                targ_class_plot.append(targ_class[jj])
                pm_ra_to_plot.append(pm_ra[jj])
                pm_dec_to_plot.append(pm_dec[jj])
                err_rvs_plot.append(round(err_rvs[jj],2))
                snr_plot.append(round(snr[jj],2))


                targsrvy_plot.append(targsrvy[jj])
                mjd_plot.append(round(MJD_OBS[jj],8))
                name_ephem_plot.append(inp.name)
                fiberid_to_plot.append(FIBERID[jj])

    if Vcm == -999:  # if there is no RRLyraes in the given L2 data

        print('No RR Lyrae stars in the OB or not programme stars in ephemerids file!')
        jd_to_plot = [0.0]
        ra_to_plot = [0.0]
        dec_to_plot = [0.0]
        c_name_to_plot = ['none']
        name_to_plot = ['none']
        name_star_to_plot = ['none']
        err_phase_to_plot = [0.0]
        sep_to_plot = [0.0]
        phase_obs_fin_plot = [0.0]
        Vcm_plot = [0.0]
        Vcm_d_m_plot = [0.0]
        rvs_plot = [0.0]

        targ_class_plot = ['none']
        pm_ra_to_plot = [0.0]
        pm_dec_to_plot = [0.0]
        err_rvs_plot = [0.0]
        snr_plot = [0.0]
        targsrvy_plot = [0.0]
        mjd_plot = [0.0]
        name_ephem_plot = [0.0]
        fiberid_to_plot = [0.0]
        epoch_coord_to_plot = [0.0]


        d_period_to_plot = [0.0]
        amplitude_to_plot = [0.0]
        period_to_plot = [0.0]

    else:
        print('OK, we have data!')
    # Making the final FITS OUTPUT TABLE
    c1 = fits.Column(name='CNAME', array=c_name_to_plot, format='25A')
    c2 = fits.Column(name='TARGNAME', array=name_to_plot, format='15A')
    c3 = fits.Column(name='FIBREID', array=fiberid_to_plot, format='I')
    c4 = fits.Column(name='TARGSRVY', array=targsrvy_plot, format='15A')
    c5 = fits.Column(name='TARGCLASS', array=targ_class_plot, format='15A')
    c6 = fits.Column(name='TARGRA', array=ra_to_plot, format='D')
    c7 = fits.Column(name='TARGDEC', array=dec_to_plot, format='D')
    c8 = fits.Column(name='TARGPMRA', array=pm_ra_to_plot, format='D')
    c9 = fits.Column(name='TARGPMDEC', array=pm_dec_to_plot, format='D')
    c10 = fits.Column(name='SNR_RVS', array=snr_plot, format='D')
    c11 = fits.Column(name='JD', array=jd_to_plot, format='D',unit='days')
    c12 = fits.Column(name='MJD_OBS', array=mjd_plot, format='D',unit='days')
    c13 = fits.Column(name='VRAD_RVS', array=rvs_plot, format='E')
    c14 = fits.Column(name='VRAD_RVS_ERR', array=err_rvs_plot, format='E')

    c15 = fits.Column(name='CAT_EPH_RRLGV', array=name_ephem_plot, format='20A')
    c16 = fits.Column(name='EPH_EPOH_RRLGV', array=epoch_coord_to_plot, format='D')
    c17 = fits.Column(name='TARG_NAME-Ephem_RRLGV', array=name_star_to_plot, format='20A')
    c18 = fits.Column(name='PERIOD_RRLGV', array=period_to_plot, format='D',unit='days')
    c19 = fits.Column(name='ERR_PERIOD_RRLGV', array=d_period_to_plot, format='D',unit='days')
    c20 = fits.Column(name='V_AMPLITUDE_RRLGV', array=amplitude_to_plot, format='E',unit='mag')
    c21 = fits.Column(name='PHASE_RRLGV', array=phase_obs_fin_plot, format='E')
    c22 = fits.Column(name='ERR_PHASE_RRLGV', array=err_phase_to_plot, format='E')
    c23 = fits.Column(name='GVRAD_RRLGV', array=Vcm_plot, format='E',unit='km/s')
    c24 = fits.Column(name='ERR_GVRAD_RRLGV', array=Vcm_d_m_plot, format='E',unit='km/s')
    c25 = fits.Column(name='COO_SEP_RRLGV', array=sep_to_plot, format='E',unit='arcseconds')

    hdr = fits.Header()
    hdr['COMMENT'] = 'WEAVE Contributed Software: RRLGV'

    hdr['CS_CODE'] = 'RRLGV'
    hdr['CS_VER'] = '5.0'
    hdr['CS_NME1'] = 'Nikolay'
    hdr['CS_NME2'] = 'Britavskiy'
    hdr['CS_MAIL'] = os.environ.get('PYAPS_CS_MAIL', '')  # contributor contact, set via env (not hardcoded)
    hdr['PROV1001'] = os.path.basename(infiles[0])
    hdr['PROV1002'] = os.path.basename(name_file_aps)






    primary = fits.PrimaryHDU(header=hdr)
    t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25]) #making FITS table
    t1 = fits.HDUList([primary, t])


    output_file = os.path.join(outpath, headname)+'_RRLGV.fits'
    t1.writeto(output_file, overwrite=True)
    print('Output file created :%s' %(output_file))



############################################################################################################

def rrlgv_weave(options=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--apsfile", help='The input fits file, contains the aps table',
                        type=none_or_str, default=none_or_str, required=True)
    parser.add_argument("--ephemfile", help='The input ephemerids file, contains the ephemerids for a given catalogue of stars',
                        type=none_or_str, default=none_or_str, required=True)
    parser.add_argument("--targclass", help='targclass argument to select RR Lyraes',
                        type=none_or_str, default=None, required=False)
    parser.add_argument("--infiles", type=none_or_str, default=None,
                        required=True, help="input files", nargs='*')
    parser.add_argument("--outpath", help='Directory to keep WEAVE_RVS outputs', type=none_or_str, default=None, required=True)
    parser.add_argument("--headname", help='Output headname. The output filenames whill be generated based on this', 
        type=none_or_str, default=None, required=True)
    parser.add_argument("--calibrate_metal", type=none_or_str, default=None,
                        required=True, help="Path to the file with calibrated synthetic rv curve Sesar et al.")
    parser.add_argument("--calibrate_liu", type=none_or_str, default=None,
                        required=True, help="input Path to the file with calibrated synthetic rv curve Liu (1991)")
    parser.add_argument('--aps_ids', help='comma-separated list of WEAVE APS_IDs',
                        type=none_or_str, default=None, required=False)


    # Check if any command-line argument has been passed to the module. It counts the number of system arguments to check this.
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')

        args = parser.parse_args(options)

    aps_ids = None
    if args.aps_ids is not None:
        aps_ids = [int(x) for x in args.aps_ids.split(",")]


    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print("OUTPATH: %s Created!" %(args.outpath))
    outpath=args.outpath+os.path.sep
    outpath=outpath.replace(' ','')



    if args.infiles is None:
        raise Exception('You need to specify the spectra you want to fit')

    if args.apsfile is None:
        raise Exception('You need to specify the APS file to proceed')

    if args.headname is None:
        raise Exception('You need to specify a headname')


    targclass = None
    if args.targclass is not None:
        targclass = args.targclass


    rrlgv_args = [args.apsfile, args.ephemfile, targclass, args.infiles, args.calibrate_metal, args.calibrate_liu, aps_ids, outpath, args.headname]

    # print args and assigned/default values on the screen
    print_args(args,module='RRLGV', version= __aps_rrlgv_version__, path=outpath, headname=args.headname)


    try:
        main_RRLGV(rrlgv_args)
    except:
        print('ERROR : %s' %(sys.exc_info()[1]))
        return


########################################################################


if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).

    debug_demo = [
        '--apsfile', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/stack_<runid>__stack_<runid>_APS.fits',
        '--infiles', '<PYAPS_DATA>/opr4_jan2022/<night>/stack_<runid>.fit', '<PYAPS_DATA>/opr4_jan2022/<night>/stack_<runid>.fit',
        '--targclass', 'STAR_RRL',
        '--aps_ids', '418,293', #'418,293,768,605,641,83,461,1005,942,457,390,260,881,33,559,345,749,680,99,334,780,981',
        '--outpath', '<PYAPS_DIR>/PyAPS_results/<night>/<obid>/',
        '--headname', 'stack_<runid>__stack_<runid>',
        '--ephemfile', '<PYAPS_DIR>/CS/RRLGV/ephemerids_files/ephemerids.txt',
        '--calibrate_metal', '<PYAPS_DIR>/CS/RRLGV/metallic.txt',
        '--calibrate_liu', '<PYAPS_DIR>/CS/RRLGV/liu.txt',

    ]

    rrlgv_weave(options=debug_demo)

