#!/usr/bin/env python3

"""
Nikolay Britavskiy, 28/06/2021 (Original version)

RR Lyraes analysis Part II. RRLEW
The code is consist of two main functions:
-- actualseefits_LR (analysing the WEAVE LR data)
-- actualseefits_HR (analysing the WEAVE HR data, green and red grating)

Version 3.0
Command line run:
python3 aps_rrlew.py --apsfile <PYAPS_DIR>/PyAPS_results/20160908/3434/stack_1002250__stack_1002249_APS.fits --infiles <PYAPS_DATA>/opr4_jan2022/20160908/stack_1002250.fit <PYAPS_DATA>/opr4_jan2022/20160908/stack_1002249.fit --outpath <PYAPS_DIR>/PyAPS_results/20160908/3434/ --headname stack_1002250__stack_1002249 --targclass STAR_RRL --aps_ids 418,293 --hr_lines <PYAPS_DIR>/CS/RRLEW/gala_test_HR_RRLYR_list.in --lr_lines <PYAPS_DIR>/CS/RRLEW/gala_test_LR_RRLYR_list.in



History:
28/06/2021: Original version (version 2.0) by Nikolay Britavskiy
09/08/2021: (Alireza Molaeinezhad) find_edges_true_regions, find_edges and between functions imported directly to the code (form the original utilities code)
09/08/2021: (Alireza Molaeinezhad) outpath and headname keywords added to the code to redirect the output files
09/08/2021: (Alireza Molaeinezhad) code description updated
16/02/2022: (Alireza Molaeinezhad) bugs fixed  and documentation updated
21/02/222: (Nikolay Britavskiy and Alireza Molaeinezhad) Use of gen_targlist from PyAPS to search for suitable aps_ids and targclasses
21/05/2022: (Nikolay Britavskiy) Fits output header fixed according to the bug description
"""




import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.io.ascii as ascii
import numpy as np
import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
import operator
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.signal import argrelextrema
import os.path
from scipy.interpolate import splrep, splev
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from PyAPS.aps_utils import APSOB
import argparse
from pyspark.sql.types import StructType as structtype

import PyAPS
from PyAPS.aps_utils import APSOB, makeR, print_args, none_or_str, str2bool, aps_ids_class,l1_fileinfo, gen_targlist, islandinfo
from PyAPS import aps_constants
APSVERS = PyAPS.__version__




# aps_utils parameters (Selecting the corresponding L1 data for a given L2 data):
# APS_UTILS parameters:
wlranges = None
aps_ids = None
targsrvy = None
targclass = None
mask_aps_ids = None
area = None
mask_areas = None
sens_corr = False
mask_gaps = True
vacuum = False
tellurics = True
fill_gap = True
arms_ratio = [1.0, 0.83]
join_arms = False
funit = 1.0e18
offset_gap_pix = 10

# Plot option, usually is 0, but if necessary the code can create the plots, with all information per analysed target. 
#To enable the option is necessary to change the flag to 1, Plot=1. However, creation of the plots is a very time-consuming process. 
plots = 0


def find_edges_true_regions(condition):
    """ Finds the indices for the edges of contiguous regions where
    condition is True.

    Examples
    --------
    >>> a = np.array([3,0,1,4,6,7,8,6,3,2,0,3,4,5,6,4,2,0,2,5,0,3])
    >>> ileft, iright = find_edges_true_regions(a > 2)
    >>> zip(ileft, iright)
    [(0, 0), (3, 8), (11, 15), (19, 19), (21, 21)]

    """
    indices, = condition.nonzero()
    if not len(indices):
        return None, None
    iright, = (indices[1:] - indices[:-1] > 1).nonzero()
    ileft = iright + 1
    iright = np.concatenate( (iright, [len(indices)-1]) )
    ileft = np.concatenate( ([0], ileft) )
    return indices[ileft], indices[iright]


def find_edges(centres, above_nmin, signif):
    """ Given the line centres and significance array, add the line
    edges, and whether it is part of a blend.

    Returns indices of left, centre, right edges.
    """
    # now find feature edges
    centres = np.asarray(centres)
    ind0,ind1 = find_edges_true_regions(above_nmin)
    cond =  ind1 - ind0 >= 2
    ind0 = ind0[cond]
    ind1 = ind1[cond]
    feature = []
    for i0,i1 in zip(ind0,ind1):
        cond = between(centres, i0, i1)
        if not np.any(cond):
            continue
        cen = centres[cond]
        if len(cen) == 1:
            # not blended
            feature.append([i0-1, cen[0], i1+1])
            continue
        #pl.vlines(cen, 0, 10, colors='y')
        right = cen[0] + signif[cen[0]:cen[1]+1].argmin()
        feature.append([i0-1, cen[0], right])
        #pl.vlines([i0-1,right[-1]], 0, 10)
        #pl.vlines([cen[0]], 0, 10, colors='r')
        #raw_input()
        for j in range(1, len(cen)-1):
            left = right
            c0,c1 = cen[j:j+2]
            right = c0 + signif[c0:c1+1].argmin()
            feature.append([left, c0, right])
            #pl.vlines([left[-1],right[-1]], 0, 10)
            #pl.vlines([c0], 0, 10, colors='r')
            #raw_input()
        left = right
        feature.append([left, cen[-1], i1+1])
        #pl.vlines([left[-1],right[-1]], 0, 10)
        #pl.vlines([cen[-1]], 0, 10, colors='r')
        #raw_input()

    # make sure the start and end indices are not negative or longer
    # than the array
    if feature == []:
        a=[0,0,0]
        return zip(a)
    else:
        feature[0][0] = max(0, feature[0][0])
        feature[-1][2] = min(len(signif)-1, feature[-1][2])
        return zip(*feature)

def between(a, vmin, vmax):
    """ Return a boolean array True where vmin <= a < vmax.

    (Careful of floating point issues when dealing with equalities
    though)
    """
    a = np.asarray(a)
    c = a < vmax
    c &= a >= vmin
    return c




# Creating a class for returning values of actualseefits_LR and actualseefits_HR functions
class ReturnValue(object):

    def __init__(self, name, rv, name_of_line, RV_local, EW_local, FWHM_local,EW_local_INT,SNR_new,EW_local_INT_T,SNR_test,fits_l_name_wave,fits_l_wave,fits_l_EW,fits_l_EW_err,fits_l_RV_local,ERR_RV_local,fits_l_ERR_RV_local):
        self.name = name
        self.rv = rv
        self.EW_local = EW_local
        self.name_of_line = name_of_line
        self.RV_local = RV_local 
        self.FWHM_local = FWHM_local
        self.EW_local_INT = EW_local_INT
        self.SNR_new = SNR_new ### SNR INT
        self.EW_INTEGR_T = EW_local_INT_T
        self.SNR_test = SNR_test #SNR FWHM gauss
        self.ERR_RV_local = ERR_RV_local 
        self.fits_l_name_wave=fits_l_name_wave
        self.fits_l_wave=fits_l_wave
        self.fits_l_EW=fits_l_EW
        self.fits_l_EW_err=fits_l_EW_err
        self.fits_l_RV_local=fits_l_RV_local
        self.fits_l_ERR_RV_local=fits_l_ERR_RV_local


def find_nearest(array, value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return array[idx], idx


def gaus(x,a,x0,sigma,y0):
    return y0-a*np.exp(-(x-x0)**2/(2*sigma**2))


def gauss_new(x, p): # p[0]==mean, p[1]==stdev
    return p[2]-(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))



def lorentz(x, *p):
    I, x0, gamma, y0 = p
    return y0 - I * gamma**2 / ((x - x0)**2 + gamma**2)


def continuum_script(my_wave,my_flux,number,order1,order_2):

    #number=45
    #yhat = savitzky_golay(my_flux,65,1)
    yhat = my_flux
    n_points=int(len(yhat)/number)
    max_w=[]
    max_y=[]

    for kk in range(0,number):

        xhat_t1=([el for el in my_wave[kk*n_points:(kk+1)*n_points]])
        yhat_t1=([el for el in yhat[kk*n_points:(kk+1)*n_points]])

        max_w_loc=[xhat_t1[0],xhat_t1[-1]]
        max_y_loc=[yhat_t1[0],yhat_t1[-1]]

        spline = splrep(max_w_loc,max_y_loc,k=1)
        continuum_loc = splev(xhat_t1,spline)
        yhat_t1_loc=yhat_t1/continuum_loc

        yhat_t1_loc[np.isnan(yhat_t1_loc)]=0
        yhat_t1_loc=([el for el in yhat_t1_loc])





        mean_noise_first=np.mean(yhat_t1_loc)
        std_noise_first=np.std(yhat_t1_loc)
        #print mean_noise_first,std_noise_first

        filtered_noise_first = [el for el in yhat_t1_loc if (el<mean_noise_first+2*std_noise_first)and(el>mean_noise_first-2*std_noise_first)]
        #print filtered_noise_first
        try:
        #mean_noise_first=np.mean(filtered_noise_first)
            while (np.mean(filtered_noise_first) - mean_noise_first)/mean_noise_first > 0.0005:
                mean_noise_first=np.mean(filtered_noise_first)
                std_noise_first=np.std(filtered_noise_first)
                filtered_noise_first = [el for el in filtered_noise_first if (el<mean_noise_first+1*std_noise_first)and(el>mean_noise_first-1*std_noise_first)]
        #print filtered_noise_first
            ind_noise_f=yhat_t1_loc.index(filtered_noise_first[0])



            xhat_t_ind=xhat_t1[ind_noise_f] 
            ind_wave_f=xhat_t1.index(xhat_t_ind)


            yhat_t_ind=yhat_t1[ind_wave_f]
    
            max_y.append(yhat_t_ind)
            max_w.append(xhat_t_ind)


        except:

            max_y.append(0)
            max_w.append(xhat_t1[0])
        
    #print 'filtered_noise_first', max_y,max_w

    yhat=([el for el in yhat])
    test_x=(my_wave[-25:])
    test_x=([el for el in test_x])
    [max_yh,max_ind]=max(enumerate(yhat[-25:]))
    #print max_yh
    #print test_x[max_yh]
    max_y.append(max_ind)
    max_w.append(test_x[int(max_yh)])

    z = np.polyfit(max_w, max_y,order1)
    p = np.poly1d(z)
    first_y = p(max_w)
    diff_y = np.asarray(abs(first_y-max_y))
    second_y = []
    second_x = []
    for kk in range(0,len(diff_y)):
        if abs(max_y[kk]-first_y[kk]) < 4*np.std(diff_y):
            second_y.append(max_y[kk])
            second_x.append(max_w[kk])


    z=np.polyfit(second_x, second_y,order1)
    p=np.poly1d(z)

    first_y=p(second_x)
    diff_y_second=np.asarray(abs(first_y-second_y))
    third_y=[]
    third_x=[]
    for kk in range(0,len(diff_y_second)):
        if (abs(second_y[kk]-first_y[kk]) < 3*np.std(diff_y_second))and(second_y[kk]!=0):
            third_y.append(second_y[kk])
            third_x.append(second_x[kk])

    z=np.polyfit(third_x,third_y,order_2)
    p=np.poly1d(z)
    continuum=p(my_wave)
    del spline,diff_y_second,first_y,p,z,second_y,second_x,diff_y,filtered_noise_first,std_noise_first,mean_noise_first,xhat_t1,yhat_t1,ind_noise_f
    return continuum, max_w, max_y, third_x, third_y


def actualseefits_LR(argumentsList, lr_list):
    c = 299792.458 #km/s
    path = os.getcwd()
    global pp1 
    global fig
    #linelist_array = np.loadtxt("line_list.dat", dtype={'names': ('Name_wave', 'wavelength'),'formats': ('|S15',float)},delimiter=' ', skiprows=0)
    #name_of_line=linelist_array['Name_wave']


    fmod_obs = ascii.read(lr_list)

    l_wavelength = np.asarray(fmod_obs['col1'])
    l_namewave = np.asarray(fmod_obs['col4'])
    l_5 = np.asarray(fmod_obs['col5'])
    l_6 = np.asarray(fmod_obs['col6'])
    l_7 = np.asarray(fmod_obs['col7'])
    l_8 = np.asarray(fmod_obs['col8'])
    l_9 = np.asarray(fmod_obs['col9'])
    l_10 = np.asarray(fmod_obs['col10'])

    name_of_line=l_namewave
    wave_test=np.asarray(argumentsList[0])
    flux_or=np.asarray(argumentsList[1])

    wave_test_red=np.asarray(argumentsList[14])
    flux_or_red=np.asarray(argumentsList[15])

    wave_step=wave_test[1]-wave_test[0]
    
    APS_RVS=float(argumentsList[11])
    plots=float(argumentsList[6])



    mask_find_blue=np.logical_and(np.asarray(wave_test)<= 6000, np.asarray(wave_test)>= 3800)
    mask_find_red=np.logical_and(np.asarray(wave_test_red)<=9200, np.asarray(wave_test_red)> 6000)

    wave_test=wave_test[mask_find_blue]
    wave_test_red=wave_test_red[mask_find_red]
    flux_or=flux_or[mask_find_blue]
    flux_or_red=flux_or_red[mask_find_red]




    myspec = [wave_test,flux_or]
    myspec_red =[wave_test_red,flux_or_red]
    myspec=np.asarray(myspec)
    myspec_red=np.asarray(myspec_red)

    #files=argumentsList[4]
    files = str(argumentsList[16])



    name_inp=argumentsList[4]
    mag_inp=argumentsList[5]


    my_wave = np.asarray(myspec[0])
    my_flux = np.asarray(myspec[1])
    my_wave_red = np.asarray(myspec_red[0])
    my_flux_red = np.asarray(myspec_red[1])

    INP_name=argumentsList[7]


    mask_find_cut0a=np.logical_and(np.asarray(my_wave)<= 4050, np.asarray(my_wave)<= 4050)
    mask_find_cut1a=np.logical_and(np.asarray(my_wave)> 4050, np.asarray(my_wave)<= 5600)
    mask_find_cut2a=np.logical_and(np.asarray(my_wave)> 5600, np.asarray(my_wave)> 5600)


    mask_find_cut1b=np.logical_and(np.asarray(my_wave_red)<= 7500, np.asarray(my_wave_red)<= 7500)
    mask_find_cut2b=np.logical_and(np.asarray(my_wave_red)> 7500, np.asarray(my_wave_red)> 7500)

    my_wave0=my_wave[mask_find_cut0a]
    my_flux0=my_flux[mask_find_cut0a]
    my_wave1=my_wave[mask_find_cut1a]
    my_wave2=my_wave[mask_find_cut2a]
    my_flux1=my_flux[mask_find_cut1a]
    my_flux2=my_flux[mask_find_cut2a]

    my_wave_red1=my_wave_red[mask_find_cut1b]
    my_wave_red2=my_wave_red[mask_find_cut2b]
    my_flux_red1=my_flux_red[mask_find_cut1b]
    my_flux_red2=my_flux_red[mask_find_cut2b]

    if all(my_flux == 0):
        bbb=[]
        print('No flux in the spectrum!')
        zzz = ([float(0) for el in l_wavelength])
        yyy = np.array(["No flux" for el in l_wavelength])
        for el in l_wavelength:
            bbb.append(float(0))
        #print bbb


        
        if plots==1:
            path_file_plots=path+"/Plots/"
            if not os.path.exists(path_file_plots):
                os.makedirs(path_file_plots)


            pp1 = PdfPages(path_file_plots+files+'.pdf')
            my_dpi=100
            plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
            ax1=plt.subplot(411)
            plt.plot(my_wave,my_flux,'b-',lw=1)
            plt.title('%s       %s       I MAG = %2.2f'%(INP_name,files,mag_inp))

        return ReturnValue(files, 0, yyy, bbb, bbb, bbb, bbb, bbb, bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb)

    else:



        continuum0,max_w0,max_y0,second_x0,second_y0=continuum_script(my_wave0,my_flux0,int(10),int(3),int(4))
        continuum1,max_w1,max_y1,second_x1,second_y1=continuum_script(my_wave1,my_flux1,20,5,3)
        continuum2,max_w2,max_y2,second_x2,second_y2=continuum_script(my_wave2,my_flux2,35,6,6)
        continuum_red1,max_w_red1,max_y_red1,second_x_red1,second_y_red1=continuum_script(my_wave_red1,my_flux_red1,30,5,3)


        continuum_red2,max_w_red2,max_y_red2,second_x_red2,second_y_red2=continuum_script(my_wave_red2,my_flux_red2,15,6,3)


        continuum_n=[]
        continuum_redn=[]
        continuum_n.extend(continuum0)
        continuum_n.extend(continuum1)
        continuum_n.extend(continuum2)

        continuum_redn.extend(continuum_red1)
        continuum_redn.extend(continuum_red2)
        continuum=np.asarray(continuum_n)
        continuum_red=np.asarray(continuum_redn)


        max_w=[max_w0+max_w1+max_w2]
        max_y=[max_y0+max_y1+max_y2]


        second_x=[second_x0+second_x1+second_x2]
        second_y=[second_y0+second_y1+second_y2]

    
        max_w_red=[max_w_red1+max_w_red2]
        max_y_red=[max_y_red1+max_y_red2]
        second_x_red=[second_x_red1+second_x_red2]
        second_y_red=[second_y_red1+second_y_red2]

        '''
        myspec_test =np.array([wave_test,flux_or])
        myspec_test=myspec_test.T
        with open(path_file_spectra+files+'.asc',"w") as File_ascii_or:
            np.savetxt(File_ascii_or,myspec_test, fmt=['%4.3f','%4.3f'])
        File_ascii_or.close()


        myspec_test =np.array([wave_test,flux_or/continuum])
        myspec_test=myspec_test.T
        with open(path_file_spectra+files+'_c.asc',"w") as File_ascii_or:
            np.savetxt(File_ascii_or,myspec_test, fmt=['%4.3f','%4.3f'])
        File_ascii_or.close()



        myspec_test =np.array([wave_test_red,flux_or_red])
        myspec_test=myspec_test.T
        with open(path_file_spectra+files+'R.asc',"w") as File_ascii_or:
            np.savetxt(File_ascii_or,myspec_test, fmt=['%4.3f','%4.3f'])
        File_ascii_or.close()


        myspec_test =np.array([wave_test_red,flux_or_red/continuum_red])
        myspec_test=myspec_test.T
        with open(path_file_spectra+files+'R_c.asc',"w") as File_ascii_or:
            np.savetxt(File_ascii_or,myspec_test, fmt=['%4.3f','%4.3f'])
        File_ascii_or.close()
        '''

        ##FIGURE1
        if plots==1:
            path_file_plots=path+"/Plots/"
            if not os.path.exists(path_file_plots):
                os.makedirs(path_file_plots)
            #global pp1 

            pp1 = PdfPages(path_file_plots+files+'.pdf')

            my_dpi=100
            plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
            ax1=plt.subplot(411)
            plt.plot(my_wave,my_flux,'b-',lw=1)
            plt.plot(my_wave,continuum,'--')
            plt.plot(max_w,max_y,'rX')
            plt.plot(second_x,second_y,'gX')
            plt.title('%s       %s       I MAG = %2.2f'%(INP_name,files,mag_inp))


        my_flux=my_flux/continuum
        myspec=[wave_test,my_flux]

        myspec_blue=myspec


    ############################################
    ###############First SNR estimations
    ############################################


        myspec[0]=np.asarray(myspec[0])
        myspec[1]=np.asarray(myspec[1])
        myspec_red[0]=np.asarray(myspec_red[0])
        myspec_red[1]=np.asarray(myspec_red[1])

############################################
###############First SNR estimations
############################################



        mask_find1=np.logical_and(np.asarray(myspec[0])>=   4160, np.asarray(myspec[0])<= 4260)
        mask_find2=np.logical_and(np.asarray(myspec[0])>=   5200, np.asarray(myspec[0])<= 5300)
    #   mask_find3=np.logical_and(myspec[0]>=   6300, myspec[0]<= 6400)
        #print myspec[0],mask_find1,mask_find2
        if any(mask_find1) == True :            

            wave_find1= my_wave[mask_find1]
            flux_find1= my_flux[mask_find1]
        ####    
            n_points=int(len(flux_find1)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_find1[0])
            max_y.append(flux_find1[0])
            for kk in range(1,9):
                xhat_t=([el for el in wave_find1[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_find1[kk*n_points:(kk+1)*n_points]])
                max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))


                max_y.append(max_d_line)
                max_w.append(xhat_t[max_index])
                y_points.append(flux_find1[kk*n_points])
                x_points.append(wave_find1[kk*n_points])
            max_w.append(wave_find1[-1])
            max_y.append(flux_find1[-1])
            spline = splrep(max_w,max_y,k=1)
            continuum1 = splev(wave_find1,spline)
            flux_find1=flux_find1/continuum1
        ####
            mean_noise = np.mean(flux_find1)
            std_noise = np.std(flux_find1)

            filtered_noise = [el for el in flux_find1 if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            #while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise1=filtered_noise
            noise1=np.mean(filtered_noise)
            std_noise1=np.std(filtered_noise)
        else:
            wave_find1=0
            noise1=0
        if any(mask_find2) == True :            

            wave_find2= my_wave[mask_find2]
            flux_find2= my_flux[mask_find2]

        ####    
            n_points=int(len(flux_find2)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_find2[0])
            max_y.append(flux_find2[0])
            for kk in range(1,9):
                xhat_t=([el for el in wave_find2[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_find2[kk*n_points:(kk+1)*n_points]])
                max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))
                max_y.append(max_d_line)
                max_w.append(xhat_t[max_index])
                y_points.append(flux_find2[kk*n_points])
                x_points.append(wave_find2[kk*n_points])
            max_w.append(wave_find2[-1])
            max_y.append(flux_find2[-1])
            spline = splrep(max_w,max_y,k=1)
            continuum2 = splev(wave_find2,spline)
            flux_find2=flux_find2/continuum2
        ####

            mean_noise = np.mean(flux_find2)
            std_noise = np.std(flux_find2)

            filtered_noise = [el for el in flux_find2 if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:#(np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise2=filtered_noise      
            noise2=np.mean(filtered_noise)
            std_noise2=np.std(filtered_noise)
        else:
            wave_find2=0
            noise2=0

        mask_find3=np.logical_and(myspec_red[0]>=   6750, myspec_red[0]<= 6850)
        #yhat_red = savitzky_golay(my_flux_red,65,1)
        if any(mask_find3) == True :            

            wave_find3 = my_wave_red[mask_find3]
            flux_find3 = my_flux_red[mask_find3]
        ####    
            n_points=int(len(flux_find3)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_find3[0])
            max_y.append(flux_find3[0])
            for kk in range(1,9):
                xhat_t=([el for el in wave_find3[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_find3[kk*n_points:(kk+1)*n_points]])
                max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))
                max_y.append(max_d_line)
                max_w.append(xhat_t[max_index])
                y_points.append(flux_find3[kk*n_points])
                x_points.append(wave_find3[kk*n_points])
            max_w.append(wave_find3[-1])
            max_y.append(flux_find3[-1])
            spline = splrep(max_w,max_y,k=1)
            continuum3 = splev(wave_find3,spline)
            flux_find3=flux_find3/continuum3
        ####

            mean_noise = np.mean(flux_find3)
            std_noise = np.std(flux_find3)

            filtered_noise = [el for el in flux_find3 if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:#(np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise3=filtered_noise
            noise3=np.mean(filtered_noise)
            std_noise3=np.std(filtered_noise)
        else:
            wave_find3=0
            noise3=0

    




    ############################################
    ############### Check if the wavelength ranges the Hydrogen lines are exist
    ############################################
        bbb=[]
        wavHeI5876=4340.47 ###### Hgamma FROM HERE starts the line minimum exploration
        wavHeI4471=4861.33


        mask1=np.logical_and(myspec[0]>=    wavHeI5876-5, myspec[0]<=wavHeI5876+5)
        mask2=np.logical_and(myspec[0]>=    wavHeI4471-5, myspec[0]<=wavHeI4471+5)

        #print 'MASK1', myspec[0],mask1
        if any(mask1) == True :
            print('Here is the H gamma!')
            mask=mask1 
            marker=0
        elif any(mask2) == True :
            print('Here is the H beta!')
            mask=mask2 
            marker=1
            wavHeI5876=4861.33

        else:
            print('No diagnostic line regions in the file!')
            zzz = ([float(0) for el in l_wavelength])
            yyy = np.array(["No H lines in file" for el in l_wavelength])
            for el in l_wavelength:
                bbb.append(float(0))



            my_wave = myspec[0]
            my_flux = myspec[1]


            if plots==1:

                ax2=plt.subplot(412)
                plt.title(files)
                plt.plot(myspec[0], myspec[1], color='blue',lw=1)

                for tt in range(0, len(l_namewave)):



                    plt.fill_between(wave_find1, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
                    plt.fill_between(wave_find2, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)

                    plt.plot(wave_find1,continuum1,'r-',linewidth=0.5)
                    plt.plot(wave_find1,flux_find1,'black',linewidth=0.5)
                    plt.plot([wave_find1[0],wave_find1[-1]],[min(filtered_noise1),min(filtered_noise1)],'cyan',linewidth=0.5)
                    plt.plot([wave_find1[0],wave_find1[-1]],[max(filtered_noise1),max(filtered_noise1)],'cyan',linewidth=0.5)

                    plt.plot(wave_find2,continuum2,'r-',linewidth=0.5)
                    plt.plot(wave_find2,flux_find2,'black',linewidth=0.5)
                    plt.plot([wave_find2[0],wave_find2[-1]],[min(filtered_noise2),min(filtered_noise2)],'cyan',linewidth=0.5)
                    plt.plot([wave_find2[0],wave_find2[-1]],[max(filtered_noise2),max(filtered_noise2)],'cyan',linewidth=0.5)

                    plt.text(wave_find1[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise1,std_noise1,1/std_noise1),rotation=90,fontsize=6,fontweight='ultralight')
                    plt.text(wave_find2[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise2,std_noise2,1/std_noise2),fontsize=6,rotation=90,fontweight='ultralight')

                    plt.plot([l_wavelength[tt],l_wavelength[tt]],[0,2],'gray',linewidth=0.2,linestyle='--') 
                    plt.text(l_wavelength[tt]-35,1.45,l_wavelength[tt],fontsize=6,rotation=90)
            


                plt.ylim([0.3,1.5])

                plt.xlim([3800,6200])

                ax3=plt.subplot(413)
                plt.title('Red part')

                plt.plot(my_wave_red,my_flux_red,'b-',lw=0.5)
                plt.plot(my_wave_red,continuum_red,'b--',lw=0.5)
                plt.plot(max_w_red,max_y_red,'rX')
                plt.plot(second_x_red,second_y_red,'gX')

                ax4=plt.subplot(414)
                plt.title('Red part')
                plt.fill_between(wave_find3, 0, 1.5, facecolor='green', interpolate=True,alpha=0.5)
                plt.plot(my_wave_red,my_flux_red/continuum_red,'b-',lw=0.5)
                plt.plot(wave_find3,continuum3,'r-',linewidth=0.5)
                plt.plot(wave_find3,flux_find3,'black',linewidth=0.5)
                plt.plot([wave_find3[0],wave_find3[-1]],[min(filtered_noise3),min(filtered_noise3)],'cyan',linewidth=0.5)
                plt.plot([wave_find3[0],wave_find3[-1]],[max(filtered_noise3),max(filtered_noise3)],'cyan',linewidth=0.5)
                plt.text(wave_find3[0]-35,0.9,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise3,std_noise3,1/std_noise3),fontsize=6,rotation=90,fontweight='ultralight')
                plt.ylim([0.3,1.5])
                plt.tight_layout()
                plt.savefig(pp1, format='pdf')
                my_dpi=90

                fig=plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
                gs1 = gridspec.GridSpec(3,4)
                my_wave_toplot = [ structtype() for tt in range(0, len(l_wavelength))]
                my_flux_toplot = [ structtype() for tt in range(0, len(l_wavelength))]

                for ttt in range(0, 12):

                    my_wave_toplot[ttt]=np.array(my_wave)
                    my_flux_toplot[ttt]=np.array(my_flux)
                    fig.add_subplot(gs1[ttt])

                    plt.plot(my_wave_toplot[ttt],my_flux_toplot[ttt],color='black')
                    #plt.plot(my_wave_toplot_INT[ttt],my_flux_toplot_INT[ttt],color='gray')

            




                    plt.plot(l_wavelength[ttt],l_wavelength[ttt],[0,2],'gray',linewidth=0.5,linestyle='--')     
            
                    wav_cen=l_wavelength[ttt]

                    plt.ylim([0.7,1.1])
                    plt.xlim([wav_cen-45,wav_cen+45])
                    plt.title(l_namewave[ttt])
                    plt.tight_layout()
            
                    plt.yticks(size=8)
                    plt.xticks(size=8)


                gs1.tight_layout(fig,rect=[0,0,1,0.95])

            return ReturnValue(files, 0, yyy, bbb, bbb, bbb, bbb, bbb, bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb)


############
############

#The RR Lyr analysis

############
############



    #wave_newSp=    my_wave[mask]
    #new_flux=my_flux[mask]

    #new_spec=[wave_newSp,new_flux]
    #min_index, min_d_line = min(enumerate(new_flux), key=operator.itemgetter(1))


    #radvel_diagnos=c*( (1.*new_spec[0][min_index]/wavHeI5876) - 1)
    radvel_diagnos=APS_RVS
    param=1.1
    step_gaus=1
    RV_local=[]
    ERR_RV_local=[]
    ERR_EW_local=[]
    FWHM_local=[]
    EW_local=[]
    EW_local_INT=[]
    wavel_local=[]
    SNR=[]
    SNR_new=[]
    SNR_test=[]
            
        
    item_x = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_int = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_noise = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_T = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_T_noise = [ structtype() for tt in range(0, len(l_namewave))]
    IND_NOISE_FWHM = [ structtype() for tt in range(0, len(l_namewave))]
    my_wave_toplot = [ structtype() for tt in range(0, len(l_namewave))]
    my_flux_toplot = [ structtype() for tt in range(0, len(l_namewave))]
###########################
############ START GAUSSIAN
###########################
    #full_path_file_vsini=path_file_plots+files+'_vsin.pdf'
    #pp3 = PdfPages(full_path_file_vsini)
    #fig1=plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))


    waveobs_shifted_blue=[]
    waveobs_shifted_red=[]

    waveobs_shifted_blue.append(-999)
    waveobs_shifted_red.append(-999)


    for t in range(0, len(l_wavelength)):
        
        if l_wavelength[t] > 6000:

            my_flux_red_n=my_flux_red/continuum_red
            myspec_red_n=[wave_test_red,my_flux_red_n]
            
            my_wave = myspec_red_n[0]
            my_flux = myspec_red_n[1]
            myspec = myspec_red_n
            waveobs_shifted_red = np.array([el*(1+(radvel_diagnos/c)) for el in my_wave])

            


        else:


            myspec=myspec_blue
            my_wave = myspec[0]
            my_flux = myspec[1]
            waveobs_shifted_blue = np.array([el*(1+(radvel_diagnos/c)) for el in my_wave])

        mask_noise=np.logical_and(myspec[0]>=l_wavelength[t]*(1+(radvel_diagnos/c))-50, myspec[0]<=l_wavelength[t]*(1+(radvel_diagnos/c))+50)
        if all(mask_noise==False):
            wave_noise=[0,0,0,0,0]
            flux_noise=[0,0,0,0,0]
        else:
            wave_noise = my_wave[mask_noise]
            flux_noise = my_flux[mask_noise]


            mask_noise_post=np.logical_and(flux_noise!=0,flux_noise>0)
            wave_noise=wave_noise[mask_noise_post]
            flux_noise=flux_noise[mask_noise_post]



            n_points=int(len(flux_noise)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_noise[0])
            max_y.append(flux_noise[0])
            for kk in range(1,9):
                #xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                #yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                #max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))


                xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                max_y.append(yhat_t[0])
                max_w.append(xhat_t[0])



                #max_y.append(max_d_line)
                #max_w.append(xhat_t[max_index])
                #y_points.append(flux_noise[kk*n_points])
                #x_points.append(wave_noise[kk*n_points])
            max_w.append(wave_noise[-1])
            max_y.append(flux_noise[-1])
            s_y=[]
            s_x=[]
            mean_noise = np.mean(max_y)
            std_noise = np.std(max_y)

            #filtered_noise = [el for el in max_y if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            #while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
            #   mean_noise=np.mean(filtered_noise)
            #   std_noise=np.std(filtered_noise)
            #   filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]


            for kk in range(0,len(max_y)):
                if abs((max_y[kk]<mean_noise+3*std_noise)and(max_y[kk]>mean_noise-3*std_noise)):
                    s_y.append(max_y[kk])
                    s_x.append(max_w[kk])
            s_y_1=[]
            s_x_1=[]
            mean_noise = np.mean(s_y)
            std_noise = np.std(s_y)
            for kk in range(0,len(s_y)):
                if abs((s_y[kk]<mean_noise+1*std_noise)and(s_y[kk]>mean_noise-1*std_noise)):
                    s_y_1.append(s_y[kk])
                    s_x_1.append(s_x[kk])


            z=np.polyfit([s_x_1[0],s_x_1[-1]],[s_y_1[0],s_y_1[-1]],1)
            #z=np.polyfit([s_x_1[0],s_x_1[-1]],[s_y_1[0],s_y_1[-1]],1)
            #z=np.polyfit(s_x_1, s_y_1,1)
            p=np.poly1d(z)


            continuum=p(wave_noise)
            #spline = splrep(max_w,max_y,k=2)
            #continuum = splev(wave_noise,spline)
            flux_noise = flux_noise/continuum


            mask_noise_post=np.logical_and(wave_noise>= l_wavelength[t]*(1+(radvel_diagnos/c))-25, wave_noise<=l_wavelength[t]*(1+(radvel_diagnos/c))+25)

            wave_noise_p = wave_noise[mask_noise_post]
            flux_noise_p = flux_noise[mask_noise_post]

            mean_noise = np.mean(flux_noise_p)
            std_noise = np.std(flux_noise_p)
            
            filtered_noise = [el for el in flux_noise_p if (el<mean_noise+8*std_noise)and(el>mean_noise-2*std_noise)]
            mean_noise = np.mean(filtered_noise)
            std_noise = np.std(filtered_noise)


            filtered_noise1_c = [el for el in filtered_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-2*std_noise)]

            filtered_noise=filtered_noise1_c

            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-1.5*std_noise)]


            #print 'FILTERED NOISE!!!!'

            #print filtered_noise5
            #print std_noise


###########Gauss noise to each line
###################################
        my_wave=wave_noise  
        my_flux=flux_noise
        if np.isnan(np.mean(filtered_noise)) == True:
            filtered_noise=[1.0 for el in mask_noise]




    ####
            ''' old
            mean_noise = np.mean(flux_noise)
            std_noise = np.std(flux_noise)
            
            filtered_noise = [el for el in flux_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-2*std_noise)]
            while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-1.5*std_noise)]
            filtered_noise5=filtered_noise
            noise=np.mean(filtered_noise)
            std_noise=np.std(filtered_noise)
            
###########Gauss noise to each line
###################################
            my_wave=wave_noise  
            my_flux=flux_noise
            '''#old


        '''
        if np.isnan(np.mean(filtered_noise)) == True:
            filtered_noise=[1.2 for el in mask_noise]
        if len(filtered_noise)==1:
            filtered_noise=[filtered_noise[0] for el in mask_noise]
        '''



        my_wave_toplot[t]=np.array(my_wave)
        my_flux_toplot[t]=np.array(my_flux)

        wav_testline,idx_testline=find_nearest(np.array(my_wave), l_wavelength[t]*(1+(radvel_diagnos/c)))

        mask_per=np.logical_and(np.array(my_wave) >= wav_testline-1.5, np.array(my_wave)<=wav_testline+1.5)




        wave_newSp_per=my_wave[mask_per]
        new_flux_per=my_flux[mask_per]

        idx_testline_per, min_d_line_per = min(enumerate(new_flux_per), key=operator.itemgetter(1))

            
        new_min_wave=wave_newSp_per[idx_testline_per]
        wav_testline,idx_testline=find_nearest(my_wave, new_min_wave)






        try:
            left,centres,right = find_edges(idx_testline, my_flux<np.mean(filtered_noise)-0.2*np.std(filtered_noise), my_flux)  ### sensitivity -0.05


            loc_wave_1=my_wave[left[0]]####final borders to gauss
            loc_wave_0=my_wave[centres[0]]
            loc_wave_2=my_wave[right[0]]####final borders to gauss
        

            signal=0

        except:
            left=[0]
            loc_wave_1=0
            loc_wave_0=0
            loc_wave_2=0
        #print('LINES!!!!!', loc_wave_1,loc_wave_0,loc_wave_2)

        if abs(loc_wave_0-loc_wave_1)>2*(loc_wave_2-loc_wave_0):
            #left[0]==0
            signal=1
        elif abs(loc_wave_2-loc_wave_0)>2*(loc_wave_0-loc_wave_1):
            left[0]==0
            signal=2

        #print(left,centres,right)
        if all(mask_noise==False):
            ewHeI5876=0
            xtoplot5876=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
            ytoplot5876=[1.2,1.2]
            wavel_local.append(0)
            RV_local.append(99999.99)
            ERR_RV_local.append(99999.99)
            ERR_EW_local.append(0)
            FWHM_local.append(0)
            EW_local.append(0)
            item_x[t].a=np.array(xtoplot5876)
            item_x[t].b=np.array(ytoplot5876)
            item_x_int[t].a=np.array(xtoplot5876)
            item_x_int[t].b=np.array(ytoplot5876)
            SNR_test.append(0)
            item_x_noise[t].a=np.array(xtoplot5876)
            item_x_noise[t].b=np.array(ytoplot5876)
            item_x_noise[t].c=np.array(ytoplot5876)
            EW_local_INT.append(0)
            SNR.append(0)
            IND_NOISE_FWHM[t]=np.array([0,0])
            xtofit=[0]
            ytofit=[0]
            
        elif left[0]==0:
            ewHeI5876=0
            xtoplot5876=[l_wavelength[t]*(1+(radvel_diagnos/c))-10, l_wavelength[t]*(1+(radvel_diagnos/c))+10]
            ytoplot5876=[1.2,1.2]
            wavel_local.append(0)
            RV_local.append(99999.99)
            ERR_RV_local.append(99999.99)
            ERR_EW_local.append(0)

            FWHM_local.append(0)
            EW_local.append(0)
            item_x[t].a=np.array(xtoplot5876)
            item_x[t].b=np.array(ytoplot5876)
            item_x_int[t].a=np.array(xtoplot5876)
            item_x_int[t].b=np.array(ytoplot5876)
            SNR_test.append(0)
            item_x_noise[t].a=np.array(xtoplot5876)
            item_x_noise[t].b=np.array(ytoplot5876)
            item_x_noise[t].c=np.array(ytoplot5876)
            EW_local_INT.append(0)
            SNR.append(0)
            IND_NOISE_FWHM[t]=np.array([0,0])
            xtofit=[0]
            ytofit=[0]

        else:



            loc_wave0c = my_wave[left[0]-15:right[0]+15:step_gaus]####final borders to gauss
            loc_flux0c = my_flux[left[0]-15:right[0]+15:step_gaus]

            loc_wave = loc_wave0c
            loc_flux = loc_flux0c
            for tk in range(0, 15):
                loc_flux[tk] = np.mean(filtered_noise)+0.1*np.std(filtered_noise)
                
            for tk in range(len(loc_flux)-15, len(loc_flux)):

                loc_flux[tk] = np.mean(filtered_noise)+0.1*np.std(filtered_noise)

            xtofit=loc_wave
            ytofit=loc_flux

            xtofit_new=np.linspace(min(xtofit),max(xtofit),700)
            ytofit_new=np.interp(xtofit_new,xtofit,ytofit)
            xtofit=xtofit_new
            ytofit=ytofit_new







            try:


                #mean_g=sum(xtofit*ytofit)/sum(ytofit)
                #sigma_g=np.sqrt(sum(ytofit*(xtofit-mean_g)**2)/sum(ytofit))
                #sigma_g=np.sqrt(sum((1-ytofit)*(xtofit-mean_g)**2))


                mean_g=sum(xtofit*ytofit)/sum(ytofit)
                sigma_g=np.sqrt(sum((ytofit)*(xtofit-mean_g)**2)/sum(ytofit))
                #sigma_g=np.sqrt(sum((1-ytofit)*(xtofit-mean_g)**2))
                '''

                '''

                #if (linelist_array['Name_wave'][t] == 'OIII5591')or(linelist_array['Name_wave'][t] == 'MgII4481')or(linelist_array['Name_wave'][t] == 'CII4267')or(linelist_array['Name_wave'][t] == 'SiII6371')or(linelist_array['Name_wave'][t] == 'SiIII4552'):
                    #popt,pcov = curve_fit(gaus,xtofit,ytofit,p0=[max(ytofit)-min(ytofit),mean_g,0.6,np.mean(filtered_noise)+0.1*np.std(filtered_noise)],method='lm')#,method='lm'  

                popt,pcov = curve_fit(lorentz,xtofit,ytofit,p0=[max(ytofit)-min(ytofit),mean_g,0.6,np.mean(filtered_noise)+0.1*np.std(filtered_noise)],method='lm')#,method='lm'    



                #print sigma_g,mean_g
                #print xtofit[1]-xtofit[0],xtofit[10]-xtofit[9]
                #print xtofit_new[1]-xtofit_new[0],xtofit[10]-xtofit[9]
                #print '!!!!!!!!!!!!!!!!!!!!!!',popt,pcov
                #print '!!!!!!!!!!!!!!!!!!!!!!',linelist_array['Name_wave'][t]

                #popt,pcov = curve_fit(gaus,xtofit,ytofit,p0=[max(ytofit)-min(ytofit),np.mean(xtofit),np.std(xtofit),max(ytofit)],method='lm')#,method='lm'
                #xtoplot=np.linspace(min(xtofit),max(xtofit),100)
                xtoplot=xtofit
                xtoplot5876=xtoplot
                popt5876=popt

                #if (linelist_array['Name_wave'][t] == 'OIII5591')or(linelist_array['Name_wave'][t] == 'MgII4481')or(linelist_array['Name_wave'][t] == 'CII4267')or(linelist_array['Name_wave'][t] == 'SiII6371')or(linelist_array['Name_wave'][t] == 'SiIII4552'):
                    #ytoplot5876=gaus(xtoplot5876,*popt5876)

                ytoplot5876=lorentz(xtoplot5876,*popt5876)






                #ytoplot5876=gaus(xtoplot5876,*popt5876)
            except:
                popt=[0,0,0,0]
                #print 'xtofit', xtofit
                if (len(xtofit) == 0):
                    #print 'xtofitttt'
                    xtofit =[5000,5001,5002,5004,5005]
                elif xtofit[0] == 0:
                    #print 'xtofitttt'
                    xtofit =[5000,5001,5002,5004,5005]
                xtoplot5876=np.linspace(min(xtofit),max(xtofit),100)
                ytoplot5876=[1.1 for el in xtoplot5876]
                ytofit=ytoplot5876




                #####checking for a line
    
                #if ((max(ytoplot5876)-min(ytoplot5876))<np.mean(filtered_noise)-0*np.std(filtered_noise)):
            if abs((max(ytoplot5876)-min(ytoplot5876))<1.0*np.std(filtered_noise)):
                ewHeI5876=0
                xtoplot5876=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
                ytoplot5876=[1.2,1.2]
                wavel_local.append(0)
                RV_local.append(0)
                ERR_RV_local.append(0)
                FWHM_local.append(0)
                EW_local.append(0)
                ERR_EW_local.append(0)
                item_x[t].a=np.array(xtoplot5876)
                item_x[t].b=np.array(ytoplot5876)
                item_x_int[t].a=np.array(xtoplot5876)
                item_x_int[t].b=np.array(ytoplot5876)
                SNR_test.append(0)
                item_x_noise[t].a=np.array(xtoplot5876)
                item_x_noise[t].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtoplot5876]
                item_x_noise[t].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtoplot5876]
                EW_local_INT.append(0)
                SNR.append(0)
                IND_NOISE_FWHM[t]=np.array([0,0])

            else:
                

            

                if ytoplot5876[0]==1.1:
                    fwhm_last = 0
                elif ytoplot5876[0]==1.2:
                    fwhm_last = 0
                else:
                    #print 'FWHM!!!',ytoplot5876
                    #fwhm_last = fwhm(xtoplot5876, 1-np.asarray(ytoplot5876))
                    fwhm_last=abs(2*popt[2])
                #print('FWHM_last', fwhm_last)


                #if (linelist_array['Name_wave'][t] == 'OIII5591')or(linelist_array['Name_wave'][t] == 'MgII4481')or(linelist_array['Name_wave'][t] == 'CII4267')or(linelist_array['Name_wave'][t] == 'SiII6371')or(linelist_array['Name_wave'][t] == 'SiIII4552'):
                    #EW=popt[0]*popt[2]*np.sqrt(2*np.pi)/max(ytofit) #a*sigma*sqrt(2pi), normalising to continuum level!
                #else:
                EW=popt[0]*popt[2]*np.sqrt(np.pi/np.log(2))/max(ytofit) 
                #print 'Name=%s WL=%8.3f  FWHM(5876)=%5.3fmA =%3.1fpx  EW(5876)=%5.3fmA' % (linelist_array['Name_wave'][t],popt[1],10*2.3548*popt[2],2.35482*popt[2]/wave_step,1000*EW)
        
                std_height, std_mean, std_sigma, std_sigma1=np.sqrt(np.diag(pcov))

                EW_local_sigma=[]
                RV_local_sigma=[]
                ss_ew1=np.random.normal(popt[0],std_height,200)
                ss_ew2=np.random.normal(popt[2],std_sigma,200)
                ss_rv=np.random.normal(popt[1],std_mean,200)
                for jk in range(0,200):
                    RV_local_sigma.append((c*( (1.*ss_rv[jk]/l_wavelength[t]) - 1)))
                    EW_local_sigma.append(ss_ew1[jk]*ss_ew2[jk]*np.sqrt(np.pi/np.log(2))/max(ytofit))

                ERR_EW_local.append((np.std(EW_local_sigma))*1000)


                wavel_local.append(popt[1])

                if (EW < 0):
                    EW=0.00
                    RV_local.append((c*( (1.*popt[1]/l_wavelength[t]) - 1)))
                    ERR_RV_local.append(np.std(RV_local_sigma))
                elif (EW > 2000):
                    EW=0.00
                    RV_local.append(99999.99)
                    ERR_RV_local.append(99999.99)
    
                elif  ytofit[0]==1.1:
                    EW=0.00     
                    RV_local.append(99999.99)   
                    ERR_RV_local.append(99999.99)

                else:
                    print('OK, line is present')
                    RV_local.append((c*( (1.*popt[1]/l_wavelength[t]) - 1)))
                    ERR_RV_local.append(np.std(RV_local_sigma))


                EW_local.append(1000*EW)    
                FWHM_local.append(fwhm_last)
                FWHM_to_int=fwhm_last
                item_x[t].a=np.array(xtoplot5876)
                item_x[t].b=np.array(ytoplot5876)


                a_integr=popt[1]-param*FWHM_to_int  
                b_integr=popt[1]+param*FWHM_to_int  
                b_noise=popt[1]+param*FWHM_to_int+2*FWHM_to_int
                bb_noise=popt[1]+param*FWHM_to_int+4*FWHM_to_int
                def  f1(x): 
                    wav_testline_f,idx_testline_f=find_nearest(my_wave, x)
            
                    return my_flux[idx_testline_f]
                    del wav_testline_f,idx_testline_f

                wav_testline_fa,idx_testline_fa=find_nearest(my_wave, a_integr)
                wav_testline_fb,idx_testline_fb=find_nearest(my_wave, b_integr)
                wav_testline_fa_noise,idx_testline_fa_noise=find_nearest(my_wave, b_noise)
                wav_testline_fb_noise,idx_testline_fb_noise=find_nearest(my_wave, bb_noise)

                x_iint_noise=[el for el in my_wave[idx_testline_fa_noise:idx_testline_fb_noise]]
                y_iint_noise=[el for el in my_flux[idx_testline_fa_noise:idx_testline_fb_noise]] #########FWHM noise value to return    
                sigma_noise_test=np.std(y_iint_noise)
                SNR_test.append(1/sigma_noise_test)
                IND_NOISE_FWHM[t]=np.array(x_iint_noise)



                x_iint=[el for el in my_wave[idx_testline_fa:idx_testline_fb]]
                y_iint=[el for el in my_flux[idx_testline_fa:idx_testline_fb]]  
            
                y_iint_noise=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtoplot5876] #####Global SN for Gauss +-45
                y_iint_noise_p=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtoplot5876]   #####Global SN for Gauss +-45
                xtopot_g=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
                item_x_noise[t].a=xtopot_g
                item_x_noise[t].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtopot_g]
                item_x_noise[t].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtopot_g]
                item_x_int[t].a=np.array(x_iint)
                item_x_int[t].b=np.array(y_iint)
                #INTEGR=integrate.quad(lambda x: f1(x),a_integr,b_integr)   

                #EW_INTEGR=((wav_testline_fb-wav_testline_fa)-INTEGR[0])*1000
                EW_local_INT.append(0)
                EW_INTEGR_ERR = 0
                del xtofit,ytofit


###########################  
############ Second part Integration 
#########################
    print('Second Part')

    my_wave=myspec[0]
    my_flux=myspec[1]
    EW_local_INT_T=[]

    my_wave_toplot_INT = [ structtype() for tt in range(0, len(l_namewave))]
    my_flux_toplot_INT = [ structtype() for tt in range(0, len(l_namewave))]



    filtered_FWHM = [el for el in FWHM_local if (el!=0)and(el>0)and(el!=float('nan'))]
    FWHM_to_analysis=np.mean(filtered_FWHM) 
    size_of_line=2.5*FWHM_to_analysis


    for j in range(0, len(l_wavelength)):   

        if  l_wavelength[j] > 6000:

            my_wave = myspec_red_n[0]
            my_flux = myspec_red_n[1]
            myspec = myspec_red_n



        else:

            myspec=myspec_blue
            my_wave = myspec[0]
            my_flux = myspec[1]

        RV_local_HeI=radvel_diagnos

###################
### New each noise for INT
###################
        mask_noise=np.logical_and(myspec[0]>= l_wavelength[j]*(1+(RV_local_HeI/c))-65, myspec[0]<=l_wavelength[j]*(1+(RV_local_HeI/c))+65)
        if all(mask_noise==False):
            wave_noise=[10,10,10,10,10]
            flux_noise=[10,10,10,10,10]
            filtered_noise=[0,0,0,0]
        else:
            wave_noise = my_wave[mask_noise]
            flux_noise = my_flux[mask_noise]

            n_points=int(len(flux_noise)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_noise[0])
            max_y.append(flux_noise[0])
            for kk in range(1,9):
                #xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                #yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                #max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))

                xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                max_y.append(yhat_t[0])
                max_w.append(xhat_t[0])



                #max_y.append(max_d_line)
                #max_w.append(xhat_t[max_index])
                #y_points.append(flux_noise[kk*n_points])
                #x_points.append(wave_noise[kk*n_points])
            max_w.append(wave_noise[-1])
            max_y.append(flux_noise[-1])
            s_y=[]
            s_x=[]
            mean_noise = np.mean(max_y)
            std_noise = np.std(max_y)
            for kk in range(0,len(max_y)):
                if abs((max_y[kk]<mean_noise+3*std_noise)and(max_y[kk]>mean_noise-3*std_noise)):
                    s_y.append(max_y[kk])
                    s_x.append(max_w[kk])


            s_y_1=[]
            s_x_1=[]
            mean_noise = np.mean(s_y)
            std_noise = np.std(s_y)
            for kk in range(0,len(s_y)):
                if abs((s_y[kk]<mean_noise+1*std_noise)and(s_y[kk]>mean_noise-1*std_noise)):
                    s_y_1.append(s_y[kk])
                    s_x_1.append(s_x[kk])

            z=np.polyfit([s_x_1[0],s_x_1[-1]],[s_y_1[0],s_y_1[-1]],1)
            #z=np.polyfit(s_x_1, s_y_1,1)
            p=np.poly1d(z)


            continuumi=p(wave_noise)

            flux_noise=flux_noise/continuumi
            mean_noise = np.mean(flux_noise)
            std_noise = np.std(flux_noise)  
            #filtered_noise=flux_noise
            
            filtered_noise = [el for el in flux_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:#(np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            
            filtered_noise6=filtered_noise

            '''
            spline = splrep(max_w,max_y,k=2)
            continuumi = splev(wave_noise,spline)
            flux_noise=flux_noise/continuumi
    ####

            mean_noise = np.mean(flux_noise)
            std_noise = np.std(flux_noise)  

            filtered_noise = [el for el in flux_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise6=filtered_noise
            '''
            
        if np.isnan(np.mean(filtered_noise)) == True:
            filtered_noise=[1.0 for el in mask_noise]
        my_wave=wave_noise  
        my_flux=flux_noise
        my_wave_toplot_INT[j]=np.array(my_wave)
        my_flux_toplot_INT[j]=np.array(my_flux)
        #RV_local_HeI=0
        SNR_new.append(1/np.std(filtered_noise))   ###################SNRRR INTEGRAL
        wav_testline_T,idx_testline_T=find_nearest(my_wave, l_wavelength[j]*(1+(RV_local_HeI/c)))   

#########################
### end each noise INT
########################

            

######################
###### EM LINE test?
####################

        if (l_namewave[j]=='2')and(my_flux[idx_testline_T] > np.mean(filtered_noise)+1.5*np.std(filtered_noise)):
            #print 'LINE', linelist_array['Name_wave'][j]
            #print 'EM!!!!!! my spectrum', my_flux[idx_testline_T] 
            #print 'EM!!!!!! noise', np.mean(filtered_noise)+1.5*np.std(filtered_noise)
            ewHeI5876_T=-100
            xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            ytoplot5876_T=[1.2,1.2]
            #SNR_new.append(0)
            EW_local_INT_T.append(-100)
            item_x_T[j].a=np.array(xtoplot5876_T)
            item_x_T[j].b=np.array(ytoplot5876_T) 
            item_x_T_noise[j].a=np.array(xtoplot5876_T)
            item_x_T_noise[j].b=[np.mean(filtered_noise)+1.5*np.std(filtered_noise) for el in xtoplot5876_T]
            item_x_T_noise[j].c=[np.mean(filtered_noise)+1.5*np.std(filtered_noise) for el in xtoplot5876_T]
        elif (l_namewave[j]=='3')and(my_flux[idx_testline_T] > np.mean(filtered_noise)+1.5*np.std(filtered_noise)):
            #print 'LINE', linelist_array['Name_wave'][j]
            #print 'EM!!!!!! my spectrum', my_flux[idx_testline_T] 
            #print 'EM!!!!!! noise', np.mean(filtered_noise)+1.5*np.std(filtered_noise)
            ewHeI5876_T=-100
            xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            ytoplot5876_T=[1.2,1.2]
            #SNR_new.append(0)
            EW_local_INT_T.append(-100)
            item_x_T[j].a=np.array(xtoplot5876_T)
            item_x_T[j].b=np.array(ytoplot5876_T) 
            item_x_T_noise[j].a=np.array(xtoplot5876_T)
            item_x_T_noise[j].b=[np.mean(filtered_noise)-2.5*np.std(filtered_noise) for el in xtoplot5876_T]
            item_x_T_noise[j].c=[np.mean(filtered_noise)+2.5*np.std(filtered_noise) for el in xtoplot5876_T]


##########################
######Check for line existance INT
############################3

        elif (my_flux[idx_testline_T] > np.mean(filtered_noise)-1.5*np.std(filtered_noise)): ###3.5 in case of it was 3...
            #### Exception
            ewHeI5876_T=0
            xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            ytoplot5876_T=[1.2,1.2]
            #SNR_new.append(0)
            EW_local_INT_T.append(0)
            item_x_T[j].a=np.array(xtoplot5876_T)
            item_x_T[j].b=np.array(ytoplot5876_T) 
            item_x_T_noise[j].a=np.array(xtoplot5876_T)
            item_x_T_noise[j].b=[np.mean(filtered_noise)-0.1*np.std(filtered_noise) for el in xtoplot5876_T]
            item_x_T_noise[j].c=[np.mean(filtered_noise)+0.1*np.std(filtered_noise) for el in xtoplot5876_T]
############################




        else:
            #wav_testline_left,idx_testline_left=find_nearest(my_wave, (linelist_array['wavelength'][j])*(1+(RV_local_HeI/c))-0.5*size_of_line)
            #wav_testline_right,idx_testline_right=find_nearest(my_wave, (linelist_array['wavelength'][j])*(1+(RV_local_HeI/c))+0.5*size_of_line)
                    ### new     

            wav_testline,idx_testline=find_nearest(my_wave, l_wavelength[j]*(1+(RV_local_HeI/c)))



            left_i,centres_i,right_i = find_edges(idx_testline, my_flux<np.mean(filtered_noise)-0.0*np.std(filtered_noise), my_flux)    #### sensitivity INTEG it was +0.1
            #left_i,centres_i,right_i = find_edges(idx_testline, my_flux<np.mean(filtered_noise)-0.0*np.std(filtered_noise), my_flux)  ### sensitivity

                


            #print 'left right', left_i[0], centres_i[0], right_i[0]
            loc_wave_T=my_wave[left_i[0]:right_i[0]:1]
            loc_flux_T=my_flux[left_i[0]:right_i[0]:1]
            #print loc_wave_T
            item_x_T_noise[j].a=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            item_x_T_noise[j].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in item_x_T_noise[j].a]   ###############Noise sensetive!!!!!!!!!!!!!!!!!!
            item_x_T_noise[j].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in item_x_T_noise[j].a]   ###############Noise sensetive!!!!!!!!!!!!!!!!!!
            #print 'left,right',left_i[0], right_i[0]
            if (left_i[0]==0)and(right_i[0]==0):
                #print 'ZERO',left_i[0], right_i[0]                 
                ewHeI5876_T=0
                xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
                ytoplot5876_T=[1.2,1.2]
                #SNR_new.append(0)
                EW_local_INT_T.append(0)
                item_x_T[j].a=np.array(xtoplot5876_T)
                item_x_T[j].b=np.array(ytoplot5876_T) 
                item_x_T_noise[j].a=np.array(xtoplot5876_T)
                item_x_T_noise[j].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtoplot5876_T]
                item_x_T_noise[j].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtoplot5876_T]

            else:
            

                                    
                xtoplot5876_T=loc_wave_T
                ytoplot5876_T=loc_flux_T
                #print xtoplot5876_T

                maxm = argrelextrema(ytoplot5876_T, np.greater)  
                

                item_x_T[j].a=np.array(xtoplot5876_T) #~ tab once
                item_x_T[j].b=np.array(ytoplot5876_T) #~ tab once - double EW INT
                
                
                
                
                #print left_i[0], right_i[0]
                a_integr_T=xtoplot5876_T[0] 
                
                b_integr_T=xtoplot5876_T[-1]        
            


                    
                #print  'IN A IN B', a_integr_T, b_integr_T
                def  f2(x):
                    wav_testline_T_int,idx_testline_T_int=find_nearest(my_wave, x)
                    flux_to_int=my_flux[idx_testline_T_int]
                    return flux_to_int

                
                
                INTEGR_T=integrate.quad(lambda x: f2(x),a_integr_T,b_integr_T)  
                EW_INTEGR_T=((b_integr_T-a_integr_T)-INTEGR_T[0])*1000
                EW_local_INT_T.append(EW_INTEGR_T)
    
                if (EW_local_INT_T[j] > 25000)or(EW_local_INT_T[j] < 0):
                    EW_local_INT_T[j]=0

                EW_INTEGR_ERR_T =   INTEGR_T[0]
                    
                


    print('Rad vel=', radvel_diagnos, 'km/s (using diagnostic line)')


    
    my_wave = myspec[0]
    my_flux = myspec[1]

    if plots==1:
        ax2=plt.subplot(412)
        plt.title(files)



        plt.plot(myspec_red[0], myspec_red[1], color='blue',lw=1)

        plt.plot(myspec_blue[0], myspec_blue[1], color='blue',lw=1)

            #print '1!!!',item_x[tt].a


        plt.fill_between(wave_find1, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
        plt.fill_between(wave_find2, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
    #   plt.fill_between(wave_find3, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
        plt.plot(wave_find1,continuum1,'r-',linewidth=0.5)
        plt.plot(wave_find1,flux_find1,'black',linewidth=0.5)
        plt.plot([wave_find1[0],wave_find1[-1]],[min(filtered_noise1),min(filtered_noise1)],'cyan',linewidth=0.5)
        plt.plot([wave_find1[0],wave_find1[-1]],[max(filtered_noise1),max(filtered_noise1)],'cyan',linewidth=0.5)
        plt.plot(wave_find2,continuum2,'r-',linewidth=0.5)
        plt.plot(wave_find2,flux_find2,'black',linewidth=0.5)
        plt.plot([wave_find2[0],wave_find2[-1]],[min(filtered_noise2),min(filtered_noise2)],'cyan',linewidth=0.5)   
        plt.plot([wave_find2[0],wave_find2[-1]],[max(filtered_noise2),max(filtered_noise2)],'cyan',linewidth=0.5)

    #       plt.plot(wave_find3,continuum3,'r-',linewidth=0.5)
    #       plt.plot(wave_find3,flux_find3,'black',linewidth=0.5)
    #       plt.plot([wave_find3[0],wave_find3[-1]],[min(filtered_noise3),min(filtered_noise3)],'cyan',linewidth=0.5)
    #       plt.plot([wave_find3[0],wave_find3[-1]],[max(filtered_noise3),max(filtered_noise3)],'cyan',linewidth=0.5)



        plt.text(wave_find1[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise1,std_noise1,1/std_noise1),rotation=90,fontsize=6,fontweight='ultralight')
        plt.text(wave_find2[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise2,std_noise2,1/std_noise2),fontsize=6,rotation=90,fontweight='ultralight')
        #   plt.text(wave_find3[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise3,std_noise3,1/std_noise3),fontsize=6,rotation=90,fontweight='ultralight')

        for tt in range(0, len(l_wavelength)):
            if l_wavelength[tt]<6000:

                plt.plot([l_wavelength[tt],l_wavelength[tt]],[0,2],'gray',linewidth=0.2,linestyle='--') 
                plt.text(l_wavelength[tt]-35,1.45,l_namewave[tt],fontsize=6,rotation=90)
                plt.plot(item_x[tt].a,item_x[tt].b,'r-',lw=1)
                plt.plot(item_x_int[tt].a,item_x_int[tt].b,'g-',lw=1)
            #   plt.plot(item_x_noise[tt].a,item_x_noise[tt].b,'cyan',lw=1)
                plt.plot(item_x_T[tt].a,item_x_T[tt].b,'orange',lw=1)
                #   plt.plot(item_x_T_noise[tt].a,item_x_T_noise[tt].b,'blue',lw=1)

        #fig2=plt.show()

        plt.ylim([0.3,1.5])
    #   plt.xlim([3800,7000])
        plt.xlim([3800,6200])

        ax3=plt.subplot(413)
        plt.title('Red part')

        plt.plot(my_wave_red,my_flux_red,'b-',lw=0.5)
        plt.plot(my_wave_red,continuum_red,'b--',lw=0.5)
        plt.plot(max_w_red,max_y_red,'rX')
        plt.plot(second_x_red,second_y_red,'gX')

        ax4=plt.subplot(414)
        plt.title('Red part')
        plt.fill_between(wave_find3, 0, 1.5, facecolor='green', interpolate=True,alpha=0.5)
        plt.plot(my_wave_red,my_flux_red/continuum_red,'b-',lw=0.5)
        plt.plot(wave_find3,continuum3,'r-',linewidth=0.5)
        plt.plot(wave_find3,flux_find3,'black',linewidth=0.5)
        plt.plot([wave_find3[0],wave_find3[-1]],[min(filtered_noise3),min(filtered_noise3)],'cyan',linewidth=0.5)
        plt.plot([wave_find3[0],wave_find3[-1]],[max(filtered_noise3),max(filtered_noise3)],'cyan',linewidth=0.5)
        plt.text(wave_find3[0]-35,0.9,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise3,std_noise3,1/std_noise3),fontsize=6,rotation=90,fontweight='ultralight')
        for tt in range(0, len(l_wavelength)):
            if l_wavelength[tt]>6000:

                plt.plot([l_wavelength[tt],l_wavelength[tt]],[0,2],'gray',linewidth=0.2,linestyle='--') 
                plt.text(l_wavelength[tt]-35,1.45,l_namewave[tt],fontsize=6,rotation=90)
                plt.plot(item_x[tt].a,item_x[tt].b,'r-',lw=1)
                plt.plot(item_x_int[tt].a,item_x_int[tt].b,'g-',lw=1)
            #   plt.plot(item_x_noise[tt].a,item_x_noise[tt].b,'cyan',lw=1)
                plt.plot(item_x_T[tt].a,item_x_T[tt].b,'orange',lw=1)
                #   plt.plot(item_x_T_noise[tt].a,item_x_T_noise[tt].b,'blue',lw=1)



        plt.ylim([0.3,1.5])
        plt.tight_layout()



        plt.savefig(pp1, format='pdf')
    #pp1.close()





    mean_noise_TEST=[1/std_noise1,1/std_noise2,1/std_noise3]   ######First S/N variable to out





    #plt.clf()

    RV_gauss_tt=[el for el in RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]
    RV_gauss_meant=np.mean(RV_gauss_tt)
    RV_gauss_sigmat=np.std(RV_gauss_tt)
    RV_gauss_tt=[el for el in RV_gauss_tt if (abs(np.mean(RV_gauss_tt)-el) < 3*np.std(RV_gauss_tt))]
    RV_gauss_meant1=np.mean(RV_gauss_tt)
    RV_gauss_sigmat1=np.std(RV_gauss_tt)


    RV_gauss_tt=[el for el in RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]

    RV_gauss_tt=[el for el in RV_gauss_tt if (abs(radvel_diagnos-el) < 1*np.std(RV_gauss_tt))]
    RV_gauss_meant=np.mean(RV_gauss_tt)
    RV_gauss_sigmat=np.std(RV_gauss_tt)




    for j in range(3, len(l_wavelength)):

        if abs(radvel_diagnos-RV_local[j])>1*RV_gauss_sigmat:
            RV_local[j]=0
            EW_local[j]=0




    RV_gauss_fin=[el for el in RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]
    for j in range(3, len(l_wavelength)):

        if abs(radvel_diagnos-RV_local[j])>2*np.std(RV_gauss_fin):
            RV_local[j]=float('nan')
            EW_local[j]=0




    if plots==1: 
        my_dpi=50
        #global fig
        fig=plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))

        num_pages=int(ceil(len(l_wavelength)/30))
        t1=0
        print(range(0,int(num_pages+1)))
        for tr in range(0,num_pages+1):
            print(tr)
            gs1 = gridspec.GridSpec(5,6)
            trtr=0
            for ttt in range(t1, t1+30):
                if ttt>=len(l_wavelength):
                    ttt=len(l_wavelength)-1
                print(ttt, len(l_wavelength),trtr,len(my_wave_toplot))

                #fig.add_subplot(3,4,ttt+1)
                fig.add_subplot(gs1[trtr])
        #       plt.plot(myspec[0], myspec[1], color='black')
                plt.plot(my_wave_toplot[ttt],my_flux_toplot[ttt],color='black')
                #plt.plot(my_wave_toplot_INT[ttt],my_flux_toplot_INT[ttt],color='gray')
            #   plt.fill_between(IND_NOISE_FWHM[ttt], 0, 1.5, facecolor='green', interpolate=True,alpha=0.3)
                


                plt.plot(waveobs_shifted_blue, myspec_blue[1],'--',linewidth=0.5,color='blue')

                plt.plot(waveobs_shifted_red, myspec_red_n[1],'--',linewidth=0.5,color='blue')
                plt.plot([l_wavelength[ttt],l_wavelength[ttt]],[0,2],'gray',linewidth=0.5,linestyle='--')       
                plt.plot(item_x[ttt].a,item_x[ttt].b,'r-',lw=2) #Gauss

                #plt.plot(item_x_int[ttt].a,item_x_int[ttt].b,'g-',lw=1)  #INT Gauss FWHM
                plt.plot(item_x_T[ttt].a,item_x_T[ttt].b,'orange',lw=2) #INT

                #print item_x_noise[ttt].a, item_x_noise[ttt].b
                plt.plot(item_x_noise[ttt].a,item_x_noise[ttt].b,'r--',lw=1)
                plt.plot(item_x_noise[ttt].a,item_x_noise[ttt].c,'r--',lw=1)
                plt.plot(item_x_T_noise[ttt].a,item_x_T_noise[ttt].b,'--',lw=1, color='orange')
                plt.plot(item_x_T_noise[ttt].a,item_x_T_noise[ttt].c,'--',lw=1, color='orange')

                wav_cen=l_wavelength[ttt]

                plt.text(wav_cen-40.5,0.75,'EW(G)= %2.1f$\pm$%2.1f'%(EW_local[ttt],ERR_EW_local[ttt]),fontsize=6)
                plt.text(wav_cen-40.5,0.73,'EW(G INT)= %2.1f'%(EW_local_INT[ttt]),fontsize=6)
                plt.text(wav_cen-40.5,0.71,'EW(INT)= %2.1f'%(EW_local_INT_T[ttt]),fontsize=6)
                

                plt.text(wav_cen+2,0.73,'SNR(FWHM)= %2.1f'%(SNR_test[ttt]),fontsize=6)
                plt.text(wav_cen+2,0.71,'SNR(INT)= %2.1f'%(SNR_new[ttt]),fontsize=6)


                plt.text(wav_cen+2,1.08,'RV first= %2.2f'%(radvel_diagnos),fontsize=6)
                plt.text(wav_cen+2,1.06,'<RV>= %2.2f$\pm$%2.2f'%(RV_gauss_meant1,RV_gauss_sigmat1),fontsize=6)
                plt.text(wav_cen+2,1.04,'RV(ind)= %2.2f$\pm$%2.2f'%(RV_local[ttt],ERR_RV_local[ttt]),fontsize=6)
                plt.ylim([0.5,1.1])
                plt.xlim([wav_cen-25,wav_cen+25])
                plt.title(l_namewave[ttt])
                plt.tight_layout()
                #plt.show()
                #plt.clf()  # Clear the figure for the next loop
                plt.yticks(size=8)
                plt.xticks(size=8)
                trtr=trtr+1

            t1=t1+30
            plt.savefig(pp1, format='pdf')
            plt.clf()




    fits_l_name_wave=[]
    fits_l_wave=[]
    fits_l_EW=[]
    fits_l_EW_err=[]
    fits_l_RV_local=[]
    fits_l_ERR_RV_local=[]

    #plt.savefig(pp1, format='pdf')

    #gs1.tight_layout(fig,rect=[0,0,1,0.95])
    #plt.show() #          !!!!!!!!!!!!!!!!!!!!!!!!!!!
    #File_gala_local=open(path+"/"+files+"_weave_gala_test.txt","w")

    err_ew=ERR_EW_local
    for j in range(3, len(l_wavelength)):
        if EW_local[j]>1:
            #File_gala_local.write('%4.4f   %4.2f  %4.2f  %4.2f  %4.2f  %4.3f  %4.2f  %4.2f  %4.2f  %4.2f  %s' %(l_wavelength[j],EW_local[j],err_ew[j],l_namewave[j],l_5[j],l_6[j],l_7[j],l_8[j],l_9[j],l_10[j], '\n'))
            fits_l_name_wave.append(round(l_namewave[j],2))
            fits_l_wave.append(round(l_wavelength[j],3))
            fits_l_EW.append(round(EW_local[j],3))
            fits_l_EW_err.append(round(err_ew[j],3))
            fits_l_RV_local.append(round(RV_local[j],2))
            fits_l_ERR_RV_local.append(round(ERR_RV_local[j],2))


    #File_gala_local.close()



    return ReturnValue(files, radvel_diagnos, name_of_line, RV_local, EW_local, FWHM_local, EW_local_INT,mean_noise_TEST,EW_local_INT_T,SNR_test,fits_l_name_wave,fits_l_wave,fits_l_EW,fits_l_EW_err,fits_l_RV_local,ERR_RV_local,fits_l_ERR_RV_local)
#   return ReturnValue(files, radvel_diagnos, name_of_line, RV_local, EW_local, FWHM_local, EW_local_INT,SNR_new,EW_local_INT_T,SNR_test)










#function that takes the arguments, reads the files and plots:
def actualseefits_HR(argumentsList,hr_list):
    c = 299792.458 #km/s
    global pp1 
    global fig
    path = os.getcwd()

    #linelist_array = np.loadtxt("line_list.dat", dtype={'names': ('Name_wave', 'wavelength'),'formats': ('|S15',float)},delimiter=' ', skiprows=0)
    #name_of_line=linelist_array['Name_wave']


    fmod_obs = ascii.read(hr_list)
    #fmod_obs = ascii.read('gala_test_HR_RRLYR.in')

    #fmod_obs = ascii.read('gala_test_HR.in')
    l_wavelength=np.asarray(fmod_obs['col1'])
    l_namewave = np.asarray(fmod_obs['col4'])
    l_5 = np.asarray(fmod_obs['col5'])
    l_6 = np.asarray(fmod_obs['col6'])
    l_7 = np.asarray(fmod_obs['col7'])
    l_8 = np.asarray(fmod_obs['col8'])
    l_9 = np.asarray(fmod_obs['col9'])
    l_10 = np.asarray(fmod_obs['col10'])

    name_of_line=l_namewave
    wave_test=np.asarray(argumentsList[0])
    flux_or=np.asarray(argumentsList[1])

    wave_test_red=np.asarray(argumentsList[14])
    flux_or_red=np.asarray(argumentsList[15])

    wave_step=wave_test[1]-wave_test[0]
    
    APS_RVS=float(argumentsList[11])
    plots=float(argumentsList[6])



    mask_find_blue=np.logical_and(np.asarray(wave_test)<= 5450, np.asarray(wave_test)>= 4730)
    mask_find_red=np.logical_and(np.asarray(wave_test_red)<=6800, np.asarray(wave_test_red)> 5950)

    wave_test=wave_test[mask_find_blue]
    wave_test_red=wave_test_red[mask_find_red]
    flux_or=flux_or[mask_find_blue]
    flux_or_red=flux_or_red[mask_find_red]




    myspec = [wave_test,flux_or]
    myspec_red =[wave_test_red,flux_or_red]
    myspec=np.asarray(myspec)
    myspec_red=np.asarray(myspec_red)

    files=str(argumentsList[16])

    mag_inp = argumentsList[5]

    my_wave = np.asarray(myspec[0])
    my_flux = np.asarray(myspec[1])

    my_wave_red = np.asarray(myspec_red[0])
    my_flux_red = np.asarray(myspec_red[1])

    INP_name=argumentsList[7]


    mask_find_cut0a=np.logical_and(np.asarray(my_wave)<=    5000, np.asarray(my_wave)<= 5000)
    mask_find_cut1a=np.logical_and(np.asarray(my_wave)> 5000, np.asarray(my_wave)<= 5250)
    mask_find_cut2a=np.logical_and(np.asarray(my_wave)> 5250, np.asarray(my_wave)> 5250)


    mask_find_cut1b=np.logical_and(np.asarray(my_wave_red)<=    6150, np.asarray(my_wave_red)<= 6150)
    mask_find_cut2b=np.logical_and(np.asarray(my_wave_red)> 6150, np.asarray(my_wave_red)> 6150)

    my_wave0=my_wave[mask_find_cut0a]
    my_flux0=my_flux[mask_find_cut0a]
    my_wave1=my_wave[mask_find_cut1a]
    my_wave2=my_wave[mask_find_cut2a]
    my_flux1=my_flux[mask_find_cut1a]
    my_flux2=my_flux[mask_find_cut2a]

    my_wave_red1=my_wave_red[mask_find_cut1b]
    my_wave_red2=my_wave_red[mask_find_cut2b]
    my_flux_red1=my_flux_red[mask_find_cut1b]
    my_flux_red2=my_flux_red[mask_find_cut2b]

    if all(my_flux == 0):
        bbb=[]
        print('No flux in the spectrum!')
        zzz = ([float(0) for el in l_wavelength])
        yyy = np.array(["No flux" for el in l_wavelength])
        for el in l_wavelength:
            bbb.append(float(0))

        if plots == 1:
            path_file_plots=path+"/Plots/"
            if not os.path.exists(path_file_plots):
                os.makedirs(path_file_plots)


            pp1 = PdfPages(path_file_plots+files+'.pdf')
            my_dpi = 100
            plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
            ax1 = plt.subplot(411)
            plt.plot(my_wave,my_flux,'b-',lw=1)
            plt.title('%s       %s       I MAG = %2.2f'%(INP_name,files,mag_inp))

        return ReturnValue(files, 0, yyy, bbb, bbb, bbb, bbb, bbb, bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb)


    else:



        continuum0,max_w0,max_y0,second_x0,second_y0=continuum_script(my_wave0,my_flux0,10,3,3)
        continuum1,max_w1,max_y1,second_x1,second_y1=continuum_script(my_wave1,my_flux1,10,5,3)
        continuum2,max_w2,max_y2,second_x2,second_y2=continuum_script(my_wave2,my_flux2,4,3,2)
        continuum_red1,max_w_red1,max_y_red1,second_x_red1,second_y_red1=continuum_script(my_wave_red1,my_flux_red1,10,5,3)
        continuum_red2,max_w_red2,max_y_red2,second_x_red2,second_y_red2=continuum_script(my_wave_red2,my_flux_red2,15,6,3)


        continuum_n=[]
        continuum_redn=[]
        continuum_n.extend(continuum0)
        continuum_n.extend(continuum1)
        continuum_n.extend(continuum2)

        continuum_redn.extend(continuum_red1)
        continuum_redn.extend(continuum_red2)
        continuum=np.asarray(continuum_n)
        continuum_red=np.asarray(continuum_redn)


        max_w=[max_w0+max_w1+max_w2]
        max_y=[max_y0+max_y1+max_y2]


        second_x=[second_x0+second_x1+second_x2]
        second_y=[second_y0+second_y1+second_y2]

    
        max_w_red=[max_w_red1+max_w_red2]
        max_y_red=[max_y_red1+max_y_red2]
        second_x_red=[second_x_red1+second_x_red2]
        second_y_red=[second_y_red1+second_y_red2]

        ##FIGURE1
        if plots==1:
            path_file_plots=path+"/Plots/"
            if not os.path.exists(path_file_plots):
                os.makedirs(path_file_plots)

            pp1 = PdfPages(path_file_plots+files+'.pdf')
            #global pp1 
            my_dpi=100
            plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
            ax1=plt.subplot(411)
            plt.plot(my_wave,my_flux,'b-',lw=1)
            plt.plot(my_wave,continuum,'--')
            plt.plot(max_w,max_y,'rX')
            plt.plot(second_x,second_y,'gX')
            plt.title('%s       %s       I MAG = %2.2f'%(INP_name,files,mag_inp))


        my_flux=my_flux/continuum
        myspec=[wave_test,my_flux]

        myspec_blue=myspec


    ############################################
    ###############First SNR estimations
    ############################################


        myspec[0]=np.asarray(myspec[0])
        myspec[1]=np.asarray(myspec[1])
        myspec_red[0]=np.asarray(myspec_red[0])
        myspec_red[1]=np.asarray(myspec_red[1])

############################################
###############First SNR estimations
############################################



        mask_find1=np.logical_and(np.asarray(myspec[0])>=   4900, np.asarray(myspec[0])<= 4950)
        mask_find2=np.logical_and(np.asarray(myspec[0])>=   5150, np.asarray(myspec[0])<= 5200)
    #   mask_find3=np.logical_and(myspec[0]>=   6300, myspec[0]<= 6400)
        #print myspec[0],mask_find1,mask_find2
        if any(mask_find1) == True :            

            wave_find1= my_wave[mask_find1]
            flux_find1= my_flux[mask_find1]
        ####    
            n_points=int(len(flux_find1)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_find1[0])
            max_y.append(flux_find1[0])
            for kk in range(1,9):
                xhat_t=([el for el in wave_find1[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_find1[kk*n_points:(kk+1)*n_points]])
                max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))


                max_y.append(max_d_line)
                max_w.append(xhat_t[max_index])
                y_points.append(flux_find1[kk*n_points])
                x_points.append(wave_find1[kk*n_points])
            max_w.append(wave_find1[-1])
            max_y.append(flux_find1[-1])
            spline = splrep(max_w,max_y,k=1)
            continuum1 = splev(wave_find1,spline)
            flux_find1=flux_find1/continuum1
        ####
            mean_noise = np.mean(flux_find1)
            std_noise = np.std(flux_find1)

            filtered_noise = [el for el in flux_find1 if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:#(np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise1=filtered_noise
            noise1=np.mean(filtered_noise)
            std_noise1=np.std(filtered_noise)
        else:
            wave_find1=0
            noise1=0
        if any(mask_find2) == True :            

            wave_find2= my_wave[mask_find2]
            flux_find2= my_flux[mask_find2]

        ####    
            n_points=int(len(flux_find2)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_find2[0])
            max_y.append(flux_find2[0])
            for kk in range(1,9):
                xhat_t=([el for el in wave_find2[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_find2[kk*n_points:(kk+1)*n_points]])
                max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))
                max_y.append(max_d_line)
                max_w.append(xhat_t[max_index])
                y_points.append(flux_find2[kk*n_points])
                x_points.append(wave_find2[kk*n_points])
            max_w.append(wave_find2[-1])
            max_y.append(flux_find2[-1])
            spline = splrep(max_w,max_y,k=1)
            continuum2 = splev(wave_find2,spline)
            flux_find2=flux_find2/continuum2
        ####

            mean_noise = np.mean(flux_find2)
            std_noise = np.std(flux_find2)

            filtered_noise = [el for el in flux_find2 if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:#(np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise2=filtered_noise      
            noise2=np.mean(filtered_noise)
            std_noise2=np.std(filtered_noise)
        else:
            wave_find2=0
            noise2=0

        mask_find3=np.logical_and(myspec_red[0]>=   6200, myspec_red[0]<= 6250)
        #yhat_red = savitzky_golay(my_flux_red,65,1)
        if any(mask_find3) == True :            

            wave_find3= my_wave_red[mask_find3]
            flux_find3= my_flux_red[mask_find3]
        ####    
            n_points=int(len(flux_find3)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_find3[0])
            max_y.append(flux_find3[0])
            for kk in range(1,9):
                xhat_t=([el for el in wave_find3[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_find3[kk*n_points:(kk+1)*n_points]])
                max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))
                max_y.append(max_d_line)
                max_w.append(xhat_t[max_index])
                y_points.append(flux_find3[kk*n_points])
                x_points.append(wave_find3[kk*n_points])
            max_w.append(wave_find3[-1])
            max_y.append(flux_find3[-1])
            spline = splrep(max_w,max_y,k=1)
            continuum3 = splev(wave_find3,spline)
            flux_find3=flux_find3/continuum3
        ####

            mean_noise = np.mean(flux_find3)
            std_noise = np.std(flux_find3)

            filtered_noise = [el for el in flux_find3 if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:#(np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise3=filtered_noise
            noise3=np.mean(filtered_noise)
            std_noise3=np.std(filtered_noise)
        else:
            wave_find3=0
            noise3=0



    ############################################
    ############### Check if the wavelength ranges the Hydrogen lines are exist
    ############################################
        bbb=[]
        wavHeI5876=4340.47 ###### Hgamma FROM HERE starts the line minimum exploration
        wavHeI4471=6562.852


        mask1=np.logical_and(myspec[0]>=    wavHeI5876-5, myspec[0]<=wavHeI5876+5)
        mask2=np.logical_and(myspec_red[0]>=    wavHeI4471-5, myspec_red[0]<=wavHeI4471+5)

        #print 'MASK1', myspec[0],mask1
        if any(mask1) == True :
            print('Here is the H gamma!')
            mask=mask1 
            marker=0
        elif any(mask2) == True :
            print('Here is the H alpha!')
            mask=mask2 
            marker=1
            wavHeI5876=6562.852

        else:
            print('No diagnostic line regions in the file!')
            zzz = ([float(0) for el in l_wavelength])
            yyy = np.array(["No H lines in file" for el in l_wavelength])
            for el in l_wavelength:
                bbb.append(float(0))



            my_wave = myspec[0]
            my_flux = myspec[1]
            if plots==1:
                ax2=plt.subplot(412)
                plt.title(files)
                plt.plot(myspec[0], myspec[1], color='blue',lw=1)

                for tt in range(0, len(l_namewave)):



                    plt.fill_between(wave_find1, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
                    plt.fill_between(wave_find2, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)

                    plt.plot(wave_find1, continuum1, 'r-',linewidth=0.5)
                    plt.plot(wave_find1, flux_find1, 'black',linewidth=0.5)
                    plt.plot([wave_find1[0],wave_find1[-1]],[min(filtered_noise1),min(filtered_noise1)],'cyan',linewidth=0.5)
                    plt.plot([wave_find1[0],wave_find1[-1]],[max(filtered_noise1),max(filtered_noise1)],'cyan',linewidth=0.5)

                    plt.plot(wave_find2,continuum2,'r-',linewidth=0.5)
                    plt.plot(wave_find2,flux_find2,'black',linewidth=0.5)
                    plt.plot([wave_find2[0],wave_find2[-1]],[min(filtered_noise2),min(filtered_noise2)],'cyan',linewidth=0.5)
                    plt.plot([wave_find2[0],wave_find2[-1]],[max(filtered_noise2),max(filtered_noise2)],'cyan',linewidth=0.5)

                    plt.text(wave_find1[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise1,std_noise1,1/std_noise1),rotation=90,fontsize=6,fontweight='ultralight')
                    plt.text(wave_find2[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise2,std_noise2,1/std_noise2),fontsize=6,rotation=90,fontweight='ultralight')

                    plt.plot([l_wavelength[tt],l_wavelength[tt]],[0,2],'gray',linewidth=0.2,linestyle='--') 
                    plt.text(l_wavelength[tt]-35,1.45,l_wavelength[tt],fontsize=6,rotation=90)
            


                plt.ylim([0.3,1.5])

                plt.xlim([4700,5500])

                ax3=plt.subplot(413)
                plt.title('Red part')

                plt.plot(my_wave_red,my_flux_red,'b-',lw=0.5)
                plt.plot(my_wave_red,continuum_red,'b--',lw=0.5)
                plt.plot(max_w_red,max_y_red,'rX')
                plt.plot(second_x_red,second_y_red,'gX')

                ax4=plt.subplot(414)
                plt.title('Red part')
                plt.fill_between(wave_find3, 0, 1.5, facecolor='green', interpolate=True,alpha=0.5)
                plt.plot(my_wave_red,my_flux_red/continuum_red,'b-',lw=0.5)
                plt.plot(wave_find3,continuum3,'r-',linewidth=0.5)
                plt.plot(wave_find3,flux_find3,'black',linewidth=0.5)
                plt.plot([wave_find3[0],wave_find3[-1]],[min(filtered_noise3),min(filtered_noise3)],'cyan',linewidth=0.5)
                plt.plot([wave_find3[0],wave_find3[-1]],[max(filtered_noise3),max(filtered_noise3)],'cyan',linewidth=0.5)
                plt.text(wave_find3[0]-35,0.9,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise3,std_noise3,1/std_noise3),fontsize=6,rotation=90,fontweight='ultralight')
                plt.ylim([0.3,1.5])
                plt.tight_layout()
                plt.savefig(pp1, format='pdf')
                my_dpi=90

                fig=plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
                gs1 = gridspec.GridSpec(3,4)
                my_wave_toplot = [ structtype() for tt in range(0, len(l_wavelength))]
                my_flux_toplot = [ structtype() for tt in range(0, len(l_wavelength))]

                for ttt in range(0, 12):

                    my_wave_toplot[ttt]=np.array(my_wave)
                    my_flux_toplot[ttt]=np.array(my_flux)
                    fig.add_subplot(gs1[ttt])

                    plt.plot(my_wave_toplot[ttt],my_flux_toplot[ttt],color='black')
                    #plt.plot(my_wave_toplot_INT[ttt],my_flux_toplot_INT[ttt],color='gray')

            




                    plt.plot(l_wavelength[ttt],l_wavelength[ttt],[0,2],'gray',linewidth=0.5,linestyle='--')     
            
                    wav_cen=l_wavelength[ttt]

                    plt.ylim([0.7,1.1])
                    plt.xlim([wav_cen-45,wav_cen+45])
                    plt.title(l_namewave[ttt])
                    plt.tight_layout()
            
                    plt.yticks(size=8)
                    plt.xticks(size=8)


                gs1.tight_layout(fig,rect=[0,0,1,0.95])

            return ReturnValue(files, 0, yyy, bbb, bbb, bbb, bbb, bbb, bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb,bbb)


############
############

#The RR Lyr analysis

############
############



    #wave_newSp=    my_wave[mask]
    #new_flux=my_flux[mask]

    #new_spec=[wave_newSp,new_flux]
    #min_index, min_d_line = min(enumerate(new_flux), key=operator.itemgetter(1))


    #radvel_diagnos=c*( (1.*new_spec[0][min_index]/wavHeI5876) - 1)
    radvel_diagnos=APS_RVS
    param=1.1
    step_gaus=1
    RV_local=[]
    ERR_RV_local=[]
    ERR_EW_local=[]

    FWHM_local=[]
    EW_local=[]
    EW_local_INT=[]
    wavel_local=[]
    SNR=[]
    SNR_new=[]
    SNR_test=[]
            
        
    item_x = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_int = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_noise = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_T = [ structtype() for tt in range(0, len(l_namewave))]
    item_x_T_noise = [ structtype() for tt in range(0, len(l_namewave))]
    IND_NOISE_FWHM = [ structtype() for tt in range(0, len(l_namewave))]
    my_wave_toplot = [ structtype() for tt in range(0, len(l_namewave))]
    my_flux_toplot = [ structtype() for tt in range(0, len(l_namewave))]
###########################
############ START GAUSSIAN
###########################
    waveobs_shifted_blue=[]
    waveobs_shifted_red=[]

    waveobs_shifted_blue.append(-999)
    waveobs_shifted_red.append(-999)
    #full_path_file_vsini=path_file_plots+files+'_vsin.pdf'
    #pp3 = PdfPages(full_path_file_vsini)
    #fig1=plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
    for t in range(0, len(l_wavelength)):
        if l_wavelength[t] > 5950:

            my_flux_red_n=my_flux_red/continuum_red
            myspec_red_n=[wave_test_red,my_flux_red_n]
            
            my_wave = myspec_red_n[0]
            my_flux = myspec_red_n[1]
            myspec = myspec_red_n
            waveobs_shifted_red = np.array([el*(1+(radvel_diagnos/c)) for el in my_wave])

            


        elif l_wavelength[t] < 5500:


            myspec=myspec_blue
            my_wave = myspec[0]
            my_flux = myspec[1]
            waveobs_shifted_blue = np.array([el*(1+(radvel_diagnos/c)) for el in my_wave])

        mask_noise=np.logical_and(myspec[0]>=   l_wavelength[t]*(1+(radvel_diagnos/c))-50, myspec[0]<=l_wavelength[t]*(1+(radvel_diagnos/c))+50)
        if all(mask_noise==False):
            wave_noise=[0,0,0,0,0]
            flux_noise=[0,0,0,0,0]
        else:
            wave_noise = my_wave[mask_noise]
            flux_noise = my_flux[mask_noise]


            mask_noise_post=np.logical_and(flux_noise!=0,flux_noise>0)
            wave_noise=wave_noise[mask_noise_post]
            flux_noise=flux_noise[mask_noise_post]



            n_points=int(len(flux_noise)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_noise[0])
            max_y.append(flux_noise[0])
            for kk in range(1,9):
                #xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                #yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                #max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))


                xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                max_y.append(yhat_t[0])
                max_w.append(xhat_t[0])



                #max_y.append(max_d_line)
                #max_w.append(xhat_t[max_index])
                #y_points.append(flux_noise[kk*n_points])
                #x_points.append(wave_noise[kk*n_points])
            max_w.append(wave_noise[-1])
            max_y.append(flux_noise[-1])
            s_y=[]
            s_x=[]
            mean_noise = np.mean(max_y)
            std_noise = np.std(max_y)

            #filtered_noise = [el for el in max_y if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            #while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
            #   mean_noise=np.mean(filtered_noise)
            #   std_noise=np.std(filtered_noise)
            #   filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]


            for kk in range(0,len(max_y)):
                if abs((max_y[kk]<mean_noise+3*std_noise)and(max_y[kk]>mean_noise-3*std_noise)):
                    s_y.append(max_y[kk])
                    s_x.append(max_w[kk])
            s_y_1=[]
            s_x_1=[]
            mean_noise = np.mean(s_y)
            std_noise = np.std(s_y)
            for kk in range(0,len(s_y)):
                if abs((s_y[kk]<mean_noise+1*std_noise)and(s_y[kk]>mean_noise-1*std_noise)):
                    s_y_1.append(s_y[kk])
                    s_x_1.append(s_x[kk])


            z=np.polyfit([s_x_1[0],s_x_1[-1]],[s_y_1[0],s_y_1[-1]],1)
            #z=np.polyfit([s_x_1[0],s_x_1[-1]],[s_y_1[0],s_y_1[-1]],1)
            #z=np.polyfit(s_x_1, s_y_1,1)
            p=np.poly1d(z)


            continuum=p(wave_noise)
            #spline = splrep(max_w,max_y,k=2)
            #continuum = splev(wave_noise,spline)
            flux_noise=flux_noise/continuum


            mask_noise_post=np.logical_and(wave_noise>= l_wavelength[t]*(1+(radvel_diagnos/c))-25, wave_noise<=l_wavelength[t]*(1+(radvel_diagnos/c))+25)

            wave_noise_p = wave_noise[mask_noise_post]
            flux_noise_p = flux_noise[mask_noise_post]

            mean_noise = np.mean(flux_noise_p)
            std_noise = np.std(flux_noise_p)
            
            filtered_noise = [el for el in flux_noise_p if (el<mean_noise+8*std_noise)and(el>mean_noise-2*std_noise)]
            mean_noise = np.mean(filtered_noise)
            std_noise = np.std(filtered_noise)


            filtered_noise1_c = [el for el in filtered_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-2*std_noise)]

            filtered_noise=filtered_noise1_c

            while abs((np.std(filtered_noise) - std_noise)/std_noise) > 0.1:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-1.5*std_noise)]


            #print 'FILTERED NOISE!!!!'

            #print filtered_noise5
            #print std_noise


###########Gauss noise to each line
###################################
        my_wave=wave_noise  
        my_flux=flux_noise
        if np.isnan(np.mean(filtered_noise)) == True:
            filtered_noise=[1.0 for el in mask_noise]




    ####
            ''' old
            mean_noise = np.mean(flux_noise)
            std_noise = np.std(flux_noise)
            
            filtered_noise = [el for el in flux_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-2*std_noise)]
            while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-1.5*std_noise)]
            filtered_noise5=filtered_noise
            noise=np.mean(filtered_noise)
            std_noise=np.std(filtered_noise)
            
###########Gauss noise to each line
###################################
            my_wave=wave_noise  
            my_flux=flux_noise
            '''#old


        '''
        if np.isnan(np.mean(filtered_noise)) == True:
            filtered_noise=[1.2 for el in mask_noise]
        if len(filtered_noise)==1:
            filtered_noise=[filtered_noise[0] for el in mask_noise]
        '''



        my_wave_toplot[t]=np.array(my_wave)
        my_flux_toplot[t]=np.array(my_flux)


        


        wav_testline,idx_testline=find_nearest(my_wave, l_wavelength[t]*(1+(radvel_diagnos/c)))
        mask_per=np.logical_and(my_wave>=   wav_testline-1.5, my_wave<=wav_testline+1.5)




        wave_newSp_per=my_wave[mask_per]
        new_flux_per=my_flux[mask_per]

        idx_testline_per, min_d_line_per = min(enumerate(new_flux_per), key=operator.itemgetter(1))

            
        new_min_wave=wave_newSp_per[idx_testline_per]
        wav_testline,idx_testline=find_nearest(my_wave, new_min_wave)






        try:
            left,centres,right = find_edges(idx_testline, my_flux<np.mean(filtered_noise)-0.20*np.std(filtered_noise), my_flux)  ### sensitivity -0.1


            loc_wave_1=my_wave[left[0]]####final borders to gauss
            loc_wave_0=my_wave[centres[0]]
            loc_wave_2=my_wave[right[0]]####final borders to gauss
        

            signal=0

        except:
            left=[0]
            loc_wave_1=0
            loc_wave_0=0
            loc_wave_2=0
        #print('LINES!!!!!', loc_wave_1,loc_wave_0,loc_wave_2)

        if abs(loc_wave_0-loc_wave_1)>2*(loc_wave_2-loc_wave_0):
            #left[0]==0
            signal=1
        elif abs(loc_wave_2-loc_wave_0)>2*(loc_wave_0-loc_wave_1):
            left[0]==0
            signal=2

        #print(left,centres,right)
        if all(mask_noise==False):
            ewHeI5876=0
            xtoplot5876=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
            ytoplot5876=[1.2,1.2]
            wavel_local.append(0)
            RV_local.append(99999.99)
            ERR_RV_local.append(99999.99)
            FWHM_local.append(0)
            EW_local.append(0)
            ERR_EW_local.append(0)

            item_x[t].a=np.array(xtoplot5876)
            item_x[t].b=np.array(ytoplot5876)
            item_x_int[t].a=np.array(xtoplot5876)
            item_x_int[t].b=np.array(ytoplot5876)
            SNR_test.append(0)
            item_x_noise[t].a=np.array(xtoplot5876)
            item_x_noise[t].b=np.array(ytoplot5876)
            item_x_noise[t].c=np.array(ytoplot5876)
            EW_local_INT.append(0)
            SNR.append(0)
            IND_NOISE_FWHM[t]=np.array([0,0])
            xtofit=[0]
            ytofit=[0]
            
        elif left[0]==0:
            ewHeI5876=0
            xtoplot5876=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
            ytoplot5876=[1.2,1.2]
            wavel_local.append(0)
            RV_local.append(99999.99)
            ERR_RV_local.append(99999.99)
            FWHM_local.append(0)
            EW_local.append(0)
            ERR_EW_local.append(0)

            item_x[t].a=np.array(xtoplot5876)
            item_x[t].b=np.array(ytoplot5876)
            item_x_int[t].a=np.array(xtoplot5876)
            item_x_int[t].b=np.array(ytoplot5876)
            SNR_test.append(0)
            item_x_noise[t].a=np.array(xtoplot5876)
            item_x_noise[t].b=np.array(ytoplot5876)
            item_x_noise[t].c=np.array(ytoplot5876)
            EW_local_INT.append(0)
            SNR.append(0)
            IND_NOISE_FWHM[t]=np.array([0,0])
            xtofit=[0]
            ytofit=[0]

        else:



            loc_wave0c=my_wave[left[0]-200:right[0]+200:step_gaus]####final borders to gauss, it was 5
            loc_flux0c=my_flux[left[0]-200:right[0]+200:step_gaus]

            loc_wave=loc_wave0c
            loc_flux=loc_flux0c
            for tk in range(0, 200):
                loc_flux[tk]=np.mean(filtered_noise)+0.1*np.std(filtered_noise)
                
            for tk in range(len(loc_flux)-200,len(loc_flux)):

                loc_flux[tk]=np.mean(filtered_noise)+0.1*np.std(filtered_noise)





            xtofit=loc_wave
            ytofit=loc_flux

            xtofit_new=np.linspace(min(xtofit),max(xtofit),200)
            ytofit_new=np.interp(xtofit_new,xtofit,ytofit)
            xtofit=xtofit_new
            ytofit=ytofit_new

            try:

                #mean_g=sum(xtofit*ytofit)/sum(ytofit)
                #sigma_g=np.sqrt(sum(ytofit*(xtofit-mean_g)**2)/sum(ytofit))
                #sigma_g=np.sqrt(sum((1-ytofit)*(xtofit-mean_g)**2))

                mean_g=sum(xtofit*ytofit)/sum(ytofit)
                sigma_g=np.sqrt(sum((ytofit)*(xtofit-mean_g)**2)/sum(ytofit))
                #sigma_g=np.sqrt(sum((1-ytofit)*(xtofit-mean_g)**2))
                '''

                '''

                #if (linelist_array['Name_wave'][t] == 'OIII5591')or(linelist_array['Name_wave'][t] == 'MgII4481')or(linelist_array['Name_wave'][t] == 'CII4267')or(linelist_array['Name_wave'][t] == 'SiII6371')or(linelist_array['Name_wave'][t] == 'SiIII4552'):
                    #popt,pcov = curve_fit(gaus,xtofit,ytofit,p0=[max(ytofit)-min(ytofit),mean_g,0.6,np.mean(filtered_noise)+0.1*np.std(filtered_noise)],method='lm')#,method='lm'  

                popt,pcov = curve_fit(lorentz,xtofit,ytofit,p0=[max(ytofit)-min(ytofit),mean_g,0.6,np.mean(filtered_noise)+0.2*np.std(filtered_noise)],method='lm')#,method='lm'    



                #print sigma_g,mean_g
                #print xtofit[1]-xtofit[0],xtofit[10]-xtofit[9]
                #print xtofit_new[1]-xtofit_new[0],xtofit[10]-xtofit[9]
                #print '!!!!!!!!!!!!!!!!!!!!!!',popt,pcov
                #print '!!!!!!!!!!!!!!!!!!!!!!',linelist_array['Name_wave'][t]

                #popt,pcov = curve_fit(gaus,xtofit,ytofit,p0=[max(ytofit)-min(ytofit),np.mean(xtofit),np.std(xtofit),max(ytofit)],method='lm')#,method='lm'
                #xtoplot=np.linspace(min(xtofit),max(xtofit),100)
                xtoplot=xtofit
                xtoplot5876=xtoplot
                popt5876=popt

                #if (linelist_array['Name_wave'][t] == 'OIII5591')or(linelist_array['Name_wave'][t] == 'MgII4481')or(linelist_array['Name_wave'][t] == 'CII4267')or(linelist_array['Name_wave'][t] == 'SiII6371')or(linelist_array['Name_wave'][t] == 'SiIII4552'):
                    #ytoplot5876=gaus(xtoplot5876,*popt5876)

                ytoplot5876=lorentz(xtoplot5876,*popt5876)






                #ytoplot5876=gaus(xtoplot5876,*popt5876)
            except:
                popt=[0,0,0,0]
                #print 'xtofit', xtofit
                if (len(xtofit) == 0):
                    #print 'xtofitttt'
                    xtofit =[5000,5001,5002,5004,5005]
                elif xtofit[0] == 0:
                    #print 'xtofitttt'
                    xtofit =[5000,5001,5002,5004,5005]
                xtoplot5876=np.linspace(min(xtofit),max(xtofit),100)
                ytoplot5876=[1.1 for el in xtoplot5876]
                ytofit=ytoplot5876




                #####checking for a line
    
                #if ((max(ytoplot5876)-min(ytoplot5876))<np.mean(filtered_noise)-0*np.std(filtered_noise)):
            if abs((max(ytoplot5876)-min(ytoplot5876))<1.0*np.std(filtered_noise)):
                ewHeI5876=0
                xtoplot5876=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
                ytoplot5876=[1.2,1.2]
                wavel_local.append(0)
                RV_local.append(99999.99)
                ERR_RV_local.append(99999.99)
                FWHM_local.append(0)
                EW_local.append(0)
                ERR_EW_local.append(0)

                item_x[t].a=np.array(xtoplot5876)
                item_x[t].b=np.array(ytoplot5876)
                item_x_int[t].a=np.array(xtoplot5876)
                item_x_int[t].b=np.array(ytoplot5876)
                SNR_test.append(0)
                item_x_noise[t].a=np.array(xtoplot5876)
                item_x_noise[t].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtoplot5876]
                item_x_noise[t].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtoplot5876]
                EW_local_INT.append(0)
                SNR.append(0)
                IND_NOISE_FWHM[t]=np.array([0,0])

            else:
                

            

                if ytoplot5876[0]==1.1:
                    fwhm_last = 0
                elif ytoplot5876[0]==1.2:
                    fwhm_last = 0
                else:
                    #print 'FWHM!!!',ytoplot5876
                    #fwhm_last = fwhm(xtoplot5876, 1-np.asarray(ytoplot5876))
                    fwhm_last=abs(2*popt[2])
                #print('FWHM_last', fwhm_last)



                #if (linelist_array['Name_wave'][t] == 'OIII5591')or(linelist_array['Name_wave'][t] == 'MgII4481')or(linelist_array['Name_wave'][t] == 'CII4267')or(linelist_array['Name_wave'][t] == 'SiII6371')or(linelist_array['Name_wave'][t] == 'SiIII4552'):
                    #EW=popt[0]*popt[2]*np.sqrt(2*np.pi)/max(ytofit) #a*sigma*sqrt(2pi), normalising to continuum level!
                #else:

                EW=popt[0]*popt[2]*np.sqrt(np.pi/np.log(2))/max(ytofit) 

                std_height, std_mean, std_sigma, std_sigma1=np.sqrt(np.diag(pcov))

                EW_local_sigma=[]
                RV_local_sigma=[]
                ss_ew1=np.random.normal(popt[0],std_height,200)
                ss_ew2=np.random.normal(popt[2],std_sigma,200)
                ss_rv=np.random.normal(popt[1],std_mean,200)
                for jk in range(0,200):
                    RV_local_sigma.append((c*( (1.*ss_rv[jk]/l_wavelength[t]) - 1)))
                    EW_local_sigma.append(ss_ew1[jk]*ss_ew2[jk]*np.sqrt(np.pi/np.log(2))/max(ytofit))

                ERR_EW_local.append((np.std(EW_local_sigma))*1000)

                #print 'Name=%s WL=%8.3f  FWHM(5876)=%5.3fmA =%3.1fpx  EW(5876)=%5.3fmA' % (linelist_array['Name_wave'][t],popt[1],10*2.3548*popt[2],2.35482*popt[2]/wave_step,1000*EW)
        
                wavel_local.append(popt[1])

                if (EW < 0):
                    EW=0.00
                    RV_local.append((c*( (1.*popt[1]/l_wavelength[t]) - 1)))
                    ERR_RV_local.append(np.std(RV_local_sigma))
                elif (EW > 2000):
                    EW=0.00
                    RV_local.append(99999.99)   
                    ERR_RV_local.append(99999.99)
                elif  ytofit[0]==1.1:
                    EW=0.00     
                    RV_local.append(99999.99)   
                    ERR_RV_local.append(99999.99)
                else:
                    print('OK, line is present')    
                    RV_local.append((c*( (1.*popt[1]/l_wavelength[t]) - 1)))
                    ERR_RV_local.append(np.std(RV_local_sigma))




                EW_local.append(1000*EW)    
                FWHM_local.append(fwhm_last)
                FWHM_to_int=fwhm_last
                item_x[t].a=np.array(xtoplot5876)
                item_x[t].b=np.array(ytoplot5876)


                a_integr=popt[1]-param*FWHM_to_int  
                b_integr=popt[1]+param*FWHM_to_int  
                b_noise=popt[1]+param*FWHM_to_int+2*FWHM_to_int
                bb_noise=popt[1]+param*FWHM_to_int+4*FWHM_to_int
                def f1(x):
                    wav_testline_f,idx_testline_f=find_nearest(my_wave, x)
            
                    return my_flux[idx_testline_f]
                    del wav_testline_f,idx_testline_f

                wav_testline_fa,idx_testline_fa=find_nearest(my_wave, a_integr)
                wav_testline_fb,idx_testline_fb=find_nearest(my_wave, b_integr)
                wav_testline_fa_noise,idx_testline_fa_noise=find_nearest(my_wave, b_noise)
                wav_testline_fb_noise,idx_testline_fb_noise=find_nearest(my_wave, bb_noise)

                x_iint_noise=[el for el in my_wave[idx_testline_fa_noise:idx_testline_fb_noise]]
                y_iint_noise=[el for el in my_flux[idx_testline_fa_noise:idx_testline_fb_noise]] #########FWHM noise value to return    
                sigma_noise_test=np.std(y_iint_noise)
                SNR_test.append(1/sigma_noise_test)
                IND_NOISE_FWHM[t]=np.array(x_iint_noise)



                x_iint=[el for el in my_wave[idx_testline_fa:idx_testline_fb]]
                y_iint=[el for el in my_flux[idx_testline_fa:idx_testline_fb]]  
            
                y_iint_noise=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtoplot5876] #####Global SN for Gauss +-45
                y_iint_noise_p=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtoplot5876]   #####Global SN for Gauss +-45
                xtopot_g=[l_wavelength[t]*(1+(radvel_diagnos/c))-10,l_wavelength[t]*(1+(radvel_diagnos/c))+10]
                item_x_noise[t].a=xtopot_g
                item_x_noise[t].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtopot_g]
                item_x_noise[t].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtopot_g]
                item_x_int[t].a=np.array(x_iint)
                item_x_int[t].b=np.array(y_iint)
                #INTEGR=integrate.quad(lambda x: f1(x),a_integr,b_integr)   

                #EW_INTEGR=((wav_testline_fb-wav_testline_fa)-INTEGR[0])*1000
                EW_local_INT.append(0)
                EW_INTEGR_ERR = 0
                del xtofit,ytofit

###########################  
############ Second part Integration 
#########################
    print('Second Part')

    my_wave=myspec[0]
    my_flux=myspec[1]
    EW_local_INT_T=[]

    my_wave_toplot_INT = [ structtype() for tt in range(0, len(l_namewave))]
    my_flux_toplot_INT = [ structtype() for tt in range(0, len(l_namewave))]



    filtered_FWHM = [el for el in FWHM_local if (el!=0)and(el>0)and(el!=float('nan'))]
    FWHM_to_analysis=np.mean(filtered_FWHM) 
    size_of_line=2.5*FWHM_to_analysis


    for j in range(0, len(l_wavelength)):   

        if  l_wavelength[j] > 5960:

            my_wave = myspec_red_n[0]
            my_flux = myspec_red_n[1]
            myspec = myspec_red_n



        else:

            myspec=myspec_blue
            my_wave = myspec[0]
            my_flux = myspec[1]

        RV_local_HeI=radvel_diagnos

###################
### New each noise for INT
###################
        mask_noise=np.logical_and(myspec[0]>= l_wavelength[j]*(1+(RV_local_HeI/c))-55, myspec[0]<=l_wavelength[j]*(1+(RV_local_HeI/c))+55)
        if all(mask_noise==False):
            wave_noise=[10,10,10,10,10]
            flux_noise=[10,10,10,10,10]
            filtered_noise=[0,0,0,0]
        else:
            wave_noise = my_wave[mask_noise]
            flux_noise = my_flux[mask_noise]

            n_points=int(len(flux_noise)/10)
            max_w=[]
            max_y=[]
            y_points=[]
            x_points=[]
            max_w.append(wave_noise[0])
            max_y.append(flux_noise[0])
            for kk in range(1,9):
                #xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                #yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                #max_index, max_d_line = max(enumerate(yhat_t), key=operator.itemgetter(1))

                xhat_t=([el for el in wave_noise[kk*n_points:(kk+1)*n_points]])
                yhat_t=([el for el in flux_noise[kk*n_points:(kk+1)*n_points]])
                max_y.append(yhat_t[0])
                max_w.append(xhat_t[0])



                #max_y.append(max_d_line)
                #max_w.append(xhat_t[max_index])
                #y_points.append(flux_noise[kk*n_points])
                #x_points.append(wave_noise[kk*n_points])
            max_w.append(wave_noise[-1])
            max_y.append(flux_noise[-1])
            s_y=[]
            s_x=[]
            mean_noise = np.mean(max_y)
            std_noise = np.std(max_y)
            for kk in range(0,len(max_y)):
                if abs((max_y[kk]<mean_noise+3*std_noise)and(max_y[kk]>mean_noise-3*std_noise)):
                    s_y.append(max_y[kk])
                    s_x.append(max_w[kk])


            s_y_1=[]
            s_x_1=[]
            mean_noise = np.mean(s_y)
            std_noise = np.std(s_y)
            for kk in range(0,len(s_y)):
                if abs((s_y[kk]<mean_noise+1*std_noise)and(s_y[kk]>mean_noise-1*std_noise)):
                    s_y_1.append(s_y[kk])
                    s_x_1.append(s_x[kk])

            z=np.polyfit([s_x_1[0],s_x_1[-1]],[s_y_1[0],s_y_1[-1]],1)
            #z=np.polyfit(s_x_1, s_y_1,1)
            p=np.poly1d(z)


            continuumi=p(wave_noise)

            flux_noise=flux_noise/continuumi
            mean_noise = np.mean(flux_noise)
            std_noise = np.std(flux_noise)  
            #filtered_noise=flux_noise
            
            filtered_noise = [el for el in flux_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            
            filtered_noise6=filtered_noise

            '''
            spline = splrep(max_w,max_y,k=2)
            continuumi = splev(wave_noise,spline)
            flux_noise=flux_noise/continuumi
    ####

            mean_noise = np.mean(flux_noise)
            std_noise = np.std(flux_noise)  

            filtered_noise = [el for el in flux_noise if (el<mean_noise+3*std_noise)and(el>mean_noise-3*std_noise)]
            while (np.mean(filtered_noise) - mean_noise)/mean_noise > 0.0005:
                mean_noise=np.mean(filtered_noise)
                std_noise=np.std(filtered_noise)
                filtered_noise = [el for el in filtered_noise if (el<mean_noise+2*std_noise)and(el>mean_noise-2*std_noise)]
            filtered_noise6=filtered_noise
            '''
            
        if np.isnan(np.mean(filtered_noise)) == True:
            filtered_noise=[1.0 for el in mask_noise]
        my_wave=wave_noise  
        my_flux=flux_noise
        my_wave_toplot_INT[j]=np.array(my_wave)
        my_flux_toplot_INT[j]=np.array(my_flux)
        #RV_local_HeI=0
        SNR_new.append(1/np.std(filtered_noise))   ###################SNRRR INTEGRAL
        wav_testline_T,idx_testline_T=find_nearest(my_wave, l_wavelength[j]*(1+(RV_local_HeI/c)))   

#########################
### end each noise INT
########################

            

######################
###### EM LINE test?
####################


        if (l_namewave[j]=='2')and(my_flux[idx_testline_T] > np.mean(filtered_noise)+1.5*np.std(filtered_noise)):
            #print 'LINE', linelist_array['Name_wave'][j]
            #print 'EM!!!!!! my spectrum', my_flux[idx_testline_T] 
            #print 'EM!!!!!! noise', np.mean(filtered_noise)+1.5*np.std(filtered_noise)
            ewHeI5876_T=-100
            xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            ytoplot5876_T=[1.2,1.2]
            #SNR_new.append(0)
            EW_local_INT_T.append(-100)
            item_x_T[j].a=np.array(xtoplot5876_T)
            item_x_T[j].b=np.array(ytoplot5876_T) 
            item_x_T_noise[j].a=np.array(xtoplot5876_T)
            item_x_T_noise[j].b=[np.mean(filtered_noise)+1.5*np.std(filtered_noise) for el in xtoplot5876_T]
            item_x_T_noise[j].c=[np.mean(filtered_noise)+1.5*np.std(filtered_noise) for el in xtoplot5876_T]
        elif (l_namewave[j]=='3')and(my_flux[idx_testline_T] > np.mean(filtered_noise)+1.5*np.std(filtered_noise)):
            #print 'LINE', linelist_array['Name_wave'][j]
            #print 'EM!!!!!! my spectrum', my_flux[idx_testline_T] 
            #print 'EM!!!!!! noise', np.mean(filtered_noise)+1.5*np.std(filtered_noise)
            ewHeI5876_T=-100
            xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            ytoplot5876_T=[1.2,1.2]
            #SNR_new.append(0)
            EW_local_INT_T.append(-100)
            item_x_T[j].a=np.array(xtoplot5876_T)
            item_x_T[j].b=np.array(ytoplot5876_T) 
            item_x_T_noise[j].a=np.array(xtoplot5876_T)
            item_x_T_noise[j].b=[np.mean(filtered_noise)-2.5*np.std(filtered_noise) for el in xtoplot5876_T]
            item_x_T_noise[j].c=[np.mean(filtered_noise)+2.5*np.std(filtered_noise) for el in xtoplot5876_T]


##########################
######Check for line existance INT
############################3

        elif (my_flux[idx_testline_T] > np.mean(filtered_noise)-1.5*np.std(filtered_noise)): ###3.5 in case of it was 3...
            #### Exception
            ewHeI5876_T=0
            xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            ytoplot5876_T=[1.2,1.2]
            #SNR_new.append(0)
            EW_local_INT_T.append(0)
            item_x_T[j].a=np.array(xtoplot5876_T)
            item_x_T[j].b=np.array(ytoplot5876_T) 
            item_x_T_noise[j].a=np.array(xtoplot5876_T)
            item_x_T_noise[j].b=[np.mean(filtered_noise)-0.1*np.std(filtered_noise) for el in xtoplot5876_T]
            item_x_T_noise[j].c=[np.mean(filtered_noise)+0.1*np.std(filtered_noise) for el in xtoplot5876_T]
############################




        else:
            #wav_testline_left,idx_testline_left=find_nearest(my_wave, (linelist_array['wavelength'][j])*(1+(RV_local_HeI/c))-0.5*size_of_line)
            #wav_testline_right,idx_testline_right=find_nearest(my_wave, (linelist_array['wavelength'][j])*(1+(RV_local_HeI/c))+0.5*size_of_line)
                    ### new     

            wav_testline,idx_testline=find_nearest(my_wave, l_wavelength[j]*(1+(RV_local_HeI/c)))



            left_i,centres_i,right_i = find_edges(idx_testline, my_flux<np.mean(filtered_noise)-0.0*np.std(filtered_noise), my_flux)    #### sensitivity INTEG it was +0.1
            #left_i,centres_i,right_i = find_edges(idx_testline, my_flux<np.mean(filtered_noise)-0.0*np.std(filtered_noise), my_flux)  ### sensitivity

                


            #print 'left right', left_i[0], centres_i[0], right_i[0]
            loc_wave_T=my_wave[left_i[0]:right_i[0]:1]
            loc_flux_T=my_flux[left_i[0]:right_i[0]:1]
            #print loc_wave_T
            item_x_T_noise[j].a=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
            item_x_T_noise[j].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in item_x_T_noise[j].a]   ###############Noise sensetive!!!!!!!!!!!!!!!!!!
            item_x_T_noise[j].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in item_x_T_noise[j].a]   ###############Noise sensetive!!!!!!!!!!!!!!!!!!
            #print 'left,right',left_i[0], right_i[0]
            if (left_i[0]==0)and(right_i[0]==0):
                #print 'ZERO',left_i[0], right_i[0]                 
                ewHeI5876_T=0
                xtoplot5876_T=[l_wavelength[j]*(1+(RV_local_HeI/c))-10,l_wavelength[j]*(1+(RV_local_HeI/c))+10]
                ytoplot5876_T=[1.2,1.2]
                #SNR_new.append(0)
                EW_local_INT_T.append(0)
                item_x_T[j].a=np.array(xtoplot5876_T)
                item_x_T[j].b=np.array(ytoplot5876_T) 
                item_x_T_noise[j].a=np.array(xtoplot5876_T)
                item_x_T_noise[j].b=[np.mean(filtered_noise)-0.5*np.std(filtered_noise) for el in xtoplot5876_T]
                item_x_T_noise[j].c=[np.mean(filtered_noise)+0.5*np.std(filtered_noise) for el in xtoplot5876_T]

            else:
            

                                    
                xtoplot5876_T=loc_wave_T
                ytoplot5876_T=loc_flux_T
                #print xtoplot5876_T

                maxm = argrelextrema(ytoplot5876_T, np.greater)  
                

                item_x_T[j].a=np.array(xtoplot5876_T) #~ tab once
                item_x_T[j].b=np.array(ytoplot5876_T) #~ tab once - double EW INT
                
                
                
                
                #print left_i[0], right_i[0]
                a_integr_T=xtoplot5876_T[0] 
                
                b_integr_T=xtoplot5876_T[-1]        
            


                    
                #print  'IN A IN B', a_integr_T, b_integr_T
                def  f2(x):
                    wav_testline_T_int,idx_testline_T_int=find_nearest(my_wave, x)
                    flux_to_int=my_flux[idx_testline_T_int]
                    return flux_to_int

                
                
                INTEGR_T=integrate.quad(lambda x: f2(x),a_integr_T,b_integr_T)  
                EW_INTEGR_T=((b_integr_T-a_integr_T)-INTEGR_T[0])*1000
                EW_local_INT_T.append(EW_INTEGR_T)
    
                if (EW_local_INT_T[j] > 25000)or(EW_local_INT_T[j] < 0):
                    EW_local_INT_T[j]=0

                EW_INTEGR_ERR_T =   INTEGR_T[0]
                    
                




    
    my_wave = myspec[0]
    my_flux = myspec[1]

    if plots==1:
        ax2=plt.subplot(412)
        plt.title(files)



        plt.plot(myspec_red[0], myspec_red[1], color='blue',lw=1)

        plt.plot(myspec_blue[0], myspec_blue[1], color='blue',lw=1)

            #print '1!!!',item_x[tt].a


        plt.fill_between(wave_find1, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
        plt.fill_between(wave_find2, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
        #   plt.fill_between(wave_find3, 0, 1.5, facecolor='green', interpolate=True,alpha=0.05)
        plt.plot(wave_find1,continuum1,'r-',linewidth=0.5)
        plt.plot(wave_find1,flux_find1,'black',linewidth=0.5)
        plt.plot([wave_find1[0],wave_find1[-1]],[min(filtered_noise1),min(filtered_noise1)],'cyan',linewidth=0.5)
        plt.plot([wave_find1[0],wave_find1[-1]],[max(filtered_noise1),max(filtered_noise1)],'cyan',linewidth=0.5)

        plt.plot(wave_find2,continuum2,'r-',linewidth=0.5)
        plt.plot(wave_find2,flux_find2,'black',linewidth=0.5)
        plt.plot([wave_find2[0],wave_find2[-1]],[min(filtered_noise2),min(filtered_noise2)],'cyan',linewidth=0.5)
        plt.plot([wave_find2[0],wave_find2[-1]],[max(filtered_noise2),max(filtered_noise2)],'cyan',linewidth=0.5)

    #       plt.plot(wave_find3,continuum3,'r-',linewidth=0.5)
    #       plt.plot(wave_find3,flux_find3,'black',linewidth=0.5)
    #       plt.plot([wave_find3[0],wave_find3[-1]],[min(filtered_noise3),min(filtered_noise3)],'cyan',linewidth=0.5)
    #       plt.plot([wave_find3[0],wave_find3[-1]],[max(filtered_noise3),max(filtered_noise3)],'cyan',linewidth=0.5)



        plt.text(wave_find1[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise1,std_noise1,1/std_noise1),rotation=90,fontsize=6,fontweight='ultralight')
        plt.text(wave_find2[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise2,std_noise2,1/std_noise2),fontsize=6,rotation=90,fontweight='ultralight')
        #   plt.text(wave_find3[0]-35,0.7,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise3,std_noise3,1/std_noise3),fontsize=6,rotation=90,fontweight='ultralight')

        #   plt.plot(item_x_T_noise[tt].a,item_x_T_noise[tt].b,'blue',lw=1)

        #fig2=plt.show()


        for tt in range(0, len(l_wavelength)):
            if l_wavelength[tt]<5500:

                plt.plot([l_wavelength[tt],l_wavelength[tt]],[0,2],'gray',linewidth=0.2,linestyle='--') 
                plt.text(l_wavelength[tt]-10,1.45,l_namewave[tt],fontsize=6,rotation=90)
                plt.plot(item_x[tt].a,item_x[tt].b,'r-',lw=1)
                plt.plot(item_x_int[tt].a,item_x_int[tt].b,'g-',lw=1)
            #   plt.plot(item_x_noise[tt].a,item_x_noise[tt].b,'cyan',lw=1)
                plt.plot(item_x_T[tt].a,item_x_T[tt].b,'orange',lw=1)




        plt.ylim([0.3,1.5])
    #   plt.xlim([3800,7000])
        plt.xlim([4700,5500])

        ax3=plt.subplot(413)
        plt.title('Red part')

        plt.plot(my_wave_red,my_flux_red,'b-',lw=0.5)
        plt.plot(my_wave_red,continuum_red,'b--',lw=0.5)
        plt.plot(max_w_red,max_y_red,'rX')
        plt.plot(second_x_red,second_y_red,'gX')

        ax4=plt.subplot(414)
        plt.title('Red part')
        plt.fill_between(wave_find3, 0, 1.5, facecolor='green', interpolate=True,alpha=0.5)
        plt.plot(my_wave_red,my_flux_red/continuum_red,'b-',lw=0.5)
        plt.plot(wave_find3,continuum3,'r-',linewidth=0.5)
        plt.plot(wave_find3,flux_find3,'black',linewidth=0.5)
        plt.plot([wave_find3[0],wave_find3[-1]],[min(filtered_noise3),min(filtered_noise3)],'cyan',linewidth=0.5)
        plt.plot([wave_find3[0],wave_find3[-1]],[max(filtered_noise3),max(filtered_noise3)],'cyan',linewidth=0.5)
        plt.text(wave_find3[0]-35,0.9,'C=%2.3f$\pm$%2.3f SNR=%2.1f'%(noise3,std_noise3,1/std_noise3),fontsize=6,rotation=90,fontweight='ultralight')
        
        for tt in range(0, len(l_wavelength)):
            if l_wavelength[tt]>5950:

                plt.plot([l_wavelength[tt],l_wavelength[tt]],[0,2],'gray',linewidth=0.2,linestyle='--') 
                plt.text(l_wavelength[tt]-10,1.45,l_namewave[tt],fontsize=6,rotation=90)
                plt.plot(item_x[tt].a,item_x[tt].b,'r-',lw=1)
                plt.plot(item_x_int[tt].a,item_x_int[tt].b,'g-',lw=1)
            #   plt.plot(item_x_noise[tt].a,item_x_noise[tt].b,'cyan',lw=1)
                plt.plot(item_x_T[tt].a,item_x_T[tt].b,'orange',lw=1)

        plt.ylim([0.3,1.5])
        plt.xlim([5900,6900])
        plt.tight_layout()



        plt.savefig(pp1, format='pdf')


    mean_noise_TEST=[1/std_noise1,1/std_noise2,1/std_noise3]   ######First S/N variable to out



    #plt.clf()

    RV_gauss_tt=[el for el in RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]
    RV_gauss_meant=np.mean(RV_gauss_tt)
    RV_gauss_sigmat=np.std(RV_gauss_tt)
    RV_gauss_tt=[el for el in RV_gauss_tt if (abs(np.mean(RV_gauss_tt)-el) < 3*np.std(RV_gauss_tt))]
    RV_gauss_meant1=np.mean(RV_gauss_tt)
    RV_gauss_sigmat1=np.std(RV_gauss_tt)


    RV_gauss_tt=[el for el in RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]

    RV_gauss_tt=[el for el in RV_gauss_tt if (abs(radvel_diagnos-el) < 1*np.std(RV_gauss_tt))]
    RV_gauss_meant=np.mean(RV_gauss_tt)
    RV_gauss_sigmat=np.std(RV_gauss_tt)




    for j in range(3, len(l_wavelength)):

        if abs(radvel_diagnos-RV_local[j])>1*RV_gauss_sigmat:
            RV_local[j]=0
            EW_local[j]=0

    RV_gauss_fin=[el for el in RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]
    for j in range(3, len(l_wavelength)):

        if abs(radvel_diagnos-RV_local[j])>2*np.std(RV_gauss_fin):
            RV_local[j]=float('nan')
            EW_local[j]=0
    # print('waveobs_shifted_blue', waveobs_shifted_blue)

    if plots==1:
        my_dpi=50
        fig=plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
        num_pages=int(ceil(len(l_wavelength)/30))
        t1=0
        print(range(0,int(num_pages+1)))
        for tr in range(0,num_pages+1):
            gs1 = gridspec.GridSpec(5,6)
            trtr=0
            for ttt in range(t1, t1+30):
                if ttt>=len(l_wavelength):
                    ttt=len(l_wavelength)-1
                print(ttt, len(l_wavelength),trtr,len(my_wave_toplot))

                #fig.add_subplot(3,4,ttt+1)
                fig.add_subplot(gs1[trtr])
        #       plt.plot(myspec[0], myspec[1], color='black')
                plt.plot(my_wave_toplot[ttt],my_flux_toplot[ttt],color='black')
                #plt.plot(my_wave_toplot_INT[ttt],my_flux_toplot_INT[ttt],color='gray')
            #   plt.fill_between(IND_NOISE_FWHM[ttt], 0, 1.5, facecolor='green', interpolate=True,alpha=0.3)
                
                if waveobs_shifted_blue[0]==-999:
                    #print 'RED!!'
                    plt.plot(waveobs_shifted_red, myspec_red_n[1],'--',linewidth=0.5,color='blue')
                elif waveobs_shifted_red[0]==-999:

                    plt.plot(waveobs_shifted_blue, myspec_blue[1],'--',linewidth=0.5,color='blue')
                else:
                    plt.plot(waveobs_shifted_red, myspec_red_n[1],'--',linewidth=0.5,color='blue')
                    plt.plot(waveobs_shifted_blue, myspec_blue[1],'--',linewidth=0.5,color='blue')




                plt.plot([l_wavelength[ttt],l_wavelength[ttt]],[0,2],'gray',linewidth=0.5,linestyle='--')       
                plt.plot(item_x[ttt].a,item_x[ttt].b,'r-',lw=2) #Gauss

                plt.plot(item_x_int[ttt].a,item_x_int[ttt].b,'g-',lw=1)  #INT Gauss FWHM
                plt.plot(item_x_T[ttt].a,item_x_T[ttt].b,'orange',lw=2) #INT

                #print item_x_noise[ttt].a, item_x_noise[ttt].b
                plt.plot(item_x_noise[ttt].a,item_x_noise[ttt].b,'r--',lw=1)
                plt.plot(item_x_noise[ttt].a,item_x_noise[ttt].c,'r--',lw=1)
                plt.plot(item_x_T_noise[ttt].a,item_x_T_noise[ttt].b,'--',lw=1, color='orange')
                plt.plot(item_x_T_noise[ttt].a,item_x_T_noise[ttt].c,'--',lw=1, color='orange')

                wav_cen=l_wavelength[ttt]

                plt.text(wav_cen-15.5,0.75,'EW(G)= %2.1f$\pm$%2.1f'%(EW_local[ttt],ERR_EW_local[ttt]),fontsize=6)
                plt.text(wav_cen-15.5,0.73,'EW(G INT)= %2.1f'%(EW_local_INT[ttt]),fontsize=6)
                plt.text(wav_cen-15.5,0.71,'EW(INT)= %2.1f'%(EW_local_INT_T[ttt]),fontsize=6)
                

                plt.text(wav_cen+2,0.73,'SNR(FWHM)= %2.1f'%(SNR_test[ttt]),fontsize=6)
                plt.text(wav_cen+2,0.71,'SNR(INT)= %2.1f'%(SNR_new[ttt]),fontsize=6)


                plt.text(wav_cen+2,1.08,'RV first= %2.2f'%(radvel_diagnos),fontsize=6)
                plt.text(wav_cen+2,1.06,'<RV>= %2.2f$\pm$%2.2f'%(RV_gauss_meant1,RV_gauss_sigmat1),fontsize=6)
                plt.text(wav_cen+2,1.04,'RV(ind)= %2.2f$\pm$%2.2f'%(RV_local[ttt],ERR_RV_local[ttt]),fontsize=6)
                plt.ylim([0.5,1.1])
                plt.xlim([wav_cen-25,wav_cen+25])
                plt.title(l_namewave[ttt])
                plt.tight_layout()
                #plt.show()
                #plt.clf()  # Clear the figure for the next loop
                plt.yticks(size=8)
                plt.xticks(size=8)
                trtr=trtr+1

            t1=t1+30
            plt.savefig(pp1, format='pdf')
            plt.clf()




    fits_l_name_wave=[]
    fits_l_wave=[]
    fits_l_EW=[]
    fits_l_EW_err=[]
    fits_l_RV_local=[]
    fits_l_ERR_RV_local=[]
    #plt.savefig(pp1, format='pdf')

    #gs1.tight_layout(fig,rect=[0,0,1,0.95])
    #plt.show() #          !!!!!!!!!!!!!!!!!!!!!!!!!!!


    #File_gala_local=open(path+"/"+files+"_HR_weave_gala_test.txt","w")


    err_ew=ERR_EW_local
    for j in range(3, len(l_wavelength)):
        if EW_local[j]>1:
            #File_gala_local.write('%4.4f   %4.2f  %4.2f  %4.2f  %4.2f  %4.3f  %4.2f  %4.2f  %4.2f  %4.2f  %s' %(l_wavelength[j],EW_local[j],err_ew[j],l_namewave[j],l_5[j],l_6[j],l_7[j],l_8[j],l_9[j],l_10[j], '\n'))
            fits_l_name_wave.append(round(l_namewave[j],2))
            fits_l_wave.append(round(l_wavelength[j],3))
            fits_l_EW.append(round(EW_local[j],3))
            fits_l_EW_err.append(round(err_ew[j],3))
            fits_l_RV_local.append(round(RV_local[j],2))
            fits_l_ERR_RV_local.append(round(ERR_RV_local[j],2))


    #File_gala_local.close()

    return ReturnValue(files, radvel_diagnos, name_of_line, RV_local, EW_local, FWHM_local, EW_local_INT,mean_noise_TEST,EW_local_INT_T,SNR_test,fits_l_name_wave,fits_l_wave,fits_l_EW,fits_l_EW_err,fits_l_RV_local,ERR_RV_local,fits_l_ERR_RV_local)

########################################################################



def main_RRLEW(rrlew_args):

    name_file_aps, infiles, mark_targ_class, hr_lines, lr_lines, aps_ids, outpath, headname = rrlew_args
    global pp1
    global fig


    assert os.path.exists(name_file_aps), 'No APS file found'


    # Reading the L1_file folder
    hdulist0 = pyfits.open(infiles[0][:])
    # hdulist0.info()

    data0 = hdulist0['FIBTABLE'].data
    header_aps0 = hdulist0[0].header
    data_sp_mode = header_aps0['MODE']

    fiber_id0 = data0['FIBREID']
    ra_aps = data0['TARGRA']
    dec_aps = data0['TARGDEC']
    targ_magni = data0['MAG_I']
    targ_magng = data0['MAG_G']
    targ_name = data0['TARGNAME']
    targ_group = data0['TARGSRVY']
    cname = data0['CNAME']
    targ_id = data0['TARGID']

    try:
        targ_class_L1 = data0['TARGCLASS']
    except:
        targ_class_L1 = data0['TARGPROG']

    hdulist1 = pyfits.open(name_file_aps, ignore_missing_end=True) # Reading L2 file
    # hdulist1.info()

    #Reading FIBTABLE table of L2 data
    # data = hdulist1['CLASS_TABLE'].data
    #Reading STELLAR RVS TABLE table of L2 data

    data_rvs = hdulist1['STAR_TABLE'].data

    # targ_id = data['TARGID']
    # fiber_id = data['APS_ID']




    # Reading the FERRE parameters
    '''
    targ_sn_ferre = data_sn['SNR_FERRE']
    teff_ferre = data_sn['TEFF']
    err_teff_s_ferre = data_sn['TEFF_ERR']
    log_ferre = data_sn['LOGG']
    err_logg_ferre = data_sn['LOGG_ERR']
    mh_ferre = data_sn['M_H']
    mh_err_ferre = data_sn['M_H_ERR']

    '''

    # Reading the STELLAR RVS parameters

    fiberid_rvs = data_rvs['APS_ID']
    vrad_rvs_all = data_rvs['VRAD']
    err_vrad_rvs_err = data_rvs['VRAD_ERR']
    snr_rvs_all = data_rvs['SNR_RVS']

    logg_rvs = data_rvs['LOGG']
    logg_rvs_err = data_rvs['LOGG_ERR']

    teff_rvs = data_rvs['TEFF']
    teff_rvs_err = data_rvs['TEFF_ERR']

    feh_rvs = data_rvs['FEH']
    feh_rvs_err = data_rvs['FEH_ERR']

    alpha_rvs = data_rvs['ALPHA']
    alpha_rvs_err = data_rvs['ALPHA_ERR']

    fiberid_rvs = list(fiberid_rvs)
    fiber_id0 = list(fiber_id0)

    vrad_rvs_all_new = []
    vrad_rvs_all_new_err = []
    targ_rv = []
    snr_rvs_new = []
    targ_class_fin = []
    targ_group_fin = []
    targ_name_fin = []
    targ_magni_fin = []
    targ_magng_fin = []
    cname_fin = []
    targ_id_fin = []
    ra_fin = []
    dec_fin = []
    fiberid_fin = []



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
            targ_rv.append(plots)
            targ_class_fin.append(targ_class_L1[ind_fiber])
            targ_group_fin.append(targ_group[ind_fiber])
            targ_name_fin.append(targ_name[ind_fiber])
            targ_magni_fin.append(targ_magni[ind_fiber])
            targ_magng_fin.append(targ_magng[ind_fiber])
            cname_fin.append(cname[ind_fiber])
            targ_id_fin.append(targ_id[ind_fiber])
            ra_fin.append(ra_aps[ind_fiber])
            dec_fin.append(dec_aps[ind_fiber])
            fiberid_fin.append(fiber_id0[ind_fiber])

            try:
                ind_targid = fiberid_rvs.index(fiber_id0[ind_fiber])
                vrad_rvs_all_new.append(vrad_rvs_all[ind_targid])
                vrad_rvs_all_new_err.append(err_vrad_rvs_err[ind_targid])
                snr_rvs_new.append(snr_rvs_all[ind_targid])

            except:
                vrad_rvs_all_new.append(float('nan'))
                vrad_rvs_all_new_err.append(float('nan'))
                snr_rvs_new.append(float('nan'))


    if len(targ_rv) == 0:
        sys.exit('No RR Lyraes in this OB! No output files created.')
    else:
        pass

    fluxes1 = []
    fluxes2 = []
    wave_bn = []
    wave_rn = []

    # Reading L1 data by APS_utils function
    targs =APSOB(infiles, aps_ids=fiberid_fin, targsrvy=targsrvy, targclass=targclass, mask_aps_ids=mask_aps_ids , area=area, mask_areas=mask_areas,
        wlranges=wlranges, sens_corr=sens_corr, mask_gaps=mask_gaps, vacuum=vacuum, tellurics=tellurics,
        fill_gap=fill_gap, arms_ratio=arms_ratio, join_arms=join_arms, funit= funit, offset_gap_pix = offset_gap_pix)


    data = targs.data()

    if data_sp_mode == 'HIGHRES':
        for kr in range(0, len(data)):
            target_id = targs.apstoid(fiberid_fin[kr])

            fluxes1.append(list(data[target_id].spectra[0].flux[10:-10]))
            fluxes2.append(list(data[target_id].spectra[1].flux[100:-30]))
            wave_bn.append(list(data[target_id].spectra[0].wave[10:-10]))
            wave_rn.append(list(data[target_id].spectra[1].wave[100:-30]))

    elif data_sp_mode == 'LOWRES':

        for kr in range(0, len(data)):
            target_id = targs.apstoid(fiberid_fin[kr])

            fluxes1.append(list(data[target_id].spectra[0].flux[0:-200]))
            fluxes2.append(list(data[target_id].spectra[1].flux[100:-30]))
            wave_bn.append(list(data[target_id].spectra[0].wave[0:-200]))
            wave_rn.append(list(data[target_id].spectra[1].wave[100:-30]))
    else:

        sys.exit('ERROR: In which resolution mode we are working?')

    test_array = [(a,b,c,d,e,f,g,h,i,k,k1,k2,ra,dec,k5,k6,k7,k8) for a,b,c,d,e,f,g,h,i,k,k1,k2,ra,dec,k5,k6,k7,k8 in zip(wave_bn,fluxes1,targ_class_fin,targ_group_fin,targ_name_fin,targ_magni_fin,targ_rv,cname_fin,targ_id_fin,targ_magng_fin,snr_rvs_new,vrad_rvs_all_new,ra_fin,dec_fin,wave_rn,fluxes2,fiberid_fin,vrad_rvs_all_new_err)]
    sub_array = test_array
    argumentsList = sub_array

    del hdulist1, data, data_rvs, hdulist0


    c_name_to_plot=[]
    name_to_plot=[]
    fiberid_to_plot=[]
    targsrvy_plot=[]
    targ_class_plot=[]
    ra_to_plot=[]
    dec_to_plot=[]
    snr_plot=[]
    magi_plot=[]
    magg_plot=[]
    rvs_plot_fin_aps=[]
    err_rvs_plot=[]
    fits_l_name_wave_plot=[]
    fits_l_wave_plot=[]
    fits_l_EW_plot=[]
    fits_l_EW_err_plot=[]
    fits_l_RV_local_plot=[]
    rv_metal_plot=[]
    err_rv_metal_plot=[]
    EW_h4340rv_plot=[]
    EW_h4861rv_plot=[]
    EW_h6562rv_plot=[]
    ERR_EW_h4340rv_plot=[]
    ERR_EW_h4861rv_plot=[]
    ERR_EW_h6562rv_plot=[]
    fits_l_ERR_RV_local_plot=[]

    #### !!!! Change the range to len(sub_array)
    for i in range(0, len(sub_array)):

        if data_sp_mode == 'HIGHRES':
            self = actualseefits_HR(argumentsList[i], hr_lines)
        elif data_sp_mode == 'LOWRES':
            self = actualseefits_LR(argumentsList[i], lr_lines)
        else:
            pass

        mag_i_fits = float(sub_array[i][5])
        rad_vel_fits = float(sub_array[i][6])
        spec_name = sub_array[i][2]
        targ_name = sub_array[i][4]

        if data_sp_mode =='HIGHRES':

            EW_h4340rv_plot.append(float('nan'))
            ERR_EW_h4340rv_plot.append(float('nan'))
            for jj in range(0, len(self.RV_local[:])):


                if self.name_of_line[jj]==2:
                    EW_h4861rv_plot.append(round(self.RV_local[jj],2))
                    ERR_EW_h4861rv_plot.append(round(self.ERR_RV_local[jj],2))

                elif self.name_of_line[jj]==3:
                    EW_h6562rv_plot.append(round(self.RV_local[jj],2))
                    ERR_EW_h6562rv_plot.append(round(self.ERR_RV_local[jj],2))


        else:

            for jj in range(0, len(self.RV_local[:])):

                if self.name_of_line[jj] == 1:
                    EW_h4340rv_plot.append(round(self.RV_local[jj],2))
                    ERR_EW_h4340rv_plot.append(round(self.ERR_RV_local[jj],2))



                elif self.name_of_line[jj] == 2:
                    EW_h4861rv_plot.append(round(self.RV_local[jj],2))
                    ERR_EW_h4861rv_plot.append(round(self.ERR_RV_local[jj],2))

                elif self.name_of_line[jj] == 3:
                    EW_h6562rv_plot.append(round(self.RV_local[jj],2))
                    ERR_EW_h6562rv_plot.append(round(self.ERR_RV_local[jj],2))




        RV_gauss_t=[el for el in self.RV_local[:] if (el != 0)and(el<2000)and(el>-2000)and(el[~np.isnan(el)])and(el[~np.isinf(el)])]

        RV_gauss_mean_new=np.mean(RV_gauss_t)
        RV_gauss_sigma_new=np.std(RV_gauss_t)

        mean_noise_TEST=self.SNR_new




        if self.name_of_line[0]=='No H lines in file':


            sys.exit('Something wrong with the spectra, no wavelength region of the H lines exits, no output file produced!')


        else:

            final_fits_l_name_wave=np.full(581,float('nan'))
            final_fits_l_wave=np.full(581,float('nan'))
            final_fits_l_EW=np.full(581,float('nan'))
            final_fits_l_EW_err=np.full(581,float('nan'))
            final_fits_l_RV_local=np.full(581,float('nan'))
            final_fits_l_ERR_RV_local=np.full(581,float('nan'))

            for k in range(0, len(self.fits_l_wave)):
                final_fits_l_name_wave[k]=self.fits_l_name_wave[k]
                final_fits_l_wave[k]=self.fits_l_wave[k]
                final_fits_l_EW[k]=self.fits_l_EW[k]
                final_fits_l_EW_err[k]=self.fits_l_EW_err[k]
                final_fits_l_RV_local[k]=self.fits_l_RV_local[k]
                final_fits_l_ERR_RV_local[k]=self.fits_l_ERR_RV_local[k]




            c_name_to_plot.append(sub_array[i][7])
            name_to_plot.append(sub_array[i][4])
            fiberid_to_plot.append(sub_array[i][16])
            targsrvy_plot.append(sub_array[i][3])
            targ_class_plot.append(sub_array[i][2])
            ra_to_plot.append(sub_array[i][12])
            dec_to_plot.append(sub_array[i][13])
            snr_plot.append(sub_array[i][10])
            magi_plot.append(sub_array[i][5])
            magg_plot.append(sub_array[i][9])
            rvs_plot_fin_aps.append(round(sub_array[i][11],2))
            err_rvs_plot.append(round(sub_array[i][17],2))



            fits_l_name_wave_plot.append(final_fits_l_name_wave)
            fits_l_wave_plot.append(final_fits_l_wave)
            fits_l_EW_plot.append(final_fits_l_EW)
            fits_l_EW_err_plot.append(final_fits_l_EW_err)
            fits_l_RV_local_plot.append(final_fits_l_RV_local)
            fits_l_ERR_RV_local_plot.append(final_fits_l_ERR_RV_local)

            #fits_l_name_wave_plot.append(self.fits_l_name_wave)
            #fits_l_wave_plot.append(self.fits_l_wave)
            #fits_l_EW_plot.append(self.fits_l_EW)
            #fits_l_EW_err_plot.append(self.fits_l_EW_err)
            #fits_l_RV_local_plot.append(self.fits_l_RV_local)
            #fits_l_ERR_RV_local_plot.append(self.fits_l_ERR_RV_local)



            rv_metal_plot.append(round(np.mean(self.fits_l_RV_local),2))
            err_rv_metal_plot.append(round(np.std(self.fits_l_RV_local),2))

        title=self.name

        print('It was the spectrum: ', title)

        if plots == 1:

            plot_ew_local=self.EW_local
            rvs_plot=self.RV_local
            names_plot=self.name_of_line

            my_dpi=100
            plt.figure(figsize=(1000/my_dpi, 1000/my_dpi))
            plt.title('%s %s Mean RVs=%2.1f; std(RVs)=%2.1f; RVS_APS=%2.1f'%(title,sub_array[i][7],RV_gauss_mean_new,RV_gauss_sigma_new,float(sub_array[i][11])),fontsize=10)
            epoch_plot=range(0,len(names_plot))
            #epoch_plot=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #rvs_plot=[EW_5876rv, EW_4471rv, EW_4713rv, EW_4541rv, EW_5411rv, EW_4686rv, EW_5592rv,EW_4552rv,EW_4481rv,EW_4267rv,EW_6371rv,EW_4116rv,EW_h4340rv,EW_h4861rv]



            plt.grid(color='black', linestyle='-', linewidth=0.1,alpha=0.5)

            plt.plot([epoch_plot[0],epoch_plot[-1]],[RV_gauss_mean_new,RV_gauss_mean_new],color='gray',marker='None',linestyle='-')#,label='Mean (RV_g Clean)'
            plt.plot([epoch_plot[0],epoch_plot[-1]],[RV_gauss_mean_new-1*RV_gauss_sigma_new,RV_gauss_mean_new-1*RV_gauss_sigma_new],color='gray',marker='None',linestyle='--')
            plt.plot([epoch_plot[0],epoch_plot[-1]],[RV_gauss_mean_new+1*RV_gauss_sigma_new,RV_gauss_mean_new+1*RV_gauss_sigma_new],color='gray',marker='None',linestyle='--')


            for kr in range(0,len(names_plot)):
                if plot_ew_local[kr]==0:
                    rvs_plot[kr]=np.nan
                plt.plot(kr,rvs_plot[kr],color='red',marker='o',linestyle='None',markersize=8)

            for kp in range(0,len(names_plot)):

                plt.text(kp,rvs_plot[kp]-5,names_plot[kp],fontsize=5)
            '''
            for kr in range(0,len(self.fits_l_RV_local)):

                plt.plot(kr,self.fits_l_RV_local[kr],color='green',marker='x',linestyle='None',markersize=5)
            '''

            plt.plot(0,float(sub_array[i][11]),color='blue',marker='x',label='APS RVS',linestyle='None',markersize=10)

            plt.ylabel('RV (km/s)')
            plt.xlabel('Spectral lines')
            plt.tick_params(labelsize=10)
            try:
                plt.ylim([RV_gauss_mean_new-7*RV_gauss_sigma_new,RV_gauss_mean_new+7*RV_gauss_sigma_new])
            except:
                plt.ylim([-70,70])
            plt.legend()
            plt.savefig(pp1, format='pdf')
            pp1.close()
            plt.clf()
            plt.close()
            del pp1
        del self

    # Making a final output fits table.
    c1 = fits.Column(name='CNAME', array=c_name_to_plot, format='25A')
    c2 = fits.Column(name='TARGNAME', array=name_to_plot, format='15A')
    c3 = fits.Column(name='FIBREID', array=fiberid_to_plot, format='I')
    c4 = fits.Column(name='TARGSRVY', array=targsrvy_plot, format='15A')
    c5 = fits.Column(name='TARGCLASS', array=targ_class_plot, format='15A')
    c6 = fits.Column(name='TARGRA', array=ra_to_plot, format='D')
    c7 = fits.Column(name='TARGDEC', array=dec_to_plot, format='D')
    c8 = fits.Column(name='SNR_RVS', array=snr_plot, format='D')
    c9 = fits.Column(name='MAG_G', array=magg_plot, format='D',unit='mag')
    c10 = fits.Column(name='MAG_I', array=magi_plot, format='D',unit='mag')
    c11 = fits.Column(name='VRAD_RVS', array=rvs_plot_fin_aps, format='E',unit='km/s')
    c12 = fits.Column(name='VRAD_RVS_ERR', array=err_rvs_plot, format='E',unit='km/s')
    c13 = fits.Column(name='NAME_LINES_RRLEW', array=fits_l_name_wave_plot, format='PE()',unit='atomic number')
    c14 = fits.Column(name='WAVE_LINES_RRLEW', array=fits_l_wave_plot, format='PE()',unit='A')
    c15 = fits.Column(name='EW_LINES_RRLEW', array=fits_l_EW_plot, format='PE()',unit='mA')
    c16 = fits.Column(name='ERR_EW_LINES_RRLEW', array=fits_l_EW_err_plot, format='PE()',unit='mA')
    c17 = fits.Column(name='RV_LINES_RRLEW', array=fits_l_RV_local_plot, format='PE()',unit='km/s')
    c18 = fits.Column(name='ERR_RV_LINES_RRLEW', array=fits_l_ERR_RV_local_plot, format='PE()',unit='km/s')

    c19 = fits.Column(name='MEAN_RV_METALLIC_RRLEW', array=rv_metal_plot, format='E',unit='km/s')
    c20 = fits.Column(name='ERR_RV_METALLIC_RRLEW', array=err_rv_metal_plot, format='E',unit='km/s')

    c21 = fits.Column(name='RV_HGAMMA_RRLEW', array=EW_h4340rv_plot, format='E',unit='km/s')
    c22 = fits.Column(name='ERR_RV_HGAMMA_RRLEW', array=ERR_EW_h4340rv_plot, format='E',unit='km/s')

    c23 = fits.Column(name='RV_HBETA_RRLEW', array=EW_h4861rv_plot, format='E',unit='km/s')
    c24 = fits.Column(name='ERR_RV_HBETA_RRLEW', array=ERR_EW_h4861rv_plot, format='E',unit='km/s')

    c25 = fits.Column(name='RV_HALPHA_RRLEW', array=EW_h6562rv_plot, format='E',unit='km/s')
    c26 = fits.Column(name='ERR_RV_HALPHA_RRLEW', array=ERR_EW_h6562rv_plot, format='E',unit='km/s')

    hdr = fits.Header()
    hdr['COMMENT'] = 'WEAVE Contributed Software: RRLEW'
    hdr['CS_CODE'] = 'RRLGV'
    hdr['CS_VER'] = '5.0'
    hdr['CS_NME1'] = 'Nikolay'
    hdr['CS_NME2'] = 'Britavskiy'
    hdr['CS_MAIL'] = os.environ.get('PYAPS_CS_MAIL', '')  # contributor contact, set via env (not hardcoded)
    hdr['PROV1001'] = os.path.basename(infiles[0])
    hdr['PROV1002'] = os.path.basename(name_file_aps)
    primary = fits.PrimaryHDU(header=hdr)


    t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26])
    t1 = fits.HDUList([primary, t])


    output_file = os.path.join(outpath, headname) +'_RRLEW.fits'
    t1.writeto(output_file , overwrite=True)

    print('Output file created :%s' %(output_file))



############################################################################################################

def rrlew_weave(options=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--apsfile", help='The input fits file, contains the aps table',
                        type=none_or_str, default=none_or_str, required=True)
    parser.add_argument("--infiles", help='The input ephemerids file, contains the ephemerids for a given catalogue of stars',
                        type=none_or_str, default=none_or_str, required=True, nargs='*')
    parser.add_argument("--targclass", help='targclass argument to select RR Lyraes',
                        type=none_or_str, default=None, required=False)
    parser.add_argument("--outpath", help='Directory to keep WEAVE_RVS outputs', type=none_or_str, default=None, required=True)
    parser.add_argument("--headname", help='Output headname. The output filenames whill be generated based on this', 
        type=none_or_str, default=None, required=True)
    parser.add_argument("--hr_lines", type=none_or_str, default=None,
                        required=True, help="Line list file for the HR spectra")
    parser.add_argument("--lr_lines", type=none_or_str, default=None,
                        required=True, help="Line list file for the LR spectra")
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

    rrlew_args = [args.apsfile, args.infiles, targclass, args.hr_lines, args.lr_lines, aps_ids, outpath, args.headname]



    try:
        main_RRLEW(rrlew_args)
    except:
        print('ERROR : %s' %(sys.exc_info()[1]))
        return

########################################################################


if __name__ == '__main__':

    debug_demo = [
        '--apsfile', '<PYAPS_DIR>/PyAPS_results/20160908/3434/stack_1002250__stack_1002249_APS.fits',
        '--infiles', '<PYAPS_DATA>/opr4_jan2022/20160908/stack_1002250.fit', '<PYAPS_DATA>/opr4_jan2022/20160908/stack_1002249.fit',
        '--targclass', 'STAR_RRL',
        '--outpath', '<PYAPS_DIR>/PyAPS_results/20160908/3434/',
        '--headname', 'stack_1002250__stack_1002249',
        '--aps_ids', '418,293',#768,605,641,83,461,1005,942,457,390,260,881,33,559,345,749,680,99,334,780,981',
        '--hr_lines', '<PYAPS_DIR>/CS/RRLEW/gala_test_HR_RRLYR_list.in',
        '--lr_lines', '<PYAPS_DIR>/CS/RRLEW/gala_test_LR_RRLYR_list.in',

    ]

    rrlew_weave(options=debug_demo)

