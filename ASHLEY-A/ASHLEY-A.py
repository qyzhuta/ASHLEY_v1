#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:21:18 2020

@author: qyzh
"""

import numpy as np
from spacepy import pycdf

from util import calc_cf
from util import determine_bt_models, determine_weights, determine_weights_small_bt
from util import rec_mlt_coeffs, rec_uniform_diff_ef
from util import calc_slope_yint,calc_slope_yint1
from util import calc_int_diff_ef
from util import calc_expansion_rate
"""
Initialization
"""
#%% Initialize Coefficient
coeff_fn='ASHLEY_A_coeffs.cdf'

cdf=pycdf.CDF(coeff_fn)
cdf_dat=cdf.copy()
cdf.close()

### Cat 1-8 IMF clock angle Fourier coefficients
all_coeff=cdf_dat['all_coeff'] 

### Cat 0 MLT Fourier coefficient
cat0_coeff=cdf_dat['cat0_coeff']

### >500 channel hemispheric integrated differential energy flux
# slope Fourier coefficients
slope_coeff=cdf_dat['slope_coeff']
# yint Fourier coefficients
yint_coeff=cdf_dat['yint_coeff']

### Expansion
# slope Fourier coefficients
exp_slope_coeff=cdf_dat['exp_slope_coeff']
# yint Fourier coefficients
exp_yint_coeff=cdf_dat['exp_yint_coeff']

#%% Model parameters 
nbt=8
nmlat=40
nmlt=24
nchannel=19
LMAX=4
NMAX=4

ref_cfs=np.array([4283.2,6073.3,7957.8,9929.7,11941.6,14254.0,17590.4,22770.4])
cf_inf=17000

mlts_in=np.arange(0,24,(24./nmlt))+12./nmlt
mlats_in=np.arange(nmlat)+50.5
#%% Model inputs
bt_in=8
sw_in=450
n_in=5
cf_in=calc_cf(bt_in,sw_in,n_in)

ca_in=180.

"""
Major part of ASHLEY-A
"""
#%% Step 1: Determine the pattern for any cf<=22770.4

### Get 2 models having cloest median cfs and their weights
imod1,imod2=determine_bt_models(bt_in=cf_in,ref_bts=ref_cfs)
coeffs1=all_coeff[imod1,:,:,:,:]
coeffs2=all_coeff[imod2,:,:,:,:]

if imod2>0: 
    weight1, weight2=determine_weights(bt_in=cf_in,ref_bts=ref_cfs,left=imod1,right=imod2)
if imod2==0:
    weight1,weight2=determine_weights_small_bt(bt_in=cf_in,bt1=2579.,bt2=ref_cfs[0])    

### If need cat0 model
all_diff_ef0=rec_uniform_diff_ef(mlt_coeffs=cat0_coeff,
                                 nchannel=nchannel,nmlat=nmlat,
                                 nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)

all_diff_ef0[all_diff_ef0<0.]=0.

### Combine two models to get final patterns 
if imod1!=imod2:
    
    
    
    # Differential energy flux for Model #1
    all_mlt_coeffs1=rec_mlt_coeffs(coeffs=coeffs1,nchannel=nchannel,nmlat=nmlat,
                                  ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
    
    all_diff_ef1=rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs1,
                                     nchannel=nchannel,nmlat=nmlat,
                                     nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
    
    # Differential energy flux for Model #2
    all_mlt_coeffs2=rec_mlt_coeffs(coeffs=coeffs2,nchannel=nchannel,nmlat=nmlat,
                                  ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
    
    all_diff_ef2=rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs2,
                                     nchannel=nchannel,nmlat=nmlat,
                                     nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
    
else:
    
    # Differential energy flux for Model #1
    all_mlt_coeffs1=rec_mlt_coeffs(coeffs=coeffs1,nchannel=nchannel,nmlat=nmlat,
                                  ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
    
    all_diff_ef2=rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs1,
                                     nchannel=nchannel,nmlat=nmlat,
                                     nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
    
    if imod2==0:        
        all_diff_ef1=all_diff_ef0
    else:
        all_diff_ef1=all_diff_ef2
    
all_diff_ef1[all_diff_ef1<0]=0.
all_diff_ef2[all_diff_ef2<0]=0.
    
rec_diff_ef=all_diff_ef1*weight1+all_diff_ef2*weight2

#%% Step 2: Extrapolation and expansion
expansion1=1.

### >500 eV (Keep it linear)
slope,yint=calc_slope_yint(slope_coeff,yint_coeff,ca_in,NMAX,nchannel)
int_diff_ef1=slope*cf_in+yint
int_diff_ef1[int_diff_ef1<=0]=1e-12

exp_slope,exp_yint=calc_slope_yint1(exp_slope_coeff,exp_yint_coeff,ca_in,NMAX)

rs=90-mlats_in
ts=(mlts_in-6)/12*np.pi

int_diff_ef2=calc_int_diff_ef(ts,rs,rec_diff_ef,nchannel)
int_diff_ef2[int_diff_ef2<=0]=1e-12    
scaling_fac=int_diff_ef1/int_diff_ef2
rec_diff_ef[:,:,:11]*=scaling_fac[:11]
    
### <500 eV
if imod1==nbt-1:
    
    coeffs1=all_coeff[imod1-1,:,:,:,:]
    
    # Differential energy flux for Model #1
    all_mlt_coeffs1=rec_mlt_coeffs(coeffs=coeffs1,nchannel=nchannel,nmlat=nmlat,
                                  ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
    
    all_diff_ef1=rec_uniform_diff_ef(mlt_coeffs=all_mlt_coeffs1,
                                     nchannel=nchannel,nmlat=nmlat,
                                     nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)

    all_diff_ef1[all_diff_ef1<0]=0.
    all_diff_ef2[all_diff_ef2<0]=0.    

    # Calculate the integrated differential energy flux for each channel    
    int_diff_ef1=calc_int_diff_ef(ts,rs,all_diff_ef1,nchannel)
    int_diff_ef2=calc_int_diff_ef(ts,rs,all_diff_ef2,nchannel)
    
    rate=(int_diff_ef2-int_diff_ef1)/(ref_cfs[-1]-ref_cfs[-2])
    
    diff_bt=cf_in-ref_cfs[-1]
    
    if (diff_bt<0):
        diff_bt=0.
    
    int_diff_ef3=int_diff_ef2+rate*diff_bt
    
    increment=(int_diff_ef3/int_diff_ef2)
    
    all_diff_ef3=all_diff_ef2*increment
    all_diff_ef3[:,:,:11]=rec_diff_ef[:,:,:11]
    
    ### Expansion
    expansion1=calc_expansion_rate(cf_in,cf_inf,ref_cfs[-1],exp_slope,exp_yint)
    
    ### Ensure hemispheric integrated differential energy flux is same before/after expansion
    int_diff_ef3=calc_int_diff_ef(ts,rs,all_diff_ef3,nchannel)
    int_diff_ef4=calc_int_diff_ef(ts,rs*expansion1,all_diff_ef3,nchannel)
    scale_fac=int_diff_ef3/int_diff_ef4
    
    rec_diff_ef=all_diff_ef3*scale_fac    

"""
ASHLEY outputs
"""
mlts_out=mlts_in # Central MLT of each MLT bin
mlats_out=90.-(90-mlats_in)*expansion1 # Central MLAT of each MLAT bin
diff_ef_out=rec_diff_ef*1e8 # Differential energy flux in each channel and each bin of MLT and MLAT (eV/(cm-2 s-1 sr-1 eV-1))

