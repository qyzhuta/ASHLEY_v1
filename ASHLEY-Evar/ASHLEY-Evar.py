#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:23:31 2020

@author: qyzh
"""

import numpy as np
from spacepy import pycdf

from util_Evar import calc_cf, determine_bt_models, determine_weights, determine_weights_small_bt
from util_Evar import rec_mlt_coeffs, rec_evar
#%% Initialize Coefficient
coeff_fn='ASHLEY_Evar_coeffs.cdf'

cdf=pycdf.CDF(coeff_fn)
cdf_dat=cdf.copy()
cdf.close()

all_coeff1=cdf_dat['all_coeff1'] 
all_coeff2=cdf_dat['all_coeff2'] 
cat0_coeff1=cdf_dat['cat0_coeff1']
cat0_coeff2=cdf_dat['cat0_coeff2']

#%%

ref_cfs=np.array([4615.3,6539.2,8456.0,10614.1,13523.9,18357.4])

nbt=6
nmlat=20
nmlt=24

LMAX=6
NMAX=4

cf_inf=40000
cf_inf1=22000

mlts_in=np.arange(0,24,(24./nmlt))+12./nmlt
mlats_in=np.arange(nmlat)+50.5
#%% Model inputs
bt_in=8
sw_in=450
n_in=5
cf_in=calc_cf(bt_in,sw_in,n_in)

ca_in=180.

### PARMS calculated from the ASHLEY-E under same cf_in and ca_in
scaling_fac=1. #(if not used, set as 1)
expansion_rate=1. # (if not used, set as 1)


#%% Step 1: Determine the pattern for any cf<=18357.4
imod1,imod2=determine_bt_models(bt_in=cf_in,ref_bts=ref_cfs)

### Coeffs for Ed1       
coeffs11=all_coeff1[imod1,:,:]
coeffs12=all_coeff1[imod2,:,:]
    
### Coeffs for Ed2
coeffs21=all_coeff2[imod1,:,:]
coeffs22=all_coeff2[imod2,:,:]
    
## Determine the weights
if imod2>0: 
    weight1, weight2=determine_weights(bt_in=cf_in,bt_inf=cf_inf,ref_bts=ref_cfs,left=imod1,right=imod2)
else:
    weight1, weight2=determine_weights_small_bt(bt_in=cf_in,bt_inf=cf_inf,bt1=2583.,bt2=ref_cfs[0])
    
all_dEd1_0=rec_evar(mlt_coeffs=cat0_coeff1,nmlat=nmlat,
                   nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
all_dEd2_0=rec_evar(mlt_coeffs=cat0_coeff2,nmlat=nmlat,
                   nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
    
### Reconstruct other Bt models 
if imod1!=imod2:
        
   ### Reconstruct Ed1
   all_mlt_coeffs11=rec_mlt_coeffs(coeffs=coeffs11,nmlat=nmlat,
                                   ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
   all_mlt_coeffs12=rec_mlt_coeffs(coeffs=coeffs12,nmlat=nmlat,
                                   ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
        
   all_dEd1_1=rec_evar(mlt_coeffs=all_mlt_coeffs11,nmlat=nmlat,
                      nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
        
   all_dEd1_2=rec_evar(mlt_coeffs=all_mlt_coeffs12,nmlat=nmlat,
                      nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
        
   ### Reconstruct Ed2
   all_mlt_coeffs21=rec_mlt_coeffs(coeffs=coeffs21,nmlat=nmlat,
                                   ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
   all_mlt_coeffs22=rec_mlt_coeffs(coeffs=coeffs22,nmlat=nmlat,
                                   ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
        
   all_dEd2_1=rec_evar(mlt_coeffs=all_mlt_coeffs21,nmlat=nmlat,
                      nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
        
   all_dEd2_2=rec_evar(mlt_coeffs=all_mlt_coeffs22,nmlat=nmlat,
                      nmlt=nmlt,mlts_in=mlts_in,LMAX=LMAX)
        
else:
        
    ### Differential energy flux for Model #1
    all_mlt_coeffs1=rec_mlt_coeffs(coeffs=coeffs11,nmlat=nmlat,
                                   ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
        
    all_dEd1_2=rec_evar(mlt_coeffs=all_mlt_coeffs1,
                       nmlat=nmlat,nmlt=nmlt,
                       mlts_in=mlts_in,LMAX=LMAX)
        
    all_mlt_coeffs2=rec_mlt_coeffs(coeffs=coeffs21,nmlat=nmlat,
                                   ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
        
    all_dEd2_2=rec_evar(mlt_coeffs=all_mlt_coeffs2,
                       nmlat=nmlat,nmlt=nmlt,
                       mlts_in=mlts_in,LMAX=LMAX)
        
    if imod2==0:
        all_dEd1_1=all_dEd1_0
        all_dEd2_1=all_dEd2_0
    else:
        all_dEd1_1=all_dEd1_2
        all_dEd2_1=all_dEd2_2
            
rec_dEd1=all_dEd1_1*weight1+all_dEd1_2*weight2
rec_dEd2=all_dEd2_1*weight1+all_dEd2_2*weight2    

"""
ASHLEY-Evar outputs
"""
mlts_out=mlts_in # Central MLT of each MLT bin
mlats_out=90.-(90-mlats_in)*expansion_rate # Central MLAT of each MLAT bin
dEd1_out=rec_dEd1*scaling_fac/expansion_rate
dEd2_out=rec_dEd2*scaling_fac/expansion_rate
