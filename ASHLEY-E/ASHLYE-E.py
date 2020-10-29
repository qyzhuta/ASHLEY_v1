#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:12:54 2020

@author: qyzh
"""
import numpy as np
from spacepy import pycdf

from util_E import calc_cf, calc_beta
from util_E import determine_bt_models, determine_weights, determine_weights_small_bt
from util_E import rec_shf,rec_epot1, calc_cpcp, calc_slope_yint
from util_E import calc_expansion_rate, calc_extraoplated_epot1
#%% Initialize Coefficient
coeff_fn='ASHLEY_E_coeffs.cdf'

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
ref_cfs=np.array([4615.3,6539.2,8456.0,10614.1,13523.9,18357.4])

nbt=6
nmlat=60
nmlt=24

NMAX=4 # Fourier fitting order
LMAX=12 # Spherical Harmonic fitting order

cf_inf=40000
cf_inf1=22000

disp=0. # always 0
#%% Model inputs
bt_in=8
sw_in=450
n_in=5
cf_in=calc_cf(bt_in,sw_in,n_in)

ca_in=180

"""
Major part of ASHLEY-E
"""
#%% Step 1: Determine the pattern for any cf<=18357.4
imod1,imod2=determine_bt_models(bt_in=cf_in,ref_bts=ref_cfs)

coeffs1=all_coeff[imod1,:,:]
coeffs2=all_coeff[imod2,:,:]

## Determine the weights
if imod2>0: 
    weight1, weight2=determine_weights(bt_in=cf_in,bt_inf=cf_inf,ref_bts=ref_cfs,left=imod1,right=imod2)
else:
    weight1, weight2=determine_weights_small_bt(bt_in=cf_in,bt_inf=cf_inf,bt1=2583.,bt2=ref_cfs[0])
 
### Reconstruct the Cat 0
reconst_epot0=rec_epot1(rec_coeffs=cat0_coeff,LMAX=LMAX,nmlat=nmlat,nmlt=nmlt)

### Reconstruct other Bt models 
if imod1!=imod2:
        
    rec_coeffs1=rec_shf(fourier_coeffs=coeffs1,ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
    rec_coeffs2=rec_shf(fourier_coeffs=coeffs2,ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
        
    reconst_epot1=rec_epot1(rec_coeffs=rec_coeffs1,LMAX=LMAX,nmlat=nmlat,nmlt=nmlt)
    reconst_epot2=rec_epot1(rec_coeffs=rec_coeffs2,LMAX=LMAX,nmlat=nmlat,nmlt=nmlt)
        
else:
        
    rec_coeffs2=rec_shf(fourier_coeffs=coeffs2,ca_in=ca_in,LMAX=LMAX,NMAX=NMAX)
    reconst_epot2=rec_epot1(rec_coeffs=rec_coeffs2,LMAX=LMAX,nmlat=nmlat,nmlt=nmlt)
        
    if imod2==0:
        reconst_epot1=reconst_epot0
    else:
        reconst_epot1=reconst_epot2

reconst_epot3=reconst_epot1*weight1+reconst_epot2*weight2

#%% Get the expected CPCP and scale the pattern accordingly
slope, yint=calc_slope_yint(slope_coeff,yint_coeff,ca_in,NMAX=4)
CPCP1=calc_cpcp(reconst_epot3)
beta_in=calc_beta(cf_in,cf_inf)
CPCP2=slope*(beta_in)+yint
scale=CPCP2/CPCP1
reconst_epot4=reconst_epot3*scale
#%%
expansion_rate=1.
exp_slope, exp_yint=calc_slope_yint(exp_slope_coeff,exp_yint_coeff,ca_in,NMAX=4)


if imod1==nbt-1:    
    ### Determine the expansion rate            
    expansion_rate=calc_expansion_rate(cf_in,cf_inf1,ref_cfs[-1],exp_slope,exp_yint)    
    
    if (ca_in<90):
        
        exp_slope1, exp_yint1=calc_slope_yint(exp_slope_coeff,exp_yint_coeff,ca_in=90,NMAX=4)
        expansion1=calc_expansion_rate(cf_in,cf_inf1,ref_cfs[-1],exp_slope1,exp_yint1)    
                
        m=(expansion1-1)*2        
        expansion_rate=m*np.sin(ca_in/360.*np.pi)**2+1
                
    if (ca_in>270.):
        
        exp_slope1, exp_yint1=calc_slope_yint(exp_slope_coeff,exp_yint_coeff,ca_in=270,NMAX=4)
        expansion1=calc_expansion_rate(cf_in,cf_inf1,ref_cfs[-1],exp_slope1,exp_yint1)    
                
        m=(expansion1-1)*2        
        expansion_rate=m*np.sin(ca_in/360.*np.pi)**2+1
        
                            
    reconst_epot4=calc_extraoplated_epot1(rec_coeffs=rec_coeffs2,LMAX=LMAX,nmlat=nmlat,nmlt=nmlt,
                                          scale=scale,er=expansion_rate,disp=disp)
    
#%% output
mlts_out=(np.arange(nmlt+1)) 
mlats_out=np.arange(nmlat+1)
epot_out=reconst_epot4