#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:32:23 2020

@author: qyzh
"""
import numpy as np

#%% Calculate the coupling function 
def calc_cf(bt,sw,n):
    
    return (n**(1./6.))*(sw**(4./3.))*(bt**(2./3.))

#%% Determine the models need to generate the final auroral model 
def determine_bt_models(bt_in,ref_bts):
        
    if bt_in<ref_bts[0]:

        left=0
        right=0
        
    elif bt_in>=ref_bts[-1]:
        left=len(ref_bts)-1
        right=left
        
    else:
        
        for ii in range(len(ref_bts)):
                    
            if (bt_in>=ref_bts[ii]) & (bt_in<ref_bts[ii+1]):
    
                left=ii
                right=ii+1
                
                break       

    return left, right 

#%% determine the weights
def determine_weights(bt_in,ref_bts,left,right):

    if left==right:
        weight1=1.
        weight2=0.

    else:
        bt1=ref_bts[left]
        bt2=ref_bts[right]

        diff=(bt2-bt1)
        diff1=(bt2-bt_in)
        weight1=diff1/diff
        weight2=1-weight1
        
    return weight1, weight2

#%% determine weights between cat0 and Model 0
def determine_weights_small_bt(bt_in,bt1,bt2):

    if bt_in<bt1:
        weight1=1.
        weight2=0.
        
    elif (bt_in<bt2) & (bt_in>=bt1):
        
        diff=(bt2-bt1)
        diff1=(bt2-bt_in)
        weight1=diff1/diff
        weight2=1-weight1
        
    return weight1, weight2

#%% Reconstruct the MLT coefficients
def rec_mlt_coeffs(coeffs,nchannel,nmlat,ca_in,LMAX,NMAX):

    """
    Parmeters:
        coeffs: all coeffiecients of each channel [nchannel,nmlat,2*LMAX+1,2*NMAX+1]
        nchannel: Number of the channels 
        nmlat: Number of the MLATs
        ca_in: IMF clock angle input (degree)
        LMAX,NMAX: Fourier fitting order for MLT and IMF clock angle
    """  
    
    from fourier_fitting import reconstruct_fourier_series
    
    all_mlt_coeffs=np.zeros([nchannel,nmlat,2*LMAX+1])
    
    for ichannel in range(nchannel):
        
        for imlat in range(nmlat):
            
            for iparm in range(2*LMAX+1):                    
                
                    PARM=coeffs[ichannel,imlat,iparm,:]            
                    phi=ca_in/180.*np.pi            
                    fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)                
                    val=np.dot(PARM,fs)
                    
                    all_mlt_coeffs[ichannel,imlat,iparm]=val
                    
    return all_mlt_coeffs

#%% reconstruct the differential energy flux if the grid is uniform
def rec_uniform_diff_ef(mlt_coeffs,nchannel,nmlat,nmlt,mlts_in,LMAX):
    
    """
    Parmeters:
        mlts_coeffs: all MLT coeffiecients of each channel [nchannel,nmlat,2*LMAX+1]
        nchannel: Number of the channels 
        nmlat: Number of the MLATs
        nmlt: Number of the MLTs
        mlts_in: MLT grid point (hour) [nmlt]
        LMAX,NMAX: Fourier fitting order for MLT and IMF clock angle
    """
        
    from fourier_fitting import reconstruct_fourier_series
    
    all_diff_ef=np.zeros([nmlt,nmlat,nchannel])
    
    for ichannel in range(nchannel):
        
        for imlat in range(nmlat):
            
            PARM=mlt_coeffs[ichannel,imlat,:]
            
            for imlt in range(nmlt):
                                
                phi=mlts_in[imlt]/12*np.pi
                fs=reconstruct_fourier_series(x=phi,LMAX=LMAX)
                
                val=np.dot(PARM,fs)
                
                all_diff_ef[imlt,imlat,ichannel]=val
                
    return all_diff_ef

#%% Get the slope and yint for the hemispheric integrated energy flux at the corresponding IMF clock angle for the first 11 channels 
def calc_slope_yint(slope_coeff,yint_coeff,ca_in,NMAX,nchannel):
    
    from fourier_fitting import reconstruct_fourier_series
    
    slope=np.zeros(nchannel)
    yint=np.zeros(nchannel)
    
    s=slope_coeff.shape;
    
    for ichannel in range(nchannel):
        
        if ichannel>=s[0]:
            continue
    
        PARM=slope_coeff[ichannel,:]
        phi=ca_in/180*np.pi
        fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)                
        val=np.dot(PARM,fs)
        
        slope[ichannel]=val*1e-4
        
        PARM=yint_coeff[ichannel,:]
        fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)                
        val=np.dot(PARM,fs)
        
        yint[ichannel]=val
        
    return slope, yint

#%% Get the slope and yint for expansion 
def calc_slope_yint1(slope_coeff,yint_coeff,ca_in,NMAX):
    
    from fourier_fitting import reconstruct_fourier_series
    
   
    
    PARM=slope_coeff[:]
    phi=ca_in/180*np.pi
    fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)                
    val=np.dot(PARM,fs)
        
    slope=val*1e-4
        
    PARM=yint_coeff[:]
    fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)                
    val=np.dot(PARM,fs)
        
    yint=val
        
    return slope, yint

#%% Calculate the integrated differential energy flux for each channel
def calc_int_diff_ef(tgrids,rgrids,diff_ef,nchannel):
    
    int_diff_ef=np.zeros(nchannel)
    
    for ichannel in range(nchannel):
        
        int_diff_ef[ichannel]=calc_hp(tgrids,rgrids[::-1],diff_ef[:,::-1,ichannel])
        
    return int_diff_ef
#%% Calculate the hemispheric power
def calc_hp(tgrids,rgrids,valin):
    
    
    n_r_bins=len(rgrids)-1
    
    rgrids1=rgrids/180*np.pi
    
    cos0=np.cos(rgrids1[:-1])
    cos1=np.cos(rgrids1[1:])
    
    sum_flux=0
    
    flux=np.zeros(n_r_bins)
    
    for ii in range(n_r_bins):
        
        dcos=cos0[ii]-cos1[ii]
        
        val=np.nanmean(valin[:,ii])
        
        val=val*2*np.pi*dcos
        
        flux[ii]=val
        
        #sum_flux+=val
        
    sum_flux=np.nansum(flux)
        
    factor=(6500**2)/1e6
    
    return sum_flux*factor

#%% Calculate the Beta value according to the formula used for the DE-2 model
def calc_beta(Bt_in,Bt_inf):
    
    return Bt_in/np.sqrt(1+(Bt_in/Bt_inf)**2)

#%% Calcuate the expansion rate
### Saturation of the PB is considered 
def calc_expansion_rate(bt_in,bt_inf,bt1,slope,yint):
    
    beta=calc_beta(bt_in,bt_inf)
    beta1=calc_beta(bt1,bt_inf)
    
    r=slope*beta+yint
    r1=slope*beta1+yint
    
    rate= r/r1
    
    if (rate<1):
        rate=1
                
    return rate