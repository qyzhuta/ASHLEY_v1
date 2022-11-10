#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:31:15 2020

@author: qyzh
"""
import numpy as np

#%% Calculate the coupling function 
def calc_cf(bt,sw,n):
    
    return (n**(1./6.))*(sw**(4./3.))*(bt**(2./3.))
#%% Determine which two models should be used
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

#%% Calculate the Beta value according to the formula used for the DE-2 model
def calc_beta(Bt_in,Bt_inf):
    
    return Bt_in/np.sqrt(1+(Bt_in/Bt_inf)**2)

#%% Determine the weights of two models (Assume CPCP is linear with the beta)
def determine_weights(bt_in,bt_inf,ref_bts,left,right):

    if left==right:
        weight1=1.
        weight2=0.

    else:
        bt1=ref_bts[left]
        bt2=ref_bts[right]
        
        beta1=calc_beta(bt1,bt_inf)
        beta2=calc_beta(bt2,bt_inf)
        
        beta=calc_beta(bt_in,bt_inf)
        
        diff=(beta2-beta1)
        diff1=(beta2-beta)
        
        weight1=diff1/diff
        weight2=1-weight1
        
    return weight1, weight2

#%% determine weights between cat0 and Model 0
def determine_weights_small_bt(bt_in,bt_inf,bt1,bt2):

    if bt_in<bt1:
        weight1=1.
        weight2=0.
        
    elif (bt_in<bt2) & (bt_in>=bt1):
        
        beta1=calc_beta(bt1,bt_inf)
        beta2=calc_beta(bt2,bt_inf)
        
        beta=calc_beta(bt_in,bt_inf)
        
        diff=(beta2-beta1)
        diff1=(beta2-beta)
        
        weight1=diff1/diff
        weight2=1-weight1
        
    return weight1, weight2

"""
Part 3: Reconstrut the single model
"""    
#%% Reconstruct the MLT coefficients
def rec_mlt_coeffs(coeffs,nmlat,ca_in,LMAX,NMAX):

    """
    Parmeters:
        coeffs: all coeffiecients of each channel [nmlat,2*LMAX+1,2*NMAX+1]
        nmlat: Number of the MLATs
        ca_in: IMF clock angle input (degree)
        LMAX,NMAX: Fourier fitting order for MLT and IMF clock angle
    """  
    
    from fourier_fitting import reconstruct_fourier_series
    
    all_mlt_coeffs=np.zeros([nmlat,2*LMAX+1])
            
    for imlat in range(nmlat):
            
        for iparm in range(2*LMAX+1):                    
                
            PARM=coeffs[imlat,iparm,:]            
            phi=ca_in/180.*np.pi            
            fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)                
            val=np.dot(PARM,fs)
                    
            all_mlt_coeffs[imlat,iparm]=val
                    
    return all_mlt_coeffs

#%% reconstruct the differential energy flux if the grid is uniform
def rec_evar(mlt_coeffs,nmlat,nmlt,mlts_in,LMAX):
    
    """
    Parmeters:
        mlts_coeffs: all MLT coeffiecients of each channel [nmlat,2*LMAX+1]
        nmlat: Number of the MLATs
        nmlt: Number of the MLTs
        mlts_in: MLT grid point (hour) [nmlt]
        LMAX: Fourier fitting order for MLT and IMF clock angle
    """
        
    from fourier_fitting import reconstruct_fourier_series
    
    all_ed=np.zeros([nmlt,nmlat])
    
    
        
    for imlat in range(nmlat):
            
        PARM=mlt_coeffs[imlat,:]
            
        for imlt in range(nmlt):
                                
            phi=mlts_in[imlt]/12*np.pi
            fs=reconstruct_fourier_series(x=phi,LMAX=LMAX)
                
            val=np.dot(PARM,fs)
            
            if val<0:
                val=0.
                
            all_ed[imlt,imlat]=val
                
    return all_ed

