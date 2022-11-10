#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:20:49 2020

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

#%% Reconstruct the coefficients of the SHF
def rec_shf(fourier_coeffs,ca_in,LMAX,NMAX):
    
    from fourier_fitting import reconstruct_fourier_series
    
    rec_coeffs=np.zeros((LMAX+1)**2)
    
    for iparm in range((LMAX+1)**2):
        
        if iparm<0:
            continue
        
        PARM=fourier_coeffs[iparm,:]
        
        phi=ca_in/180.*np.pi
        fs=reconstruct_fourier_series(x=phi,LMAX=NMAX)
                
        val=np.dot(PARM,fs)
        
        rec_coeffs[iparm]=val
        
    return rec_coeffs

#%% Reconstruct the Potential pattern by using the SHF coefficients
def rec_epot1(rec_coeffs,LMAX,nmlat,nmlt):
    
    from rshf_1d import rshf_reconst_2d_array
    
    rs=np.arange(nmlat+1);rs=rs/45.*np.pi
    ts=np.arange(nmlt+1);ts=ts/12*np.pi
    
    reconst_pot=rshf_reconst_2d_array(ts,rs,rec_coeffs,LMAX)
    
    fp=rs>np.pi
    reconst_pot[:,fp]=0.
    
    return reconst_pot

#%% Calculate the CPCP 
def calc_cpcp(epot):
    
    return np.nanmax(epot)-np.nanmin(epot)

#%% Get the slope and yint for the CPCP
def calc_slope_yint(slope_coeff,yint_coeff,ca_in,NMAX):
    
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

def calc_expansion_rate(bt_in,bt_inf,bt1,slope,yint):
    
    beta=calc_beta(bt_in,bt_inf)
    beta1=calc_beta(bt1,bt_inf)
    
    r=slope*beta+yint
    r1=slope*beta1+yint
    
    rate= r/r1
    
    
    if (rate<1):
        rate=1
        
    return rate

#%% Cartetian --> polar
def cat2pol(x,y,x0,y0):
    
    r=np.sqrt((x-x0)**2+(y-y0)**2)
    theta=np.arctan2(y-y0,x-x0)
    
    return r, theta

#%% Calculate the new epot pattern after the expansion, extrapolation and displacement 
def calc_extraoplated_epot1(rec_coeffs,LMAX,nmlat,nmlt,scale,er,disp):
    
    from rshf_1d import rshf_reconst_2d_array_2
    
    ### Grid
    ts1=(np.arange(nmlt+1)-6)/12*np.pi
    rs1=np.arange(nmlat+1)*1.
    ts1,rs1=np.meshgrid(ts1,rs1)
    ts2,rs2=ts1.T,rs1.T
    
    ts3=np.zeros_like(ts2)
    rs3=np.zeros_like(rs2)
    
    ### Reference point
    ref_r0=0.
    ref_t0=np.pi
    
    ref_x0=ref_r0*np.cos(ref_t0)
    ref_y0=ref_r0*np.sin(ref_t0)
    
    for ii in range(nmlt+1):
    
        for jj in range(nmlat+1):
            
            ts_1=ts2[ii,jj]
            rs_1=rs2[ii,jj]
            
            # Find the mapped ts, rs
            x1=rs_1*np.cos(ts_1)
            y1=rs_1*np.sin(ts_1)
            
            dx=x1-ref_x0
            dy=y1-ref_y0
            
            dx1=dx/er
            dy1=dy/er
            
            x2=ref_x0+dx1-disp
            y2=ref_y0+dy1
            
            rs_2,ts_2=cat2pol(x2,y2,0,0)
            
            ts_2+=np.pi/2
            
            if ts_2<0:
                ts_2+=2*np.pi
                
            rs_2=rs_2/45.*np.pi
            
            ts3[ii,jj]=ts_2
            rs3[ii,jj]=rs_2
            
    reconst_pot=rshf_reconst_2d_array_2(ts3,rs3,rec_coeffs,LMAX)
    reconst_pot*=scale
    
    fp=rs3>np.pi
    reconst_pot[fp]=0.
    
    return reconst_pot