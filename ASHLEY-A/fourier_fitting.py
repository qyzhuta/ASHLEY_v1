#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:47:45 2019

@author: qyzh
"""

import numpy as np
from numpy.linalg import lstsq

def fourier_fitting(valin,x_grid,LMAX):
    
    """
    Input valin array must be a 1D array    
    
    Input valin size should be len(x_grid)
    """
    
    """Initial setups and checks """
    
    l1=len(x_grid)
    
    if l1<=(LMAX+1)*2:
        
        print ("Cannot apply the least square fitting since number of rows is not sufficient !!!")
        fitting_status=-1
        fitting_params=np.nan
        fitted_val=np.nan
        
        return fitting_status, fitting_params, fitted_val
    
    """ Ax = B, Construct Matrices A and B"""
    # Note G0=0, so ncol=(LMAX+1)*2-1
    
    B=np.zeros(l1)
    A=np.zeros([l1,(LMAX+1)*2-1])
    
    ### Construct the Matrix B
    for ii in range(l1):
        
        B[ii]=valin[ii]
        
    ### Construct the Matrix A
    for ii in range(l1):
        
        x=x_grid[ii]
        
        A[ii,0]=1 #F0
        
        for jj in range(1,LMAX+1):
            
            nx = jj * x 
            
            cosnx=np.cos(nx)
            sinnx=np.sin(nx)
            
            icol=2*jj-1
            A[ii,icol]=cosnx
            
            icol=2*jj
            A[ii,icol]=sinnx
            
    
    ### Remove rows corresponds to NAN data
    fp=B==B
    B1=B[fp]
    A1=A[fp,:]
    
    """ Least square fitting"""
    try:
        Coeff=(lstsq(A1,B1)[0])
        
        fitted_val=np.dot(A,Coeff)       
        
        fitting_status = 1
        fitting_params = Coeff
        
        return fitting_status, fitting_params, fitted_val
        
    except (IndexError, ValueError):
        
        fitting_status=-1
        fitting_params=np.nan
        fitted_val=np.nan
        
        return fitting_status, fitting_params, fitted_val
        
        
#%% Reconstruct the Fourier Function 
def reconstruct_fourier_series(x,LMAX):
    
    len_series=2 * LMAX + 1
    
    fourier_series=np.zeros(len_series)
    
    fourier_series[0]=1

    for ii in range(1,LMAX+1):
        
        nx= ii * x
        
        cosnx=np.cos(nx)
        pos=2*ii-1
        fourier_series[pos]=cosnx
        
        sinnx=np.sin(nx)
        pos=2*ii
        fourier_series[pos]=sinnx
        
    return fourier_series
        
    
    