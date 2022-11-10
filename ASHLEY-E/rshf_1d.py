#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:46:12 2019

@author: qyzh
"""
import numpy as np
from scipy.special import lpmv
from numpy.linalg import lstsq

def rshf_1d(valin,phi_in,theta_in,LMAX):
    
    l=len(valin)
    
    if (l<(LMAX+1)**2):
        
        print ("Cannot apply the least square fitting since number of rows is not sufficient !!!")
        fitting_status=-1
        fitting_params=np.nan
        fitted_val=np.nan
        
        return fitting_status, fitting_params, fitted_val
    
    B=np.zeros(l)
    A=np.zeros([l,(LMAX+1)**2])
    
    B[:]=valin[:]
    
    for ii in range(l):
        
        t=theta_in[ii];x=np.cos(t)
        p=phi_in[ii]
        
        ### First LMAX+1 Rows 
        for ll in range(LMAX+1):
                
            L=ll
            M=0
                
            # Provide the Associated Legendre polynomial
            alp=lpmv(M,L,x)
            #print (alp)
            #alp=alp[0][0][ll]
                
            A[ii,ll]=alp
                
        icol=LMAX+1
        
        ### L>=1; M>=1
        for ll in range(1,LMAX+1):
                
            for mm in range(1,ll+1):
                    
                #print (ll,mm)
                L=ll
                M=mm
                    
                alp=lpmv(M,L,x)
                    
                mp=M*p
                    
                cosmp=np.cos(mp)
                sinmp=np.sin(mp)
                    
                # COEFF for A_{LM}
                coeff1=cosmp*alp
                A[ii,icol]=coeff1                
                icol+=1
                    
                # COEFF for B_{LM}
                coeff2=sinmp*alp
                A[ii,icol]=coeff2
                icol+=1
                
    ### Remove rows corresponds to NAN data
    fp=B==B
    B1=B[fp]
    A1=A[fp,:]
    
    """ Least square fitting"""
    try:
        Coeff=(lstsq(A1,B1,rcond=-1)[0])
        
        fitted_val=np.dot(A,Coeff)       
        
        fitting_status = 1
        fitting_params = Coeff
        
        return fitting_status, fitting_params, fitted_val
        
    except (IndexError, ValueError):
        
        fitting_status=-1
        fitting_params=np.nan
        fitted_val=np.nan
        
        return fitting_status, fitting_params, fitted_val
#%%
def remove_fitting_coeffcient(fitting_parms,LMAX,MMAX):
    
    l=len(fitting_parms)
    
    if (l<(LMAX+1)**2):
        
        print ("Cannot apply the least square fitting since number of rows is not sufficient !!!")
        fitting_status=-1
        fitting_params=np.nan
        fitted_val=np.nan
        
        return fitting_status, fitting_params, fitted_val
    
    new_fitting_parms=np.zeros(l)
    
    
    for ll in range(LMAX+1):
                
        new_fitting_parms[ll]=fitting_parms[ll]
                
    icol=LMAX+1
        
        ### L>=1; M>=1
    for ll in range(1,LMAX+1):
                
        for mm in range(1,ll+1):
                    
            #print (ll,mm)
            L=ll
            M=mm
                
                
                
            if M>MMAX:
                icol+=2
                continue
                
            #print (ll,mm,icol)
                
            new_fitting_parms[icol]=fitting_parms[icol]
            icol+=1
                    
                # COEFF for B_{LM}
            new_fitting_parms[icol]=fitting_parms[icol]
            icol+=1
                
    return new_fitting_parms

#%% 
def rshf_reconst_coeff(phi,theta,LMAX):    
    
    
    A=np.zeros((LMAX+1)**2)
    
    ### Construct the Matrix A
    for ii in range(1):
        
        
        for jj in range(1):
            
            
            t=theta;x=np.cos(t)
            p=phi
            
            ### First LMAX+1 Rows 
            for ll in range(LMAX+1):
                
                L=ll
                M=0
                
                # Provide the Associated Legendre polynomial
                alp=lpmv(M,L,x)
                #print (alp)
                #alp=alp[0][0][ll]
                
                A[ll]=alp
                
            icol=LMAX+1
            
            ### L>=1; M>=1
            for ll in range(1,LMAX+1):
                
                for mm in range(1,ll+1):
                    
                    #print (ll,mm)
                    L=ll
                    M=mm
                    
                    if M>3:
                        continue
                    
                    alp=lpmv(M,L,x)
                    
                    mp=M*p
                    
                    cosmp=np.cos(mp)
                    sinmp=np.sin(mp)
                    
                    # COEFF for A_{LM}
                    coeff1=cosmp*alp
                    A[icol]=coeff1                
                    icol+=1
                    
                    # COEFF for B_{LM}
                    coeff2=sinmp*alp
                    A[icol]=coeff2
                    icol+=1
                    
    return A
#%%    
def rshf_reconst_2d_array(phi_grid,theta_grid,fitting_parms,LMAX):
    

        
    l1=len(phi_grid)
    l2=len(theta_grid)
    
    if len(fitting_parms)!=(LMAX+1)**2:
        
        print ("Check the fitting order !!!")

        fitted_val=np.nan
        
        return fitted_val
    
    
    A=np.zeros([l1*l2,(LMAX+1)**2])
    
    ### Construct the Matrix A
    for ii in range(l1):
        
        
        for jj in range(l2):
            
            
            irow=ii*l2+jj
            
            t=theta_grid[jj];x=np.cos(t)
            p=phi_grid[ii]
            
            ### First LMAX+1 Rows 
            for ll in range(LMAX+1):
                
                L=ll
                M=0
                
                # Provide the Associated Legendre polynomial
                alp=lpmv(M,L,x)
                #print (alp)
                #alp=alp[0][0][ll]
                
                A[irow,ll]=alp
                
            icol=LMAX+1
            
            ### L>=1; M>=1
            for ll in range(1,LMAX+1):
                
                for mm in range(1,ll+1):
                    
                    #print (ll,mm)
                    L=ll
                    M=mm
                    
                    alp=lpmv(M,L,x)
                    
                    mp=M*p
                    
                    cosmp=np.cos(mp)
                    sinmp=np.sin(mp)
                    
                    # COEFF for A_{LM}
                    coeff1=cosmp*alp
                    A[irow,icol]=coeff1                
                    icol+=1
                    
                    # COEFF for B_{LM}
                    coeff2=sinmp*alp
                    A[irow,icol]=coeff2
                    icol+=1
    
                
    fitted_val=np.dot(A,fitting_parms)       
    fitted_val=np.reshape(fitted_val,[l1,l2]) 
    
    return fitted_val
 
#%% Reconstruct the pattern on 2D grids 
def rshf_reconst_2d_array_2(phi_grid,theta_grid,fitting_parms,LMAX):

    """
    Phi_grid and theta_grid are 2D-grid
    """    
    
    l1=phi_grid.shape[0]
    l2=phi_grid.shape[1]
    
    if len(fitting_parms)!=(LMAX+1)**2:
        
        print ("Check the fitting order !!!")

        fitted_val=np.nan
        
        return fitted_val
    
    
    A=np.zeros([l1*l2,(LMAX+1)**2])
    
    ### Construct the Matrix A
    for ii in range(l1):
        
        
        for jj in range(l2):
            
            
            irow=ii*l2+jj
            
            t=theta_grid[ii,jj];x=np.cos(t)
            p=phi_grid[ii,jj]
            
            ### First LMAX+1 Rows 
            for ll in range(LMAX+1):
                
                L=ll
                M=0
                
                # Provide the Associated Legendre polynomial
                alp=lpmv(M,L,x)
                #print (alp)
                #alp=alp[0][0][ll]
                
                A[irow,ll]=alp
                
            icol=LMAX+1
            
            ### L>=1; M>=1
            for ll in range(1,LMAX+1):
                
                for mm in range(1,ll+1):
                    
                    #print (ll,mm)
                    L=ll
                    M=mm
                    
                    alp=lpmv(M,L,x)
                    
                    mp=M*p
                    
                    cosmp=np.cos(mp)
                    sinmp=np.sin(mp)
                    
                    # COEFF for A_{LM}
                    coeff1=cosmp*alp
                    A[irow,icol]=coeff1                
                    icol+=1
                    
                    # COEFF for B_{LM}
                    coeff2=sinmp*alp
                    A[irow,icol]=coeff2
                    icol+=1
                    
    fitted_val=np.dot(A,fitting_parms)       
    fitted_val=np.reshape(fitted_val,[l1,l2]) 
    
    return fitted_val    