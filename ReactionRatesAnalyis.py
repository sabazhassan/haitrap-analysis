# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:56:44 2021

@author: Saba
Script to determine the reaction rate coefficients
"""
import os

os.chdir("C:\\Users\\Saba\\Documents\\Python")
from AnalysisClass3 import import_files
from AnalysisClass3 import analysis_data

import numpy as np
from scipy.integrate import nquad
from scipy.integrate import quad
import scipy.integrate as integrate
import scipy.constants as con
import scipy.special as special
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy  import optimize
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from time import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
options={'limit':100}
#%% define constant values

T0=370 #intial temperature
sIz0=0.00111 #axial distribution
sIr0=0.00083 #radial distribution

k_bgr=0.0033 #background losses

muOH=17*85/(17+85)  #redcued mass of the OH-Rb complex
muOHcluster=35*85/(85+35) #redcued mass of the OH(H2O)-Rb complex

LangevinS_OHcluster=4.3*1e-9/100**3*np.sqrt(muOH/muOHcluster)  #Langevin Ground
LangevinP_OHcluster=7.2*1e-9/100**3*np.sqrt(muOH/muOHcluster)  #Langevin Excited

DetEff=0.0079  #detection efficiency of the MCP
#%% functions

def AtomDensity(t,x,y,z,zA,nA_peak,kA_load,sAx,sAy,sAz):
    return(nA_peak*(1-np.exp(-kA_load*t))*np.exp(-x**2/2/sAx**2)*np.exp(-y**2/2/sAy**2)*np.exp(-(z-zA)**2/2/sAz**2))
    
def Potential(x,y,z,sz0,sr0):
    r=np.sqrt(x**2+y**2)
    A=con.k*T0/(2*sz0**2)
    #B=con.k*T0/(8*sr0**6)
    #,fill_value = 'extrapolate')
    return((A*z**2+radialpot(r)))

def NormIonDenT(x,y,z,T,sz0,sr0):
    r=np.sqrt(x**2+y**2)
    A=con.k*T0/(2*sz0**2)
    B=(con.k*T0/(8*sr0**6))
    IntA=np.sqrt(con.k*T*np.pi/A)
    radialdensity = np.exp(-y1/(con.k*T))
    IntB1=2*np.pi*special.gamma(1/3)/(6*(B/con.k/T)**(1/3))   
    IntB=np.pi*np.trapz(y=np.abs(x1)*radialdensity,x=x1, axis=0)
    return(np.exp(-Potential(x,y,z,sz0,sr0)/con.k/T)/ (IntA*IntB))
def CalcTemp(sigma):
    out_a=np.array([])
    for s in sigma:
        if (s**2-0.02404)/0.00231>60:
            out=(s**2-0.02404)/0.00231
        else:
            out=(s**2-0.0029)/0.0028
        out_a=np.append(out_a,out)
    return(out_a)    
   
#%% overlap

def overlap(t,zA,sAy0,k_sAy,sAz0,k_sAz,nA,kA,ion_temp):
    sAy=sAy0*(1-np.exp(-k_sAy*t))
    sAz=sAz0*(1-np.exp(-k_sAz*t))
    sAx=(sAy+sAz)/2
    Lim=5
    T=ion_temp(t);
    if t==0:
        out = 0
    else:
        out=nquad(lambda x,y,z: NormIonDenT(x,y,z,T,sIz0,sIr0)*AtomDensity(t,x,y,z,zA,nA,kA,sAx,sAy,sAz),[[-Lim*sAx,Lim*sAx],[-Lim*sAy,Lim*sAy],[-Lim*sAz+zA,Lim*sAz+zA]],opts={"epsabs":1e21})[0]
    return(out)

#%% ion loss

def f_IonLoss(t,k,f_overlap,k_bgr,ion_temp):
    if isinstance(t,(list,np.ndarray)):
        out=np.array([np.exp(-(k*quad(f_overlap,0,t_it)[0]+(k_bgr)*t_it)) for t_it in t])
    else:
        out=np.exp(-(k*quad(f_overlap,0,t)[0]+(k_bgr)*t))
    return(out)
    
#%% all measurements

position="0_0"
datapath="D:/Analysis/2021/09_September/03_ClusterwithMOT_FirstDataSet"
folderlist=['0']
#%%RadialPotential
x1,y1 = np.loadtxt(datapath+"/radial_potential_y.txt",skiprows=8,unpack=True)
#x1=x1[0::2]
#y1=y1[0::2]

y1 -= y1.min()
y1 *= con.e
x1 *= 1e3 # m -> mm 
x1 -= 3.
x1  = x1/1000.

cond = x1[:]<0
x1,y1 = x1[cond],y1[cond]

radialpot=interp1d(x1,y1)

plt.figure()
T=50
sr0=0.00083
T0=370
#cond = x1[:]>0
#x1,y1 = x1[cond], y1[cond]
B=con.k*T0/(8*sr0**6)
density = np.exp((-B*np.abs(x1)**6)/(con.k*T))
IntB1=2*np.pi*special.gamma(1/3)/(6*(B/(con.k*T))**(1/3))
print(IntB1)
radialdensity = np.exp((-B*np.abs(x1)**6)/(con.k*T))
norm = np.pi*2*np.trapz(y=np.abs(x1)*radialdensity,x=x1)/IntB1
print("norm check")
print(norm)

y_plot = density/IntB1
radialdensity = np.exp(-radialpot(x1)/(con.k*T))
norm = np.pi*2*np.trapz(y=np.abs(x1)*radialdensity,x=x1)
print(norm)
y_plot2 = radialdensity/norm
plt.plot(x1,y_plot2)
plt.show()
normi = np.pi*2*np.trapz(y=np.abs(x1)*radialdensity,x=x1)/norm
print("norm check 2")
print(normi)
plt.figure()
radialdensity2= np.exp(-y1/(con.k*T))
plt.plot(x1,radialdensity)
plt.plot(x1,radialdensity2)
plt.show()

#%%
for i,position in enumerate(folderlist):

    zA=0.00
    #ions
    data_E=np.loadtxt(datapath+"/Results/Fitparameters.dat")
    #data_E=data_E[1::]
    data_T=CalcTemp(data_E[:,1])
    data_N=np.loadtxt(datapath+"/Results/MOT duration_ionlossMOT.dat")
    #data_N=data_N[1::]
    IonInit=data_N[0,3]/DetEff
    ion_temp=interp1d(data_E[:,0]/1000,data_T,kind='cubic',fill_value = 'extrapolate')
    
    #atoms
    data_MOT=np.loadtxt(datapath+"/Results/data_mot_mean_final.dat",skiprows=1)
    [n_fit,n_cov]=curve_fit(lambda t,n_peak,k: n_peak*(1-np.exp(-k*t)),data_MOT[:,0],data_MOT[:,4]*100**3,p0=[data_MOT[-1,4]*100**3,0.5])
    [sy_fit,sy_cov]=curve_fit(lambda t,sy_peak,k: sy_peak*(1-np.exp(-k*t)),data_MOT[:,0],data_MOT[:,7]*1e-3,p0=[data_MOT[-1,7]*1e-3,0.5])
    [sz_fit,sz_cov]=curve_fit(lambda t,sz_peak,k: sz_peak*(1-np.exp(-k*t)),data_MOT[:,0],data_MOT[:,10]*1e-3,p0=[data_MOT[-1,10]*1e-3,0.5])

    #interpolate overlap
    tmax=np.max([data_MOT[-1,0],data_E[-1,0]/1000])
    test=np.array([[t,overlap(t,zA,*sy_fit,*sz_fit,n_fit[0],n_fit[1],ion_temp)] for t in np.arange(0,tmax+tmax/100,tmax/100)])
    f_overlap=interp1d(test[:,0],test[:,1],kind='cubic',fill_value = 'extrapolate')
    #f_temp=interp1d(test[:,0],data_T)
    R_constant=8.31
    k_r=2*1e-16
    
    [k_fit,k_cov]=curve_fit(lambda t,k: f_IonLoss(t,k,f_overlap,k_bgr,ion_temp),data_N[:,0]/1000,data_N[:,1],p0=[k_r])#,sigma=data_N[:,2])
    plt.figure()
    plott=np.arange(0/999,data_N[-1,0]/999,0.1)
    plt.errorbar(data_N[:,0]/1000,data_N[:,1],yerr=data_N[:,2],Marker="o",linestyle="")
    plt.plot(plott,f_IonLoss(plott,k_fit,f_overlap,k_bgr,ion_temp),color='r')
    print(k_fit[0])
    
