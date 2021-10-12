# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:10:45 2021

@author: Saba
This script bins and plots the data obtained from the COMSOL simulations.

To check convergence, there were 3 sweeps performed:
    buffer gas density
    electric field grid resolution
    time steps size
    
Use the selector variable to define which of the 3 you wanna plot
"""
import numpy as np
import scipy as sp
from scipy.interpolate import interpn
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import norm
import pandas as pd
import h5py
import matplotlib.ticker as mticker

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sb



"constants"
#q = 1.602176565e-19;"C"
k = 1.3806488e-23;
m = 17*1.6605e-27;"kg"
kb = 8.6173303 * (10.0**-5); "in eV"
kb_t = 1.380649 * (10.**-23); "in SI"

def normalize(x,y):
    y_norm = y/np.trapz(y=y ,x=x, axis=0)
    
    return y_norm

def maxwell_norm(V_eff,temp):
    V_eff -= V_eff.min()
    density = np.exp(-V_eff/(kb*temp))
    density /= np.trapz(y=density ,x=x, axis=0)
    
    return density

def tsallis_norm(V_eff,temp,q):
    V_eff -= V_eff.min()
    density = (1. - (1.-q)*(V_eff/(1.*kb*temp)))**(1./(1.-q))
    density /= np.trapz(y=density, x=x ,axis=0)

    return density

# better than numpy load for loading big files
def load_big_file(fname,skiprows=0,length=2):

    rows = []  # unknown number of lines, so use list
    with open(fname) as f:
        for _ in range(skiprows):
            next(f)
        
        for line in f:
            line = [float(s) for s in line.split()]
                        
            if len(line) == length:
                rows.append(np.array(line, dtype = np.double))

            
    return np.vstack(rows)  # convert list of vectors to array


temperature_average = lambda v: (np.pi * m * v**2) / (8*kb)

gauss = lambda x, x0,s,A: A*(1/(s*np.sqrt(2*np.pi)))*np.exp(-(x-x0)**2/(2*(s**2)))
q_gauss = lambda x, x0,s,q,A: A*(1./s)*(np.sqrt((q-1.)/np.pi)*sp.special.gamma(1./(q-1.))/sp.special.gamma((1./(q-1.))-.5)) *(1. - (1.-q)*((x-x0)/s)**2)**(1./(1.-q))

maxwell_func = lambda x,T,A: A*4*np.pi*((m/(2*np.pi*kb*T))**(1.5)) * (x**2) *np.exp(-(m*x**2)/(2*kb*T))
maxwell_func2 = lambda x,T,A: A*4*np.pi*((m/(2*np.pi*kb_t*T))**(1.5)) * (x**2) *np.exp(-(m*x**2)/(2*kb_t*T))
maxwell_func2_no_A = lambda x,T: 4*np.pi*((m/(2*np.pi*kb_t*T))**(1.5)) * (x**2) *np.exp(-(m*x**2)/(2*kb_t*T))

tsallis_func = lambda x,T,q,A: A*(((q-1.))**(1.5)) * (sp.special.gamma(1./(q-1.)) / sp.special.gamma((1./(q-1.))-1.5)) * ((m/(2.*np.pi*kb_t*T))**(1.5)) *4*np.pi*x**2 * (1. - (1.-q)*(m*x**2)/(2.*kb_t*T))**(1./(1.-q))
tsallis_func_no_A = lambda x,T,q: (((q-1.))**(1.5)) * (sp.special.gamma(1./(q-1.)) / sp.special.gamma((1./(q-1.))-1.5)) * ((m/(2.*np.pi*kb_t*T))**(1.5)) *4*np.pi*x**2 * (1. - (1.-q)*(m*x**2)/(2.*kb_t*T))**(1./(1.-q))

###############################################################################

selector = 2

if selector == 1:
    filelist=["s1m8.txt","s1m9.txt","s3m8.txt","s4m9.txt",
              "s5m8.txt","s7m9.txt","s7m10.txt"]
    x_values = np.array([1e-8,1e-9,3e-8,4e-9,5e-8,7e-9,7e-10])
    for i,file in enumerate(filelist):
        filelist[i]="sweep_timesteps/"+file

elif selector == 2:
    filelist=["986c5.txt","645c5.txt","213c5.txt","843c6.txt",
              "479c6.txt","162c6.txt","799c7.txt","512c7.txt",
              "244c7.txt"]
    x_values = np.array([9.86e-5,6.45e-5,2.13e-5,8.43e-6,
                         4.79e-6,1.62e-6,7.99e-7,5.12e-7,
                         2.44e-7])
    for i,file in enumerate(filelist):
        filelist[i]="sweep_buffergas_density/"+file

elif selector == 3:
    filelist=["r100.txt","r1000.txt","r2000.txt","r3000.txt",
              "r4000.txt","r5000.txt","r6000.txt"]
    x_values = np.array([100,1000,2000,3000,
                         4000,5000,6000]) /6. # grid points over 6mm
    for i,file in enumerate(filelist):
        filelist[i]="sweep_rf_resolution/"+file



temperatures = []
temperatures_err = []
q_values = []
q_values_err = []
A_values = []
A_values_err = []
temp_maxwell = []
temp_maxwell_err = []
temp_maxwell2 = []
temp_maxwell_err2 = []

for filename in filelist:
    ionN,x,y,z,vx,vy,vz,v_rel,t,pot_energy,kin_energy = load_big_file(filename,skiprows=0,length=11).T
    #kin_energy = kin_energy[kin_energy<0.4]
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    #number_bins = int(75)
    
    #speed = speed[(int(len(speed)/2)):]
    #
    speed = speed[speed[:]<2500]
    #
    print("mean speed")
    print(speed.mean())
    print("  ")
  
    bin_end = 4000
    bin_range = np.arange(0,bin_end,50)
    number_bins = len(bin_range)-1
    
    bin_heights,bin_edges = np.histogram(speed,bins = bin_range)
    bins = np.array([])
    
    for i in range(number_bins):
     bins = np.append(bins,(bin_edges[i] + bin_edges[i+1])/2.)
    
    bin_err = np.sqrt(bin_heights)
    normal = np.trapz(y=bin_heights ,x=bins, axis=0)
    bin_heights = bin_heights/normal
    bin_err = bin_err/normal
    
    bins = bins[(bin_heights[:]>0)]
    bin_err = bin_err[(bin_heights[:]>0)]
    bin_heights = bin_heights[(bin_heights[:]>0)]

    
 #%%       
    plt.figure()
    plt.errorbar(bins, bin_heights, yerr=bin_err, fmt='o')
        
    x_fit_plot = np.arange(0,bin_end,10)
        
    params, params_cov = curve_fit(tsallis_func, bins,bin_heights, p0=[300.,1.05,1.])
    perr = np.sqrt(np.diag(params_cov))
    print("Parameters:")     
    print(params[0],params[1],params[2],params_cov[1])
    plt.plot( x_fit_plot,tsallis_func(x_fit_plot,*params),color = "red")
    
    temperatures = np.append(temperatures,params[0])
    temperatures_err = np.append(temperatures_err,perr[0])
    q_values = np.append(q_values,params[1])
    q_values_err = np.append(q_values_err,perr[1])
    
    params, params_cov = curve_fit(maxwell_func2_no_A, bins,bin_heights, p0=[340.])
    perr = np.sqrt(np.diag(params_cov))
    print("Parameters:")     
    print(params[0],params_cov[0])
    plt.plot( x_fit_plot,maxwell_func2_no_A(x_fit_plot,*params),color = "orange")
    
    
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Probability (-)")
    #plt.ylim([-0.0001,0.0013])
        
    plt.yscale("log")
    

  #%%       
    plt.figure()
    plt.errorbar(bins, bin_heights, yerr=bin_err, fmt='o')
        
    x_fit_plot = np.arange(0,bin_end,10)
        
    params, params_cov = curve_fit(tsallis_func, bins,bin_heights, p0=[300.,1.05,1.])
    perr = np.sqrt(np.diag(params_cov))
    print("Parameters:")     
    print(params[0],params[1],params[2],params_cov[1])
    plt.plot( x_fit_plot,tsallis_func(x_fit_plot,*params),color = "red")
        
        
    params, params_cov = curve_fit(maxwell_func2, bins,bin_heights, p0=[340.,1.])
    perr = np.sqrt(np.diag(params_cov))
    print("Parameters:")     
    print(params[0],params[1],params_cov[1])
    

    temp_maxwell= np.append(temp_maxwell,params[0])
    temp_maxwell_err = np.append(temp_maxwell_err,perr[0])
    

    params, params_cov = curve_fit(maxwell_func2_no_A, bins,bin_heights, p0=[340.])
    perr = np.sqrt(np.diag(params_cov))
    print("Parameters:")     
    print(params[0],params_cov[0])
    temp_maxwell2 = np.append(temp_maxwell2,params[0])
    temp_maxwell_err2 = np.append(temp_maxwell_err2,perr[0])    
    
    #plt.scatter(bins,bin_heights)
    plt.plot( x_fit_plot,maxwell_func2_no_A(x_fit_plot,*params),color = "orange")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Probability (-)")
    #plt.ylim([-0.0001,0.0013])
    
#%%
nn=len(filelist)+1

plt.figure()
plt.errorbar(np.arange(1,nn,1),temperatures,yerr=temperatures_err, fmt='o')
plt.figure()
plt.errorbar(np.arange(1,nn,1),q_values,yerr=q_values_err, fmt='o')
plt.figure()
plt.errorbar(np.arange(1,nn,1),temp_maxwell,yerr=temp_maxwell_err, fmt='o')
plt.figure()
plt.errorbar(np.arange(1,nn,1),temp_maxwell2,yerr=temp_maxwell_err2, fmt='o')
plt.figure()
#plt.plot(sigmas)
plt.show()
print("mean amp corrected maxwell")
print(temp_maxwell.mean())
print("mean T only maxwell")
print(temp_maxwell2.mean())

#%%


"""

    Plotting of the final graphs starts here
    
"""


fig, ax1 = plt.subplots()
fig.set_size_inches(7.2,4.45)

#fig.set_size_inches(7.2,4.45)
# setup seaborn layout
sb.set_context("paper",
                   rc={"font.size":10,
                       "axes.titlesize":8,
                       "axes.labelsize":11})
sb.set_style("whitegrid")
colors = sb.color_palette("muted",1)

y_values = temp_maxwell2
y_values_err = temp_maxwell_err2


if selector == 1:
    ax1.errorbar(1./x_values,y_values,yerr=y_values_err,
                 fmt='o',
                 color=colors[0])
    
    ax1.set_xlabel("Reciprocal step size (1/s)")
    ax1.set_ylabel("Temperature (K)")
    
    # save the graphic
    plt.savefig("convergence_step_size.pdf",bbox_inches='tight')
    plt.savefig("convergence_step_size.png",bbox_inches='tight')

 
elif selector == 2:
    
    
    
    ax1.errorbar(x_values,y_values,yerr=y_values_err,
                 fmt='o',
                 color=colors[0])
    
    ax1.set_xlabel("Average time between collisions (s)")
    ax1.set_ylabel("Temperature (K)")
    

    
    ax1.set_xlim([1e-7,2e-4])
    ax1.set_xscale("log")

    plt.grid(True,which="both",ls="-",c='.8')
    locmaj = mticker.LogLocator(base=10,numticks=12) 
    
    ax1.xaxis.set_major_locator(locmaj)
    locmin = mticker.LogLocator(base=10,
                                           subs=(0.2,0.4,0.6,0.8),
                                           numticks=12)
    ax1.xaxis.set_minor_locator(locmin)
    ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
    
    

    
    # save the graphic
    plt.savefig("convergence_density.pdf",bbox_inches='tight')
    plt.savefig("convergence_density.png",bbox_inches='tight')

elif selector == 3:
    ax1.errorbar(x_values,y_values,yerr=y_values_err,
                 fmt='o',
                 color=colors[0])
    
    ax1.set_xlabel("Grid points per mm (1/mm)")
    ax1.set_ylabel("Temperature (K)")
    
 
    
    # save the graphic
    plt.savefig("convergence_resolution.pdf",bbox_inches='tight')
    plt.savefig("convergence_resolution.png",bbox_inches='tight')   



