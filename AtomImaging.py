# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:13:45 2021

@author: Saba
Parts of MOT image analysis scripts
"""
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import matplotlib.image as mpimg
from AnalysisClass3 import import_files
from AnalysisClass3 import analysis_data
import numpy as np
import matplotlib.pyplot as plt
#import os
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset
from matplotlib.gridspec import GridSpec
#from matplotlib import ticker
#import matplotlib.image as mpimg
#import scipy.constants as con
#from scipy import special
from scipy.optimize import curve_fit
#from matplotlib.gridspec import GridSpec
#from scipy.integrate import quad,nquad
#from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
#from matplotlib.patches import Rectangle
#from scipy.integrate import odeint
bin_s=0.5
center_y=101
center_x=209
w_t = 15/2.54 # typical width in inches
mydpi=200
datapath="D:\\Analysis"
imgBgd=mpimg.imread(datapath+"\\Data\\AtomImaging\\Bgd_old.bmp")
imgAbs=mpimg.imread(datapath+"\\Data\\AtomImaging\\Abs_old.bmp")-imgBgd
imgDiv=mpimg.imread(datapath+"\\Data\\AtomImaging\\Div_old.bmp")-imgBgd


def twoD_Gaussian(inp, amplitude, xo, yo, sigma_x, sigma_y):
    (x, y)=inp  
    g = amplitude*np.exp( -(x-xo)**2/(2*sigma_x**2) - (y-yo)**2/(2*sigma_y**2))
    return(g.ravel())

OD=gaussian_filter(-np.log(imgAbs/imgDiv),sigma=bin_s)
#OD=-np.log(imgAbs/imgDiv)
x_axis=(np.arange(0,len(OD[center_y,:]))-center_x)*9.8e-3*4#to mm
y_axis=(np.arange(0,len(OD[:,center_x]))-center_y)*8.4e-3*4#to mm

X_fit,Y_fit=np.meshgrid(x_axis[center_x-50:center_x+50],y_axis[center_y-50:center_y+50])
OD[np.isnan(OD)]=0
OD[OD==np.inf]=0
OD[OD==-np.inf]=0

G2D_fit=curve_fit(twoD_Gaussian, (X_fit,Y_fit), OD[center_y-50:center_y+50,center_x-50:center_x+50].ravel(), p0=[2,0,0,0.3,0.3])

fig,ax=plt.subplots(5,1,figsize=(w_t,w_t/1.5),dpi=mydpi)

ax[0].pcolormesh(X_fit,Y_fit,imgAbs[center_y-50:center_y+50,center_x-50:center_x+50],cmap=cm.Blues_r)
ax[1].pcolormesh(X_fit,Y_fit,imgDiv[center_y-50:center_y+50,center_x-50:center_x+50],cmap=cm.Blues_r)
ax[0].set_xlim([x_axis[center_x-50],x_axis[center_x+50]])
ax[0].set_ylim([y_axis[center_y-50],y_axis[center_y+50]])
ax[1].set_xlim([x_axis[center_x-50],x_axis[center_x+50]])
ax[1].set_ylim([y_axis[center_y-50],y_axis[center_y+50]])
ax[0].set_xticks([-2,-1,0,1,2])
ax[0].set_yticks([-1,0,1])
ax[1].set_xticks([-2,-1,0,1,2])
ax[1].set_yticks([-1,0,1])
ax[0].set_xticklabels([])

ax[0].set_position([.01,0.51,0.32,0.48])
ax[1].set_position([.01,0.01,0.32,0.48])

OD=gaussian_filter(-np.log(imgAbs/imgDiv),sigma=bin_s)
ax[2].pcolormesh(OD,cmap=cm.coolwarm)
ax[2].set_position([0.35,0.39,0.4,0.6])
ax[2].set_xlim([center_x-50,center_x+50])
ax[2].set_ylim([center_y-50,center_y+50])
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].plot([center_x,center_x],[0,300],color="k",ls="--",lw=1)
ax[2].plot([0,300],[center_y,center_y],color="k",ls="--",lw=1)


ax[3].plot(x_axis,OD[center_y,:],".",color="C0")
ax[3].set_position([0.35,0.15,0.4,0.24])
ax[3].set_xlim([x_axis[center_x-50],x_axis[center_x+50]])
ax[3].set_xlabel("x [mm]")
ax[3].set_xticks([-2,-1,0,1,2])
ax[3].set_xticklabels(["-2","-1","0","1","2"])
ax[3].set_ylim([-.15,3])
ax[3].set_ylabel("OD")
ax[3].yaxis.set_label_position("right")
ax[3].yaxis.tick_right()
ax[3].plot([x_axis[center_x],x_axis[center_x]],[0,3],color="k",ls="--",lw=1)
ax[3].plot(x_axis,G2D_fit[0][0]*np.exp(-(x_axis-G2D_fit[0][1])**2/(2*G2D_fit[0][3]**2)-(0-G2D_fit[0][2])**2/(2*G2D_fit[0][4])**2),color="C1")


ax[4].plot(OD[:,center_x],y_axis,".",color="C0")
ax[4].set_position([0.75,0.39,0.16,0.6])
ax[4].set_ylim([y_axis[center_y-50],y_axis[center_y+50]])
ax[4].yaxis.tick_right()
ax[4].yaxis.set_label_position("right")
ax[4].set_ylabel("y [mm]")
ax[4].set_yticks([-1,0,1])
ax[4].set_yticklabels(["-1","0","1"],rotation=90,verticalalignment="center")
ax[4].set_xlim([-.15,3])
ax[4].set_xlabel("OD")
ax[4].plot([0,3],[y_axis[center_y],y_axis[center_y]],color="k",ls="--",lw=1)
ax[4].plot(G2D_fit[0][0]*np.exp(-(0-G2D_fit[0][1])**2/(2*G2D_fit[0][3]**2)-(y_axis-G2D_fit[0][2])**2/(2*G2D_fit[0][4]**2)),y_axis,color="C1")


ax[0].text(0.02,0.98,"a",verticalalignment="top",color="k",transform=ax[0].transAxes,fontweight="bold", bbox=dict(facecolor='white',edgecolor="white",boxstyle='square,pad=0.15',))
ax[1].text(0.02,0.98,"b",verticalalignment="top",color="k",transform=ax[1].transAxes,fontweight="bold", bbox=dict(facecolor='white',edgecolor="white",boxstyle='square,pad=0.15',))
ax[2].text(0.02,0.98,"c",verticalalignment="top",color="k",transform=ax[2].transAxes,fontweight="bold", bbox=dict(facecolor='white',edgecolor="white",boxstyle='square,pad=0.25',))

fig.savefig(datapath+"/Document/Figures/Absorbi.png")
fig.savefig(datapath+"/Document/Figures/Absorbi.pdf")
fig.savefig(datapath+"/Document/Figures/Absorbi.svg")

#%%Alpha Factor
alpha = 3.29465
 
data_alpha_min=np.loadtxt(datapath+"/Data/AlphaFactor/ResultsAlphaMeasurementAll.dat")
alphas_fine=np.unique(data_alpha_min[:,0])
data_alpha_min_all=[data_alpha_min[data_alpha_min[:,0]==a] for a in alphas_fine]

I_Isat=np.unique(data_alpha_min[:,1])

min_data=[]
for i in range(len(alphas_fine)):
    min_data_I=[]
    temp=data_alpha_min_all[i]
    for I in I_Isat:
            min_data_I.append([
                     np.mean(temp[temp[:,1]==I,2]),
                     np.mean(temp[temp[:,1]==I,3]),
                     np.sqrt(
                             (np.std(temp[temp[:,1]==I,3])/np.sqrt(len(temp[temp[:,1]==I,3])))**2+
                       np.sqrt(np.mean(temp[temp[:,1]==I,4]**2)/len(temp[temp[:,1]==I,4]))**2)
                     ])
    min_data.append(np.array(min_data_I))

devs=[np.std(min_data[i][:,1]) for i in range(len(min_data))]
devsE=[np.sqrt(np.mean(min_data[i][:,2]**2)/len(min_data[i][:,2])) for i in range(len(min_data))]

plot_inds=[0,10,20,23,30,40,50]

AlphaCList=["grey","grey","grey","C1","grey","grey","grey"]
AlphaLabel=["$\\alpha=1$","$\\alpha=2$","$\\alpha=3$","$\\alpha=3.29$","$\\alpha=4$","$\\alpha=5$","$\\alpha=6$"]
fig,ax=plt.subplots(1,1,figsize=(w_t,w_t/1.5),dpi=mydpi/2)
for i,plot_d in enumerate(plot_inds):
    ax.errorbar(min_data[plot_d][:,0],min_data[plot_d][:,1],yerr=min_data[plot_d][:,2]
                ,Marker="o",MarkerSize=3,ls="-",color=AlphaCList[i],ecolor=AlphaCList[i],label=AlphaLabel[i])
ax.set_xscale("log")
ax.set_xlim([1e-2,10e1])
ax.set_ylim([0,14])
ax.set_ylabel("optical density")
ax.set_xlabel("I/I$_{sat}$")

from labellines import labelLine, labelLines
xvals=[0.04,0.04,0.04,0.04,0.04,0.04]
lines=plt.gca().get_lines()
la=lines[3]
lines.pop(3)
labelLines(lines,xvals=xvals, backgroundcolor='none',yoffsets=-0.4)
labelLine(la,0.5,yoffset=0.4,backgroundcolor="none",label=AlphaLabel[3])

axin = plt.axes([0,0,1,1])
ip = InsetPosition(ax, [0.6,0.6,0.39,0.375])
axin.set_axes_locator(ip)
axin.errorbar(alphas_fine,devs,yerr=devsE,Marker="o",MarkerSize=3,ls="",color="grey",ecolor="grey")
axin.plot([3.29465,3.29465],[0,1],ls="--",color="C1",lw=1.5)
axin.set_xlabel("$\\alpha$",labelpad=0)
axin.set_xlim([1.95,4.65])
axin.set_ylabel("$std(od)$",labelpad=1)
axin.set_ylim([0,0.81])
axin.set_yticks([0,0.2,0.4,0.6,0.8])

plt.tight_layout()

fig.savefig(datapath+"/Document/Figures/Alpha.png")
fig.savefig(datapath+"/Document/Figures/Alpha.pdf")
fig.savefig(datapath+"/Document/Figures/Alpha.svg")