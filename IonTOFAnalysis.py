# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:59:48 2021

@author: Saba
Analysis for extracting information about the ions' distribution from the time of flight of ions to the detector
"""
import os

os.chdir("C:\\Users\\Saba\\Documents\\Python")

from AnalysisClass3 import import_files
from AnalysisClass3 import analysis_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special
from lmfit.models import GaussianModel
from lmfit import Model
import glob, os
import matplotlib.dates as mdates
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator)
import scipy.constants as con

###############################################################################
#functions
##############################################################################
def getHist(Du,tmin,tmax,steps,normed):
    ind=analysis_data.getrun2(variablesarray,[Du],[0])
    whatever=[]
    if len(ind)>0:
        for i in ind:
            if len(toflist[i])<maxions:
                whatever=whatever+toflist[i]  
    finalTOF=analysis_data.histarray(whatever,steps,(tmin,tmax),normed)
    MeanNumber=np.sum(analysis_data.histarray(whatever,steps,(tmin,tmax),True)[1])/len(ind)
    np.savetxt(resultpath+"/Histogram_"+labelX+str(int(Du))+"ms.dat",np.transpose(np.array(analysis_data.histarray(whatever,steps,(tmin,tmax),True))))
    return([finalTOF,np.round(MeanNumber)])
    
def fitgaussianTOF(time,trace,affix,tmin,tmax,EnableSave):
    tmodel = GaussianModel()
    tparams = tmodel.guess(trace,time)
    tresult = tmodel.fit(trace, tparams, x=time)

    bv=tresult.best_values
    plott=np.arange(tmin,tmax,(tmax-tmin)/1000)
    tpopt=[bv["amplitude"]/np.sqrt(2*np.pi)/bv["sigma"],bv["center"],bv["sigma"]]
    tpcov=tresult.covar
    
    if EnableSave:

        tfitreport=open(resultpath+"/tof_fitresult_"+affix+".txt","w")
        tfitreport.write(tresult.fit_report())
        tfitreport.close()
           
        fig,ax= plt.subplots()
        ax.plot(time,trace,Marker=".",LineStyle="")
        ax.plot(plott,analysis_data.gaussian_func(plott,*tpopt,0),color="Red")
        plt.xlabel("time of flight [us]")
        plt.ylabel("prob.")
        plt.title("time of flight "+affix)
        plt.xlim([tmin,tmax])
        plt.grid(True)
        fig.savefig(resultpath+"/tof_"+affix+".pdf")
        fig.savefig(resultpath+"/tof_"+affix+".png")
    
        np.savetxt(resultpath+"/TofMeanTrace_GaussianFit_"+affix+".dat",np.transpose(np.array([time,trace,analysis_data.gaussian_func(time,*tpopt,0)])),header="time of flight,norm prob.,gaussianfit",fmt="%.3e")

    return [tpopt,tpcov]

def fittsallisTOF(time,trace,affix,tmin,tmax,p0,EnableSave):
    tmodel = Model(analysis_data.tsallis_func)
    tparams=tmodel.make_params()
    tparams.add("amp",value=p0[0],min=0)
    tparams.add("t0",value=p0[1])
    tparams.add("q",value=p0[2],min=1.001,max=3)
    tparams.add("sigma",value=p0[3])
    tresult = tmodel.fit(trace, tparams, x=time)
    
    bv=tresult.best_values
    plott=np.arange(tmin,tmax,(tmax-tmin)/1000)
    tpopt=[bv["amp"],bv["t0"],bv["q"],bv["sigma"]]
    tpcov=tresult.covar
    
    if EnableSave:
    
        tfitreport=open(resultpath+"/tof_fitresult_tsallis_"+affix+".txt","w")
        tfitreport.write(tresult.fit_report())
        tfitreport.close()
        
        fig,ax= plt.subplots()
        ax.plot(time,trace,Marker=".",LineStyle="")
        ax.plot(plott,analysis_data.tsallis_func(plott,*tpopt),color="Red")
        plt.xlabel("time of flight [us]")
        plt.ylabel("prob.")
        plt.title("time of flight"+affix)
        plt.xlim([tmin,tmax])
        plt.grid(True)
        fig.savefig(resultpath+"/tof_tsallis_"+affix+".pdf")
        fig.savefig(resultpath+"/tof_tsallis_"+affix+".png")
        
        np.savetxt(resultpath+"/TofMeanTrace_TsallisFit_"+affix+".dat",np.transpose(np.array([time,trace,analysis_data.tsallis_func(time,*tpopt)])),header="time of flight,norm prob.,tsallisfit",fmt="%.3e")

    return [tpopt,tpcov]

def dynamicSigma(maxi,MOTDu,fitmodel):
    ind=analysis_data.getrun2(variablesarray,[MOTDu],[0])  
    whatever=[]
    if len(ind)>0:
        for i in range(0,maxi):
            if len(toflist[ind[i]])<maxions:
                whatever=whatever+toflist[ind[i]]  
    finalTOF=analysis_data.histarray(whatever,steps,(tmin,tmax),True)
    if fitmodel=="gaussian":
        [tpopt,tpcov]=fitgaussianTOF(finalTOF[0],finalTOF[1],"",tmin,tmax,False)
        out=[maxi,tpopt[2],tpcov[2][2],len(whatever)]
    else:
        [tpopt,tpcov]=fittsallisTOF(finalTOF[0],finalTOF[1],"",tmin,tmax,[0.04,49.5,1.2,1],False)
        out=[maxi,tpopt[3],tpcov[3][3],len(whatever)]
    return(out)
    
def getconvergence(Du,fitmodel):
    dyS=[]
    maxind=len(analysis_data.getrun2(variablesarray,[Du],[0]))
    for i in range(10,maxind,10):
        dyS.append(dynamicSigma(i,Du,fitmodel))
    dys=np.array(dyS)
    return(dys)

def numofions(DU,variables,tofs):
    ind=analysis_data.getrun2(variables,[Du],[0])  
    tofarray=np.array(tofs)
    numofions=[]
    for l in tofarray[ind]:
        numofions.append(len(l))
    return(numofions)

def changetoftomass(tof1):
    return((tof1/8.3684)**2)
def masstotof(mass):
    return((np.sqrt(mass)*8.3684))
###############################################################################
#### import data
###############################################################################

plt.close("all")
filepath=os.path.join(os.path.dirname(__file__))

datapath=filepath+"/Data.zip"
resultpath=filepath+"/Results"
detachpath=filepath+"/PDloss"

variables=["MOTDu"]
datanames=["PicoTOF_Ion_count","PeakOD"]

print("Import variables...")
[runnumbers,variablesarray]=import_files.makevaluetableZIP(datapath,variables)
print("Done!")
print("Import data...")
datatable=import_files.makedatatableZIP(datapath,datanames)
print("Done!")
print("Import tofs...")
[tofruns,toflist]=import_files.singleiontofZIP(datapath)
print("Done!")
#integrallist=np.sort(integrallist,axis=0)
#datatable=np.sort(datatable,axis=0)
print("Length of variablelist datalist toflist")
print(str(len(runnumbers))+" "+str(len(datatable))+" "+str(len(tofruns)))
print("First runnumber:")
print(str(int(runnumbers[0]))+" "+str(int(datatable[0][0]))+" "+str(int(tofruns[0])))
variablesarray=variablesarray[:np.min([len(datatable),len(toflist)])]


####import the loss data due to photodetachment
if os.path.isdir(detachpath):
    print("Import variables...")
    [runnumbersPD,variablesarrayPD]=import_files.makevaluetable(detachpath,variables)
    print("Done!")
    print("Import data...")
    datatablePD=import_files.makedatatable(detachpath,datanames)
    print("Done!")
    print("Import tofs...")
    [tofrunsPD,toflistPD]=import_files.singleiontof(detachpath)
    print("Done!")
    variablesarrayPD=variablesarrayPD[:np.min([len(datatablePD),len(toflistPD)])]
else:
    print("There is no photodetachment data")

###############################################################################
#### analysis parameters:
###############################################################################

#takes only tof data below maxions
maxions=90000
#parameters to create histogram
tmin=45
tmax=55
steps=350
#variables to sort data with
var0=np.unique(variablesarray[:,0])
#label of the x axis, corresponding to var0
labelX="MOT duration"
#title of the measurement
graphtitle="Interaction of Water Cluster with MOT"
filetitle=""

###############################################################################
###### analysis
###############################################################################

#create histograms
alltoflist=[]
for Du in var0:
    temp=getHist(Du,tmin,tmax,steps,True)
    alltoflist.append(temp[0])

timestep=(tmax-tmin)/steps

fig,ax= plt.subplots()
for i in range(0,len(alltoflist)):
    ax.plot(alltoflist[i][0],alltoflist[i][1]/timestep)
plt.xlim([tmin,tmax])
#plt.ylim([0,0.8])
ax.legend(var0,title=labelX)
plt.xlabel("time of flight [us]")
plt.ylabel("norm. prob. [1/us]")
#plt.title(graphtitle)
secax = ax.secondary_xaxis('top', functions=(changetoftomass,masstotof))
secax.set_xlabel('mass [a.u.]')
secax.xaxis.label.set_color('gray')
secax.tick_params(axis='x', colors='gray')
fig.savefig(resultpath+"/alltofs.png")
fig.savefig(resultpath+"/alltofs.pdf")


# fit data
sigma=[]
sigmaErr=[]

qpar=[]
qparErr=[]

t0=[]
t0Err=[]
for i in range(0,len(alltoflist)):
    tempg=(fitgaussianTOF(alltoflist[i][0],alltoflist[i][1],labelX+"_"+str(int(var0[i]))+"ms",tmin,tmax,True))
    temp=(fittsallisTOF(alltoflist[i][0],alltoflist[i][1],labelX+"_"+str(int(var0[i]))+"ms",tmin,tmax,[tempg[0][0]/np.sqrt(2*np.pi)/tempg[0][2],tempg[0][1],1.01,tempg[0][2]],True))
    sigma.append([tempg[0][2],temp[0][3]])
    sigmaErr.append([tempg[1][2][2],temp[1][3][3]])
    qpar.append(temp[0][2])
    qparErr.append(temp[1][2][2])
    t0.append([tempg[0][1],temp[0][1]])
    t0Err.append([tempg[1][1][1],temp[1][1][1]])
sigma=np.array(sigma)    
sigmaErr=np.array(sigmaErr) 
t0=np.array(t0)
t0Err=np.array(t0Err)    

#plot fit results

#width
fig,ax=plt.subplots()
ax.errorbar(var0,sigma[:,0],yerr=np.sqrt(sigmaErr[:,0]),linestyle="",marker="o")
ax.errorbar(var0,sigma[:,1],yerr=np.sqrt(sigmaErr[:,1]),linestyle="",marker="o")
ax.legend(["Gaussian fit","Tsallis Fit"])
plt.xlabel(labelX)
plt.ylabel("width tof [us]")
plt.grid(True)
fig.savefig(resultpath+"/"+labelX+"vsWidth.pdf")
fig.savefig(resultpath+"/"+labelX+"vsWidth.png")

#q parameter of tsallis fit
fig,ax=plt.subplots()
ax.errorbar(var0,qpar,yerr=np.sqrt(qparErr),linestyle="",marker="o")
ax.legend(["Tsallis Fit"])
plt.xlabel(labelX)
plt.ylabel("q parameter tsallis")
plt.grid(True)
fig.savefig(resultpath+"/"+labelX+"vsQ.pdf")
fig.savefig(resultpath+"/"+labelX+"vsQ.png")

#mean position of tof
fig,ax=plt.subplots()
ax.errorbar(var0,t0[:,0],yerr=np.sqrt(t0Err[:,0]),linestyle="",marker="o")
ax.errorbar(var0,t0[:,1],yerr=np.sqrt(t0Err[:,1]),linestyle="",marker="o")
ax.legend(["Gaussian fit","Tsallis Fit"])
plt.xlabel(labelX)
plt.ylabel("position tof [us]")
plt.grid(True)
fig.savefig(resultpath+"/"+labelX+"vsPosition.pdf")
fig.savefig(resultpath+"/"+labelX+"vsPosition.png")

#export fitdata
exportFitArray=np.transpose(np.array([var0,sigma[:,0],sigmaErr[:,0],sigma[:,1],sigmaErr[:,1],qpar,qparErr,t0[:,0],t0Err[:,0],t0[:,1],t0Err[:,1]]))
np.savetxt(resultpath+"/"+filetitle+"vsFitparameters.dat",exportFitArray,header="tof width gauss,error,tof width tsallis,error,q parameter,error,tof mean gauss,error,tof mean tsallis,error,mean integral [C],std integral")

#check the convergence of the measurement
fig,ax=plt.subplots()
for i in range(0,len(var0)):
    dys=getconvergence(var0[i],"gaussian")
    plt.errorbar(dys[:,3],np.abs(dys[:,1]),yerr=np.sqrt(dys[:,2]),linestyle="",marker="o")
plt.xlabel("total number of ions")
plt.ylabel("sigma")
plt.legend(var0,title=labelX)
fig.savefig(resultpath+"/WidthConvergence.pdf")
fig.savefig(resultpath+"/WidthConvergence.png")

#check number of ions per var0. Good way to check if import is correct
plt.figure()
for Du in var0:  
    temp=numofions(Du,variablesarray,toflist)
    plt.plot(temp,'.')
    
plt.legend(var0)

####### ion loss

#loss with mot
loss=[]
for Du in var0:
    temp=numofions(Du,variablesarray,toflist)
    loss.append([Du,np.mean(temp),np.std(temp)/np.sqrt(len(temp))])
loss=np.array(loss)

#loss via photodetachment
if os.path.isdir(detachpath):
    var0PD=np.unique(variablesarrayPD[:,0])
    lossPD=[]
    for Du in var0PD:
        temp=numofions(Du,variablesarrayPD,toflistPD)
        lossPD.append([Du,np.mean(temp),np.std(temp)/np.sqrt(len(temp))])
    lossPD=np.array(lossPD)
    exportlossPD=np.transpose(np.array([lossPD[:,0],lossPD[:,1]/lossPD[0,1],(lossPD[:,2]/lossPD[0,1]+lossPD[:,2]/lossPD[0,1]**2*lossPD[0,2])]))
    np.savetxt(resultpath+"/"+labelX+"_ionlossPD.dat",exportlossPD,header=labelX+",mean ion number,std ion number")


fig,ax=plt.subplots()
ax.errorbar(loss[:,0],loss[:,1]/loss[0,1],yerr=(loss[:,2]/loss[0,1]+loss[:,2]/loss[0,1]**2*loss[0,2]),linestyle="",marker="o")
if os.path.isdir(detachpath):
    ax.errorbar(lossPD[:,0],lossPD[:,1]/lossPD[0,1],yerr=(lossPD[:,2]/lossPD[0,1]+lossPD[:,2]/lossPD[0,1]**2*lossPD[0,2]),linestyle="",marker="o")
plt.xlabel("time [ms]")
plt.ylabel("norm. ion number")
plt.legend(["Ion loss with MOT","Ion loss without MOT"])
fig.savefig(resultpath+"/NormIonLoss.pdf")
fig.savefig(resultpath+"/NormIonLoss.png")

exportlossMOT=np.transpose(np.array([loss[:,0],loss[:,1]/loss[0,1],(loss[:,2]/loss[0,1]+loss[:,2]/loss[0,1]**2*loss[0,2]),loss[:,1],loss[:,2]]))

np.savetxt(resultpath+"/"+labelX+"_ionlossMOT.dat",exportlossMOT,header=labelX+",norm mean ion number,norm ste ion number,mean ion number,ste ion number")




#%% plot tof nicely last time

tmodel = Model(analysis_data.tsallis_func)
tparams=tmodel.make_params()
tparams.add("amp",value=0.02,min=0)
tparams.add("t0",value=49.5)
tparams.add("q",value=1.1,min=1.001,max=3)
tparams.add("sigma",value=0.9)
tresult = tmodel.fit(alltoflist[-1][1], tparams, x=alltoflist[-1][0])

bv=tresult.best_values
plott=np.arange(tmin,tmax,(tmax-tmin)/10000)
tpopt=[bv["amp"],bv["t0"],bv["q"],bv["sigma"]]
tpcov=tresult.covar

gmodel = GaussianModel()
gparams = gmodel.guess(alltoflist[-1][1],alltoflist[-1][0])
gresult = gmodel.fit(alltoflist[-1][1], gparams, x=alltoflist[-1][0])

bv=gresult.best_values
gpopt=[bv["amplitude"]/np.sqrt(2*np.pi)/bv["sigma"],bv["center"],bv["sigma"]]
gpcov=gresult.covar

fig,ax= plt.subplots()
ax.plot(alltoflist[-1][0],alltoflist[-1][1],Marker=".",LineStyle="")
ax.plot(plott,analysis_data.tsallis_func(plott,*tpopt),color="C1")
ax.plot(plott,analysis_data.gaussian_func(plott,*gpopt,0),color="C2")
plt.xlabel("time of flight [us]")
plt.xlim([38,55])
plt.legend(["data","Tsallis","Gaussian"])
plt.tight_layout()

fig.savefig(resultpath+"/nice_tof_last_time.pdf")
fig.savefig(resultpath+"/nice_tof_last_time.png")

#%% first time

tmodel = Model(analysis_data.tsallis_func)
tparams=tmodel.make_params()
tparams.add("amp",value=0.02,min=0)
tparams.add("t0",value=49.5)
tparams.add("q",value=1.1,min=1.001,max=3)
tparams.add("sigma",value=0.9)
tresult = tmodel.fit(alltoflist[0][1], tparams, x=alltoflist[0][0])

bv=tresult.best_values
plott=np.arange(tmin,tmax,(tmax-tmin)/10000)
tpopt=[bv["amp"],bv["t0"],bv["q"],bv["sigma"]]
tpcov=tresult.covar

gmodel = GaussianModel()
gparams = gmodel.guess(alltoflist[0][1],alltoflist[-1][0])
gresult = gmodel.fit(alltoflist[0][1], gparams, x=alltoflist[0][0])

bv=gresult.best_values
gpopt=[bv["amplitude"]/np.sqrt(2*np.pi)/bv["sigma"],bv["center"],bv["sigma"]]
gpcov=gresult.covar

fig,ax= plt.subplots()
ax.plot(alltoflist[0][0],alltoflist[0][1],Marker=".",LineStyle="")
ax.plot(plott,analysis_data.tsallis_func(plott,*tpopt),color="C1")
ax.plot(plott,analysis_data.gaussian_func(plott,*gpopt,0),color="C2")
plt.xlabel("time of flight [us]")
plt.xlim([38,55])
plt.legend(["data","Tsallis","Gaussian"])
plt.tight_layout()

fig.savefig(resultpath+"/nice_tof_first_time.pdf")
fig.savefig(resultpath+"/nice_tof_first_time.png")

tempoffset=0.02404
def CalcTemp(sigma):
    out_a=np.array([])
    for s in sigma:
        if (s**2-tempoffset*1)/0.00231>30:
            out=(s**2-tempoffset*1)/0.00231
        else:
            out=(s**2-0.02404)/0.005
        out_a=np.append(out_a,out)
    return(out_a)    
test=CalcTemp(sigma[:,0])
