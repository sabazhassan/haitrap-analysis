# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:07:44 2021

@author: Saba
Analysis Class for ion distribution analysis, Includes functions for TOF Analysis, extracting data from files, functions for the tomographic analysis
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import glob, os
import itertools
import re
import io
import scipy.integrate as integrate
from zipfile import ZipFile

class import_files:
    def correct_trace(datapath,resultpath,R1,R2,Cap,OffsetBoundry,intboundlow,intboundhigh):
        """ this function calcutates the output current with Kirchhoffs law for each MCP trace and returns a list of the charge (integral of the current) 
        between intboundlow and intboundhigh. 
        A folder in resultpath is created with all current traces, named with the runnumber.
        There is an offset current flowing, which has to be substracted. So the mean of all values before OffsetBoundry is taken and substracted for each trace.
        Cap: Measurement capacitor
        R2: Resistance to load anode of mcp
        R1: Resistance to ground after measurement capacitor"""
        
        os.chdir(datapath)
        
        folderlist=os.listdir()
        filelist=[]
        for i in range(0,len(folderlist)):
            filelist.append(glob.glob(folderlist[i]+"/*.txt"))
        filelist=list(itertools.chain(*filelist))
        
        integrallist=[]
        try:
            os.mkdir(resultpath+"/CorrectedTraces/")
        except:
            print("")
        for file in filelist:
            try:
                runnumber=int(re.split("R|_",file)[3])
            except:
                try:
                    runnumber=int(re.split("R|_",file)[4])
                except:
                    print("Measurement name has to many capital Rs :)")
            s=open(file).read().replace(",",".")
            trace=np.loadtxt(io.StringIO(s),skiprows=1)
            signal=trace[:,1]-np.mean(trace[np.where(trace[:,0]<OffsetBoundry),1])
            stepsize=trace[1,0]-trace[0,0]
            corrected=[]
            for i in range(0,len(signal)):
                corrected.append(-signal[i]*(1/R1+1/R2)-1/(R1*R2*Cap)*np.sum(signal[0:i,])*stepsize)
            boundind=np.where((trace[:,0]<intboundhigh)&(trace[:,0]>intboundlow))
            integrallist.append([runnumber,np.sum(np.array(corrected)[boundind[0]])*stepsize])
            print(runnumber)
            np.savetxt(resultpath+"/CorrectedTraces/R"+str(runnumber)+"_correctedTrace.txt",np.transpose(np.array([list(trace[:,0]),corrected])))
        np.savetxt(resultpath+"/integralresults.txt",integrallist)
        return np.array(integrallist)
        
    def makevaluetable(datapath,variables):
        """
        Takes all VariableValues.dat in datafolder and returns a list of the runnumbers and a list of the wanted variables.
        """
        
        os.chdir(datapath)
        
        folderlist=os.listdir()
        filelist=[]
        for i in range(0,len(folderlist)):
            filelist.append(glob.glob(folderlist[i]+"/*VariableValues.dat"))
        filelist=list(itertools.chain(*filelist))
        
        variableslist=[]
        runnumberlist=[]
        for file in filelist:
            try:
                runnumber=int(re.split("R|_",file)[-2])
            except:
                try:
                    runnumber=int(re.split("R|_",file)[-3])
                except:
                    print("Measurement name has to many capital Rs :)")
    
            varnames=np.array(open(file).readlines()[0].split("\t"))
            pos=[]
            for var in variables:
                pos.append(np.where(varnames==var)[0][0])
            s=open(file).read().replace(",",".")
            temp=np.loadtxt(io.StringIO(s),skiprows=1,delimiter="\t")   
            runnumberlist.append(temp[:,0]+runnumber)
            variableslist.append(temp[:,pos])  
            
        
        return [np.concatenate(runnumberlist),np.concatenate(variableslist)]    

    def makevaluetableZIP(dataZIP,variables):
        """
        Takes all VariableValues.dat in datafolder and returns a list of the runnumbers and a list of the wanted variables.
        """
   
        zip=ZipFile(dataZIP,"r")
        ziplist=zip.namelist()
        
        filelist=[]
        for line in ziplist:
            if line.find("VariableValue")!=-1:
                filelist.append(line)
            
        variableslist=[]
        runnumberlist=[]
        for file in filelist:
            try:
                runnumber=int(re.split("R|_",file)[-2])
            except:
                try:
                    runnumber=int(re.split("R|_",file)[-3])
                except:
                    print("Measurement name has to many capital Rs :)")
            
            s=str(zip.open(file).read()).replace(",",".")
            temp=[line.split("\\t") for line in s.split("\\r\\n")]
                            
            varnames=np.array(temp[0])
            pos=[]
            for var in variables:
                pos.append(np.where(varnames==var)[0][0])
            runnumberlist.append(np.transpose(np.array([float(temp[1:][i][0]) for i in range(0,len(temp)-2)]))+runnumber)
            variableslist.append(np.transpose(([(np.array([float(temp[1:][i][p]) for i in range(0,len(temp)-2)])) for p in pos])))  
           
        return [np.concatenate(runnumberlist),np.concatenate(variableslist)]
    

    def makedatatable(datapath,datanames):
        """
        Takes all DataValues.dat in datafolder and returns a list of the wanted data values.
        """
        os.chdir(datapath)
        
        folderlist=os.listdir()
        filelist=[]
        for i in range(0,len(folderlist)):
            filelist.append(glob.glob(folderlist[i]+"/*DataValues.dat"))
        filelist=list(itertools.chain(*filelist))
        
        datalist=[]
        for file in filelist:
            runnumber=int(re.split("R|_",file)[-2])
            temp=np.array(np.loadtxt(file,dtype=str,delimiter="\t"))
            dataN=temp[:,0]
            dataselect=[runnumber]
            for var in datanames:
                try:
                    dataselect.append(float(temp[np.where(dataN==var)[0][0],1].replace(",",".")))
                except:
                    print("dataname does not exist")
                    break
            datalist.append(dataselect)
        return np.array(datalist)
    
    def makedatatableZIP(dataZIP,datanames):
        """
        Takes all DataValues.dat in datafolder and returns a list of the wanted data values.
        """
        zip=ZipFile(dataZIP,"r")
        ziplist=zip.namelist()
        
        filelist=[]
        for line in ziplist:
            if line.find("DataValues.dat")!=-1:
                filelist.append(line)
        
        datalist=[]
        for file in filelist:
            runnumber=int(re.split("R|_",file)[-2])
            
            s=str(zip.open(file).read()).replace(",",".")
            temp=[line.split("\\t") for line in s.split("\\r\\n")]
            dataN=np.array([line[0] for line in temp])
            dataselect=[runnumber]
            for var in datanames:
                try:
                    dataselect.append(float(temp[np.where(dataN==var)[0][0]][1]))
                except:
                    print("dataname does not exist")
                    break
            datalist.append(dataselect)
        return np.array(datalist)    

    
    def singleiontof(datapath):
        """
        Takes all PicoTOF.dat files in folder datapath and returns a list with runnumber an arrival times in each run.
        """
        
        os.chdir(datapath)
        
        folderlist=os.listdir()
        filelist=[]
        for i in range(0,len(folderlist)):
            filelist.append(glob.glob(folderlist[i]+"/*PicoTOF*.dat"))
        filelist=list(itertools.chain(*filelist))
        
        toflist=[]
        runnumber=[]
        for file in filelist:
            s=open(file)
            counter=0
            for line in s:
                if counter == 0:
                    toflist.append(list(np.float_(line.replace(",",".").split("\t"))[1:]))
                    runnumber.append(np.float_(line.replace(",",".").split("\t"))[0])
                    counter=1
                else:
                    counter=0
            s.close()
        return([runnumber,toflist])
        
        
    def singleiontofZIP(dataZIP):
        """
        Takes all PicoTOF.dat files in folder datapath and returns a list with runnumber an arrival times in each run.
        """
       
        zip=ZipFile(dataZIP,"r")
        ziplist=zip.namelist()
        
        filelist=[]
        for line in ziplist:
            if line.find("PTOF")!=-1:
                filelist.append(line)
        
        toflist=[]
        runnumber=[]
        for file in filelist:
            s=str(zip.open(file).read()).replace(",",".").split("\\r\\n")
            counter=0
            for i in range(0,len(s)-1):
                if counter == 0:
                    toflist.append(list(np.float_(s[i].split("\\t")[1:])))
                    runnumber.append(np.float_(s[i].split("\\t")[0].split("b'")[-1]))
                    counter=1
                else:
                    counter=0
        return([runnumber,toflist])    
    
    
class analysis_data:
    def getrun2(vararray,variables,position):
        """
        Returns all runnumbers for the condition the element at position (row) in vararray is equal to variables.
        variables and position are array like, so multiple rows can be connected to a condition.
        Example:
            TomoXpos is row 0 and TomoDu row 1. You want to select all runs at TomoXpos=15 and TomoDu=0:
            variables=[15,0]
            position=[0,1]
        """
        for i in range(0,len(position)):
            if i == 0:
                conditionslist=(vararray[:,position[i]]==variables[i])
            else:
               conditionslist=np.logical_and(conditionslist,(vararray[:,position[i]]==variables[i]))
        return np.where(conditionslist==True)[0]
    
    def gaussian_func(x,amp,cen,wid,off):
        """
        If you want to set the offset to 0 you can use:
            lambda x,amp,cen,wed: gaussian_func(x,amp,cen,wid,0)
        """
        return amp*np.exp(-0.5*(x-cen)**2/wid**2)+off

    def tsallis_func(x,amp,t0,q,sigma):
        return(amp*(1+(1-q)*(-(x-t0)**2/(2*sigma**2)))**(1/(1-q)))

    def radial_func(x,amp,cen,wid,off):
        return amp*np.exp(-0.5*(x-cen)**6/wid**6)+off
    
    def tomo_func(x,y,amp,cenx,ceny,widx,widy,off):
        return amp*np.exp(-0.5*(x-cenx)**2/widx**2)*np.exp(-0.5*(y-ceny)**6/widy**6)+off
    
    def decay_func(x,amp,k,cen,off):
        """
        If you want to set the offset and t0 to 0 you can use:
            lambda x,amp,k: decay_func(x,amp,k,0,0)
        """
        return(amp*np.exp(-k*(x-cen))+off)
        
    def flatten(inp):
        return list(itertools.chain(*inp))
    
    def histarray(inplist,bins,ran,norm):
        """
        returns x and y of a histogram.
        inplist: input list
        bins: number of bins in the range ran
        norm: boolean normalize histogram or not
        """
        [y,x]=np.histogram(inplist,bins=bins,range=ran)
        x=(x[1]-x[0])/2+x[:-1]
        if norm==True:
            y=y/sum(y)
        return [x,y]
    
    def integrand2D(y, x,z,xs,zs,x0,z0):
        """
        function of distribution in axial (z) direction quatratic potential
        in radial direction (r^2=x^2+y^2) in an r^6 potential
        """
        return np.exp(-((x-x0)**2+y**2)**3/(8*xs**6))*np.exp(-(z-z0)**2/(2*zs**2))
    
    def tomo2D(xz_mesh,xs,zs,x0,z0,amp):
        """tomo[x,s,cen,amp]:=amp*Integral[Exp[-((x-x0)**2+y**2)**3/(8*xs**6)]*Exp[-(z-z0)**2/(2*zs**2)],{y,-5s,5s}]"""
        bound=5*xs
        [x,z]=xz_mesh
        array=x*0
        for j in range(0,len(z[0,:])):
            for i in range(0,len(x[:,0])):
                array[i,j]=(amp*integrate.quad(integrand2D,-bound,bound,args=(x[i,j],z[i,j],xs,zs,x0,z0))[0])
        return array
    
    def integrand(y, x, s,cen):
        return np.exp(-((x-cen)**2+y**2)**3/(8*s**6))
    
    def tomo(x,s,cen,amp):
        """tomo[x,s,cen,amp]:=amp*Integral[Exp[np.exp(-((x-cen)**2+y**2)**3/(8*s**6))],{y,-5s,5s}]"""
        bound=5*s
        array=[]
        for xx in x:
            array.append(amp*integrate.quad(integrand,-bound,bound,args=(xx,s,cen))[0])
        return np.array(array)
        


