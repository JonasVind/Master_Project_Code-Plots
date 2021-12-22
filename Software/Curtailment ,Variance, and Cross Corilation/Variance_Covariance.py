# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:43:40 2021

@author: jones
"""

#%% Create path from code
# Load current directory
import os
import sys
path = os.path.split(os.path.split(os.getcwd())[0])[0] # Creates a path to the general onedrive
sys.path.append(path+"\\Programming") # Adds the Programming folder to used by functions_file

#%% Import libaries

import os
import numpy as np
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
from functions_file import *


#%% Functions

def MatrixSorter(names,matrix):

    # Define output
    sortedMatrix = []        

    for i in names:
        
        # create empty list
        newMatrix = pd.DataFrame(data = np.zeros([len(matrix)]))
        
        for j in range(len(matrix)):
            
            newMatrix.T[j] = matrix[j][i]
        
        # Save into matrix
        sortedMatrix.append(newMatrix)
        
    return sortedMatrix


def ContComponent(sortedMatrix):
    
    loadCovData = (+sortedMatrix[7].values
                   +sortedMatrix[8].values
                   +sortedMatrix[9].values).ravel()
    genCovData =  (+sortedMatrix[4].values
                   +sortedMatrix[5].values
                   +sortedMatrix[6].values).ravel() + loadCovData
    windVarData =   sortedMatrix[0].values.ravel()
    SolarVarData =  sortedMatrix[1].values.ravel() + windVarData
    RoRVarnData =   sortedMatrix[2].values.ravel() + SolarVarData
    loadVarData =   sortedMatrix[3].values.ravel() + RoRVarnData
    
    return loadCovData, genCovData, windVarData, SolarVarData, RoRVarnData, loadVarData

def ResComponent(sortedMatrixResponse, networkType):
    
    if networkType == "elec_only":
        StorageVarData =sortedMatrixResponse[0].values.ravel()
        linksVarData  = sortedMatrixResponse[1].values.ravel() +StorageVarData
        hydroVarData  = sortedMatrixResponse[2].values.ravel() +linksVarData
        backupVarData = sortedMatrixResponse[3].values.ravel() +hydroVarData
        covData =     (+sortedMatrixResponse[4].values
                       +sortedMatrixResponse[5].values
                       +sortedMatrixResponse[6].values
                       +sortedMatrixResponse[7].values
                       +sortedMatrixResponse[8].values
                       +sortedMatrixResponse[9].values).ravel() +backupVarData
        
        return StorageVarData, linksVarData, hydroVarData, backupVarData, covData 
    
    if networkType == "elec_heat":
        StorageVarData =sortedMatrixResponse[0].values.ravel()
        linksVarData  = sortedMatrixResponse[1].values.ravel() +StorageVarData
        hydroVarData  = sortedMatrixResponse[2].values.ravel() +linksVarData
        backupVarData = sortedMatrixResponse[3].values.ravel() +hydroVarData
        heatVarData   = sortedMatrixResponse[4].values.ravel() +backupVarData
        CHPVarData    = sortedMatrixResponse[5].values.ravel() +heatVarData
        covData =     (+sortedMatrixResponse[6].values
                       +sortedMatrixResponse[7].values
                       +sortedMatrixResponse[8].values
                       +sortedMatrixResponse[9].values
                       +sortedMatrixResponse[10].values
                       +sortedMatrixResponse[11].values
                       +sortedMatrixResponse[12].values
                       +sortedMatrixResponse[13].values
                       +sortedMatrixResponse[14].values
                       +sortedMatrixResponse[15].values
                       +sortedMatrixResponse[16].values
                       +sortedMatrixResponse[17].values
                       +sortedMatrixResponse[18].values
                       +sortedMatrixResponse[19].values
                       +sortedMatrixResponse[20].values).ravel() +CHPVarData
        
        return StorageVarData, linksVarData, hydroVarData, backupVarData, heatVarData, CHPVarData, covData 
    
    if networkType == "elec_v2g50":
        StorageVarData =sortedMatrixResponse[0].values.ravel()
        linksVarData  = sortedMatrixResponse[1].values.ravel() +StorageVarData
        hydroVarData  = sortedMatrixResponse[2].values.ravel() +linksVarData
        backupVarData = sortedMatrixResponse[3].values.ravel() +hydroVarData
        transVarData  = sortedMatrixResponse[4].values.ravel() +backupVarData
        covData =     (+sortedMatrixResponse[5].values
                       +sortedMatrixResponse[6].values
                       +sortedMatrixResponse[7].values
                       +sortedMatrixResponse[8].values
                       +sortedMatrixResponse[9].values
                       +sortedMatrixResponse[10].values
                       +sortedMatrixResponse[11].values
                       +sortedMatrixResponse[12].values
                       +sortedMatrixResponse[13].values
                       +sortedMatrixResponse[14].values).ravel() +transVarData
        return StorageVarData, linksVarData, hydroVarData, backupVarData, transVarData, covData 
    
    if networkType == "elec_heat_v2g50":
        StorageVarData =sortedMatrixResponse[0].values.ravel()
        linksVarData  = sortedMatrixResponse[1].values.ravel() +StorageVarData
        hydroVarData  = sortedMatrixResponse[2].values.ravel() +linksVarData
        backupVarData = sortedMatrixResponse[3].values.ravel() +hydroVarData
        heatVarData   = sortedMatrixResponse[4].values.ravel() +backupVarData
        CHPVarData    = sortedMatrixResponse[5].values.ravel() +heatVarData
        transVarData  = sortedMatrixResponse[6].values.ravel() +CHPVarData
        covData =     (+sortedMatrixResponse[7].values
                       +sortedMatrixResponse[8].values
                       +sortedMatrixResponse[9].values
                       +sortedMatrixResponse[10].values
                       +sortedMatrixResponse[11].values
                       +sortedMatrixResponse[12].values
                       +sortedMatrixResponse[13].values
                       +sortedMatrixResponse[14].values
                       +sortedMatrixResponse[15].values
                       +sortedMatrixResponse[16].values
                       +sortedMatrixResponse[17].values
                       +sortedMatrixResponse[18].values
                       +sortedMatrixResponse[19].values
                       +sortedMatrixResponse[20].values
                       +sortedMatrixResponse[21].values
                       +sortedMatrixResponse[22].values
                       +sortedMatrixResponse[23].values
                       +sortedMatrixResponse[24].values
                       +sortedMatrixResponse[25].values
                       +sortedMatrixResponse[26].values
                       +sortedMatrixResponse[27].values).ravel() +transVarData
        return StorageVarData, linksVarData, hydroVarData, backupVarData, heatVarData, CHPVarData, transVarData, covData 

    if networkType == "brownfield":
        StorageVarData =sortedMatrixResponse[0].values.ravel()
        linksVarData  = sortedMatrixResponse[1].values.ravel() +StorageVarData
        hydroVarData  = sortedMatrixResponse[2].values.ravel() +linksVarData
        backupVarData = sortedMatrixResponse[3].values.ravel() +hydroVarData
        heatVarData   = sortedMatrixResponse[4].values.ravel() +backupVarData
        CHPVarData    = sortedMatrixResponse[5].values.ravel() +heatVarData
        covData =     (+sortedMatrixResponse[6].values
                       +sortedMatrixResponse[7].values
                       +sortedMatrixResponse[8].values
                       +sortedMatrixResponse[9].values
                       +sortedMatrixResponse[10].values
                       +sortedMatrixResponse[11].values
                       +sortedMatrixResponse[12].values
                       +sortedMatrixResponse[13].values
                       +sortedMatrixResponse[14].values
                       +sortedMatrixResponse[15].values
                       +sortedMatrixResponse[16].values
                       +sortedMatrixResponse[17].values
                       +sortedMatrixResponse[18].values
                       +sortedMatrixResponse[19].values
                       +sortedMatrixResponse[20].values).ravel() +CHPVarData
        
        return StorageVarData, linksVarData, hydroVarData, backupVarData, heatVarData, CHPVarData, covData


def CovComponent(sortedMatrixCovariance, networkType):
    
    if networkType == "elec_only":
        windStorageCov = (sortedMatrixCovariance[0].values).ravel()
        windLinksCov =   (sortedMatrixCovariance[1].values).ravel() + windStorageCov
        windHydroCov =   (sortedMatrixCovariance[2].values).ravel() + windLinksCov
        windBackupCov =  (sortedMatrixCovariance[3].values).ravel() + windHydroCov
        
        solarStorageCov =(sortedMatrixCovariance[4].values).ravel() + windBackupCov
        solarLinksCov =  (sortedMatrixCovariance[5].values).ravel() + solarStorageCov
        solarHydroCov =  (sortedMatrixCovariance[6].values).ravel() + solarLinksCov
        solarBackupCov = (sortedMatrixCovariance[7].values).ravel() + solarHydroCov
        
        rorCovData =   (+sortedMatrixCovariance[8].values
                        +sortedMatrixCovariance[9].values
                        +sortedMatrixCovariance[10].values
                        +sortedMatrixCovariance[11].values).ravel() + solarBackupCov
        loadCovData =  (+sortedMatrixCovariance[12].values
                        +sortedMatrixCovariance[13].values
                        +sortedMatrixCovariance[14].values
                        +sortedMatrixCovariance[15].values).ravel() + rorCovData
    
        return windStorageCov, windLinksCov, windHydroCov, windBackupCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, rorCovData, loadCovData

    if networkType == "elec_heat":
        WindStorageCov = (sortedMatrixCovariance[0].values).ravel()
        WindLinksCov =   (sortedMatrixCovariance[1].values).ravel() + WindStorageCov
        WindHydroCov =   (sortedMatrixCovariance[2].values).ravel() + WindLinksCov
        WindBackupCov =  (sortedMatrixCovariance[3].values).ravel() + WindHydroCov
        WindHeatCov =    (sortedMatrixCovariance[4].values).ravel() + WindBackupCov
        WindCHPCov =     (sortedMatrixCovariance[5].values).ravel() + WindHeatCov
        
        solarStorageCov =(sortedMatrixCovariance[6].values).ravel() + WindCHPCov
        solarLinksCov =  (sortedMatrixCovariance[7].values).ravel() + solarStorageCov
        solarHydroCov =  (sortedMatrixCovariance[8].values).ravel() + solarLinksCov
        solarBackupCov = (sortedMatrixCovariance[9].values).ravel() + solarHydroCov
        solarHeatCov =   (sortedMatrixCovariance[10].values).ravel() + solarBackupCov
        solarCHPCov =    (sortedMatrixCovariance[11].values).ravel() + solarHeatCov
        
        RoRCovData =   (+sortedMatrixCovariance[12].values
                        +sortedMatrixCovariance[13].values
                        +sortedMatrixCovariance[14].values
                        +sortedMatrixCovariance[15].values
                        +sortedMatrixCovariance[16].values
                        +sortedMatrixCovariance[17].values).ravel() + solarCHPCov
        LoadCovData =  (+sortedMatrixCovariance[18].values
                        +sortedMatrixCovariance[19].values
                        +sortedMatrixCovariance[20].values
                        +sortedMatrixCovariance[21].values
                        +sortedMatrixCovariance[22].values
                        +sortedMatrixCovariance[23].values).ravel() + RoRCovData
        
        return WindStorageCov, WindLinksCov, WindHydroCov, WindBackupCov, WindHeatCov, WindCHPCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarHeatCov, solarCHPCov, RoRCovData, LoadCovData
    
    if networkType == "elec_v2g50":
        WindStorageCov = (sortedMatrixCovariance[0].values).ravel()
        WindLinksCov =   (sortedMatrixCovariance[1].values).ravel() + WindStorageCov
        WindHydroCov =   (sortedMatrixCovariance[2].values).ravel() + WindLinksCov
        WindBackupCov =  (sortedMatrixCovariance[3].values).ravel() + WindHydroCov
        WindTransCov =   (sortedMatrixCovariance[4].values).ravel() + WindBackupCov
    
        solarStorageCov =(sortedMatrixCovariance[5].values).ravel() + WindTransCov
        solarLinksCov =  (sortedMatrixCovariance[6].values).ravel() + solarStorageCov
        solarHydroCov =  (sortedMatrixCovariance[7].values).ravel() + solarLinksCov
        solarBackupCov = (sortedMatrixCovariance[8].values).ravel() + solarHydroCov
        solarTransCov =  (sortedMatrixCovariance[9].values).ravel() + solarBackupCov
    
        RoRCovData =   (+sortedMatrixCovariance[10].values
                        +sortedMatrixCovariance[11].values
                        +sortedMatrixCovariance[12].values
                        +sortedMatrixCovariance[13].values
                        +sortedMatrixCovariance[14].values).ravel() + solarTransCov
        LoadCovData =  (+sortedMatrixCovariance[15].values
                        +sortedMatrixCovariance[16].values
                        +sortedMatrixCovariance[17].values
                        +sortedMatrixCovariance[18].values
                        +sortedMatrixCovariance[19].values).ravel() + RoRCovData   
        
        return WindStorageCov, WindLinksCov, WindHydroCov, WindBackupCov, WindTransCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarTransCov, RoRCovData, LoadCovData

    if networkType == "elec_heat_v2g50":
        WindStorageCov = (sortedMatrixCovariance[0].values).ravel()
        WindLinksCov =   (sortedMatrixCovariance[1].values).ravel() + WindStorageCov
        WindHydroCov =   (sortedMatrixCovariance[2].values).ravel() + WindLinksCov
        WindBackupCov =  (sortedMatrixCovariance[3].values).ravel() + WindHydroCov
        WindHeatCov =    (sortedMatrixCovariance[4].values).ravel() + WindBackupCov
        WindCHPCov =     (sortedMatrixCovariance[5].values).ravel() + WindHeatCov
        WindTransCov =   (sortedMatrixCovariance[6].values).ravel() + WindCHPCov
        
        solarStorageCov =(sortedMatrixCovariance[7].values).ravel() + WindTransCov
        solarLinksCov =  (sortedMatrixCovariance[8].values).ravel() + solarStorageCov
        solarHydroCov =  (sortedMatrixCovariance[9].values).ravel() + solarLinksCov
        solarBackupCov = (sortedMatrixCovariance[10].values).ravel() + solarHydroCov
        solarHeatCov =   (sortedMatrixCovariance[11].values).ravel() + solarBackupCov
        solarCHPCov =    (sortedMatrixCovariance[12].values).ravel() + solarHeatCov
        solarTransCov =  (sortedMatrixCovariance[13].values).ravel() + solarCHPCov
        
        RoRCovData =   (+sortedMatrixCovariance[14].values
                        +sortedMatrixCovariance[15].values
                        +sortedMatrixCovariance[16].values
                        +sortedMatrixCovariance[17].values
                        +sortedMatrixCovariance[18].values
                        +sortedMatrixCovariance[19].values
                        +sortedMatrixCovariance[20].values).ravel() + solarTransCov
        LoadCovData =  (+sortedMatrixCovariance[21].values
                        +sortedMatrixCovariance[22].values
                        +sortedMatrixCovariance[23].values
                        +sortedMatrixCovariance[24].values
                        +sortedMatrixCovariance[25].values
                        +sortedMatrixCovariance[26].values
                        +sortedMatrixCovariance[27].values).ravel() + RoRCovData
        
        return WindStorageCov, WindLinksCov, WindHydroCov, WindBackupCov, WindHeatCov, WindCHPCov, WindTransCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarHeatCov, solarCHPCov, solarTransCov, RoRCovData, LoadCovData
    
    if networkType == "brownfield":
        windStorageCov = (sortedMatrixCovariance[0].values).ravel()
        windLinksCov =   (sortedMatrixCovariance[1].values).ravel() + windStorageCov
        windHydroCov =   (sortedMatrixCovariance[2].values).ravel() + windLinksCov
        windBackupCov =  (sortedMatrixCovariance[3].values).ravel() + windHydroCov
        windHeatCov =    (sortedMatrixCovariance[4].values).ravel() + windBackupCov
        windCHPCov =     (sortedMatrixCovariance[5].values).ravel() + windHeatCov
        
        solarStorageCov =(sortedMatrixCovariance[6].values).ravel() + windCHPCov
        solarLinksCov =  (sortedMatrixCovariance[7].values).ravel() + solarStorageCov
        solarHydroCov =  (sortedMatrixCovariance[8].values).ravel() + solarLinksCov
        solarBackupCov = (sortedMatrixCovariance[9].values).ravel() + solarHydroCov
        solarHeatCov =   (sortedMatrixCovariance[10].values).ravel() + solarBackupCov
        solarCHPCov =    (sortedMatrixCovariance[11].values).ravel() + solarHeatCov
        
        RoRCovData =   (+sortedMatrixCovariance[12].values
                        +sortedMatrixCovariance[13].values
                        +sortedMatrixCovariance[14].values
                        +sortedMatrixCovariance[15].values
                        +sortedMatrixCovariance[16].values
                        +sortedMatrixCovariance[17].values).ravel() + solarCHPCov
        loadCovData =  (+sortedMatrixCovariance[18].values
                        +sortedMatrixCovariance[19].values
                        +sortedMatrixCovariance[20].values
                        +sortedMatrixCovariance[21].values
                        +sortedMatrixCovariance[22].values
                        +sortedMatrixCovariance[23].values).ravel() + RoRCovData

        return windStorageCov, windLinksCov, windHydroCov, windBackupCov, windHeatCov, windCHPCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarHeatCov, solarCHPCov, RoRCovData, loadCovData



#%% Elec_only
#%%% Import Data

# CO2 CONSTRAINTS
# Load data - CO2 constraint

# Load current directory
path = os.path.split(os.path.split(os.getcwd())[0])[0]

# Folder name of data files
directory = path+"\Data\elec_only\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

filename_links =   ["postnetwork-elec_only_0_0.05.h5",
                    "postnetwork-elec_only_0.0625_0.05.h5",
                    "postnetwork-elec_only_0.125_0.05.h5",
                    "postnetwork-elec_only_0.25_0.05.h5",
                    "postnetwork-elec_only_0.375_0.05.h5"]

# Network
network = pypsa.Network(directory+filename_CO2[-1])


#%%% Define index and columns

# Index
timeIndex = network.loads_t.p_set.index 

# Columns
countryColumn = network.loads.index[:30]

#%%% Networks

varMismatchChangeCO2 = []
varChangeCO2 = []
varResponseChangeCO2 = []
covChangeCO2 = []

for networks in filename_CO2:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    Load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - Load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeCO2.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((Load - np.mean(Load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }

    varChangeCO2.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
  
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
  
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
  
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                           varStorage,
                        "Import-Export":                     varLinks,
                        "Hydro Reservoir":                   varHydro,
                        "Backup Generator":                  varBackup,
                        "Storage\nImport-Export":            covStorageLinks,
                        "Storage\nHydro Reservoir":          covStorageHydro,
                        "Storage\nBackup Generator":         covStorageBackup,
                        "Import-Export\nHydro Reservoir":    covLinksHydro,
                        "Import-Export\nBackup Generator":   covLinksBackup,
                        "Backup Generator\nHydro Reservoir": covBackupHydro
                        }

    varResponseChangeCO2.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
  
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
   
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
      
    covLoadStorage  = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
       
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":            covWindStorage,
                        "Wind\nlinks":              covWindLinks,
                        "Wind\nHydro Reservoir":    covWindDisp,
                        "Wind\nBackup Generator":   covWindBackup,
                        
                        "Solar PV\nStorage":        covSolarStorage,
                        "Solar PV\nImport-Export":  covSolarLinks,
                        "Solar PV\nhydro":          covSolarDisp,
                        "Solar PV\nbackup":         covSolarBackup,
                        
                        "RoR\nStorage":             covRoRStorage,
                        "RoR\nImport-Export":       covRoRLinks,
                        "RoR\nHydro Reservoir":     covRoRDisp,
                        "RoR\nBackup Generator":    covRoRBackup,
                        
                        "Load\nStorage":            covLoadStorage,
                        "Load\nImport-Export":      covLoadLinks,
                        "Load\nHydro Reservoir":    covLoadDisp,
                        "Load\nBackup Generator":   covLoadBackup
                        }
    
    
    covChangeCO2.append(covMismatchSplit)
    

#%%% Networks

varMismatchChangeLinks = []
varChangeLinks = []
varResponseChangeLinks = []
covChangeLinks = []

for networks in filename_links:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    Load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - Load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeLinks.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((Load - np.mean(Load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }

    varChangeLinks.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
   
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                           varStorage,
                        "Import-Export":                     varLinks,
                        "Hydro Reservoir":                   varHydro,
                        "Backup Generator":                  varBackup,
                        "Storage\nImport-Export":            covStorageLinks,
                        "Storage\nHydro Reservoir":          covStorageHydro,
                        "Storage\nBackup Generator":         covStorageBackup,
                        "Import-Export\nHydro Reservoir":    covLinksHydro,
                        "Import-Export\nBackup Generator":   covLinksBackup,
                        "Backup Generator\nHydro Reservoir": covBackupHydro
                        }

    varResponseChangeLinks.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
   
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))

    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
      
    covLoadStorage  = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))

    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":            covWindStorage,
                        "Wind\nlinks":              covWindLinks,
                        "Wind\nHydro Reservoir":    covWindDisp,
                        "Wind\nBackup Generator":   covWindBackup,
                        
                        "Solar PV\nStorage":        covSolarStorage,
                        "Solar PV\nImport-Export":  covSolarLinks,
                        "Solar PV\nhydro":          covSolarDisp,
                        "Solar PV\nbackup":         covSolarBackup,
                        
                        "RoR\nStorage":             covRoRStorage,
                        "RoR\nImport-Export":       covRoRLinks,
                        "RoR\nHydro Reservoir":     covRoRDisp,
                        "RoR\nBackup Generator":    covRoRBackup,
                        
                        "Load\nStorage":            covLoadStorage,
                        "Load\nImport-Export":      covLoadLinks,
                        "Load\nHydro Reservoir":    covLoadDisp,
                        "Load\nBackup Generator":   covLoadBackup
                        }
    
    covChangeLinks.append(covMismatchSplit)
    

#%%% Sort matrix

# CO2 mismatch
names = list(varChangeCO2[0].keys())
sortedMatrixCO2 = MatrixSorter(names,varChangeCO2)
sortedMatrixLinks = MatrixSorter(names,varChangeLinks)

# CO2 response
names = list(varResponseChangeCO2[0].keys())
sortedMatrixResponseCO2 = MatrixSorter(names,varResponseChangeCO2)
sortedMatrixResponseLinks = MatrixSorter(names,varResponseChangeLinks)


# CO2 Covariance
names = list(covChangeCO2[0].keys())
sortedMatrixCovarianceCO2 = MatrixSorter(names,covChangeCO2)
sortedMatrixCovarianceLinks = MatrixSorter(names,covChangeLinks)

#%% Elec_only plot
#%%% Plot Var Mismatch

# quality
dpi = 200

# plot figure
fig = plt.figure(figsize=(10,8),dpi=dpi)

# grid
gs = fig.add_gridspec(21, 4)
axs = []
axs.append( fig.add_subplot(gs[0:5,0:2]) )   # plot 1
axs.append( fig.add_subplot(gs[0:5,2:4]) )   # plot 2
axs.append( fig.add_subplot(gs[8:13,0:2]) )   # plot 3
axs.append( fig.add_subplot(gs[8:13,2:4]) )   # plot 4
axs.append( fig.add_subplot(gs[16:21,0:2]) )   # plot 5
axs.append( fig.add_subplot(gs[16:21,2:4]) )   # plot 6

# Rotation of current
degrees = -12.5


###### CONTRIBUTION ######

for i in range(2):
    i += 0

    # Data for components
    sortedMatrix = []
    if i == 0: 
        sortedMatrix = sortedMatrixCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrix = sortedMatrixLinks
        varMismatchChange = varMismatchChangeLinks
    loadCovData, genCovData, windVarData, SolarVarData, RoRVarnData, loadVarData = ContComponent(sortedMatrix)

    # length af plot
    length = len(varMismatchChange)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(SolarVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(RoRVarnData,color='k',alpha=0.5,linewidth=0.2)
    axs[i].plot(loadVarData,color='k',alpha=0.5,linewidth=0.2)
    axs[i].plot(genCovData,color='k',alpha=0.5,linewidth=0.5)


    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), loadCovData,
                     label='Load\ncovariance',
                     color='slategray',
                     alpha=0.5)
    axs[i].fill_between(range(length), loadCovData, genCovData,
                     label='Generator\ncovariance',
                     color='black',
                     alpha=0.5)
    axs[i].fill_between(range(length), np.zeros(length), windVarData,
                     label='Wind',
                     color='dodgerblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), windVarData, SolarVarData,
                     label='Solar PV',
                     color='gold',
                     alpha=0.5)
    axs[i].fill_between(range(length), SolarVarData, RoRVarnData,
                     label='RoR',
                     color='limegreen',
                     alpha=0.5)
    axs[i].fill_between(range(length), RoRVarnData, loadVarData,
                     label='Load',
                     color='goldenrod',
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-2*1e9,5.2*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    if i == 0:
        axs[i].set_xticks(np.arange(0,7))
        axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    else:
        axs[i].set_xticks(np.arange(0,5))
        axs[i].set_xticklabels(['Zero', 'Current', '2x Current', '4x Current', '6x Current'],rotation=degrees)
    
    # Extra text
    if i == 0:
        axs[i].text(-1.0,1.5*1e9,"Mismatch\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        axs[i].text(3,5.5*1e9,"CO$_2$ Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,5.5*1e9,"(a)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        axs[i].text(2,5.5*1e9,"Transmission Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.6,5.5*1e9,"(b)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.3), # placement relative to the graph
           ncol = 7, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.8, # Horizontal spacing between labels
           labelspacing = 0.75 # Vertical spacing between label
           )

# Space between subplot
plt.subplots_adjust(wspace=0.3, hspace=30)

###### RESPONSE ######

for i in range(2):
    i += 2

    # Data for components
    sortedMatrix = []
    if i == 2: 
        sortedMatrixResponse = sortedMatrixResponseCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrixResponse = sortedMatrixResponseLinks
        varMismatchChange = varMismatchChangeLinks
    StorageVarData, linksVarData, hydroVarData, backupVarData, covData =  ResComponent(sortedMatrixResponse, "elec_only")

    # length af plot
    length = len(varMismatchChange)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(StorageVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(linksVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(backupVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(hydroVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(covData,color='k',alpha=0.5,linewidth=0.5)
    
    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), StorageVarData,
                     label='Storage',
                     color='orange',
                     alpha=0.5)
    axs[i].fill_between(range(length), StorageVarData, linksVarData,
                     label='Import-\nExport',
                     color='darkgreen',
                     alpha=0.5)
    axs[i].fill_between(range(length), linksVarData, hydroVarData,
                     label='Hydro\nReservoir',
                     color='lightblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), hydroVarData, backupVarData,
                     label='Backup\nGenerator',
                     color='darkgray',
                     alpha=0.5)
    axs[i].fill_between(range(length), backupVarData, covData,
                     label='Covariance',
                     color='olive',
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-0.5*1e9,4.2*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    if i == 2:
        axs[i].set_xticks(np.arange(0,7))
        axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    else:
        axs[i].set_xticks(np.arange(0,5))
        axs[i].set_xticklabels(['Zero', 'Current', '2x Current', '4x Current', '6x Current'],rotation=degrees)
    
    # Extra text
    if i == 2:
        axs[i].text(-1.0,2.2*1e9,"Response\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        #axs[i].text(3,5.5*1e9,"CO$_2$ Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,4.5*1e9,"(c)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        #axs[i].text(2,5.5*1e9,"Transmission Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.6,4.5*1e9,"(d)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.3), # placement relative to the graph
           ncol = 7, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.8, # Horizontal spacing between labels
           labelspacing = 0.75 # Vertical spacing between label
           )


###### Covariance ######

color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
         'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','tan',
         'darkgreen','brown','fuchsia','yellow','purple','black',
         'olivedrab','teal','gainsboro']

for i in range(2):
    i += 4

    # Data for components
    sortedMatrix = []
    if i == 4: 
        sortedMatrixCovariance = sortedMatrixCovarianceCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrixCovariance = sortedMatrixCovarianceLinks
        varMismatchChange = varMismatchChangeLinks
    windStorageCov, windLinksCov, windHydroCov, windBackupCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, rorCovData, loadCovData = CovComponent(sortedMatrixCovariance, "elec_only")

    # length af plot
    length = len(varMismatchChange)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(windStorageCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windLinksCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windHydroCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windBackupCov,color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(solarStorageCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarLinksCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarHydroCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarBackupCov,color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(rorCovData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
    
    
    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), windStorageCov,
                     label='Wind/\nStorage',
                     color=color[1],
                     alpha=0.5)
    axs[i].fill_between(range(length), windStorageCov, windLinksCov,
                     label='Wind/\nImport-Export',
                     color=color[2],
                     alpha=0.5)
    axs[i].fill_between(range(length), windLinksCov, windHydroCov,
                     label='Wind/\nHydro Reservoir',
                     color=color[3],
                     alpha=0.5)
    axs[i].fill_between(range(length), windLinksCov, windBackupCov,
                     label='Wind/\nBackup Generator',
                     color=color[4],
                     alpha=0.5)
    axs[i].fill_between(range(length), windBackupCov, solarStorageCov,
                     label='Solar/\nStorage',
                     color=color[7],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarStorageCov, solarLinksCov,
                     label='Solar/\nImport-Export',
                     color=color[8],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarLinksCov, solarHydroCov,
                     label='Solar/\nHydro Reservoir',
                     color=color[9],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarHydroCov, solarBackupCov,
                     label='Solar/\nBackup Generator',
                     color=color[10],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarBackupCov, rorCovData,
                     label='RoR\ncovariance',
                     color=color[13],
                     alpha=0.5)
    axs[i].fill_between(range(length), rorCovData, loadCovData,
                     label='Load\ncovariance',
                     color=color[14],
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-0.5*1e9,4.2*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    if i == 4:
        axs[i].set_xticks(np.arange(0,7))
        axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    else:
        axs[i].set_xticks(np.arange(0,5))
        axs[i].set_xticklabels(['Zero', 'Current', '2x Current', '4x Current', '6x Current'],rotation=degrees)
    
    # Extra text
    if i == 4:
        axs[i].text(-1.0,2.2*1e9,"Covariance\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        #axs[i].text(3,5.5*1e9,"CO$_2$ Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,4.5*1e9,"(e)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        #axs[i].text(2,5.5*1e9,"Transmission Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.6,4.5*1e9,"(f)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.3), # placement relative to the graph
           ncol = 5, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.6, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Save figure
title = "elec_only - Electricity Variance and Cross Correlation"
pathPlot = path + "\\Figures\\elec_only\\Pre Analysis\\"
SavePlot(fig,pathPlot,title)

plt.show(all)

#%% Elec_Heat
#%%% Import Data

# CO2 CONSTRAINTS
# Load data - CO2 constraint

# Load current directory
path = os.path.split(os.path.split(os.getcwd())[0])[0]

# Folder name of data files
directory = path+"\Data\elec_heat\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_heat_0.125_0.6.h5",
                "postnetwork-elec_heat_0.125_0.5.h5",
                "postnetwork-elec_heat_0.125_0.4.h5",
                "postnetwork-elec_heat_0.125_0.3.h5",
                "postnetwork-elec_heat_0.125_0.2.h5",
                "postnetwork-elec_heat_0.125_0.1.h5",
                "postnetwork-elec_heat_0.125_0.05.h5"]

filename_links =   ["postnetwork-elec_heat_0_0.05.h5",
                    "postnetwork-elec_heat_0.0625_0.05.h5",
                    "postnetwork-elec_heat_0.125_0.05.h5",
                    "postnetwork-elec_heat_0.25_0.05.h5",
                    "postnetwork-elec_heat_0.375_0.05.h5"]

# Network
network = pypsa.Network(directory+filename_CO2[-1])


#%%% Define index and columns

# Index
timeIndex = network.loads_t.p_set.index 

# Columns
countryColumn = network.loads.index[:30]

#%%% Networks

varMismatchChangeCO2 = []
varChangeCO2 = []
varResponseChangeCO2 = []
covChangeCO2 = []

for networks in filename_CO2:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    Load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - Load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeCO2.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((Load - np.mean(Load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }
    varChangeCO2.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))
    varHeat = sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))**2, axis=0))
    varCHP = sum(np.mean((response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covStorageHeat      = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covStorageCHP       = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLinksHeat        = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLinksCHP         = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covBackupHeat       = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covBackupCHP        = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHydroHeat        = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covHydroCHP         = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHeatCHP          = 2* sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
      
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                          varStorage,
                        "Import-Export":                    varLinks,
                        "Hydro Reservoir":                  varHydro,
                        "Backup Generator":                 varBackup,
                        "Heat Couple":                      varHeat,
                        "CHP Electric":                     varCHP,
                        "Storage\nImport-Export":           covStorageLinks,
                        "Storage\nHydro":                   covStorageHydro,
                        "Storage\nBackup Generator":        covStorageBackup,
                        "Storage\nHeat Couple":             covStorageHeat,
                        "Storage\nCHP Electric":            covStorageCHP,
                        "Import-Export\nHydro":             covLinksHydro,
                        "Import-Export\nBackup Generator":  covLinksBackup,
                        "Import-Export\nHeat Couple":       covLinksHeat,
                        "Import-Export\nCHP Electric":      covLinksCHP,
                        "Backup Generator\nHydro":          covBackupHydro,
                        "Backup Generator\nHeat Couple":    covBackupHeat,
                        "Backup Generator\nCHP Electric":   covBackupCHP,
                        "hydro\nHeat Couple":               covHydroHeat,
                        "hydro\nCHP Electric":              covHydroCHP,
                        "Heat Couple\nCHP Electric":        covHeatCHP
                        }

    varResponseChangeCO2.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covWindHeat     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covWindCHP      = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covSolarHeat    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covSolarCHP     = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covRoRHeat      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covRoRCHP       = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
        
    covLoadStorage  = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLoadHeat     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLoadCHP      = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
            
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":                covWindStorage,
                        "Wind\nImport-Export":          covWindLinks,
                        "Wind\nhydro":                  covWindDisp,
                        "Wind\nBackup Generator":       covWindBackup,
                        "Wind\nHeat Couple":            covWindHeat,
                        "Wind\nCHP Electric":           covWindCHP,
                        
                        "Solar PV\nStorage":            covSolarStorage,
                        "Solar PV\nImport-Export":      covSolarLinks,
                        "Solar PV\nhydro":              covSolarDisp,
                        "Solar PV\nBackup Generator":   covSolarBackup,
                        "Solar PV\nHeat Couple":          covSolarHeat,
                        "Solar PV\nCHP Electric":       covSolarCHP,
                        
                        "RoR\nStorage":                 covRoRStorage,
                        "RoR\nImport-Export":           covRoRLinks,
                        "RoR\nhydro":                   covRoRDisp,
                        "RoR\nBackup Generator":        covRoRBackup,
                        "RoR\nHeat Couple":             covRoRHeat,
                        "RoR\nCHP Electric":            covRoRCHP,
                        
                        "Load\nStorage":                covLoadStorage,
                        "Load\nImport-Export":          covLoadLinks,
                        "Load\nhydro":                  covLoadDisp,
                        "Load\nBackup Generator":       covLoadBackup,
                        "Load\nHeat Couple":            covLoadHeat,
                        "Load\nCHP Electric":           covLoadCHP
                        }
    
    
    covChangeCO2.append(covMismatchSplit)
    

#%%% Networks

varMismatchChangeLinks = []
varChangeLinks = []
varResponseChangeLinks = []
covChangeLinks = []

for networks in filename_links:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    Load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - Load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeLinks.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((Load - np.mean(Load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }

    varChangeLinks.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))
    varHeat = sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))**2, axis=0))
    varCHP = sum(np.mean((response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covStorageHeat      = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covStorageCHP       = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLinksHeat        = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLinksCHP         = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covBackupHeat       = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covBackupCHP        = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHydroHeat        = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covHydroCHP         = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHeatCHP          = 2* sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                          varStorage,
                        "Import-Export":                    varLinks,
                        "Hydro Reservoir":                  varHydro,
                        "Backup Generator":                 varBackup,
                        "Heat Couple":                      varHeat,
                        "CHP Electric":                     varCHP,
                        "Storage\nImport-Export":           covStorageLinks,
                        "Storage\nHydro":                   covStorageHydro,
                        "Storage\nBackup Generator":        covStorageBackup,
                        "Storage\nHeat Couple":             covStorageHeat,
                        "Storage\nCHP Electric":            covStorageCHP,
                        "Import-Export\nHydro":             covLinksHydro,
                        "Import-Export\nBackup Generator":  covLinksBackup,
                        "Import-Export\nHeat Couple":       covLinksHeat,
                        "Import-Export\nCHP Electric":      covLinksCHP,
                        "Backup Generator\nHydro":          covBackupHydro,
                        "Backup Generator\nHeat Couple":    covBackupHeat,
                        "Backup Generator\nCHP Electric":   covBackupCHP,
                        "hydro\nHeat Couple":               covHydroHeat,
                        "hydro\nCHP Electric":              covHydroCHP,
                        "Heat Couple\nCHP Electric":        covHeatCHP
                        }

    varResponseChangeLinks.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covWindHeat     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covWindCHP      = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covSolarHeat    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covSolarCHP     = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covRoRHeat      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covRoRCHP       = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
        
    covLoadStorage  = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLoadHeat     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLoadCHP      = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
            
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":                covWindStorage,
                        "Wind\nImport-Export":          covWindLinks,
                        "Wind\nhydro":                  covWindDisp,
                        "Wind\nBackup Generator":       covWindBackup,
                        "Wind\nHeat Couple":            covWindHeat,
                        "Wind\nCHP Electric":           covWindCHP,
                        
                        "Solar PV\nStorage":            covSolarStorage,
                        "Solar PV\nImport-Export":      covSolarLinks,
                        "Solar PV\nhydro":              covSolarDisp,
                        "Solar PV\nBackup Generator":   covSolarBackup,
                        "Solar PV\nHeat Couple":          covSolarHeat,
                        "Solar PV\nCHP Electric":       covSolarCHP,
                        
                        "RoR\nStorage":                 covRoRStorage,
                        "RoR\nImport-Export":           covRoRLinks,
                        "RoR\nhydro":                   covRoRDisp,
                        "RoR\nBackup Generator":        covRoRBackup,
                        "RoR\nHeat Couple":             covRoRHeat,
                        "RoR\nCHP Electric":            covRoRCHP,
                        
                        "Load\nStorage":                covLoadStorage,
                        "Load\nImport-Export":          covLoadLinks,
                        "Load\nhydro":                  covLoadDisp,
                        "Load\nBackup Generator":       covLoadBackup,
                        "Load\nHeat Couple":            covLoadHeat,
                        "Load\nCHP Electric":           covLoadCHP
                        }
    
    covChangeLinks.append(covMismatchSplit)
    



#%%% Sort matrix

# CO2 mismatch
names = list(varChangeCO2[0].keys())
sortedMatrixCO2 = MatrixSorter(names,varChangeCO2)
sortedMatrixLinks = MatrixSorter(names,varChangeLinks)

# CO2 response
names = list(varResponseChangeCO2[0].keys())
sortedMatrixResponseCO2 = MatrixSorter(names,varResponseChangeCO2)
sortedMatrixResponseLinks = MatrixSorter(names,varResponseChangeLinks)


# CO2 Covariance
names = list(covChangeCO2[0].keys())
sortedMatrixCovarianceCO2 = MatrixSorter(names,covChangeCO2)
sortedMatrixCovarianceLinks = MatrixSorter(names,covChangeLinks)

#%% Elec_Heat plot
#%%% Plot Var Mismatch

# quality
dpi = 200

# plot figure
fig = plt.figure(figsize=(15,6),dpi=dpi)

# Rotation of current
degrees = -12.5

#%%% Plot Var Mismatch

# quality
dpi = 200

# plot figure
fig = plt.figure(figsize=(10,9),dpi=dpi)

# grid
gs = fig.add_gridspec(46, 4)
axs = []
axs.append( fig.add_subplot(gs[0:10,0:2]) )   # plot 1
axs.append( fig.add_subplot(gs[0:10,2:4]) )   # plot 2
axs.append( fig.add_subplot(gs[17:27,0:2]) )   # plot 3
axs.append( fig.add_subplot(gs[17:27,2:4]) )   # plot 4
axs.append( fig.add_subplot(gs[36:46,0:2]) )   # plot 5
axs.append( fig.add_subplot(gs[36:46,2:4]) )   # plot 6

# Rotation of current
degrees = -12.5


###### CONTRIBUTION ######

for i in range(2):
    i += 0

    # Data for components
    sortedMatrix = []
    if i == 0: 
        sortedMatrix = sortedMatrixCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrix = sortedMatrixLinks
        varMismatchChange = varMismatchChangeLinks
    loadCovData, genCovData, windVarData, SolarVarData, RoRVarnData, loadVarData = ContComponent(sortedMatrix)

    # length af plot
    length = len(varMismatchChange)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(SolarVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(RoRVarnData,color='k',alpha=0.5,linewidth=0.2)
    axs[i].plot(loadVarData,color='k',alpha=0.5,linewidth=0.2)
    axs[i].plot(genCovData,color='k',alpha=0.5,linewidth=0.5)


    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), loadCovData,
                     label='Load\ncovariance',
                     color='slategray',
                     alpha=0.5)
    axs[i].fill_between(range(length), loadCovData, genCovData,
                     label='Generator\ncovariance',
                     color='black',
                     alpha=0.5)
    axs[i].fill_between(range(length), np.zeros(length), windVarData,
                     label='Wind',
                     color='dodgerblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), windVarData, SolarVarData,
                     label='Solar PV',
                     color='gold',
                     alpha=0.5)
    axs[i].fill_between(range(length), SolarVarData, RoRVarnData,
                     label='RoR',
                     color='limegreen',
                     alpha=0.5)
    axs[i].fill_between(range(length), RoRVarnData, loadVarData,
                     label='Load',
                     color='goldenrod',
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-2*1e9,12*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    if i == 0:
        axs[i].set_xticks(np.arange(0,7))
        axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    else:
        axs[i].set_xticks(np.arange(0,5))
        axs[i].set_xticklabels(['Zero', 'Current', '2x Current', '4x Current', '6x Current'],rotation=degrees)
    
    # Extra text
    if i == 0:
        axs[i].text(-1.2,5.5*1e9,"Mismatch\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        axs[i].text(3,12.7*1e9,"CO$_2$ Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,12.7*1e9,"(a)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        axs[i].text(2,12.7*1e9,"Transmission Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.6,12.7*1e9,"(b)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.27), # placement relative to the graph
           ncol = 7, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.6, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Space between subplot
plt.subplots_adjust(wspace=0.3, hspace=30)

###### RESPONSE ######

for i in range(2):
    i += 2

    # Data for components
    sortedMatrix = []
    if i == 2: 
        sortedMatrixResponse = sortedMatrixResponseCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrixResponse = sortedMatrixResponseLinks
        varMismatchChange = varMismatchChangeLinks
    StorageVarData, linksVarData, hydroVarData, backupVarData, heatVarData, CHPVarData, covData  =  ResComponent(sortedMatrixResponse, "elec_heat")

    # length af plot
    length = len(varMismatchChange)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(StorageVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(linksVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(backupVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(hydroVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(heatVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(CHPVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(covData,color='k',alpha=0.5,linewidth=0.5)
    
    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), StorageVarData,
                     label='Storage',
                     color='orange',
                     alpha=0.5)
    axs[i].fill_between(range(length), StorageVarData, linksVarData,
                     label='Import-\nExport',
                     color='darkgreen',
                     alpha=0.5)
    axs[i].fill_between(range(length), linksVarData, hydroVarData,
                     label='Hydro\nReservoir',
                     color='lightblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), hydroVarData, backupVarData,
                     label='Backup\nGenerator',
                     color='darkgray',
                     alpha=0.5)
    axs[i].fill_between(range(length), backupVarData, heatVarData,
                     label='Heat\nCouple',
                     color='mediumblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), heatVarData, CHPVarData,
                     label='CHP Electric',
                     color='aqua',
                     alpha=0.5)
    axs[i].fill_between(range(length), CHPVarData, covData,
                     label='Covariance',
                     color='olive',
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-0.5*1e9,10*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    if i == 2:
        axs[i].set_xticks(np.arange(0,7))
        axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    else:
        axs[i].set_xticks(np.arange(0,5))
        axs[i].set_xticklabels(['Zero', 'Current', '2x Current', '4x Current', '6x Current'],rotation=degrees)
    
    # Extra text
    if i == 2:
        axs[i].text(-1.2,5.5*1e9,"Response\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        #axs[i].text(3,5.5*1e9,"CO$_2$ Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,10.7*1e9,"(c)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        #axs[i].text(2,5.5*1e9,"Transmission Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.6,10.7*1e9,"(d)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.065,-0.27), # placement relative to the graph
           ncol = 6, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.8, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )


###### Covariance ######

color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
         'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','tan',
         'darkgreen','brown','fuchsia','yellow','purple','black',
         'olivedrab','teal','gainsboro']

for i in range(2):
    i += 4

    # Data for components
    sortedMatrix = []
    if i == 4: 
        sortedMatrixCovariance = sortedMatrixCovarianceCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrixCovariance = sortedMatrixCovarianceLinks
        varMismatchChange = varMismatchChangeLinks
    WindStorageCov, WindLinksCov, WindHydroCov, WindBackupCov, WindHeatCov, WindCHPCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarHeatCov, solarCHPCov, RoRCovData, LoadCovData = CovComponent(sortedMatrixCovariance, "elec_heat")

    # length af plot
    length = len(varMismatchChange)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)

    axs[i].plot(WindStorageCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(WindLinksCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(WindHydroCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(WindBackupCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(WindHeatCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(WindCHPCov,color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(solarStorageCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarLinksCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarHydroCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarBackupCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarHeatCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarCHPCov,color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(RoRCovData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(LoadCovData,color='k',alpha=0.5,linewidth=0.5)
    
    
    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), WindStorageCov,
                     label='Wind/\nStorage',
                     color=color[1],
                     alpha=0.5)
    axs[i].fill_between(range(length), WindStorageCov, WindLinksCov,
                     label='Wind/\nImport-Export',
                     color=color[2],
                     alpha=0.5)
    axs[i].fill_between(range(length), WindLinksCov, WindHydroCov,
                     label='Wind/\nHydro Reservoir',
                     color=color[3],
                     alpha=0.5)
    axs[i].fill_between(range(length), WindLinksCov, WindBackupCov,
                     label='Wind/\nBackup Generator',
                     color=color[4],
                     alpha=0.5)
    axs[i].fill_between(range(length), WindBackupCov, WindHeatCov,
                     label='Wind/\Heat Couple',
                     color=color[5],
                     alpha=0.5)
    axs[i].fill_between(range(length), WindHeatCov, WindCHPCov,
                     label='Wind/\nCHP Electric',
                     color=color[6],
                     alpha=0.5)
    axs[i].fill_between(range(length), WindCHPCov, solarStorageCov,
                     label='Solar PV/\nStorage',
                     color=color[7],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarStorageCov, solarLinksCov,
                     label='Solar PV/\nImport-Export',
                     color=color[8],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarLinksCov, solarHydroCov,
                     label='Solar PV/\nHydro Reservoir',
                     color=color[9],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarHydroCov, solarBackupCov,
                     label='Solar PV/\nBackup Generator',
                     color=color[10],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarBackupCov, solarHeatCov,
                     label='Solar PV/\nHeat Couple',
                     color=color[11],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarHeatCov, solarCHPCov,
                     label='Solar PV/\nCHP Electric',
                     color=color[12],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarCHPCov, RoRCovData,
                     label='RoR\ncovariance',
                     color=color[13],
                     alpha=0.5)
    axs[i].fill_between(range(length), RoRCovData, LoadCovData,
                     label='Load\ncovariance',
                     color=color[15],
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-0.5*1e9,10*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    if i == 4:
        axs[i].set_xticks(np.arange(0,7))
        axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    else:
        axs[i].set_xticks(np.arange(0,5))
        axs[i].set_xticklabels(['Zero', 'Current', '2x Current', '4x Current', '6x Current'],rotation=degrees)
    
    # Extra text
    if i == 4:
        axs[i].text(-1.2,5.5*1e9,"Covariance\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        #axs[i].text(3,5.5*1e9,"CO$_2$ Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,10.7*1e9,"(e)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        #axs[i].text(2,5.5*1e9,"Transmission Constrain",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.6,10.7*1e9,"(f)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.27), # placement relative to the graph
           ncol = 5, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.4, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Save figure
title = "elec_heat - Electricity Variance and Cross Correlation"
pathPlot = path + "\\Figures\\elec_heat\\Pre Analysis\\"
SavePlot(fig,pathPlot,title)

plt.show(all)

#%% Elec_v2g50
#%%% Import Data

# CO2 CONSTRAINTS
# Load data - CO2 constraint

# Load current directory
path = os.path.split(os.path.split(os.getcwd())[0])[0]

# Folder name of data files
directory = path+"\Data\elec_v2g50\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_v2g50_0.125_0.6.h5",
                "postnetwork-elec_v2g50_0.125_0.5.h5",
                "postnetwork-elec_v2g50_0.125_0.4.h5",
                "postnetwork-elec_v2g50_0.125_0.3.h5",
                "postnetwork-elec_v2g50_0.125_0.2.h5",
                "postnetwork-elec_v2g50_0.125_0.1.h5",
                "postnetwork-elec_v2g50_0.125_0.05.h5"]

# Network
network = pypsa.Network(directory+filename_CO2[-1])


#%%% Define index and columns

# Index
timeIndex = network.loads_t.p_set.index 

# Columns
countryColumn = network.loads.index[:30]

#%%% Networks

varMismatchChangeCO2 = []
varChangeCO2 = []
varResponseChangeCO2 = []
covChangeCO2 = []

for networks in filename_CO2:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    Load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - Load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeCO2.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((Load - np.mean(Load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }

    varChangeCO2.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))
    varTrans = sum(np.mean((response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covStorageTrans     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
    
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLinksTrans       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
   
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covBackupTrans      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
   
    covHydroTrans       = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
    
   
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                              varStorage,
                        "Import-Export":                        varLinks,
                        "Hydro Reservoir":                      varHydro,
                        "Backup Generator":                     varBackup,
                        "Transport Couple":                     varTrans,
                        "Storage\nImport-Export":               covStorageLinks,
                        "Storage\nHydro Reservoir":             covStorageHydro,
                        "Storage\nBackup Generator":            covStorageBackup,
                        "Storage\nTransport Couple":            covStorageTrans,
                        "Import-Export\nHydro Reservoir":       covLinksHydro,
                        "Import-Export\nBackup Generator":      covLinksBackup,
                        "Import-Export\nTransport Couple":      covLinksTrans,
                        "Backup Generator\nHydro Reservoir":    covBackupHydro,
                        "Backup Generator\nTransport Couple":   covBackupTrans,
                        "Hydro Reservoir\nTransport Couple":    covHydroTrans
                        }

    varResponseChangeCO2.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covWindTrans    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
 
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covSolarTrans   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
   
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covRoRTrans     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
      
    covLoadStorage  = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLoadTrans    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
       
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":                covWindStorage,
                        "Wind\nImport-Export":          covWindLinks,
                        "Wind\nhydro":                  covWindDisp,
                        "Wind\nBackup Generator":       covWindBackup,
                        "Wind\nTransport Couple":       covWindTrans,
                        
                        "Solar PV\nStorage":            covSolarStorage,
                        "Solar PV\nImport-Export":      covSolarLinks,
                        "Solar PV\nhydro":              covSolarDisp,
                        "Solar PV\nBackup Generator":   covSolarBackup,
                        "solar\nTransport Couple":      covSolarTrans,
                        
                        "RoR\nStorage":                 covRoRStorage,
                        "RoR\nImport-Export":           covRoRLinks,
                        "RoR\nhydro":                   covRoRDisp,
                        "RoR\nBackup Generator":        covRoRBackup,
                        "RoR\nTransport Couple":        covRoRTrans,
                       
                        "Load\nStorage":                covLoadStorage,
                        "Load\nImport-Export":          covLoadLinks,
                        "Load\nhydro":                  covLoadDisp,
                        "Load\nBackup Generator":       covLoadBackup,
                        "Load\nTransport Couple":       covLoadTrans
                        }
    
    
    covChangeCO2.append(covMismatchSplit)
    
#%%% Sort matrix

# CO2 mismatch
names = list(varChangeCO2[0].keys())
sortedMatrixCO2 = MatrixSorter(names,varChangeCO2)

# CO2 response
names = list(varResponseChangeCO2[0].keys())
sortedMatrixResponseCO2 = MatrixSorter(names,varResponseChangeCO2)

# CO2 Covariance
names = list(covChangeCO2[0].keys())
sortedMatrixCovarianceCO2 = MatrixSorter(names,covChangeCO2)

#%% Elec_v2g50 plot
#%%% Plot Var Mismatch

# quality
dpi = 200

# plot figure
fig = plt.figure(figsize=(10,6),dpi=dpi)

# grid
gs = fig.add_gridspec(10, 6)
axs = []
axs.append( fig.add_subplot(gs[0:4,0:3]) )   # plot 1
axs.append( fig.add_subplot(gs[0:4,3:6]) )   # plot 2
axs.append( fig.add_subplot(gs[6:10,1:5]) )   # plot 3

# Rotation of current
degrees = -12.5


###### CONTRIBUTION ######


# Data for components
i=0
sortedMatrix = []
sortedMatrix = sortedMatrixCO2 
varMismatchChange = varMismatchChangeCO2
loadCovData, genCovData, windVarData, SolarVarData, RoRVarnData, loadVarData = ContComponent(sortedMatrix)

# length af plot
length = len(varMismatchChange)

# plot
axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(windVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(SolarVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(RoRVarnData,color='k',alpha=0.5,linewidth=0.2)
axs[i].plot(loadVarData,color='k',alpha=0.5,linewidth=0.2)
axs[i].plot(genCovData,color='k',alpha=0.5,linewidth=0.5)


# Fill lines
axs[i].fill_between(range(length), np.zeros(length), windVarData,
                 label='Wind',
                 color='dodgerblue',
                 alpha=0.5)
axs[i].fill_between(range(length), windVarData, SolarVarData,
                 label='Solar PV',
                 color='gold',
                 alpha=0.5)
axs[i].fill_between(range(length), SolarVarData, RoRVarnData,
                 label='RoR',
                 color='limegreen',
                 alpha=0.5)
axs[i].fill_between(range(length), RoRVarnData, loadVarData,
                 label='Load',
                 color='goldenrod',
                 alpha=0.5)
axs[i].fill_between(range(length), np.zeros(length), loadCovData,
                 label='Load\ncovariance',
                 color='slategray',
                 alpha=0.5)
axs[i].fill_between(range(length), loadCovData, genCovData,
                 label='Generator\ncovariance',
                 color='black',
                 alpha=0.5)


# Mismatch variance
axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

# Y axis
axs[i].set(ylim = [-2.2*1e9,10*1e9])
axs[i].tick_params(axis='both',
                   labelsize=10)
axs[i].yaxis.offsetText.set_fontsize(10)
    
# X axis
axs[i].set_xticks(np.arange(0,7))
axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    
# Extra text
#axs[i].text(-1.2,5.5*1e9,"Mismatch\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
axs[i].text(3,10.7*1e9,"Mismatch Variance",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
axs[i].text(-0.8,10.7*1e9,"(a)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (0.5,-0.15), # placement relative to the graph
           ncol = 3, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 2, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Space between subplot
plt.subplots_adjust(wspace=0.5, hspace=3000)

###### RESPONSE ######


# Data for components
i = 1
sortedMatrix = []
sortedMatrixResponse = sortedMatrixResponseCO2 
varMismatchChange = varMismatchChangeCO2
StorageVarData, linksVarData, hydroVarData, backupVarData, transVarData, covData   =  ResComponent(sortedMatrixResponse, "elec_v2g50")

# length af plot
length = len(varMismatchChange)

# plot
axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(StorageVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(linksVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(backupVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(hydroVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(transVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(covData,color='k',alpha=0.5,linewidth=0.5)

# Fill lines
axs[i].fill_between(range(length), np.zeros(length), StorageVarData,
                 label='Storage',
                 color='orange',
                 alpha=0.5)
axs[i].fill_between(range(length), transVarData, covData,
                 label='Covariance',
                 color='olive',
                 alpha=0.5)
axs[i].fill_between(range(length), StorageVarData, linksVarData,
                 label='Import-\nExport',
                 color='darkgreen',
                 alpha=0.5)
axs[i].fill_between(range(length), linksVarData, hydroVarData,
                 label='Hydro\nReservoir',
                 color='lightblue',
                 alpha=0.5)
axs[i].fill_between(range(length), hydroVarData, backupVarData,
                 label='Backup\nGenerator',
                 color='darkgray',
                 alpha=0.5)
axs[i].fill_between(range(length), backupVarData, transVarData,
                 label='Transport\nCouple',
                 color='lawngreen',
                 alpha=0.5)


# Mismatch variance
axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

# Y axis
axs[i].set(ylim = [-0.5*1e9,8*1e9])
axs[i].tick_params(axis='both',
                   labelsize=10)
axs[i].yaxis.offsetText.set_fontsize(10)
    
# X axis
axs[i].set_xticks(np.arange(0,7))
axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    
# Extra text
#axs[i].text(-1.2,5.5*1e9,"Response\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
axs[i].text(3,8.5*1e9,"Response Variance",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
axs[i].text(-0.8,8.5*1e9,"(b)",rotation="horizontal",fontsize=12, fontweight="bold")
      

    
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (0.5,-0.15), # placement relative to the graph
           ncol = 3, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.4, # Horizontal spacing between labels
           labelspacing = 0.5, # Vertical spacing between label
           )

#axs[i].get_legend()._legend_box.align = "left"


###### Covariance ######

color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
         'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','tan',
         'darkgreen','brown','fuchsia','yellow','purple','black',
         'olivedrab','teal','gainsboro']


# Data for components
i = 2
sortedMatrix = []
sortedMatrixCovariance = sortedMatrixCovarianceCO2 
varMismatchChange = varMismatchChangeCO2
WindStorageCov, WindLinksCov, WindHydroCov, WindBackupCov, WindTransCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarTransCov, RoRCovData, LoadCovData = CovComponent(sortedMatrixCovariance, "elec_v2g50")

# length af plot
length = len(varMismatchChange)

# plot
axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)

axs[i].plot(WindStorageCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindLinksCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindHydroCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindBackupCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindTransCov,color='k',alpha=0.5,linewidth=0.5)

axs[i].plot(solarStorageCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarLinksCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarHydroCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarBackupCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarTransCov,color='k',alpha=0.5,linewidth=0.5)

axs[i].plot(RoRCovData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(LoadCovData,color='k',alpha=0.5,linewidth=0.5)


# Fill lines
axs[i].fill_between(range(length), np.zeros(length), WindStorageCov,
                 label='Wind/\nStorage',
                 color=color[1],
                 alpha=0.5)
axs[i].fill_between(range(length), WindStorageCov, WindLinksCov,
                 label='Wind/\nImport-Export',
                 color=color[2],
                 alpha=0.5)
axs[i].fill_between(range(length), WindLinksCov, WindHydroCov,
                 label='Wind/\nHydro Reservoir',
                 color=color[3],
                 alpha=0.5)
axs[i].fill_between(range(length), WindLinksCov, WindBackupCov,
                 label='Wind/\nBackup Generator',
                 color=color[4],
                 alpha=0.5)
axs[i].fill_between(range(length), WindBackupCov, WindTransCov,
                 label='Wind/\nTransport Couple',
                 color=color[5],
                 alpha=0.5)
axs[i].fill_between(range(length), WindTransCov, solarStorageCov,
                 label='Solar PV/\nStorage',
                 color=color[7],
                 alpha=0.5)
axs[i].fill_between(range(length), solarStorageCov, solarLinksCov,
                 label='Solar PV/\nImport-Export',
                 color=color[8],
                 alpha=0.5)
axs[i].fill_between(range(length), solarLinksCov, solarHydroCov,
                 label='Solar PV/\nHydro Reservoir',
                 color=color[9],
                 alpha=0.5)
axs[i].fill_between(range(length), solarHydroCov, solarBackupCov,
                 label='Solar PV/\nBackup Generator',
                 color=color[10],
                 alpha=0.5)
axs[i].fill_between(range(length), solarBackupCov, solarTransCov,
                 label='Solar PV/\nTransport Couple',
                 color=color[11],
                 alpha=0.5)
axs[i].fill_between(range(length), solarTransCov, RoRCovData,
                 label='RoR\ncovariance',
                 color=color[13],
                 alpha=0.5)
axs[i].fill_between(range(length), RoRCovData, LoadCovData,
                 label='Load\ncovariance',
                 color=color[15],
                 alpha=0.5)

# Mismatch variance
axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

# Y axis
axs[i].set(ylim = [-0.5*1e9,8*1e9])
axs[i].tick_params(axis='both',
                   labelsize=10)
axs[i].yaxis.offsetText.set_fontsize(10)
    
# X axis
axs[i].set_xticks(np.arange(0,7))
axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    
# Extra text
#axs[i].text(-1.2,5.5*1e9,"Covariance\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
axs[i].text(3,8.5*1e9,"Covariance Variance",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
axs[i].text(-0.7,8.5*1e9,"(c)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (0.47,-0.125), # placement relative to the graph
           ncol = 5, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.5, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Save figure
title = "elec_v2g50 - Electricity Variance and Cross Correlation"
pathPlot = path + "\\Figures\\elec_v2g50\\Pre Analysis\\"
SavePlot(fig,pathPlot,title)

plt.show(all)


#%% Elec_Heat_v2g50
#%%% Import Data

# CO2 CONSTRAINTS
# Load data - CO2 constraint

# Load current directory
path = os.path.split(os.path.split(os.getcwd())[0])[0]

# Folder name of data files
directory = path+"\Data\elec_heat_v2g50\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

# Network
network = pypsa.Network(directory+filename_CO2[-1])


#%%% Define index and columns

# Index
timeIndex = network.loads_t.p_set.index 

# Columns
countryColumn = network.loads.index[:30]

#%%% Networks

varMismatchChangeCO2 = []
varChangeCO2 = []
varResponseChangeCO2 = []
covChangeCO2 = []

for networks in filename_CO2:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    Load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - Load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeCO2.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((Load - np.mean(Load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-Load - np.mean(-Load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":        varWind,
                        "solar":        varSolar,
                        "RoR":          varRoR,
                        "Load":         varLoad,
                        "Wind\nsolar":  covWindSolar,
                        "Wind\nRoR":    covWindRoR,
                        "solar\nRoR":   covSolarRoR,
                        "Wind\nLoad":   covWindLoad,
                        "solar\nLoad":  covSolarLoad,
                        "RoR\nLoad":    covRoRLoad
                        }

    varChangeCO2.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))
    varHeat = sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))**2, axis=0))
    varCHP = sum(np.mean((response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0))**2, axis=0))
    varTrans = sum(np.mean((response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covStorageHeat      = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covStorageCHP       = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covStorageTrans     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
    
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLinksHeat        = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLinksCHP         = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covLinksTrans       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
   
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covBackupHeat       = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covBackupCHP        = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covBackupTrans      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
   
    covHydroHeat        = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covHydroCHP         = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covHydroTrans       = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
    
    covHeatCHP          = 2* sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covHeatTrans        = 2* sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
    
    covCHPTrans         = 2* sum(np.mean((response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                              varStorage,
                        "Import-Export":                        varLinks,
                        "Hydro Reservoir":                      varHydro,
                        "Backup Generator":                     varBackup,
                        "Heat Couple":                          varHeat,
                        "CHP Electic":                          varCHP,
                        "Transport Couple":                     varTrans,
                        "Storage\nImport-Export":               covStorageLinks,
                        "Storage\nHydro Reservoir":             covStorageHydro,
                        "Storage\nBackup Generator":            covStorageBackup,
                        "Storage\nHeat Couple":                 covStorageHeat,
                        "Storage\nCHP Electic":                 covStorageCHP,
                        "Storage\nTransport Couple":            covStorageTrans,
                        "Import-Export\nHydro Reservoir":       covLinksHydro,
                        "Import-Export\nBackup Generator":      covLinksBackup,
                        "Import-Export\nHeat Couple":           covLinksHeat,
                        "Import-Export\nCHP Electic":           covLinksCHP,
                        "Import-Export\nTransport Couple":      covLinksTrans,
                        "Backup Generator\nHydro":              covBackupHydro,
                        "Backup Generator\nHeat Couple":        covBackupHeat,
                        "Backup Generator\nCHP Electic":        covBackupCHP,
                        "Backup Generator\nTransport Couple":   covBackupTrans,
                        "Hydro Reservoir\nHydro":               covHydroHeat,
                        "Hydro Reservoir\nHeat Couple":         covHydroCHP,
                        "Hydro Reservoir\nTransport Couple":    covHydroTrans,
                        "Heat Couple\nCHP Electic":             covHeatCHP,
                        "Heat Couple\nTransport Couple":        covHeatTrans,
                        "CHP Electric\nTransport Couple":       covCHPTrans,
                        
                        }
    
    varResponseChangeCO2.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covWindHeat     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covWindCHP      = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))    
    covWindTrans    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
 
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covSolarHeat    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covSolarCHP     = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covSolarTrans   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
   
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covRoRHeat      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covRoRCHP       = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covRoRTrans     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
      
    covLoadStorage  = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLoadHeat     = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLoadCHP      = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    covLoadTrans    = - sum(np.mean((-Load - np.mean(-Load, axis=0))*(response["Transport Couple"] - np.mean(response["Transport Couple"], axis=0)), axis=0))
       
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":                covWindStorage,
                        "Wind\nImport-Export":          covWindLinks,
                        "Wind\nHydro Reservoir":        covWindDisp,
                        "Wind\nBackup Generator":       covWindBackup,
                        "Wind\nHeat Couple":            covWindHeat,
                        "Wind\nCHP Electric":           covWindCHP,
                        "Wind\nTransport Couple":       covWindTrans,
                        
                        "Solar PV\nStorage":            covSolarStorage,
                        "Solar PV\nImport-Export":      covSolarLinks,
                        "Solar PV\nHydro Reservoir":    covSolarDisp,
                        "Solar PV\nBackup Generator":   covSolarBackup,
                        "Solar PV\nHeat Couple":        covSolarHeat,
                        "Solar PV\nCHP Electric":           covSolarCHP,
                        "solar\nTransport Couple":      covSolarTrans,
                        
                        "RoR\nStorage":                 covRoRStorage,
                        "RoR\nImport-Export":           covRoRLinks,
                        "RoR\nHydro Reservoir":         covRoRDisp,
                        "RoR\nBackup Generator":        covRoRBackup,
                        "RoR\nHeat Couple":             covRoRHeat,
                        "RoR\nCHP Electric":           covRoRCHP,
                        "RoR\nTransport Couple":        covRoRTrans,
                       
                        "Load\nStorage":                covLoadStorage,
                        "Load\nImport-Export":          covLoadLinks,
                        "Load\nHydro Reservoir":        covLoadDisp,
                        "Load\nBackup Generator":       covLoadBackup,
                        "Load\nHeat Couple":            covLoadHeat,
                        "Load\nCHP Electric":           covLoadCHP,
                        "Load\nTransport Couple":       covLoadTrans
                        }
    
    
    covChangeCO2.append(covMismatchSplit)
    
#%%% Sort matrix

# CO2 mismatch
names = list(varChangeCO2[0].keys())
sortedMatrixCO2 = MatrixSorter(names,varChangeCO2)

# CO2 response
names = list(varResponseChangeCO2[0].keys())
sortedMatrixResponseCO2 = MatrixSorter(names,varResponseChangeCO2)

# CO2 Covariance
names = list(covChangeCO2[0].keys())
sortedMatrixCovarianceCO2 = MatrixSorter(names,covChangeCO2)

#%% Elec_Heat_v2g50 plot
#%%% Plot Var Mismatch

# quality
dpi = 200

# plot figure
fig = plt.figure(figsize=(10,6),dpi=dpi)

# grid
gs = fig.add_gridspec(21, 6)
axs = []
axs.append( fig.add_subplot(gs[0:7,0:3]) )   # plot 1
axs.append( fig.add_subplot(gs[0:7,3:6]) )   # plot 2
axs.append( fig.add_subplot(gs[14:21,1:5]) )   # plot 3

# Rotation of current
degrees = -12.5


###### CONTRIBUTION ######


# Data for components
i=0
sortedMatrix = []
sortedMatrix = sortedMatrixCO2 
varMismatchChange = varMismatchChangeCO2
loadCovData, genCovData, windVarData, SolarVarData, RoRVarnData, loadVarData = ContComponent(sortedMatrix)

# length af plot
length = len(varMismatchChange)

# plot
axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(windVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(SolarVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(RoRVarnData,color='k',alpha=0.5,linewidth=0.2)
axs[i].plot(loadVarData,color='k',alpha=0.5,linewidth=0.2)
axs[i].plot(genCovData,color='k',alpha=0.5,linewidth=0.5)


# Fill lines
axs[i].fill_between(range(length), np.zeros(length), loadCovData,
                 label='Load\ncovariance',
                 color='slategray',
                 alpha=0.5)
axs[i].fill_between(range(length), loadCovData, genCovData,
                 label='Generator\ncovariance',
                 color='black',
                 alpha=0.5)
axs[i].fill_between(range(length), np.zeros(length), windVarData,
                 label='Wind',
                 color='dodgerblue',
                 alpha=0.5)
axs[i].fill_between(range(length), windVarData, SolarVarData,
                 label='Solar PV',
                 color='gold',
                 alpha=0.5)
axs[i].fill_between(range(length), SolarVarData, RoRVarnData,
                 label='RoR',
                 color='limegreen',
                 alpha=0.5)
axs[i].fill_between(range(length), RoRVarnData, loadVarData,
                 label='Load',
                 color='goldenrod',
                 alpha=0.5)


# Mismatch variance
axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

# Y axis
axs[i].set(ylim = [-3.5*1e9,16.5*1e9])
axs[i].tick_params(axis='both',
                   labelsize=10)
axs[i].yaxis.offsetText.set_fontsize(10)
    
# X axis
axs[i].set_xticks(np.arange(0,7))
axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    
# Extra text
#axs[i].text(-1.2,5.5*1e9,"Mismatch\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
axs[i].text(3,17.3*1e9,"Mismatch Variance",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
axs[i].text(-0.8,17.3*1e9,"(a)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (0.5,-0.16), # placement relative to the graph
           ncol = 3, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 2, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Space between subplot
plt.subplots_adjust(wspace=0.5, hspace=3000)

###### RESPONSE ######


# Data for components
i = 1
sortedMatrix = []
sortedMatrixResponse = sortedMatrixResponseCO2 
varMismatchChange = varMismatchChangeCO2
StorageVarData, linksVarData, hydroVarData, backupVarData, heatVarData, CHPVarData, transVarData, covData   =  ResComponent(sortedMatrixResponse, "elec_heat_v2g50")

# length af plot
length = len(varMismatchChange)

# plot
axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(StorageVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(linksVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(backupVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(hydroVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(heatVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(CHPVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(transVarData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(covData,color='k',alpha=0.5,linewidth=0.5)

# Fill lines
axs[i].fill_between(range(length), np.zeros(length), StorageVarData,
                 label='Storage',
                 color='orange',
                 alpha=0.5)
axs[i].fill_between(range(length), StorageVarData, linksVarData,
                 label='Import-\nExport',
                 color='darkgreen',
                 alpha=0.5)
axs[i].fill_between(range(length), linksVarData, hydroVarData,
                 label='Hydro\nTeservoir',
                 color='lightblue',
                 alpha=0.5)
axs[i].fill_between(range(length), hydroVarData, backupVarData,
                 label='Backup\nGenerator',
                 color='darkgray',
                 alpha=0.5)
axs[i].fill_between(range(length), backupVarData, heatVarData,
                 label='Heat\nCouple',
                 color='mediumblue',
                 alpha=0.5)
axs[i].fill_between(range(length), heatVarData, CHPVarData,
                 label='CHP Electric',
                 color='aqua',
                 alpha=0.5)
axs[i].fill_between(range(length), CHPVarData, transVarData,
                 label='Transport\nCouple',
                 color='lawngreen',
                 alpha=0.5)
axs[i].fill_between(range(length), transVarData, covData,
                 label='Covariance',
                 color='olive',
                 alpha=0.5)


# Mismatch variance
axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

# Y axis
axs[i].set(ylim = [-0.5*1e9,14*1e9])
axs[i].tick_params(axis='both',
                   labelsize=10)
axs[i].yaxis.offsetText.set_fontsize(10)
    
# X axis
axs[i].set_xticks(np.arange(0,7))
axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    
# Extra text
#axs[i].text(-1.2,5.5*1e9,"Response\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
axs[i].text(3,14.5*1e9,"Response Variance",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
axs[i].text(-0.8,14.5*1e9,"(b)",rotation="horizontal",fontsize=12, fontweight="bold")
      

    
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (0.5,-0.16), # placement relative to the graph
           ncol = 3, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.4, # Horizontal spacing between labels
           labelspacing = 0.5, # Vertical spacing between label
           )

#axs[i].get_legend()._legend_box.align = "left"


###### Covariance ######

color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
         'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','tan',
         'darkgreen','brown','fuchsia','yellow','purple','black',
         'olivedrab','teal','gainsboro']


# Data for components
i = 2
sortedMatrix = []
sortedMatrixCovariance = sortedMatrixCovarianceCO2 
varMismatchChange = varMismatchChangeCO2
WindStorageCov, WindLinksCov, WindHydroCov, WindBackupCov, WindHeatCov, WindCHPCov, WindTransCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarHeatCov, solarCHPCov, solarTransCov, RoRCovData, LoadCovData = CovComponent(sortedMatrixCovariance, "elec_heat_v2g50")

# length af plot
length = len(varMismatchChange)

# plot
axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)

axs[i].plot(WindStorageCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindLinksCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindHydroCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindBackupCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindHeatCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindCHPCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(WindTransCov,color='k',alpha=0.5,linewidth=0.5)

axs[i].plot(solarStorageCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarLinksCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarHydroCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarBackupCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarHeatCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarCHPCov,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(solarTransCov,color='k',alpha=0.5,linewidth=0.5)

axs[i].plot(RoRCovData,color='k',alpha=0.5,linewidth=0.5)
axs[i].plot(LoadCovData,color='k',alpha=0.5,linewidth=0.5)


# Fill lines
axs[i].fill_between(range(length), np.zeros(length), WindStorageCov,
                 label='Wind/\nStorage',
                 color=color[1],
                 alpha=0.5)
axs[i].fill_between(range(length), WindStorageCov, WindLinksCov,
                 label='Wind/\nImport-Export',
                 color=color[2],
                 alpha=0.5)
axs[i].fill_between(range(length), WindLinksCov, WindHydroCov,
                 label='Wind/\nHydro Reservoir',
                 color=color[3],
                 alpha=0.5)
axs[i].fill_between(range(length), WindHydroCov, WindBackupCov,
                 label='Wind/\nBackup Generator',
                 color=color[4],
                 alpha=0.5)
axs[i].fill_between(range(length), WindBackupCov, WindHeatCov,
                 label='Wind/\nHeat Couple',
                 color=color[5],
                 alpha=0.5)
axs[i].fill_between(range(length), WindHeatCov, WindCHPCov,
                 label='Wind/\nCHP Electric',
                 color=color[6],
                 alpha=0.5)
axs[i].fill_between(range(length), WindCHPCov, WindTransCov,
                 label='Wind/\nTransport Couple',
                 color=color[7],
                 alpha=0.5)
axs[i].fill_between(range(length), WindTransCov, solarStorageCov,
                 label='Solar PV/\nStorage',
                 color=color[8],
                 alpha=0.5)
axs[i].fill_between(range(length), solarStorageCov, solarLinksCov,
                 label='Solar PV/\nImport-Export',
                 color=color[9],
                 alpha=0.5)
axs[i].fill_between(range(length), solarLinksCov, solarHydroCov,
                 label='Solar PV/\nHydro Reservoir',
                 color=color[10],
                 alpha=0.5)
axs[i].fill_between(range(length), solarHydroCov, solarBackupCov,
                 label='Solar PV/\nBackup Generator',
                 color=color[11],
                 alpha=0.5)
axs[i].fill_between(range(length), solarBackupCov, solarHeatCov,
                 label='Solar PV/\nHeat Couple',
                 color=color[12],
                 alpha=0.5)
axs[i].fill_between(range(length), solarHeatCov, solarCHPCov,
                 label='Solar PV/\nCHP Electric',
                 color=color[13],
                 alpha=0.5)
axs[i].fill_between(range(length), solarCHPCov, solarTransCov,
                 label='Solar PV/\nTransport Couple',
                 color=color[15],
                 alpha=0.5)

axs[i].fill_between(range(length), solarTransCov, RoRCovData,
                 label='RoR/\ncovariance',
                 color=color[16],
                 alpha=0.5)
axs[i].fill_between(range(length), RoRCovData, LoadCovData,
                 label='Load/\ncovariance',
                 color=color[17],
                 alpha=0.5)

# Mismatch variance
axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

# Y axis
axs[i].set(ylim = [-0.5*1e9,14*1e9])
axs[i].tick_params(axis='both',
                   labelsize=10)
axs[i].yaxis.offsetText.set_fontsize(10)
    
# X axis
axs[i].set_xticks(np.arange(0,7))
axs[i].set_xticklabels(['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
    
# Extra text
#axs[i].text(-1.2,5.5*1e9,"Covariance\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
axs[i].text(3,14.5*1e9,"Covariance Variance",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
axs[i].text(-0.7,14.5*1e9,"(c)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (0.47,-0.16), # placement relative to the graph
           ncol = 5, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 0.6, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Save figure
title = "elec_heat_v2g50 - Electricity Variance and Cross Correlation"
pathPlot = path + "\\Figures\\elec_heat_v2g50\\Pre Analysis\\"
SavePlot(fig,pathPlot,title)

plt.show(all)


#%% brownfield_Heat
#%%% Import Data

# CO2 CONSTRAINTS
# Load data - CO2 constraint

# Load current directory
path = os.path.split(os.path.split(os.getcwd())[0])[0]

# Folder name of data files
directory = path + "\\Data\\brownfield_heat\\version-Base\\postnetworks\\"

# Name of file (must be in correct folder location)
filenamesGo = ["postnetwork-go_TYNDP_2020.nc",
               "postnetwork-go_TYNDP_2025.nc",
               "postnetwork-go_TYNDP_2030.nc",
               "postnetwork-go_TYNDP_2035.nc",
               "postnetwork-go_TYNDP_2040.nc",
               "postnetwork-go_TYNDP_2045.nc",
               "postnetwork-go_TYNDP_2050.nc"]

filenamesWait = ["postnetwork-wait_TYNDP_2020.nc",
                 "postnetwork-wait_TYNDP_2025.nc",
                 "postnetwork-wait_TYNDP_2030.nc",
                 "postnetwork-wait_TYNDP_2035.nc",
                 "postnetwork-wait_TYNDP_2040.nc",
                 "postnetwork-wait_TYNDP_2045.nc",
                 "postnetwork-wait_TYNDP_2050.nc"]

# Network
network = pypsa.Network(directory+filenamesGo[-1])

#%%% Define index and columns

# Index
timeIndex = network.loads_t.p_set.index 

# Columns
countryColumn = network.loads.index[:30]

#%%% Networks

varMismatchChangeCO2 = []
varChangeCO2 = []
varResponseChangeCO2 = []
covChangeCO2 = []

for networks in filenamesGo:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeCO2.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((load - np.mean(load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-load - np.mean(-load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-load - np.mean(-load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-load - np.mean(-load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }

    varChangeCO2.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))
    varHeat = sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))**2, axis=0))
    varCHP = sum(np.mean((response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covStorageHeat      = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covStorageCHP       = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLinksHeat        = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLinksCHP         = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covBackupHeat       = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covBackupCHP        = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHydroHeat        = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covHydroCHP         = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHeatCHP          = 2* sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                          varStorage,
                        "Import-Export":                    varLinks,
                        "Hydro Reservoir":                  varHydro,
                        "Backup Generator":                 varBackup,
                        "Heat Couple":                      varHeat,
                        "CHP Electric":                     varCHP,
                        "Storage\nImport-Export":           covStorageLinks,
                        "Storage\nHydro":                   covStorageHydro,
                        "Storage\nBackup Generator":        covStorageBackup,
                        "Storage\nHeat Couple":             covStorageHeat,
                        "Storage\nCHP Electric":            covStorageCHP,
                        "Import-Export\nHydro":             covLinksHydro,
                        "Import-Export\nBackup Generator":  covLinksBackup,
                        "Import-Export\nHeat Couple":       covLinksHeat,
                        "Import-Export\nCHP Electric":      covLinksCHP,
                        "Backup Generator\nHydro":          covBackupHydro,
                        "Backup Generator\nHeat Couple":    covBackupHeat,
                        "Backup Generator\nCHP Electric":   covBackupCHP,
                        "hydro\nHeat Couple":               covHydroHeat,
                        "hydro\nCHP Electric":              covHydroCHP,
                        "Heat Couple\nCHP Electric":        covHeatCHP
                        }

    varResponseChangeCO2.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covWindHeat     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covWindCHP      = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covSolarHeat    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covSolarCHP     = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covRoRHeat      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covRoRCHP       = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
        
    covLoadStorage  = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLoadHeat     = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLoadCHP      = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
         
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":                covWindStorage,
                        "Wind\nImport-Export":          covWindLinks,
                        "Wind\nhydro":                  covWindDisp,
                        "Wind\nBackup Generator":       covWindBackup,
                        "Wind\nHeat Couple":            covWindHeat,
                        "Wind\nCHP Electric":           covWindCHP,
                        
                        "Solar PV\nStorage":            covSolarStorage,
                        "Solar PV\nImport-Export":      covSolarLinks,
                        "Solar PV\nhydro":              covSolarDisp,
                        "Solar PV\nBackup Generator":   covSolarBackup,
                        "Solar PV\nHeat Couple":        covSolarHeat,
                        "Solar PV\nCHP Electric":       covSolarCHP,
                        
                        "RoR\nStorage":                 covRoRStorage,
                        "RoR\nImport-Export":           covRoRLinks,
                        "RoR\nhydro":                   covRoRDisp,
                        "RoR\nBackup Generator":        covRoRBackup,
                        "RoR\nHeat Couple":             covRoRHeat,
                        "RoR\nCHP Electric":            covRoRCHP,
                        
                        "Load\nStorage":                covLoadStorage,
                        "Load\nImport-Export":          covLoadLinks,
                        "Load\nhydro":                  covLoadDisp,
                        "Load\nBackup Generator":       covLoadBackup,
                        "Load\nHeat Couple":            covLoadHeat,
                        "Load\nCHP Electric":           covLoadCHP
                        }
    
    
    covChangeCO2.append(covMismatchSplit)
    

#%%% Networks

varMismatchChangeLinks = []
varChangeLinks = []
varResponseChangeLinks = []
covChangeLinks = []

for networks in filenamesWait:
    
    network = pypsa.Network(directory+networks)
    
    #%%% mismatch
    
    generatorWind = GeneratorSplit(network,"wind").values
    generatorSolar = GeneratorSplit(network,"solar").values
    generatorRoR = GeneratorSplit(network,"ror").values
    
    load = network.loads_t.p_set[network.loads.index[:30]].values
    
    mismatch = generatorWind + generatorSolar + generatorRoR - load
    
    #%%% Mismatch Variance
    
    varMismatch = sum(np.mean((mismatch - np.mean(mismatch, axis=0))**2, axis=0))
    
    varMismatchChangeLinks.append(varMismatch)
    
    #%%% Split variance
    
    varWind = sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))**2, axis=0))
    varSolar = sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))**2, axis=0))
    varRoR = sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))**2, axis=0))
    varLoad = sum(np.mean((load - np.mean(load, axis=0))**2, axis=0))
    
    #%%% Covariance 
    
    covWindSolar    = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covWindRoR      = 2* sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covSolarRoR     = 2* sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    covWindLoad     = 2* sum(np.mean((-load - np.mean(-load, axis=0))*(generatorWind - np.mean(generatorWind, axis=0)), axis=0))
    covSolarLoad    = 2* sum(np.mean((-load - np.mean(-load, axis=0))*(generatorSolar - np.mean(generatorSolar, axis=0)), axis=0))
    covRoRLoad      = 2* sum(np.mean((-load - np.mean(-load, axis=0))*(generatorRoR - np.mean(generatorRoR, axis=0)), axis=0))
    
    #%%% Save values
    
    varMismatchSplit = {"Wind":             varWind,
                        "Solar PV":         varSolar,
                        "RoR":              varRoR,
                        "Load":             varLoad,
                        "Wind\nSolar PV":   covWindSolar,
                        "Wind\nRoR":        covWindRoR,
                        "Solar PV\nRoR":    covSolarRoR,
                        "Wind\nLoad":       covWindLoad,
                        "Solar PV\nLoad":   covSolarLoad,
                        "RoR\nLoad":        covRoRLoad
                        }

    varChangeLinks.append(varMismatchSplit)

    #%%% Find backup
    
    response = ElecResponse(network,True)

    #%%% backup variance
    
    varStorage = sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))**2, axis=0))
    varLinks = sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))**2, axis=0))
    varHydro = sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))**2, axis=0))
    varBackup = sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))**2, axis=0))
    varHeat = sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))**2, axis=0))
    varCHP = sum(np.mean((response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0))**2, axis=0))

    #%%% backup Covariance
    
    covStorageLinks     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covStorageHydro     = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covStorageBackup    = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covStorageHeat      = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covStorageCHP       = 2* sum(np.mean((response["Storage"] - np.mean(response["Storage"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covLinksHydro       = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLinksBackup      = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLinksHeat        = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLinksCHP         = 2* sum(np.mean((response["Import-Export"] - np.mean(response["Import-Export"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covBackupHydro      = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covBackupHeat       = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covBackupCHP        = 2* sum(np.mean((response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHydroHeat        = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covHydroCHP         = 2* sum(np.mean((response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covHeatCHP          = 2* sum(np.mean((response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    
    #%%% Save values
    
    varMismatchSplit = {"Storage":                          varStorage,
                        "Import-Export":                    varLinks,
                        "Hydro Reservoir":                  varHydro,
                        "Backup Generator":                 varBackup,
                        "Heat Couple":                      varHeat,
                        "CHP Electric":                     varCHP,
                        "Storage\nImport-Export":           covStorageLinks,
                        "Storage\nHydro":                   covStorageHydro,
                        "Storage\nBackup Generator":        covStorageBackup,
                        "Storage\nHeat Couple":             covStorageHeat,
                        "Storage\nCHP Electric":            covStorageCHP,
                        "Import-Export\nHydro":             covLinksHydro,
                        "Import-Export\nBackup Generator":  covLinksBackup,
                        "Import-Export\nHeat Couple":       covLinksHeat,
                        "Import-Export\nCHP Electric":      covLinksCHP,
                        "Backup Generator\nHydro":          covBackupHydro,
                        "Backup Generator\nHeat Couple":    covBackupHeat,
                        "Backup Generator\nCHP Electric":   covBackupCHP,
                        "hydro\nHeat Couple":               covHydroHeat,
                        "hydro\nCHP Electric":              covHydroCHP,
                        "Heat Couple\nCHP Electric":        covHeatCHP
                        }

    varResponseChangeLinks.append(varMismatchSplit)

    #%%% Generation & Storage covariance
    
    covWindStorage  = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covWindLinks    = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covWindDisp     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covWindBackup   = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covWindHeat     = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covWindCHP      = - sum(np.mean((generatorWind - np.mean(generatorWind, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covSolarStorage = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covSolarLinks   = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covSolarDisp    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covSolarBackup  = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covSolarHeat    = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covSolarCHP     = - sum(np.mean((generatorSolar - np.mean(generatorSolar, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
    
    covRoRStorage   = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covRoRLinks     = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covRoRDisp      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covRoRBackup    = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covRoRHeat      = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covRoRCHP       = - sum(np.mean((generatorRoR - np.mean(generatorRoR, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
        
    covLoadStorage  = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Storage"] - np.mean(response["Storage"], axis=0)), axis=0))
    covLoadLinks    = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Import-Export"] - np.mean(response["Import-Export"], axis=0)), axis=0))
    covLoadDisp     = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Hydro Reservoir"] - np.mean(response["Hydro Reservoir"], axis=0)), axis=0))
    covLoadBackup   = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Backup Generator"] - np.mean(response["Backup Generator"], axis=0)), axis=0))
    covLoadHeat     = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["Heat Couple"] - np.mean(response["Heat Couple"], axis=0)), axis=0))
    covLoadCHP      = - sum(np.mean((-load - np.mean(-load, axis=0))*(response["CHP Electric"] - np.mean(response["CHP Electric"], axis=0)), axis=0))
         
    
    
    #%%% Save cov values
    
    covMismatchSplit = {"Wind\nStorage":                covWindStorage,
                        "Wind\nImport-Export":          covWindLinks,
                        "Wind\nhydro":                  covWindDisp,
                        "Wind\nBackup Generator":       covWindBackup,
                        "Wind\nHeat Couple":            covWindHeat,
                        "Wind\nCHP Electric":           covWindCHP,
                        
                        "Solar PV\nStorage":            covSolarStorage,
                        "Solar PV\nImport-Export":      covSolarLinks,
                        "Solar PV\nhydro":              covSolarDisp,
                        "Solar PV\nBackup Generator":   covSolarBackup,
                        "Solar PV\nHeat Couple":          covSolarHeat,
                        "Solar PV\nCHP Electric":       covSolarCHP,
                        
                        "RoR\nStorage":                 covRoRStorage,
                        "RoR\nImport-Export":           covRoRLinks,
                        "RoR\nhydro":                   covRoRDisp,
                        "RoR\nBackup Generator":        covRoRBackup,
                        "RoR\nHeat Couple":             covRoRHeat,
                        "RoR\nCHP Electric":            covRoRCHP,
                        
                        "Load\nStorage":                covLoadStorage,
                        "Load\nImport-Export":          covLoadLinks,
                        "Load\nhydro":                  covLoadDisp,
                        "Load\nBackup Generator":       covLoadBackup,
                        "Load\nHeat Couple":            covLoadHeat,
                        "Load\nCHP Electric":           covLoadCHP
                        }
    
    covChangeLinks.append(covMismatchSplit)
    



#%%% Sort matrix

# CO2 mismatch
names = list(varChangeCO2[0].keys())
sortedMatrixCO2 = MatrixSorter(names,varChangeCO2)
sortedMatrixLinks = MatrixSorter(names,varChangeLinks)

# CO2 response
names = list(varResponseChangeCO2[0].keys())
sortedMatrixResponseCO2 = MatrixSorter(names,varResponseChangeCO2)
sortedMatrixResponseLinks = MatrixSorter(names,varResponseChangeLinks)


# CO2 Covariance
names = list(covChangeCO2[0].keys())
sortedMatrixCovarianceCO2 = MatrixSorter(names,covChangeCO2)
sortedMatrixCovarianceLinks = MatrixSorter(names,covChangeLinks)

#%% brownfield_Heat plot
#%%% Plot Var Mismatch

# quality
dpi = 200

# plot figure
fig = plt.figure(figsize=(10,9),dpi=dpi)

# grid
gs = fig.add_gridspec(46, 4)
axs = []
axs.append( fig.add_subplot(gs[0:10,0:2]) )   # plot 1
axs.append( fig.add_subplot(gs[0:10,2:4]) )   # plot 2
axs.append( fig.add_subplot(gs[17:27,0:2]) )   # plot 3
axs.append( fig.add_subplot(gs[17:27,2:4]) )   # plot 4
axs.append( fig.add_subplot(gs[36:46,0:2]) )   # plot 5
axs.append( fig.add_subplot(gs[36:46,2:4]) )   # plot 6

# Rotation of current
degrees = -12.5


for i in range(2):
    i += 0

    # length af plot
    length = len(varChangeCO2)

    # Data for components
    sortedMatrix = []
    if i == 0: 
        sortedMatrix = sortedMatrixCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrix = sortedMatrixLinks
        varMismatchChange = varMismatchChangeLinks
    loadCovData, genCovData, windVarData, SolarVarData, RoRVarnData, loadVarData = ContComponent(sortedMatrix)

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(SolarVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(RoRVarnData,color='k',alpha=0.5,linewidth=0.2)
    axs[i].plot(loadVarData,color='k',alpha=0.5,linewidth=0.2)
    axs[i].plot(genCovData,color='k',alpha=0.5,linewidth=0.5)


    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), loadCovData,
                     label='Load\ncovariance',
                     color='slategray',
                     alpha=0.5)
    axs[i].fill_between(range(length), loadCovData, genCovData,
                     label='Generator\ncovariance',
                     color='black',
                     alpha=0.5)
    axs[i].fill_between(range(length), np.zeros(length), windVarData,
                     label='Wind',
                     color='dodgerblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), windVarData, SolarVarData,
                     label='Solar PV',
                     color='gold',
                     alpha=0.5)
    axs[i].fill_between(range(length), SolarVarData, RoRVarnData,
                     label='RoR',
                     color='limegreen',
                     alpha=0.5)
    axs[i].fill_between(range(length), RoRVarnData, loadVarData,
                     label='Load',
                     color='goldenrod',
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-5*1e9,20*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    axs[i].set_xticks(np.arange(0,7))
    axs[i].set_xticklabels(["2020","2025","2030","2035","2040","2045","2050"])
    
    # Extra text
    if i == 0:
        axs[i].text(-1.2,10*1e9,"Mismatch\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        axs[i].text(3,21*1e9,"Early Transition Path",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,22*1e9,"(a)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        axs[i].text(3,21*1e9,"Late Transition Path",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,22*1e9,"(b)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.175), # placement relative to the graph
           ncol = 7, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.6, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

# Space between subplot
plt.subplots_adjust(wspace=0.3, hspace=30)


for i in range(2):
    i += 2

    # length af plot
    length = len(varChangeCO2)

    # Data for components
    sortedMatrix = []
    if i == 2: 
        sortedMatrixResponse = sortedMatrixResponseCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrixResponse = sortedMatrixResponseLinks
        varMismatchChange = varMismatchChangeLinks
    StorageVarData, linksVarData, hydroVarData, backupVarData, heatVarData, CHPVarData, covData = ResComponent(sortedMatrixResponse,"brownfield")

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(StorageVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(linksVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(backupVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(hydroVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(heatVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(CHPVarData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(covData,color='k',alpha=0.5,linewidth=0.5)

    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), StorageVarData,
                     label='Storage',
                     color='orange',
                     alpha=0.5)
    axs[i].fill_between(range(length), StorageVarData, linksVarData,
                     label='Import-\nExport',
                     color='darkgreen',
                     alpha=0.5)
    axs[i].fill_between(range(length), linksVarData, hydroVarData,
                     label='Hydro\nreservoir',
                     color='lightblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), hydroVarData, backupVarData,
                     label='Backup\nGenerator',
                     color='darkgray',
                     alpha=0.5)
    axs[i].fill_between(range(length), backupVarData, heatVarData,
                     label='Heat\nCouple',
                     color='mediumblue',
                     alpha=0.5)
    axs[i].fill_between(range(length), heatVarData, CHPVarData,
                     label='CHP Electric',
                     color='aqua',
                     alpha=0.5)
    axs[i].fill_between(range(length), CHPVarData, covData,
                     label='Covariance',
                     color='olive',
                     alpha=0.5)

    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")

    # Y axis
    axs[i].set(ylim = [-1*1e9,16.5*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    axs[i].set_xticks(np.arange(0,7))
    axs[i].set_xticklabels(["2020","2025","2030","2035","2040","2045","2050"])
    
    # Extra text
    if i == 2:
        axs[i].text(-1.2,10*1e9,"Response\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        #axs[i].text(3,21*1e9,"Early Transition Path",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,17.5*1e9,"(c)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        #axs[i].text(3,21*1e9,"Late Transition Path",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,17.5*1e9,"(d)",rotation="horizontal",fontsize=12, fontweight="bold")
        
# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.065,-0.175), # placement relative to the graph
           ncol = 6, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.8, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )

for i in range(2):
    i += 4

    # length af plot
    length = len(varChangeCO2)

    # Data for components
    sortedMatrix = []
    if i == 4: 
        sortedMatrixCovariance = sortedMatrixCovarianceCO2 
        varMismatchChange = varMismatchChangeCO2
    else: 
        sortedMatrixCovariance = sortedMatrixCovarianceLinks
        varMismatchChange = varMismatchChangeLinks

    windStorageCov, windLinksCov, windHydroCov, windBackupCov, windHeatCov, windCHPCov, solarStorageCov, solarLinksCov, solarHydroCov, solarBackupCov, solarHeatCov, solarCHPCov, RoRCovData, loadCovData = CovComponent(sortedMatrixCovariance,"brownfield")

    # plot
    axs[i].plot(range(length),color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(windStorageCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windLinksCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windHydroCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windBackupCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windHeatCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(windCHPCov,color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(solarStorageCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarLinksCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarHydroCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarBackupCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarHeatCov,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(solarCHPCov,color='k',alpha=0.5,linewidth=0.5)
    
    axs[i].plot(RoRCovData,color='k',alpha=0.5,linewidth=0.5)
    axs[i].plot(loadCovData,color='k',alpha=0.5,linewidth=0.5)
    
    
    # Fill lines
    axs[i].fill_between(range(length), np.zeros(length), windStorageCov,
                     label='Wind\nStorage',
                     color=color[1],
                     alpha=0.5)
    axs[i].fill_between(range(length), windStorageCov, windLinksCov,
                     label='Wind\nImport-Export',
                     color=color[2],
                     alpha=0.5)
    axs[i].fill_between(range(length), windLinksCov, windHydroCov,
                     label='Wind\nHydro Reservoir',
                     color=color[3],
                     alpha=0.5)
    axs[i].fill_between(range(length), windLinksCov, windBackupCov,
                     label='Wind\nBackup Generator',
                     color=color[4],
                     alpha=0.5)
    axs[i].fill_between(range(length), windBackupCov, windHeatCov,
                     label='Wind\Heat Couple',
                     color=color[5],
                     alpha=0.5)
    axs[i].fill_between(range(length), windHeatCov, windCHPCov,
                     label='Wind\nCHP Electric',
                     color=color[6],
                     alpha=0.5)
    axs[i].fill_between(range(length), windCHPCov, solarStorageCov,
                     label='Solar PV\nStorage',
                     color=color[7],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarStorageCov, solarLinksCov,
                     label='Solar PV\nImport-Export',
                     color=color[8],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarLinksCov, solarHydroCov,
                     label='Solar PV\nHydro Reservoir',
                     color=color[9],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarHydroCov, solarBackupCov,
                     label='Solar PV\nBackup Generator',
                     color=color[10],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarBackupCov, solarHeatCov,
                     label='Solar PV\nHeat Couple',
                     color=color[11],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarHeatCov, solarCHPCov,
                     label='Solar PV\nCHP Electric',
                     color=color[12],
                     alpha=0.5)
    axs[i].fill_between(range(length), solarCHPCov, RoRCovData,
                     label='RoR\ncovariance',
                     color=color[13],
                     alpha=0.5)
    axs[i].fill_between(range(length), RoRCovData, loadCovData,
                     label='Load\ncovariance',
                     color=color[15],
                     alpha=0.5)
    
    # Mismatch variance
    axs[i].plot(varMismatchChange,color='k', linestyle='dashed',alpha=1,linewidth=2, label="Mismatch\nvariance")
    
    # Y axis
    axs[i].set(ylim = [-1*1e9,16.5*1e9])
    axs[i].tick_params(axis='both',
                       labelsize=10)
    axs[i].yaxis.offsetText.set_fontsize(10)
    
    # X axis
    axs[i].set_xticks(np.arange(0,7))
    axs[i].set_xticklabels(["2020","2025","2030","2035","2040","2045","2050"])

    # Extra text
    if i == 4:
        axs[i].text(-1.2,10*1e9,"Covariance\nVariance",rotation="vertical",fontsize=12, fontweight="bold",horizontalalignment='center', verticalalignment='center')
        #axs[i].text(3,21*1e9,"Early Transition Path",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,17.5*1e9,"(e)",rotation="horizontal",fontsize=12, fontweight="bold")
    else:
        #axs[i].text(3,21*1e9,"Late Transition Path",rotation="horizontal",fontsize=12, fontweight="bold",horizontalalignment='center')
        axs[i].text(-0.8,17.5*1e9,"(f)",rotation="horizontal",fontsize=12, fontweight="bold")
        


# legend
axs[i].legend(loc = 'upper center', # How the label should be places according to the placement
           bbox_to_anchor = (-0.08,-0.175), # placement relative to the graph
           ncol = 5, # Amount of columns
           markerscale = 30,
           fontsize = 10, # Size of text
           framealpha = 1, # Box edge alpha
           columnspacing = 1.4, # Horizontal spacing between labels
           labelspacing = 0.5 # Vertical spacing between label
           )


# Save figureø
title = "brownfield_heat - Electricity Variance and Cross Correlation"
pathPlot = path + "\\Figures\\brownfield_heat\\pre analysis\\"
SavePlot(fig,pathPlot,title)

plt.show(all)