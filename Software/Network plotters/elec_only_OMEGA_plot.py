# Import libraries
import os
import sys
import pypsa
import numpy as np
import pandas as pd
#from sympy import latex

import time
import math

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *         


# Directory of file
#directory = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Data\\elec_only\\"
directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\elec_only\\"

# File name
file = "postnetwork-elec_only_0.125_0.05.h5"

# Generic file name
titleFileName = file

# Figure path
#figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\elec_only\\"
figurePath = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Figures\\elec_only\\"



##############################################################################
##############################################################################

################################# Pre Analysis ###############################

##############################################################################
##############################################################################

# ------------------- Curtailment - CO2 constraints (Elec) ------------------#
# Path to save files
path = figurePath + "Pre Analysis\\"

# List of file names
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

# List of constraints
constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
title = ""#"Electricity Curtailment - " + file[12:-14]
fig = Curtailment(directory=directory, files=filename_CO2, title=title, constraints=constraints, fontsize=14, ylim=[-1,20], figsize=[6, 4.5])

SavePlot(fig, path, title=(file[12:-14] + " - Curtailment Elec (CO2)"))



# --------------- Curtailment - Transmission constraints (Elec) --------------#

# List of file names
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

# List of constraints
constraints = ["Zero", "Current", "2x Current", "4x Current", "6x Current"]
title = ""#"Electricity Curtailment - " + file[12:-14]

fig = Curtailment(directory=directory, files=filename_trans, title=title, constraints=constraints, fontsize=14, rotation=-17.5, ylim=[-2,60], legendLoc="upper right", figsize=[6, 4.8])

SavePlot(fig, path, title=(file[12:-14] + " - Curtailment Elec (trans)"))



##############################################################################
##############################################################################

################################### MISMATCH #################################

##############################################################################
##############################################################################

# ------- Electricity produced by technology (Elec CO2 and Trans) -----------#
# Path to save files
path = figurePath + "Mismatch\\"

# List of file names
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

# List of constraints
constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]

fig = ElecProductionOvernight(directory=directory, filenames=filename_CO2, constraints=constraints, fontsize=14,  figsize=[6,6])
SavePlot(fig, path, title=(file[12:-14] + " - total elec generation (CO2)"))




# List of file names
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

# List of constraints
constraints = ["Zero", "Current", "2x Current", "4x Current", "6x Current"]

fig = ElecProductionOvernight(directory=directory, filenames=filename_trans, constraints=constraints, rotation=-25, fontsize=14, figsize=[6,6.3])
SavePlot(fig, path, title=(file[12:-14] + " - total elec generation (trans)"))



# ------------------ Map Capacity Plots (Elec) ------------------#
# Path to save files
path = figurePath + "Mismatch\\Map Capacity\\"

# --- Elec ---
# Import network
network = pypsa.Network(directory+file)
fig1, fig2, fig3 = MapCapacityOriginal(network, titleFileName, ncol=3)
SavePlot(fig1, path, title=(file[12:-3] + " - Map Capacity Elec Generator"))
SavePlot(fig2, path, title=(file[12:-3] + " - Map Capacity Elec Storage Energy"))
SavePlot(fig3, path, title=(file[12:-3] + " - Map Capacity Elec Storage Power"))


# -------------------- Map Energy Plot (Elec) -------------------#
# Path for saving file
path = figurePath + "Mismatch\\Map Energy Distribution\\"

# Import network
network = pypsa.Network(directory+file)

# --- Elec ---
figElec = MapCapacityElectricityEnergy(network, file)
SavePlot(figElec, path, title=(file[12:-3] + " - Elec Energy Production"))


# --------------------- Map PC Plot (Elec) ----------------------#
# Path to save plots
path = figurePath + "Mismatch\\Map PC\\"

# Import network
network = pypsa.Network(directory+file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# Get time stamps
timeIndex = network.loads_t.p_set.index


# --- Elec ---
# Electricity load for each country
loadElec = network.loads_t.p_set[dataNames]

# Solar PV generation
generationSolar = network.generators_t.p[dataNames + " solar"]
generationSolar.columns = generationSolar.columns.str.slice(0,2)

# Onshore wind generation
generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()

# Offshore wind generation
# Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationOffwind = generationOffwind.replace(np.nan,0)

# RoR generations
# Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationRoR = generationRoR.replace(np.nan,0)

# Combined generation for electricity
generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR

# Mismatch electricity
mismatchElec = generationElec - loadElec

# PCA on mismatch for electricity
eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)

# ----------------------- Map Plot (Elec) -----------------------#
# Plot map PC for mismatch electricity
titlePlotElec = "Mismatch for electricity only"
for i in np.arange(6):
    fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1))#, titlePlotElec, titleFileName)
    title = (file[12:-3] + " - Map PC Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ----------------------- FFT Plot (Elec) -----------------------#
# Path to save FFT plots
path = figurePath + "Mismatch\\FFT\\"

# --- Elec ---
file_name = "Electricity mismatch - " + file
for i in np.arange(6):
    fig = FFTPlot(TElec.T, varianceExplainedElec, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)
    

# -------------------- Seasonal Plot (Elec) ---------------------#
# Path to save seasonal plots
path = figurePath + "Mismatch\\Seasonal\\"

# --- Elec ---
file_name = "Electricity mismatch - " + file
for i in np.arange(6):
    fig = seasonPlot(TElec, timeIndex, PC_NO=(i+1), PC_amount=6,dpi=400)
    title = (file[12:-3] + " - Seasonal Plot Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# -------------------- FFT + Seasonal Plot (Elec) ---------------------#
# Path to save seasonal plots
path = figurePath + "Mismatch\\Timeseries\\"

# --- Elec ---
file_name = "Electricity mismatch - " + file
for i in np.arange(6):
    fig = FFTseasonPlot(TElec, timeIndex, varianceExplainedElec, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file[12:-3] + " - Timeseries Plot Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# ----------------- Contribution plot (Elec) ------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Contribution\\"

# --- Elec ---
# Contribution
dircConElec = Contribution(network, "elec")
lambdaCollectedConElec = ConValueGenerator(normConstElec, dircConElec, eigenVectorsElec)

for i in range(6):
    fig = ConPlot(eigenValuesElec,lambdaCollectedConElec,i+1,10,suptitle=("Electricity Contribution - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Contribution Plot Elec (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ------------------- Response plot (Elec) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Response\\"

# --- Elec ---
# Response
dircResElec = ElecResponse(network,True)
lambdaCollectedResElec = ConValueGenerator(normConstElec, dircResElec, eigenVectorsElec)

for i in range(6):
    fig = ConPlot(eigenValuesElec,lambdaCollectedResElec,i+1,10,suptitle=("Electricity Response - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Response Plot Elec (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# ------------------- Covariance plot (Elec) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Covariance\\"

# --- Elec ---
# Covariance
covMatrixElec = CovValueGenerator(dircConElec, dircResElec , True, normConstElec, eigenVectorsElec).T

for i in range(6):
    fig = ConPlot(eigenValuesElec,covMatrixElec,i+1,10,suptitle=("Electricity Covariance - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Covariance Plot Elec (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# ------------------- Combined Projection plot (Elec) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Projection\\"

# --- Elec ---
for i in range(6):
    fig = CombConPlot(eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec, i+1, depth = 6) #, suptitle=("Electricity Projection - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Projection Plot Elec (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# ------------------- PC1 and PC2 combined plot (Elec) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Combined Plot\\"

# --- Elec --- 
fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withProjection")#,suptitle=("Electricity Mismatch - " + file[12:-3]),dpi=200)
title = (file[12:-3] + " - Combined Plot Elec (lambda 1 & 2)")
SavePlot(fig, path, title)



# ---------------------- Bar plot CO2 constraint --------------------------- #
# Path to save bar plots
path = figurePath + "Mismatch\\Bar\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]


# Variable to store mismatch PC componentns for each network
barMatrixCO2Elec = []

for file in filename_CO2:
    # --------------------------- Electricity -------------------------------#
    # Network
    network = pypsa.Network(directory + file)
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get time stamps
    timeIndex = network.loads_t.p_set.index
    
    # Electricity load for each country
    loadElec = network.loads_t.p_set[dataNames]
    
    # Solar PV generation
    generationSolar = network.generators_t.p[dataNames + " solar"]
    generationSolar.columns = generationSolar.columns.str.slice(0,2)
    
    # Onshore wind generation
    generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    
    # Offshore wind generation
    # Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
    generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationOffwind = generationOffwind.replace(np.nan,0)
    
    # RoR generations
    # Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
    generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationRoR = generationRoR.replace(np.nan,0)
    
    # Combined generation for electricity
    generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR
    
    # Mismatch electricity
    mismatchElec = generationElec - loadElec
    
    # PCA on mismatch for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)
    
    # Append value to matrix
    barMatrixCO2Elec.append(varianceExplainedElec)


constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
title = "" #"Number of PC describing variance of network as a function of $CO_{2}$ constraint"
xlabel = "$CO_{2}$ constraint"


suptitleElec = "" #("Electricity Mismatch - " + file[12:-14])
fig = BAR(barMatrixCO2Elec, 7, filename_CO2, constraints, title, xlabel, suptitleElec, fontsize=18, figsize=[6, 3], ncol=4, bbox=(0.5,-0.28))
titleBarCO2Elec = (file[12:-14] + " - Bar CO2 Elec Mismatch")
SavePlot(fig, path, titleBarCO2Elec)



# ------------------ Bar plot Transmission constraint ----------------------- #
# Path
#path = "C:/Users/jense/OneDrive - Aarhus Universitet/Dokumenter/Århus Universitet/Kandidat - Civilingeniør/11. Semester/Master Thesis/Shared Documents/Figures/elec_only/Bar/"
path = figurePath + "Mismatch\\Bar\\"

# Name of file (must be in correct folder location)
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

# Variable to store mismatch PC componentns for each network
barMatrixTransmissionElec = []
for file in filename_trans:
    # --------------------------- Electricity -------------------------------#
    # Network
    network = pypsa.Network(directory + file)
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get time stamps
    timeIndex = network.loads_t.p_set.index
    
    # Electricity load for each country
    loadElec = network.loads_t.p_set[dataNames]
    
    # Solar PV generation
    generationSolar = network.generators_t.p[dataNames + " solar"]
    generationSolar.columns = generationSolar.columns.str.slice(0,2)
    
    # Onshore wind generation
    generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    
    # Offshore wind generation
    # Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
    generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationOffwind = generationOffwind.replace(np.nan,0)
    
    # RoR generations
    # Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
    generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationRoR = generationRoR.replace(np.nan,0)
    
    # Combined generation for electricity
    generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR
    
    # Mismatch electricity
    mismatchElec = generationElec - loadElec
    
    # PCA on mismatch for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)
    
    # Append value to matrix
    barMatrixTransmissionElec.append(varianceExplainedElec)
    

constraints = ["Zero", "Current", "2x Current", "4x Current", "6x Current"]
title = "Number of PC describing variance of network as a function of transmission constraint"
xlabel = "Transmission constraint"



suptitleElec = ("Electricity Mismatch - " + file[12:-14])
fig = BAR(barMatrixTransmissionElec, 7, filename_trans, constraints, title, xlabel, suptitleElec, fontsize=18, figsize=[6, 3], ncol=4, rotation=-17.5, bbox=(0.5,-0.28))
titleBarTransmissionElec = (file[12:-14] + " - Bar Trans Elec Mismatch")
SavePlot(fig, path, titleBarTransmissionElec)

  
# ------------------ Change in contribution and response CO2 ----------------------- #


# Variable to store lambda values
lambdaContributionElec = []
lambdaResponseElec     = []
lambdaCovarianceElec   = []

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]


for file in filename_CO2:
    # --------------------------- Electricity -------------------------------#
    # Network
    network = pypsa.Network(directory + file)
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get time stamps
    timeIndex = network.loads_t.p_set.index
    
    # Electricity load for each country
    loadElec = network.loads_t.p_set[dataNames]
    
    # Solar PV generation
    generationSolar = network.generators_t.p[dataNames + " solar"]
    generationSolar.columns = generationSolar.columns.str.slice(0,2)
    
    # Onshore wind generation
    generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    
    # Offshore wind generation
    # Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
    generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationOffwind = generationOffwind.replace(np.nan,0)
    
    # RoR generations
    # Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
    generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationRoR = generationRoR.replace(np.nan,0)
    
    # Combined generation for electricity
    generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR
    
    # Mismatch electricity
    mismatchElec = generationElec - loadElec
    
    # PCA on mismatch for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)

    # Contribution Elec
    dircConElec = Contribution(network, "elec")
    lambdaCollectedConElec = ConValueGenerator(normConstElec, dircConElec, eigenVectorsElec)
    lambdaContributionElec.append(lambdaCollectedConElec)
    
    # Response Elec
    dircResElec = ElecResponse(network,True)
    lambdaCollectedResElec = ConValueGenerator(normConstElec, dircResElec, eigenVectorsElec)
    lambdaResponseElec.append(lambdaCollectedResElec)
    
    # Covariance Elec
    covMatrix = CovValueGenerator(dircConElec, dircResElec , True, normConstElec, eigenVectorsElec)
    lambdaCovarianceElec.append(covMatrix.T)    

#%%
from functions_file import *  
# general terms
pathContibution = figurePath + "Mismatch\\Change in Contribution\\"
pathResponse = figurePath + "Mismatch\\Change in Response\\"
pathCovariance = figurePath + "Mismatch\\Change in Covariance\\"

# Plot change in elec contribution
figtitle = "Change in electricity contribution as a function of CO2 constraint"
fig = ChangeContributionElec(lambdaContributionElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec cont (CO2)"
SavePlot(fig, pathContibution, saveTitle)

# Plot change in elec contribution
figtitle = "Change in electricity contribution as a function of CO2 constraint"
fig = ChangeContributionElec(lambdaContributionElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec cont app (CO2)"
SavePlot(fig, pathContibution, saveTitle)

# Plot change in elec response
figtitle = "Change in electricity response as a function of CO2 constraint"
fig = ChangeResponseElec(lambdaResponseElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec response (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in elec response
figtitle = "Change in electricity response as a function of CO2 constraint"
fig = ChangeResponseElec(lambdaResponseElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec response app (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in elec covariance response
figtitle = "Change in electricity covariance response as a function of CO2 constraint"
fig = ChangeResponseCov(lambdaResponseElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec cov response (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in elec covariance response
figtitle = "Change in electricity covariance response as a function of CO2 constraint"
fig = ChangeResponseCov(lambdaResponseElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec cov response app (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in covariance'

figtitle = "Change in electricity covariance between mismatch and response as a function of CO2 constraint"
fig = ChangeCovariance(lambdaCovarianceElec, collectTerms=True, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec covariance (CO2)"
SavePlot(fig, pathCovariance, saveTitle)

# Plot change in covariance
figtitle = "Change in electricity covariance between mismatch and response as a function of CO2 constraint"
fig = ChangeCovariance(lambdaCovarianceElec, collectTerms=True, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec covariance app (CO2)"
SavePlot(fig, pathCovariance, saveTitle)


#%%
# ------------------ Change in contribution and response transmission ----------------------- #

# Variable to store lambda values
lambdaContributionElec = []
lambdaResponseElec     = []
lambdaCovarianceElec   = []

# Name of file (must be in correct folder location)
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

for file in filename_trans:
    # --------------------------- Electricity -------------------------------#
    # Network
    network = pypsa.Network(directory + file)
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get time stamps
    timeIndex = network.loads_t.p_set.index
    
    # Electricity load for each country
    loadElec = network.loads_t.p_set[dataNames]
    
    # Solar PV generation
    generationSolar = network.generators_t.p[dataNames + " solar"]
    generationSolar.columns = generationSolar.columns.str.slice(0,2)
    
    # Onshore wind generation
    generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    
    # Offshore wind generation
    # Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
    generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationOffwind = generationOffwind.replace(np.nan,0)
    
    # RoR generations
    # Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
    generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
    generationRoR = generationRoR.replace(np.nan,0)
    
    # Combined generation for electricity
    generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR
    
    # Mismatch electricity
    mismatchElec = generationElec - loadElec
    
    # PCA on mismatch for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)

    # Contribution Elec
    dircConElec = Contribution(network, "elec")
    lambdaCollectedConElec = ConValueGenerator(normConstElec, dircConElec, eigenVectorsElec)
    lambdaContributionElec.append(lambdaCollectedConElec)
    
    # Response Elec
    dircResElec = ElecResponse(network,True)
    lambdaCollectedResElec = ConValueGenerator(normConstElec, dircResElec, eigenVectorsElec)
    lambdaResponseElec.append(lambdaCollectedResElec)
    
    # Covariance Elec
    covMatrix = CovValueGenerator(dircConElec, dircResElec , True, normConstElec,eigenVectorsElec)
    lambdaCovarianceElec.append(covMatrix.T)   
    
# general terms
pathContibution = figurePath + "Mismatch\\Change in Contribution\\"
pathResponse = figurePath + "Mismatch\\Change in Response\\"
pathCovariance = figurePath + "Mismatch\\Change in Covariance\\"

# Plot change in elec contribution
figtitle = "Change in electricity contribution as a function of transmission constraint"
fig = ChangeContributionElec(lambdaContributionElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec cont (trans)"
SavePlot(fig, pathContibution, saveTitle)

# Plot change in elec contribution
figtitle = "Change in electricity contribution as a function of transmission constraint"
fig = ChangeContributionElec(lambdaContributionElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec cont app (trans)"
SavePlot(fig, pathContibution, saveTitle)

# Plot change in elec contribution
figtitle = "Change in electricity response as a function of transmission constraint"
fig = ChangeResponseElec(lambdaResponseElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec response (trans)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in elec contribution
figtitle = "Change in electricity response as a function of transmission constraint"
fig = ChangeResponseElec(lambdaResponseElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec response app (trans)"
SavePlot(fig, pathResponse, saveTitle)


# Plot change in elec covariance response
figtitle = "Change in electricity covariance response as a function of transmission constraint"
fig = ChangeResponseCov(lambdaResponseElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec cov response (trans)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in elec covariance response
figtitle = "Change in electricity covariance response as a function of transmission constraint"
fig = ChangeResponseCov(lambdaResponseElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec cov response app (trans)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in covariance
figtitle = "Change in electricity covariance as a function of transmission constraint"
fig = ChangeCovariance(lambdaCovarianceElec, collectTerms=True, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec covariance (trans)"
SavePlot(fig, pathCovariance, saveTitle)

# Plot change in covariance
figtitle = "Change in electricity covariance as a function of transmission constraint"
fig = ChangeCovariance(lambdaCovarianceElec, collectTerms=True, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec covariance app (trans)"
SavePlot(fig, pathCovariance, saveTitle)


#%%

##############################################################################
##############################################################################

################################ NODAL PRICE #################################

##############################################################################
##############################################################################

# File name
file = "postnetwork-elec_only_0.125_0.05.h5"

# Import network
network = pypsa.Network(directory+file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# ----------------------- Map PC Plot (Elec + Heat) --------------------#
# Path to save plots
path = figurePath + "Nodal Price\\Map PC\\"

# --- Elec ---
# Prices for electricity for each country (restricted to 1000 €/MWh)
priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)

# PCA on nodal prices for electricity
eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)

# Plot map PC for electricity nodal prices
titlePlotElec = "Nodal price for electricity only"
for i in np.arange(6):
    fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1),size="medium")#, titlePlotElec, titleFileName)
    title = (file[12:-3] + " - Map PC Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ------------------------ FFT Plot (Elec + Heat) -----------------------#
# Path to save FFT plots
path = figurePath + "Nodal Price\\FFT\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + file
for i in np.arange(6):
    fig = FFTPlot(TElec.T, varianceExplainedElec, PC_NO = i+1, title=file_name)
    title = (file[12:-3] + " - FFT Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ----------------------- Seasonal Plot (Elec + Heat) ------------------------#
# Path to save seasonal plots
path = figurePath + "Nodal Price\\Seasonal\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + file
for i in np.arange(6):
    fig = seasonPlot(TElec, timeIndex, PC_NO=(i+1), PC_amount=6, title=file_name)
    title = (file[12:-3] + " - Seasonal Plot Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# -------------------- FFT + Seasonal Plot (Elec) ---------------------#
# Path to save seasonal plots
path = figurePath + "Nodal Price\\Timeseries\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + file
for i in np.arange(6):
    fig = FFTseasonPlot(TElec, timeIndex, varianceExplainedElec, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file[12:-3] + " - Timeseries Plot Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# ------------------- PC1 and PC2 combined plot (Elec) -------------------- #
# Path to save contribution plots
path = figurePath + "Nodal Price\\Combined Plot\\"

# --- Elec --- 
fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withoutProjection")#,suptitle=("Electricity Mismatch - " + file[12:-3]),dpi=200)
title = (file[12:-3] + " - Combined Plot Elec NP (lambda 1 & 2)")
SavePlot(fig, path, title)

#%%
# ---------------------- Bar plot CO2 constraint --------------------------- #
# Path to save bar plots
path = figurePath + "Nodal Price\\Bar\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_only_0.125_0.6.h5",
                "postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]

# Variable to store nodal price PC componentns for each network
barMatrixCO2Elec = []

# Variable to store nodal price mean and standard variation
meanPriceElec = []
stdMeanPriceElec = []
quantileMeanPriceElec = []
quantileMinPriceElec = []

for file in filename_CO2:
    # Network
    network = pypsa.Network(directory + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    
    # --- Elec ---
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)

    # PCA on nodal prices for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)
    
    # Append value to matrix
    barMatrixCO2Elec.append(varianceExplainedElec)
    
    # ----------------------- NP Mean (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    minPrice = priceElec.min().mean()
    meanPrice = priceElec.mean().mean()
    
    # append min, max and mean to matrix
    meanPriceElec.append([minPrice, meanPrice])
    
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    quantileMinPrice = np.quantile(priceElec.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPrice = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # append min, max and mean to matrix
    quantileMeanPriceElec.append(quantileMeanPrice)
    quantileMinPriceElec.append(quantileMinPrice)
    


constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
title = "" #"Number of PC describing variance of network as a function of $CO_{2}$ constraint"
xlabel = "" #"$CO_{2}$ constraint"


suptitleElec = "" #("Electricity Nodal Price - " + file[12:-14])
fig = BAR(barMatrixCO2Elec, 7, filename_CO2, constraints, title, xlabel, suptitleElec, fontsize=18, figsize=[6, 3], ncol=4, bbox=(0.5,-0.28))
titleBarCO2Elec = (file[12:-14] + " - Bar CO2 Elec NP")
SavePlot(fig, path, titleBarCO2Elec)

# ----------------------- Price evalution (Elec) --------------------#
path = figurePath + "Nodal Price\\Price Evolution\\"
title =  "" #("Electricity Nodal Price Evalution - "  + file[12:-14])
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElec, quantileMinPriceElec, networktype="green", title=title, figsize=[6,3], fontsize=16)
title =  (file[12:-14] + " - Elec NP CO2 Evolution")
SavePlot(fig, path, title)



# ------------------ Bar plot Transmission constraint ----------------------- #
# Path
path = figurePath + "Nodal Price\\Bar\\"

# Name of file (must be in correct folder location)
filename_trans = ["postnetwork-elec_only_0_0.05.h5",
                  "postnetwork-elec_only_0.0625_0.05.h5",
                  "postnetwork-elec_only_0.125_0.05.h5",
                  "postnetwork-elec_only_0.25_0.05.h5",
                  "postnetwork-elec_only_0.375_0.05.h5"]

# Variable to store nodal price PC componentns for each network
barMatrixTransmissionElec = []

# Variable to store nodal price mean and standard variation
meanPriceElec = []
quantileMeanPriceElec = []
quantileMinPriceElec = []

for file in filename_trans:
        # Network
    network = pypsa.Network(directory + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # --- Elec ---
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)
    
    # PCA on nodal prices for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)
    
    # Append value to matrix
    barMatrixTransmissionElec.append(varianceExplainedElec)

    # ----------------------- NP Mean (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    minPrice = priceElec.min().mean()
    meanPrice = priceElec.mean().mean()
    
    # append min, max and mean to matrix
    meanPriceElec.append([minPrice, meanPrice])
    
    # ----------------------- NP Quantile (Elec) --------------------#
    # --- Elec ---
    # Mean price for country
    quantileMinPrice = np.quantile(priceElec.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPrice = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # append min, max and mean to matrix
    quantileMeanPriceElec.append(quantileMeanPrice)
    quantileMinPriceElec.append(quantileMinPrice)



# ----------------------- Bar plot (Elec) --------------------#
constraints = ["Zero", "Current", "2x Current", "4x Current", "6x Current"]
title = "" #"Number of PC describing variance of network as a function of transmission constraint"
xlabel = "" #"Transmission constraint"


suptitleElec = "" #("Electricity Nodal Price - "  + file[12:-14])
fig = BAR(barMatrixTransmissionElec, 7, filename_trans, constraints, title, xlabel, suptitleElec, fontsize=18, figsize=[6, 3], ncol=4, rotation=-17.5, bbox=(0.5,-0.28))
titleBarTransmissionElec =  (file[12:-14] + " - Bar Trans Elec NP")
SavePlot(fig, path, titleBarTransmissionElec)

# ----------------------- Price evalution (Elec) --------------------#
path = figurePath + "Nodal Price\\Price Evolution\\"
title = "" #("Electricity Nodal Price Evalution - "  + file[12:-14])
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElec, quantileMinPriceElec, networktype="green", title=title, figsize=[6,3.2], fontsize=16)
title =  (file[12:-14] + " - Elec NP Trans Evolution")
SavePlot(fig, path, title)

#%%
##############################################################################
##############################################################################

################################# Coherence ##################################

##############################################################################
##############################################################################

# -------------------- Coherence Plot (Elec + Heat) ---------------------#
# File name
file = "postnetwork-elec_only_0.125_0.05.h5"

# Import network
network = pypsa.Network(directory+file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# Get time stamps
timeIndex = network.loads_t.p_set.index

# Path to save contribution plots
path = figurePath + "Coherence\\"

# --- Elec ---
# Electricity load for each country
loadElec = network.loads_t.p_set[dataNames]

# Solar PV generation
generationSolar = network.generators_t.p[dataNames + " solar"]
generationSolar.columns = generationSolar.columns.str.slice(0,2)

# Onshore wind generation
generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()

# Offshore wind generation
# Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationOffwind = generationOffwind.replace(np.nan,0)

# RoR generations
# Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationRoR = generationRoR.replace(np.nan,0)

# Combined generation for electricity
generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR

# Mismatch electricity
mismatchElec = generationElec - loadElec

# Prices for each country
priceElec = FilterPrice(network.buses_t.marginal_price[dataNames], 465)

# Coherence between prices and mismatch
c1Elec, c2Elec, c3Elec = Coherence(mismatchElec, priceElec)

# Plot properties
title1 = "" #"Coherence 1: Electricity mismatch and nodal price"
title2 = "" #"Coherence 2: Electricity mismatch and nodal price"
title3 = "" #"Coherence 3: Electricity mismatch and nodal price"
xlabel = "Electricity Mismatch"
ylabel="Electricity Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1Elec.T, übertitle="", title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2Elec.T, übertitle="", title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3Elec.T, übertitle="", title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file[12:-3] + " - C1 elec mismatch and ENP"))
SavePlot(fig2, path, title = (file[12:-3] + " - C2 elec mismatch and ENP"))
SavePlot(fig3, path, title = (file[12:-3] + " - C3 elec mismatch and ENP"))

# Combined Plot
fig = CoherencePlotCombined(c1Elec.T, c2Elec.T, c3Elec.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file[12:-3] + " - C123 combined elec mismatch and ENP"))


# Finish timer
t1 = time.time() # End timer
total_time = round(t1-t0)
total_time_min = math.floor(total_time/60)
total_time_sec = round(total_time-(total_time_min*60))
print("\n \nThe code is now done running. It took %s min and %s sec." %(total_time_min,total_time_sec))



