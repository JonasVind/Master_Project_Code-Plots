# Import libraries
import os
import sys
import pypsa
import numpy as np
import pandas as pd

import time
import math

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *           


# Directory of file
directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\elec_heat_v2g50\\"

# File name
file = "postnetwork-elec_heat_v2g50_0.125_0.05.h5"

# Generic file name
titleFileName = file

# Figure path
figurePath = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Figures\\elec_heat_v2g50\\"


##############################################################################
##############################################################################

################################# Pre Analysis ###############################

##############################################################################
##############################################################################

# ------------------- Curtailment - CO2 constraints (Elec) ------------------#
# Path to save files
path = figurePath + "Pre Analysis\\"

# List of file names
filename_CO2 = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

# List of constraints
constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
title= "" #("Electricity Curtailment - " + file[12:-14])
fig = Curtailment(directory=directory, files=filename_CO2, title=title, constraints=constraints, figsize=[10,3], ylim=[-1,24])
SavePlot(fig, path, title=(file[12:-14] + " - Curtailment Elec (CO2)"))


title = "" #("Heat Curtailment - " + file[12:-14])
fig = CurtailmentHeat(directory=directory, files=filename_CO2, title=title, constraints=constraints, figsize=[10,3], ylim=[-1,20], legendLoc="upper left")
SavePlot(fig, path, title=(file[12:-14] + " - Curtailment Heat (CO2)"))



##############################################################################
##############################################################################

################################### MISMATCH #################################

##############################################################################
##############################################################################

# ------- Electricity produced by technology (Elec CO2 and Trans) -----------#
# Path to save files
path = figurePath + "Mismatch\\"

# Name of file (must be in correct folder location)
filenames = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
             "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
             "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
             "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
             "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
             "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
             "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

# List of constraints
constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]

fig = ElecProductionOvernight(directory=directory, filenames=filenames, constraints=constraints, figsize=[10,4])
SavePlot(fig, path, title=(file[12:-14] + " - total elec generation (CO2)"))




# ------------------ Total Installed Heating Capacity (CO2) -----------------#

# Path to save files
path = figurePath + "Mismatch\\"

fig = OvernightHeatCapacityInstalled(path=directory, filenames=filenames, constraints=constraints)

SavePlot(fig, path, title=(file[12:-14] + " - Total Heat Cap Inst (CO2)"))


for files in filenames[-1:]:
    # ------------------ Map Capacity Plots (Elec + Heat +Transport) ------------------#
    # Path to save files
    path = figurePath + "Mismatch\\Map Capacity\\"

    # --- Elec ---
    # Import network
    network = pypsa.Network(directory+files)
    fig1, fig2, fig3 = MapCapacityOriginal(network, files, ncol=3)
    SavePlot(fig1, path, title=(files[12:-3] + " - Map Capacity Elec Generator"))
    SavePlot(fig2, path, title=(files[12:-3] + " - Map Capacity Elec Storage Energy"))
    SavePlot(fig3, path, title=(files[12:-3] + " - Map Capacity Elec Storage Power"))
    
    # --- Heat ---
    # Import network
    network = pypsa.Network(directory+files)
    fig4, fig5, fig6 = MapCapacityHeat(network, files, ncol=2)
    SavePlot(fig4, path, title=(files[12:-3] + " - Map Capacity Heat Generator"))
    SavePlot(fig5, path, title=(files[12:-3] + " - Map Capacity Heat Storage Energy"))
    SavePlot(fig6, path, title=(files[12:-3] + " - Map Capacity Heat Storage Power"))



    # -------------------- Map Energy Plot (Elec + Heat + Transport) -------------------#
    # Path for saving file
    path = figurePath + "Mismatch\\Map Energy Distribution\\"
    
    # Import network
    network = pypsa.Network(directory+files)
    
    # --- Elec ---
    figElec = MapCapacityElectricityEnergy(network, files)
    SavePlot(figElec, path, title=(files[12:-3] + " - Elec Energy Production"))
    
    # --- Heat ---
    # Heating energy
    figHeat = MapCapacityHeatEnergy(network, files)
    SavePlot(figHeat, path, title=(files[12:-3] + " - Heat Energy Production"))
    

#%%
# --------------------- Map PC Plot (Elec + Heat + Transport) ----------------------#
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

# Plot map PC for mismatch electricity
titlePlotElec = "Mismatch for electricity only"
for i in np.arange(6):
    fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1)) #, titlePlotElec, titleFileName)
    title = (file[12:-3] + " - Map PC Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Heat ---
# Heat load for each country
loadHeat = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "heat" in country]].groupby(network.loads.bus.str.slice(0,2),axis=1).sum()

# Heat generators for each country (solar collectors)
# Because some countries have urban collectors, while other have central collectors, 
# additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the heat generators and remove 'NaN' values.
generationHeatSolar = network.generators_t.p[dataNames + " solar thermal collector"]
generationHeatSolar.columns = generationHeatSolar.columns.str.slice(0,2)

# Urban heat
generationHeatUrbanSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "urban" in country]]
generationHeatUrbanSingle.columns = generationHeatUrbanSingle.columns.str.slice(0,2)
generationHeatUrban = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationHeatUrban += generationHeatUrbanSingle
generationHeatUrban = generationHeatUrban.replace(np.nan,0)

# Central heat
generationHeatCentralSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "central" in country]]
generationHeatCentralSingle.columns = generationHeatCentralSingle.columns.str.slice(0,2)
generationHeatCentral = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationHeatCentral += generationHeatCentralSingle
generationHeatCentral = generationHeatCentral.replace(np.nan,0)

# Combine generation for heat
generationHeat = generationHeatSolar + generationHeatUrban + generationHeatCentral

# Mismatch electricity
mismatchHeat = generationHeat - loadHeat

# PCA on mismatch for Heating
eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(mismatchHeat)

# Plot map PC for mismatch heat
titlePlotHeat = "Mismatch for heating only"
for i in np.arange(6):
    fig = MAP(eigenVectorsHeat, eigenValuesHeat, dataNames, (i + 1)) #, titlePlotHeat, titleFileName)
    title = (file[12:-3] + " - Map PC Heat Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Transport ---
# Transport load for each country
loadTransport = network.loads_t.p_set[dataNames + ' transport']

# Generation transport
generationTransport = pd.DataFrame(data=np.zeros([8760,30]), index=timeIndex, columns=(dataNames + ' transport'))

# Mismatch transport
mismatchTransport = generationTransport - loadTransport

# PCA on mismatch for transport
eigenValuesTrans, eigenVectorsTrans, varianceExplainedTrans, normConstTrans, TTrans = PCA(mismatchTransport)

# Plot map PC for mismatch transport
titlePlotTrans = "Mismatch for transport only"
for i in np.arange(6):
    fig = MAP(eigenVectorsTrans, eigenValuesTrans, dataNames, (i + 1)) #, titlePlotTrans, titleFileName)
    title = (file[12:-3] + " - Map PC v2g Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# ----------------------- FFT Plot (Elec + Heat + Transport) -----------------------#
# Path to save FFT plots
path = figurePath + "Mismatch\\FFT\\"

# --- Elec ---
file_name = "Electricity mismatch - " + file
for i in np.arange(6):
    fig = FFTPlot(TElec.T, varianceExplainedElec, title=file_name, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Heat ---
file_name = "Heating mismatch - " + file
for i in np.arange(6):
    fig = FFTPlot(THeat.T, varianceExplainedHeat, title=file_name, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT Heat Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Transport ---
file_name = "Transport mismatch - " + file
for i in np.arange(6):
    fig = FFTPlot(TTrans.T, varianceExplainedTrans, title=file_name, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT v2g Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# -------------------- Seasonal Plot (Elec + Heat + Transport) ---------------------#
# Path to save seasonal plots
path = figurePath + "Mismatch\\Seasonal\\"

# --- Elec ---
file_name = "Electricity mismatch - " + file
for i in np.arange(6):
    fig = seasonPlot(TElec, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=6)
    title = (file[12:-3] + " - Seasonal Plot Elec Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Heat ---
file_name = "Heating mismatch - " + file
for i in np.arange(6):
    fig = seasonPlot(THeat, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=6)
    title = (file[12:-3] + " - Seasonal Plot Heat Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Transport ---
file_name = "Transport mismatch - " + file
for i in np.arange(6):
    fig = seasonPlot(TTrans, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=6)
    title = (file[12:-3] + " - Seasonal Plot v2g Mismatch (lambda " + str(i+1) + ")")
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

# --- Heat ---
file_name = "Heating mismatch - " + file
for i in np.arange(6):
    fig = FFTseasonPlot(THeat, timeIndex, varianceExplainedHeat, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file[12:-3] + " - Timeseries Plot Heat Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)
    
# --- Transport ---
file_name = "Transport mismatch - " + file
for i in np.arange(6):
    fig = FFTseasonPlot(TTrans, timeIndex, varianceExplainedTrans, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file[12:-3] + " - Timeseries Plot v2g Mismatch (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ----------------- Contribution plot (Elec + Heat + Transport) ------------------- #
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


# --- Heat ---
# Contribution
dircConHeat = Contribution(network, "heat")
lambdaCollectedConHeat = ConValueGenerator(normConstHeat, dircConHeat, eigenVectorsHeat)

for i in range(6):
    fig = ConPlot(eigenValuesHeat,lambdaCollectedConHeat,i+1,10,suptitle=("Heating Contribution - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Contribution Plot Heat (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)
    

# ------------------- Response plot (Elec + Heat + Transport) -------------------- #
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
    

# --- Heat ---
# Response
dircResHeat = HeatResponse(network,True)
lambdaCollectedResHeat = ConValueGenerator(normConstHeat, dircResHeat, eigenVectorsHeat)

for i in range(6):
    fig = ConPlot(eigenValuesHeat,lambdaCollectedResHeat,i+1,10,suptitle=("Heating Response - " + file[12:-3]),dpi=100)
    title = (file[12:-3] + " - Response Plot Heat (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# ------------------- Covariance plot (Elec + Heat + Transport) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Covariance\\"

# --- Elec ---
# Covariance
covMatrixElec = CovValueGenerator(dircConElec, dircResElec , True, normConstElec,eigenVectorsElec).T

for i in range(6):
    fig = ConPlot(eigenValuesElec,covMatrixElec,i+1,10,suptitle=("Electricity Covariance - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Covariance Plot Elec (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)
    

# --- Heat ---
# Covariance
covMatrixHeat = CovValueGenerator(dircConHeat, dircResHeat , True, normConstHeat, eigenVectorsHeat).T

for i in range(6):
    fig = ConPlot(eigenValuesHeat,covMatrixHeat,i+1,10,suptitle=("Heating Covariance - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Covariance Plot Heat (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# ------------------- Combined Projection plot (Elec + Heat + Transport) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Projection\\"

# --- Elec ---
for i in range(6):
    fig = CombConPlot(eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec, i+1, depth = 6)#, suptitle=("Electricity Projection - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Projection Plot Elec (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)
    

# --- Heat ---
for i in range(6):
    fig = CombConPlot(eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat, i+1, depth = 5)#, suptitle=("Heating Projection - " + file[12:-3]),dpi=200)
    title = (file[12:-3] + " - Projection Plot Heat (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ------------------- PC1 and PC2 combined plot (Elec + Heat + Transport) -------------------- #
# Path to save contribution plots
path = figurePath + "Mismatch\\Combined Plot\\"

# --- Elec --- 
fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withProjection")#,suptitle=("Electricity Mismatch - " + file[12:-3]),dpi=200,depth=3)
title = (file[12:-3] + " - Combined Plot Elec (lambda 1 & 2)")
SavePlot(fig, path, title)

fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="onlyProjection")#,suptitle=("Electricity Mismatch - " + file[12:-3]),dpi=200,depth=3)
title = (file[12:-3] + " - Comb Plot Elec only Proj (lambda 1 & 2)")
SavePlot(fig, path, title)



# --- Heat --- 
fig = PC1and2Plotter(THeat, timeIndex, [1,2], eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat,PCType="withProjection")#,suptitle=("Heating Mismatch - " + file[12:-3]),dpi=200,depth=3)
title = (file[12:-3] + " - Combined Plot Heat (lambda 1 & 2)")
SavePlot(fig, path, title)

fig = PC1and2Plotter(THeat, timeIndex, [1,2], eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat,PCType="onlyProjection")#,suptitle=("Heating Mismatch - " + file[12:-3]),dpi=200,depth=3)
title = (file[12:-3] + " - Comb Plot Heat only Proj (lambda 1 & 2)")
SavePlot(fig, path, title)



# --- Transport --- 
fig = PC1and2Plotter(TTrans, timeIndex, [1,2], eigenValuesTrans, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withoutProjection")#,suptitle=("Transport Mismatch - " + file[12:-3]),dpi=200,depth=3)
title = (file[12:-3] + " - Combined Plot v2g (lambda 1 & 2)")
SavePlot(fig, path, title)


#%%


# ---------------------- Bar plot CO2 constraint --------------------------- #
# Path to save bar plots
path = figurePath + "Mismatch\\Bar\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]


# Variable to store mismatch PC componentns for each network
barMatrixCO2Elec = []
barMatrixCO2Heat = []
barMatrixCO2Trans = []

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
    
    
    # --------------------------- Heat -------------------------------#
    # Heat load for each country
    loadHeat = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "heat" in country]].groupby(network.loads.bus.str.slice(0,2),axis=1).sum()
    
    # Heat generators for each country (solar collectors)
    # Because some countries have urban collectors, while other have central collectors, 
    # additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the heat generators and remove 'NaN' values.
    generationHeatSolar = network.generators_t.p[dataNames + " solar thermal collector"]
    generationHeatSolar.columns = generationHeatSolar.columns.str.slice(0,2)
    
    # Urban heat
    generationHeatUrbanSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "urban" in country]]
    generationHeatUrbanSingle.columns = generationHeatUrbanSingle.columns.str.slice(0,2)
    generationHeatUrban = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationHeatUrban += generationHeatUrbanSingle
    generationHeatUrban = generationHeatUrban.replace(np.nan,0)
    
    # Central heat
    generationHeatCentralSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "central" in country]]
    generationHeatCentralSingle.columns = generationHeatCentralSingle.columns.str.slice(0,2)
    generationHeatCentral = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationHeatCentral += generationHeatCentralSingle
    generationHeatCentral = generationHeatCentral.replace(np.nan,0)
    
    # Combine generation for heat
    generationHeat = generationHeatSolar + generationHeatUrban + generationHeatCentral
    
    # Mismatch electricity
    mismatchHeat = generationHeat - loadHeat
    
    # PCA on mismatch for electricity
    eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(mismatchHeat)
    
    # Append value to matrix
    barMatrixCO2Heat.append(varianceExplainedHeat)
    
    
    
    
    # ----------------------------- Transport --------------------------------#
    # Transport load for each country
    loadTransport = network.loads_t.p_set[dataNames + ' transport']
    
    # Generation transport
    generationTransport = pd.DataFrame(data=np.zeros([8760,30]), index=timeIndex, columns=(dataNames + ' transport'))
    
    # Mismatch transport
    mismatchTransport = generationTransport - loadTransport
    
    # PCA on mismatch for transport
    eigenValuesTrans, eigenVectorsTrans, varianceExplainedTrans, normConstTrans, TTrans = PCA(mismatchTransport)
    
    # Append value to matrix
    barMatrixCO2Trans.append(varianceExplainedTrans)

constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
title = "Number of PC describing variance of network as a function of $CO_{2}$ constraint"
xlabel = "$CO_{2}$ constraint"


suptitleElec = ("Electricity Mismatch - " + file[12:-14])
fig = BAR(barMatrixCO2Elec, 10, filename_CO2, constraints, title, xlabel, suptitleElec)
titleBarCO2Elec = (file[12:-14] + " - Bar CO2 Elec Mismatch")
SavePlot(fig, path, titleBarCO2Elec)

suptitleHeat = ("Heating Mismatch - " + file[12:-14])
fig = BAR(barMatrixCO2Heat, 10, filename_CO2, constraints, title, xlabel, suptitleHeat)
titleBarCO2Heat = (file[12:-14] + " - Bar CO2 Heat Mismatch")
SavePlot(fig, path, titleBarCO2Heat)

suptitleTrans = ("Transport Mismatch - " + file[12:-14])
fig = BAR(barMatrixCO2Trans, 10, filename_CO2, constraints, title, xlabel, suptitleTrans)
titleBarCO2Trans = (file[12:-14] + " - Bar CO2 v2g Mismatch")
SavePlot(fig, path, titleBarCO2Trans)

# ------------------ Change in contribution and response CO2 ----------------------- #


# Variable to store lambda values
lambdaContributionElec = []
lambdaContributionHeat = []
lambdaResponseElec     = []
lambdaResponseHeat     = []
lambdaCovarianceElec   = []
lambdaCovarianceHeat   = []



# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]


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
    lambdaCollected = ConValueGenerator(normConstElec, dircConElec, eigenVectorsElec)
    lambdaContributionElec.append(lambdaCollected)
    
    # Response Elec
    dircResElec = ElecResponse(network,True)
    lambdaCollected = ConValueGenerator(normConstElec, dircResElec, eigenVectorsElec)
    lambdaResponseElec.append(lambdaCollected)
    
    # Covariance Elec
    covMatrix = CovValueGenerator(dircConElec, dircResElec , True, normConstElec,eigenVectorsElec)
    lambdaCovarianceElec.append(covMatrix.T)    
    
    # --------------------------- Heat -------------------------------#
    # Heat load for each country
    loadHeat = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "heat" in country]].groupby(network.loads.bus.str.slice(0,2),axis=1).sum()
    
    # Heat generators for each country (solar collectors)
    # Because some countries have urban collectors, while other have central collectors, 
    # additional methods have to be implemented to make it at 8760 x 30 matrix
    # Create empty array of 8760 x 30, add the heat generators and remove 'NaN' values.
    generationHeatSolar = network.generators_t.p[dataNames + " solar thermal collector"]
    generationHeatSolar.columns = generationHeatSolar.columns.str.slice(0,2)
    
    # Urban heat
    generationHeatUrbanSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "urban" in country]]
    generationHeatUrbanSingle.columns = generationHeatUrbanSingle.columns.str.slice(0,2)
    generationHeatUrban = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationHeatUrban += generationHeatUrbanSingle
    generationHeatUrban = generationHeatUrban.replace(np.nan,0)
    
    # Central heat
    generationHeatCentralSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "central" in country]]
    generationHeatCentralSingle.columns = generationHeatCentralSingle.columns.str.slice(0,2)
    generationHeatCentral = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
    generationHeatCentral += generationHeatCentralSingle
    generationHeatCentral = generationHeatCentral.replace(np.nan,0)
    
    # Combine generation for heat
    generationHeat = generationHeatSolar + generationHeatUrban + generationHeatCentral
    
    # Mismatch electricity
    mismatchHeat = generationHeat - loadHeat
    
    # PCA on mismatch for electricity
    eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(mismatchHeat)

    # Contribution Heat
    dircConHeat = Contribution(network, "heat")
    lambdaCollected = ConValueGenerator(normConstHeat, dircConHeat, eigenVectorsHeat)
    lambdaContributionHeat.append(lambdaCollected)

    # Response Heat
    dircResHeat = HeatResponse(network,True)
    lambdaCollected = ConValueGenerator(normConstHeat, dircResHeat, eigenVectorsHeat)
    lambdaResponseHeat.append(lambdaCollected)

    # Covariance Heat
    covMatrix = CovValueGenerator(dircConHeat, dircResHeat , True, normConstHeat,eigenVectorsHeat)
    lambdaCovarianceHeat.append(covMatrix.T)


# general terms
pathContibution = figurePath + "Mismatch\\Change in Contribution\\"
pathResponse = figurePath + "Mismatch\\Change in Response\\"
pathCovariance = figurePath + "Mismatch\\Change in Covariance\\"
#%%
# Plot change in elec contribution
figtitle = "Change in electricity contribution as a function of CO2 constraint"
fig = ChangeContributionElec(lambdaContributionElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec cont (CO2)"
SavePlot(fig, pathContibution, saveTitle)

figtitle = "Change in electricity contribution as a function of CO2 constraint"
fig = ChangeContributionElec(lambdaContributionElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec cont app (CO2)"
SavePlot(fig, pathContibution, saveTitle)

# Plot change in heat contribution
figtitle = "Change in heating contribution as a function of CO2 constraint"
fig = ChangeContributionHeat(lambdaContributionHeat, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in heat cont (CO2)"
SavePlot(fig, pathContibution, saveTitle)

figtitle = "Change in heating contribution as a function of CO2 constraint"
fig = ChangeContributionHeat(lambdaContributionHeat, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in heat cont app (CO2)"
SavePlot(fig, pathContibution, saveTitle)

# Plot change in elec response
figtitle = "Change in electricity response as a function of CO2 constraint"
fig = ChangeResponseElec(lambdaResponseElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec response (CO2)"
SavePlot(fig, pathResponse, saveTitle)

figtitle = "Change in electricity response as a function of CO2 constraint"
fig = ChangeResponseElec(lambdaResponseElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec response app (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in heat response
figtitle = "Change in heating response as a function of CO2 constraint"
fig = ChangeResponseHeat(lambdaResponseHeat, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in heat response (CO2)"
SavePlot(fig, pathResponse, saveTitle)

figtitle = "Change in heating response as a function of CO2 constraint"
fig = ChangeResponseHeat(lambdaResponseHeat, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in heat response app (CO2)"
SavePlot(fig, pathResponse, saveTitle)


# Plot change in elec covariance response
figtitle = "Change in electricity covariance response as a function of CO2 constraint"
fig = ChangeResponseCov(lambdaResponseElec, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec cov response (CO2)"
SavePlot(fig, pathResponse, saveTitle)

figtitle = "Change in electricity covariance response as a function of CO2 constraint"
fig = ChangeResponseCov(lambdaResponseElec, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec cov response app (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in heat covariance response
figtitle = "Change in heating covariance response as a function of CO2 constraint"
fig = ChangeResponseCov(lambdaResponseHeat, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in heat cov response (CO2)"
SavePlot(fig, pathResponse, saveTitle)

figtitle = "Change in heating covariance response as a function of CO2 constraint"
fig = ChangeResponseCov(lambdaResponseHeat, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in heat cov response app (CO2)"
SavePlot(fig, pathResponse, saveTitle)

# Plot change in elec covariance
figtitle = "Change in electricity covariance between mismatch and response as a function of CO2 constraint"
fig = ChangeCovariance(lambdaCovarianceElec, collectTerms=True, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in elec covariance (CO2)"
SavePlot(fig, pathCovariance, saveTitle)

figtitle = "Change in electricity covariance between mismatch and response as a function of CO2 constraint"
fig = ChangeCovariance(lambdaCovarianceElec, collectTerms=True, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in elec covariance app (CO2)"
SavePlot(fig, pathCovariance, saveTitle)

# Plot change in heat covariance
figtitle = "Change in heating covariance between mismatch and response as a function of CO2 constraint"
fig = ChangeCovariance(lambdaCovarianceHeat, collectTerms=True, rotate=True, PC=2) #figtitle
saveTitle = file[12:-14] + " - Change in heat covariance (CO2)"
SavePlot(fig, pathCovariance, saveTitle)

figtitle = "Change in heating covariance between mismatch and response as a function of CO2 constraint"
fig = ChangeCovariance(lambdaCovarianceHeat, collectTerms=True, rotate=False, PC=6) #figtitle
saveTitle = file[12:-14] + " - Change in heat covariance app (CO2)"
SavePlot(fig, pathCovariance, saveTitle)



#%%


##############################################################################
##############################################################################

################################ NODAL PRICE #################################

##############################################################################
##############################################################################

# File name
file = "postnetwork-elec_heat_v2g50_0.125_0.05.h5"

# Import network
network = pypsa.Network(directory+file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# ----------------------- Map PC Plot (Elec + Heat + Transport) --------------------#
# Path to save plots
path = figurePath + "Nodal Price\\Map PC\\"

# --- Elec ---
# Prices for electricity for each country (restricted to 1000 €/MWh)
priceElec = FilterPrice(network.buses_t.marginal_price[dataNames],465)

# PCA on nodal prices for electricity
eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)

# Plot map PC for electricity nodal prices
titlePlotElec = "Nodal price for electricity only"
for i in np.arange(6):
    fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1)) #, titlePlotElec, titleFileName)
    title = (file[12:-3] + " - Map PC Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Heat ---
# Prices for heat for each country (restricted to 1000 €/MWh)
priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
priceHeat.columns = priceHeat.columns + " heat"
priceHeat = FilterPrice(priceHeat,465)

# PCA on nodal prices for heating
eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(priceHeat)

# Plot map PC for heating nodal prices
titlePlotHeat = "Nodal price for heating only"
for i in np.arange(6):
    fig = MAP(eigenVectorsHeat, eigenValuesHeat, dataNames, (i + 1)) #, titlePlotHeat, titleFileName)
    title = (file[12:-3] + " - Map PC Heat NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Transport ---
# Prices for heat for each country (restricted to 1000 €/MWh)
priceTrans = FilterPrice(network.buses_t.marginal_price[dataNames + " EV battery"],465)

# PCA on nodal prices for heating
eigenValuesTrans, eigenVectorsTrans, varianceExplainedTrans, normConstTrans, TTrans = PCA(priceTrans)

# Plot map PC for heating nodal prices
titlePlotTrans = "Nodal price for transport only"
for i in np.arange(6):
    fig = MAP(eigenVectorsTrans, eigenValuesTrans, dataNames, (i + 1)) #, titlePlotTrans, titleFileName)
    title = (file[12:-3] + " - Map PC v2g NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# ------------------------ FFT Plot (Elec + Heat + Transport) -----------------------#
# Path to save FFT plots
path = figurePath + "Nodal Price\\FFT\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + file
for i in np.arange(6):
    fig = FFTPlot(TElec.T, varianceExplainedElec, title=file_name, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# --- Heat ---
file_name = "Heating Nodal Price - " + file
for i in np.arange(6):
    fig = FFTPlot(THeat.T, varianceExplainedHeat, title=file_name, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT Heat NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Transport ---
file_name = "Transport Nodal Price - " + file
for i in np.arange(6):
    fig = FFTPlot(TTrans.T, varianceExplainedTrans, title=file_name, PC_NO = (i+1))
    title = (file[12:-3] + " - FFT v2g NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ----------------------- Seasonal Plot (Elec + Heat + Transport) ------------------------#
# Path to save seasonal plots
path = figurePath + "Nodal Price\\Seasonal\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + file
for i in np.arange(6):
    fig = seasonPlot(TElec, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=6)
    title = (file[12:-3] + " - Seasonal Plot Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Heat ---
file_name = "Heating Nodal Price - " + file
for i in np.arange(6):
    fig = seasonPlot(THeat, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=6)
    title = (file[12:-3] + " - Seasonal Plot Heat NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# --- Transport ---
file_name = "Transport Nodal Price - " + file
for i in np.arange(6):
    fig = seasonPlot(TTrans, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=6)
    title = (file[12:-3] + " - Seasonal Plot v2g NP (lambda " + str(i+1) + ")")
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

# --- Heat ---
file_name = "Heating Nodal Price - " + file
for i in np.arange(6):
    fig = FFTseasonPlot(THeat, timeIndex, varianceExplainedHeat, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file[12:-3] + " - Timeseries Plot Heat NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)
    
# --- Transport ---
file_name = "Transport Nodal Price - " + file
for i in np.arange(6):
    fig = FFTseasonPlot(TTrans, timeIndex, varianceExplainedTrans, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file[12:-3] + " - Timeseries Plot v2g NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ------------------- PC1 and PC2 combined plot (Elec + Heat + Transport) -------------------- #
# Path to save contribution plots
path = figurePath + "Nodal Price\\Combined Plot\\"

# --- Elec --- 
fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withoutProjection")#,suptitle=("Electricity Nodal Price - " + file[12:-3]),dpi=200)
title = (file[12:-3] + " - Combined Plot Elec NP (lambda 1 & 2)")
SavePlot(fig, path, title)

# --- Heat --- 
fig = PC1and2Plotter(THeat, timeIndex, [1,2], eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat,PCType="withoutProjection")#,suptitle=("Heating Nodal Price - " + file[12:-3]),dpi=200)
title = (file[12:-3] + " - Combined Plot Heat (lambda 1 & 2)")
SavePlot(fig, path, title)

# --- Transport --- 
fig = PC1and2Plotter(TTrans, timeIndex, [1,2], eigenValuesTrans, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withoutProjection")#,suptitle=("Transport Nodal Price - " + file[12:-3]),dpi=200)
title = (file[12:-3] + " - Combined Plot v2g NP (lambda 1 & 2)")
SavePlot(fig, path, title)


# ---------------------- Bar plot CO2 constraint --------------------------- #
# Path to save bar plots
path = figurePath + "Nodal Price\\Bar\\"

# Name of file (must be in correct folder location)
filename_CO2 = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

# Variable to store nodal price PC componentns for each network
barMatrixCO2Elec = []
barMatrixCO2Heat = []
barMatrixCO2Trans = []

# Variable to store nodal price mean and standard variation
meanPriceElec = []
quantileMeanPriceElec = []
quantileMinPriceElec = []
meanPriceHeat = []
quantileMeanPriceHeat = []
quantileMinPriceHeat = []
meanPriceTrans = []
quantileMeanPriceTrans = []
quantileMinPriceTrans = []

for file in filename_CO2:
    # Network
    network = pypsa.Network(directory + file)

    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    
    # --- Elec ---
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames],465)

    # PCA on nodal prices for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)
    
    # Append value to matrix
    barMatrixCO2Elec.append(varianceExplainedElec)
    
    
    # --- Heat ---
    # Prices for heat for each country (restricted to 1000 €/MWh)
    priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
    priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
    priceHeat.columns = priceHeat.columns + " heat"
    priceHeat = FilterPrice(priceHeat,465)
    
    # PCA on nodal prices for heating
    eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(priceHeat)
    
    # Append value to matrix
    barMatrixCO2Heat.append(varianceExplainedHeat)
    
    
    # --- Transport ---
    # Prices for transport for each country (restricted to 1000 €/MWh)
    priceTrans = FilterPrice(network.buses_t.marginal_price[dataNames + " EV battery"],465)
    
    # PCA on nodal prices for heating
    eigenValuesTrans, eigenVectorsTrans, varianceExplainedTrans, normConstTrans, TTrans = PCA(priceTrans)
    
    # Append value to matrix
    barMatrixCO2Trans.append(varianceExplainedTrans)
    
    # ----------------------- NP Mean (Elec + Heat + Transport) --------------------#
    # --- Elec ---
    # Mean price for country
    minPrice = priceElec.min().mean()
    meanPrice = priceElec.mean().mean()
    
    # append min, max and mean to matrix
    meanPriceElec.append([minPrice, meanPrice])

    # --- Heat ---
    # Mean price for country
    minPrice = priceHeat.min().mean()
    meanPrice = priceHeat.mean().mean()
    
    # append min, max and mean to matrix
    meanPriceHeat.append([minPrice, meanPrice])
    
    # --- Transport ---
    # Mean price for country
    minPrice = priceTrans.min().mean()
    meanPrice = priceTrans.mean().mean()
    
    # append min, max and mean to matrix
    meanPriceTrans.append([minPrice, meanPrice])
    
    # ----------------------- NP Quantile (Elec+Heat+Transport) --------------------#
    # --- Elec ---
    # Mean price for country
    quantileMinPrice = np.quantile(priceElec.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPrice = np.quantile(priceElec.mean(),[0.05,0.25,0.75,0.95])
    
    # append min, max and mean to matrix
    quantileMeanPriceElec.append(quantileMeanPrice)
    quantileMinPriceElec.append(quantileMinPrice)
    
    # --- Heat ---
    # Mean price for country
    quantileMinPrice = np.quantile(priceHeat.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPrice = np.quantile(priceHeat.mean(),[0.05,0.25,0.75,0.95])
    
    # append min, max and mean to matrix
    quantileMeanPriceHeat.append(quantileMeanPrice)
    quantileMinPriceHeat.append(quantileMinPrice)
    
     # --- Transport ---
    # Mean price for country
    quantileMinPrice = np.quantile(priceTrans.min(),[0.05,0.25,0.75,0.95])
    quantileMeanPrice = np.quantile(priceTrans.mean(),[0.05,0.25,0.75,0.95])
    
    # append min, max and mean to matrix
    quantileMeanPriceTrans.append(quantileMeanPrice)
    quantileMinPriceTrans.append(quantileMinPrice)   


constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
title = "Number of PC describing variance of network as a function of $CO_{2}$ constraint"
xlabel = "$CO_{2}$ constraint"


suptitleElec = ("Electricity Nodal Price - " + file[12:-14])
fig = BAR(barMatrixCO2Elec, 10, filename_CO2, constraints, title, xlabel, suptitleElec)
titleBarCO2Elec = (file[12:-14] + " - Bar CO2 Elec NP")
SavePlot(fig, path, titleBarCO2Elec)


suptitleHeat = ("Heating Nodal Price - "  + file[12:-14])
fig = BAR(barMatrixCO2Heat, 10, filename_CO2, constraints, title, xlabel, suptitleHeat)
titleBarCO2Heat = (file[12:-14] + " - Bar CO2 Heat NP")
SavePlot(fig, path, titleBarCO2Heat)


suptitleTrans = ("Transport Nodal Price - "  + file[12:-14])
fig = BAR(barMatrixCO2Trans, 10, filename_CO2, constraints, title, xlabel, suptitleTrans)
titleBarCO2Trans = (file[12:-14] + " - Bar CO2 v2g NP")
SavePlot(fig, path, titleBarCO2Trans)


# ----------------------- Price evalution (Elec) --------------------#
path = figurePath + "Nodal Price\\Price Evolution\\"
title =  ("Electricity Nodal Price Evalution - "  + file[12:-14])
fig = PriceEvolution(meanPriceElec,quantileMeanPriceElec,quantileMinPriceElec, networktype="green", figsize=[10,4])#, title=title)
title =  (file[12:-14] + " - Elec NP CO2 Evolution")
SavePlot(fig, path, title)

# ----------------------- Price evalution (Heat) --------------------#
path = figurePath + "Nodal Price\\Price Evolution\\"
title =  ("Heating Nodal Price Evalution - "  + file[12:-14])
fig = PriceEvolution(meanPriceHeat,quantileMeanPriceHeat,quantileMinPriceHeat,networktype="green")#,title=title)
title =  (file[12:-14] + " - Heating NP CO2 Evolution")
SavePlot(fig, path, title)

# ----------------------- Price evalution (Transport) --------------------#
path = figurePath + "Nodal Price\\Price Evolution\\"
title =  ("Transport Nodal Price Evalution - "  + file[12:-14])
fig = PriceEvolution(meanPriceTrans,quantileMeanPriceTrans,quantileMinPriceTrans,networktype="green")#,title=title)
title =  (file[12:-14] + " - Transport NP CO2 Evolution")
SavePlot(fig, path, title)







#%%
##############################################################################
##############################################################################

################################# Coherence ##################################

##############################################################################
##############################################################################

# -------------------- Coherence Plot (Elec + Heat + Transport) ---------------------#
# File name
file = "postnetwork-elec_heat_v2g50_0.125_0.05.h5"

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

# Prices for each country (restricted to 1000 €/MWh)
priceElec = FilterPrice(network.buses_t.marginal_price[dataNames],465)

# Coherence between prices and mismatch
c1Elec, c2Elec, c3Elec = Coherence(mismatchElec, priceElec)

# Plot properties
title1 = "" # "Coherence 1: Electricity mismatch and nodal price"
title2 = "" # "Coherence 2: Electricity mismatch and nodal price"
title3 = "" # "Coherence 3: Electricity mismatch and nodal price"
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


# --- Heat ---
# Heat load for each country
loadHeat = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "heat" in country]].groupby(network.loads.bus.str.slice(0,2),axis=1).sum()

# Heat generators for each country (solar collectors)
# Because some countries have urban collectors, while other have central collectors, 
# additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the heat generators and remove 'NaN' values.
generationHeatSolar = network.generators_t.p[dataNames + " solar thermal collector"]
generationHeatSolar.columns = generationHeatSolar.columns.str.slice(0,2)

# Urban heat
generationHeatUrbanSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "urban" in country]]
generationHeatUrbanSingle.columns = generationHeatUrbanSingle.columns.str.slice(0,2)
generationHeatUrban = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationHeatUrban += generationHeatUrbanSingle
generationHeatUrban = generationHeatUrban.replace(np.nan,0)

# Central heat
generationHeatCentralSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "central" in country]]
generationHeatCentralSingle.columns = generationHeatCentralSingle.columns.str.slice(0,2)
generationHeatCentral = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationHeatCentral += generationHeatCentralSingle
generationHeatCentral = generationHeatCentral.replace(np.nan,0)

# Combine generation for heat
generationHeat = generationHeatSolar + generationHeatUrban + generationHeatCentral

# Mismatch electricity
mismatchHeat = generationHeat - loadHeat

# Prices for heat for each country (restricted to 1000 €/MWh)
priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
priceHeat.columns = priceHeat.columns + " heat"
priceHeat = FilterPrice(priceHeat,465)

# Coherence between prices and mismatch
c1Heat, c2Heat, c3Heat = Coherence(mismatchHeat, priceHeat)

# Plot properties
title1 = "" # "Coherence 1: Heating mismatch and nodal price"
title2 = "" # "Coherence 2: Heating mismatch and nodal price"
title3 = "" # "Coherence 3: Heating mismatch and nodal price"
xlabel = "Heat Mismatch"
ylabel="Heating Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1Heat.T, übertitle="", title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2Heat.T, übertitle="", title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3Heat.T, übertitle="", title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file[12:-3] + " - C1 heat mismatch and HNP"))
SavePlot(fig2, path, title = (file[12:-3] + " - C2 heat mismatch and HNP"))
SavePlot(fig3, path, title = (file[12:-3] + " - C3 heat mismatch and HNP"))

# Combined Plot
fig = CoherencePlotCombined(c1Heat.T, c2Heat.T, c3Heat.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file[12:-3] + " - C123 combined heat mismatch and HNP"))


# --- Transport ---
# Transport load for each country
loadTrans = network.loads_t.p_set[dataNames + ' transport']

# Generation transport
generationTrans = pd.DataFrame(data=np.zeros([8760,30]), index=timeIndex, columns=(dataNames + ' transport'))

# Mismatch transport
mismatchTrans = generationTrans - loadTrans

# Prices for transport for each country (restricted to 1000 €/MWh)
priceTrans = FilterPrice(network.buses_t.marginal_price[dataNames + " EV battery"],465)

# Coherence between prices and mismatch
c1Trans, c2Trans, c3Trans = Coherence(mismatchTrans, priceTrans)

# Plot properties
title1 = "" # "Coherence 1: Transport mismatch and nodal price"
title2 = "" # "Coherence 2: Transport mismatch and nodal price"
title3 = "" # "Coherence 3: Transport mismatch and nodal price"
xlabel = "Transport Mismatch"
ylabel="Transport Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1Trans.T, übertitle="", title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2Trans.T, übertitle="", title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3Trans.T, übertitle="", title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file[12:-3] + " - C1 v2g mismatch and TNP"))
SavePlot(fig2, path, title = (file[12:-3] + " - C2 v2g mismatch and TNP"))
SavePlot(fig3, path, title = (file[12:-3] + " - C3 v2g mismatch and TNP"))

# Combined Plot
fig = CoherencePlotCombined(c1Trans.T, c2Trans.T, c3Trans.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file[12:-3] + " - C123 combined v2g mismatch and TNP"))



# --- Elec/Heat Prices ---
# Coherence between elec prices and heat prices 
c1Price, c2Price, c3Price = Coherence(priceElec, priceHeat)

# Plot properties
title1 = "" # "Coherence 1: Electricity and heating nodal prices"
title2 = "" # "Coherence 2: Electricity and heating nodal prices"
title3 = "" # "Coherence 3: Electricity and heating nodal prices"
xlabel = "Electricity Prices"
ylabel = "Heating Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1Price.T, übertitle="", title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2Price.T, übertitle="", title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3Price.T, übertitle="", title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file[12:-3] + " - C1 ENP and HNP"))
SavePlot(fig2, path, title = (file[12:-3] + " - C2 ENP and HNP"))
SavePlot(fig3, path, title = (file[12:-3] + " - C3 ENP and HNP"))


# Combined Plot
fig = CoherencePlotCombined(c1Price.T, c2Price.T, c3Price.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file[12:-3] + " - C123 combined ENP and HNP"))




# --- Elec/Transport Prices ---
# Coherence between elec prices and heat prices 
c1Price, c2Price, c3Price = Coherence(priceElec, priceTrans)

# Plot properties
title1 = "" # "Coherence 1: Electricity and transport nodal prices"
title2 = "" # "Coherence 2: Electricity and transport nodal prices"
title3 = "" # "Coherence 3: Electricity and transport nodal prices"
xlabel = "Electricity Prices"
ylabel = "Transport Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1Price.T, übertitle="", title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2Price.T, übertitle="", title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3Price.T, übertitle="", title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file[12:-3] + " - C1 ENP and TNP"))
SavePlot(fig2, path, title = (file[12:-3] + " - C2 ENP and TNP"))
SavePlot(fig3, path, title = (file[12:-3] + " - C3 ENP and TNP"))

# Combined Plot
fig = CoherencePlotCombined(c1Price.T, c2Price.T, c3Price.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file[12:-3] + " - C123 combined ENP and TNP"))


# --- Heat/Transport Prices ---
# Coherence between elec prices and heat prices 
c1Price, c2Price, c3Price = Coherence(priceHeat, priceTrans)

# Plot properties
title1 = "" # "Coherence 1: Heating and transport nodal prices"
title2 = "" # "Coherence 2: Heating and transport nodal prices"
title3 = "" # "Coherence 3: Heating and transport nodal prices"
xlabel = "Heating Prices"
ylabel = "Transport Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1Price.T, übertitle="", title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2Price.T, übertitle="", title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3Price.T, übertitle="", title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file[12:-3] + " - C1 HNP and TNP"))
SavePlot(fig2, path, title = (file[12:-3] + " - C2 HNP and TNP"))
SavePlot(fig3, path, title = (file[12:-3] + " - C3 HNP and TNP"))



# Combined Plot
fig = CoherencePlotCombined(c1Price.T, c2Price.T, c3Price.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file[12:-3] + " - C123 combined HNP and TNP"))



# Finish timer
t1 = time.time() # End timer
total_time = round(t1-t0)
total_time_min = math.floor(total_time/60)
total_time_sec = round(total_time-(total_time_min*60))
print("\n \nThe code is now done running. It took %s min and %s sec." %(total_time_min,total_time_sec))





# 8 min og 40 sek








