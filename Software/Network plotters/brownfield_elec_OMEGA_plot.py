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
directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\brownfield_elec\\"


# Name of baseline network
filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
              "postnetwork-global_tyndp1.0_0_0.3.nc",
              "postnetwork-global_tyndp1.0_0_0.15.nc"]

for file in filenames:
    
    # Figure path
    figurePath = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Figures\\brownfield_elec\\"
    
    # Import network
    network = pypsa.Network()
    network.import_from_netcdf(directory + file)
    
    # Data names
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Time index
    timeIndex = network.buses_t.p.index
    
    ##############################################################################
    ##############################################################################
    
    ################################### MISMATCH #################################
    
    ##############################################################################
    ##############################################################################
    
    
    # ------------------ Map Capacity Plots (Elec + Heat) ------------------#
    # Path to save files
    path = figurePath + "Mismatch\\Map Capacity\\"
    
    # --- Elec ---
    # Import network
    network = pypsa.Network(directory+file)
    fig1, fig2, fig3 = MapCapacityOriginal(network, file)
    SavePlot(fig1, path, title=(file[12:-3] + " - Map Capacity Elec Generator"))
    SavePlot(fig2, path, title=(file[12:-3] + " - Map Capacity Elec Storage Energy"))
    SavePlot(fig3, path, title=(file[12:-3] + " - Map Capacity Elec Storage Power"))
     
    
    #------------- Map Backup Capacity Plot(Elec + Heat)---------------------#
    # Path to save files
    path = figurePath + "Mismatch\\Map Capacity\\"
    
    # Import network
    network = pypsa.Network(directory+file)
    
    # --- Elec ---
    fig7 = MapBackupElec(network, file)
    SavePlot(fig7, path, title=(file[12:-3] + " - Map Elec Backup Capacity"))
    
    
    
    # -------------------- Map Energy Plot (Elec + Heat) -------------------#
    # Path for saving file
    path = figurePath + "Mismatch\\Map Energy Distribution\\"
    
    # Import network
    network = pypsa.Network(directory+file)
    
    # --- Elec ---
    figElec = MapCapacityElectricityEnergy(network, file)
    SavePlot(figElec, path, title=(file[12:-3] + " - Elec Energy Production"))
    
    
    
    # --------------------- Map PC Plot (Elec + Heat) ----------------------#
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
        fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1), titlePlotElec, file)
        title = (file[12:-3] + " - Map PC Elec Mismatch (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)
    
    
    # ----------------------- FFT Plot (Elec + Heat) -----------------------#
    # Path to save FFT plots
    path = figurePath + "Mismatch\\FFT\\"
    
    # --- Elec ---
    file_name = "Electricity mismatch - " + file
    for i in np.arange(6):
        fig = FFTPlot(TElec.T, varianceExplainedElec, file_name, PC_NO = (i+1))
        title = (file[12:-3] + " - FFT Elec Mismatch (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)
    
    
    # -------------------- Seasonal Plot (Elec + Heat) ---------------------#
    # Path to save seasonal plots
    path = figurePath + "Mismatch\\Seasonal\\"
    
    # --- Elec ---
    file_name = "Electricity mismatch - " + file
    for i in np.arange(6):
        fig = seasonPlot(TElec, timeIndex, file_name, PC_NO=(i+1), PC_amount=6)
        title = (file[12:-3] + " - Seasonal Plot Elec Mismatch (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)

    
    
    # ----------------- Contribution plot (Elec + Heat) ------------------- #
    # Path to save contribution plots
    path = figurePath + "Mismatch\\Contribution\\"
    
    # --- Elec ---
    # Contribution
    dirc = Contribution(network, "elec")
    lambdaCollected = ConValueGenerator(normConstElec, dirc, eigenVectorsElec)
    
    for i in range(6):
        fig = ConPlot(eigenValuesElec,lambdaCollected,i+1,10,suptitle=("Electricity Contribution - " + file[12:-3]),dpi=200)
        title = (file[12:-3] + " - Contribution Plot Elec (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)
    
        
    
    # # ------------------- Response plot (Elec + Heat) -------------------- #
    # # Path to save contribution plots
    # path = figurePath + "Mismatch\\Response\\"
    
    # # --- Elec ---
    # # Response
    # dirc = ElecResponse(network,True)
    # lambdaCollected = ConValueGenerator(normConstElec, dirc, eigenVectorsElec)
    
    # for i in range(6):
    #     fig = ConPlot(eigenValuesElec,lambdaCollected,i+1,10,suptitle=("Electricity Response - " + file[12:-3]),dpi=200)
    #     title = (file[12:-3] + " - Response Plot Elec (lambda " + str(i+1) + ")")
    #     SavePlot(fig, path, title)
        

#%%

# Figure path for change in decarbonization plots
figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_elec\\"

# ---------------------- Bar plot CO2 constraint --------------------------- #
# Path to save bar plots
path = figurePath + "Bar\\"

filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
             "postnetwork-global_tyndp1.0_0_0.3.nc",
             "postnetwork-global_tyndp1.0_0_0.15.nc"]

# Variable to store mismatch PC componentns for each network
barMatrixElec = []

for file in filenames:
    
    # Import network
    network = pypsa.Network()
    network.import_from_netcdf(directory + file)
    
    # Data names
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Time index
    timeIndex = network.buses_t.p.index
    
    
    # --- Elec ---
    # Load electricity 
    loadElec = network.loads_t.p[dataNames]
    
    # Generation electricity
    generationElec = network.generators_t.p.groupby(network.generators_t.p.columns.str.slice(0,2), axis=1).sum()
    
    # Mismatch electricity
    mismatchElec = generationElec - loadElec
    
    # PCA on mismatch for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)
    
    # Append value to matrix
    barMatrixElec.append(varianceExplainedElec)
    



constraints = ["15%", "30%", "45%"]
title = "Number of PC describing variance of network as a function of decarbonization"
xlabel = "$CO_{2}$ constraint"

suptitleElec = ("Electricity Mismatch - " + filenames[0][12:-12])
fig = BAR(barMatrixElec, 10, filenames, constraints, title, xlabel, suptitleElec)
SavePlot(fig, path, (file[12:-8] + " - Bar Year Elec Mismatch"))



# # ------------------ Change in contribution and response CO2 ----------------------- #
# # Paths
# pathContibution = figurePath + "Change in Contribution\\"
# pathResponse = figurePath + "Change in Response\\"

# # Filenames
# filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
#              "postnetwork-global_tyndp1.0_0_0.3.nc",
#              "postnetwork-global_tyndp1.0_0_0.15.nc"]

# # Variable to store lambda values
# lambdaContributionElec = []
# lambdaResponseElec     = []

# for file in filenames:
    
#     # Import network
#     network = pypsa.Network()
#     network.import_from_netcdf(directory + file)
    
#     # Data names
#     dataNames = network.buses.index.str.slice(0,2).unique()
    
#     # Time index
#     timeIndex = network.buses_t.p.index
    
    
#     # --- Elec ---
#     # Load electricity 
#     loadElec = network.loads_t.p[dataNames]
    
#     # Generation electricity
#     generationElec = network.generators_t.p.groupby(network.generators_t.p.columns.str.slice(0,2), axis=1).sum()
    
#     # Mismatch electricity
#     mismatchElec = generationElec - loadElec
    
#     # PCA on mismatch for electricity
#     eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)
    
#     # Contribution Elec
#     dirc = Contribution(network, "elec")
#     lambdaCollected = ConValueGenerator(normConstElec, dirc, eigenVectorsElec)
#     lambdaContributionElec.append(lambdaCollected)
    
#     # Response Elec
#     dirc = ElecResponse(network,True)
#     lambdaCollected = ConValueGenerator(normConstElec, dirc, eigenVectorsElec)
#     lambdaResponseElec.append(lambdaCollected)
    
    

# # Plot change in elec contribution
# figtitle = "Change in electricity contribution as a function of decarbonization"
# fig = ChangeContributionElec(lambdaContributionElec, figtitle, networktype="brown")
# saveTitle = file[12:-8] + " - Change in elec cont (year)"
# SavePlot(fig, pathContibution, saveTitle)


# # Plot change in elec response
# figtitle = "Change in electricity response as a function of decarbonization"
# fig = ChangeResponseElec(lambdaResponseElec, figtitle, networktype="brown")
# saveTitle = file[12:-8] + " - Change in elec response (year)"
# SavePlot(fig, pathResponse, saveTitle)


# # Plot change in elec covariance response
# figtitle = "Change in electricity covariance response as a function of decarbonization"
# fig = ChangeResponseCov(lambdaResponseElec, figtitle)
# saveTitle = file[12:-8] + " - Change in elec cov response (year)"
# SavePlot(fig, pathResponse, saveTitle)



# # ------- Energy Production by different technologies as function of decarboinzation ----#
# # Path to save plots
# path = figurePath + ""

# # Filenames
# filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
#              "postnetwork-global_tyndp1.0_0_0.3.nc",
#              "postnetwork-global_tyndp1.0_0_0.15.nc"]

# fig1, fig2, fig3, fig4 = EnergyProductionBrownfield(directory, filenames)

# SavePlot(fig1, path, (filenames[0][12:-12] + " - total elec generation (year)"))
# SavePlot(fig2, path, (filenames[0][12:-12] + " - total elec storage (year)"))
# SavePlot(fig3, path, (filenames[0][12:-12] + " - total heat generation (year)"))
# SavePlot(fig4, path, (filenames[0][12:-12] + " - total heat storage (year)"))








#%%
##############################################################################
##############################################################################

################################ NODAL PRICE #################################

##############################################################################
##############################################################################

# Filenames
filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
             "postnetwork-global_tyndp1.0_0_0.3.nc",
             "postnetwork-global_tyndp1.0_0_0.15.nc"]

for file in filenames:
    
    # Figure path
    figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_elec\\"
    
    # Import network
    network = pypsa.Network()
    network.import_from_netcdf(directory + file)
    
    # Data names
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Time index
    timeIndex = network.buses_t.p.index

    # ----------------------- Map PC Plot (Elec + Heat) --------------------#
    # Path to save plots
    path = figurePath + "Nodal Price\\Map PC\\"
    
    # --- Elec ---
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = network.buses_t.marginal_price[dataNames].clip(-1000,1000)
    
    # PCA on nodal prices for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)
    
    # Plot map PC for electricity nodal prices
    titlePlotElec = "Nodal price for electricity only"
    for i in np.arange(6):
        fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1), titlePlotElec, file)
        title = (file[12:-3] + " - Map PC Elec NP (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)
    
    
    # ------------------------ FFT Plot (Elec + Heat) -----------------------#
    # Path to save FFT plots
    path = figurePath + "Nodal Price\\FFT\\"
    
    # --- Elec ---
    file_name = "Electricity Nodal Price - " + file
    for i in np.arange(6):
        fig = FFTPlot(TElec.T, varianceExplainedElec, file_name, PC_NO = (i+1))
        title = (file[12:-3] + " - FFT Elec NP (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)
    
    
    # ----------------------- Seasonal Plot (Elec + Heat) ------------------------#
    # Path to save seasonal plots
    path = figurePath + "Nodal Price\\Seasonal\\"
    
    # --- Elec ---
    file_name = "Electricity Nodal Price - " + file
    for i in np.arange(6):
        fig = seasonPlot(TElec, timeIndex, file_name, PC_NO=(i+1), PC_amount=6)
        title = (file[12:-3] + " - Seasonal Plot Elec NP (lambda " + str(i+1) + ")")
        SavePlot(fig, path, title)
    



# Figure path for change in decarbonization plots
figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_elec\\"

# ---------------------- Bar plot CO2 constraint --------------------------- #
# Path to save bar plots
path = figurePath + "Bar\\"

# Filenames
filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
             "postnetwork-global_tyndp1.0_0_0.3.nc",
             "postnetwork-global_tyndp1.0_0_0.15.nc"]

# Variable to store mismatch PC componentns for each network
barMatrixElec = []

for file in filenames:
    
    # Import network
    network = pypsa.Network()
    network.import_from_netcdf(directory + file)
    
    # Data names
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Time index
    timeIndex = network.buses_t.p.index
    
    
    # --- Elec ---
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = network.buses_t.marginal_price[dataNames].clip(-1000,1000)
    
    # PCA on nodal prices for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)
    
    # Append value to matrix
    barMatrixElec.append(varianceExplainedElec)



constraints = ["15%", "30%", "45%"]
title = "Number of PC describing variance of network as a function of decarbonization"
xlabel = "$CO_{2}$ constraint"

suptitleElec = ("Electricity Nodal Price - " + filenames[0][12:-12])
fig = BAR(barMatrixElec, 10, filenames, constraints, title, xlabel, suptitleElec)
SavePlot(fig, path, (file[12:-8] + " - Bar Year Elec NP"))




#%%

##############################################################################
##############################################################################

################################# Coherence ##################################

##############################################################################
##############################################################################


# Filenames
filenames = ["postnetwork-global_tyndp1.0_0_0.45.nc",
             "postnetwork-global_tyndp1.0_0_0.3.nc",
             "postnetwork-global_tyndp1.0_0_0.15.nc"]

for file in filenames:
    
    # Figure path
    figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_elec\\"
    
    # Path to save contribution plots
    path = figurePath + "Coherence\\"
    
    # Import network
    network = pypsa.Network()
    network.import_from_netcdf(directory + file)
    
    # Data names
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Time index
    timeIndex = network.buses_t.p.index

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
    
    # Prices for electricity for each country (restricted to 1000 €/MWh)
    priceElec = network.buses_t.marginal_price[dataNames].clip(-1000,1000)
    
    # Coherence between prices and mismatch
    c1Elec, c2Elec, c3Elec = Coherence(mismatchElec, priceElec)
    
    # Plot properties
    title1 = "Coherence 1: Electricity mismatch and nodal price"
    title2 = "Coherence 2: Electricity mismatch and nodal price"
    title3 = "Coherence 3: Electricity mismatch and nodal price"
    xlabel = "Electricity Mismatch"
    ylabel = "Electricity Nodal Prices"
    noX = 6
    noY = 6
    fig1 = CoherencePlot(dataMatrix=c1Elec.T, übertitle=file, title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
    fig2 = CoherencePlot(dataMatrix=c2Elec.T, übertitle=file, title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
    fig3 = CoherencePlot(dataMatrix=c3Elec.T, übertitle=file, title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
    SavePlot(fig1, path, title = (file[12:-3] + " - C1 elec mismatch and ENP"))
    SavePlot(fig2, path, title = (file[12:-3] + " - C2 elec mismatch and ENP"))
    SavePlot(fig3, path, title = (file[12:-3] + " - C3 elec mismatch and ENP"))
    
    
# Finish timer
t1 = time.time() # End timer
total_time = round(t1-t0)
total_time_min = math.floor(total_time/60)
total_time_sec = round(total_time-(total_time_min*60))
print("\n \nThe code is now done running. It took %s min and %s sec." %(total_time_min,total_time_sec))







