# Import libraries
import os
import sys
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import math

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *           


# Directory of file
directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\brownfield_heat\\version-Base\\postnetworks\\"

# Figure path
figurePath = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Figures\\brownfield_heat\\"
        

# Name of baseline network
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

constraints = ['2020', '2025', '2030', '2035', '2040', '2045', '2050']

# ------------------------- Curtailment (elec) ----------------------------- %
# Path to save bar plots
path = figurePath + "pre analysis\\"

title="Early Transition Path"
fig = Curtailment(directory=directory, files=filenamesGo, title=title, constraints=constraints, ylim=[-1,20], legendLoc="upper left", figsize=[5.2, 4], dpi=300)
SavePlot(fig, path, title=("Transition Path - elec curtailment (go)"))
         

title="Late Transition Path"
fig = Curtailment(directory=directory, files=filenamesWait, title=title, constraints=constraints, ylim=[-1,20], legendLoc="upper left", figsize=[5.2, 4], dpi=300)
SavePlot(fig, path, title=("Transition Path - elec curtailment (wait)"))


# ---------------------- Beginning of OMEGA for-loop ----------------------- %
# Which years to save plots from
safeYear = []
# years to choose from: ["2020","2025","2030","2035","2040","2045","2050"]

# Which type of brownfield to run
brownfieldType = ["Go"] 
# Brownfield to choose from: ["Go", "Wait"] 

# Create empty list
curtailment = []

# Create empty list
coherenceTypes = []

# Loop to run both go and wait
for i in range(len(brownfieldType)):
    
    if brownfieldType[i] == "Go":
        filenames = filenamesGo
        networkType = "Go"
    elif brownfieldType[i] == "Wait":
        filenames = filenamesWait
        networkType = "Wait"


    # Create empty list
    barMatrixGoElecMismatch = [] # Used for Mismatch
    barMatrixGoHeatMismatch = [] # Used for Mismatch
    lambdaContributionElec  = [] # Used for Mismatch
    lambdaContributionHeat  = [] # Used for Mismatch
    lambdaResponseElec      = [] # Used for Mismatch
    lambdaResponseHeat      = [] # Used for Mismatch
    lambdaCovarianceElec    = [] # Used for Mismatch
    lambdaCovarianceHeat    = [] # Used for Mismatch
    collectedMismatchElec   = [] # Used for Mismatch
    collectedMismatchHeat   = [] # Used for Mismatch
    windDis                 = [] # Used for Mismatch
    solarDis                = [] # Used for Mismatch
    rorDis                  = [] # Used for Mismatch
    barMatrixGoElecNP       = [] # Used for Nodal Price
    barMatrixGoHeatNP       = [] # Used for Nodal Price
    meanPriceElec           = [] # Used for Nodal Price
    quantileMeanPriceElec   = [] # Used for Nodal Price
    quantileMinPriceElec    = [] # Used for Nodal Price
    meanPriceHeat           = [] # Used for Nodal Price
    quantileMeanPriceHeat   = [] # Used for Nodal Price
    quantileMinPriceHeat    = [] # Used for Nodal Price
    collectedPriceElec      = [] # Used for Nodal Price
    collectedPriceHeat      = [] # Used for Nodal Price
    coherenceElecC1         = [] # Used for Coherence
    coherenceElecC2         = [] # Used for Coherence
    coherenceElecC3         = [] # Used for Coherence
    coherenceHeatC1         = [] # Used for Coherence
    coherenceHeatC2         = [] # Used for Coherence
    coherenceHeatC3         = [] # Used for Coherence
    coherencePriceC1        = [] # Used for Coherence
    coherencePriceC2        = [] # Used for Coherence
    coherencePriceC3        = [] # Used for Coherence
    
    
    
    #%% For loop through different networks
    
    # Loop through networks
    for file in filenames:
        
        # Year of current file (used for file path)
        if "go" in file:
            year = file[21:-3]
        elif "wait" in file:
            year = file[23:-3]
        
        # Figure path
        figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_heat\\" + networkType + "\\" + year + "\\"
        
        # Import network
        network = pypsa.Network()
        network.import_from_netcdf(directory + file)
        
        # Data names
        dataNames = network.buses.index.str.slice(0,2).unique()
        
        # Time index
        timeIndex = network.buses_t.p.index
        
        if year in safeYear:
        
            ##############################################################################   
            ################################### MISMATCH #################################    
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
            
            # --- Heat ---
            # Import network
            network = pypsa.Network(directory+file)
            fig5, fig6, fig7 = MapCapacityHeat(network, file)
            SavePlot(fig5, path, title=(file[12:-3] + " - Map Capacity Heat Generator"))
            SavePlot(fig6, path, title=(file[12:-3] + " - Map Capacity Heat Storage Energy"))
            SavePlot(fig7, path, title=(file[12:-3] + " - Map Capacity Heat Storage Power"))
            
            
            #------------- Map Backup Capacity Plot(Elec + Heat)---------------------#
            # Path to save files
            path = figurePath + "Mismatch\\Map Capacity\\"
            
            # Import network
            network = pypsa.Network(directory+file)
            
            # --- Elec ---
            fig8 = MapBackupElec(network, file)
            SavePlot(fig8, path, title=(file[12:-3] + " - Map Elec Backup Capacity"))
            
            # --- Heat ---
            fig9 = MapBackupHeat(network, file)
            SavePlot(fig9, path, title=(file[12:-3] + " - Map Heat Backup Capacity"))
            
            
            
            # -------------------- Map Energy Plot (Elec + Heat) -------------------#
            # Path for saving file
            path = figurePath + "Mismatch\\Map Energy Distribution\\"
            
            # Import network
            network = pypsa.Network(directory+file)
            
            # --- Elec ---
            figElec = MapCapacityElectricityEnergy(network, file)
            SavePlot(figElec, path, title=(file[12:-3] + " - Elec Energy Production"))
            
            # --- Heat ---
            # Heating energy
            figHeat = MapCapacityHeatEnergy(network, file)
            SavePlot(figHeat, path, title=(file[12:-3] + " - Heat Energy Production"))
        
        
        # --------------------- PCA (Elec + Heat) ----------------------#    
        # Import network
        #network = pypsa.Network(directory+file)
        
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
        
        # --- Heat ---
        # Heating load
        heat = network.loads_t.p[[x for x in network.loads_t.p.columns if "heat" in x]]
        cooling = network.loads_t.p[[x for x in network.loads_t.p.columns if "cooling" in x]]
        loadHeat = pd.concat([heat, cooling], axis=1).groupby(pd.concat([heat, cooling], axis=1).columns.str.slice(0,2), axis=1).sum()
        
        # Combine generation for heat
        generationHeat = pd.DataFrame(data=np.zeros(loadHeat.shape), index=timeIndex, columns=dataNames)
        
        # Mismatch electricity
        mismatchHeat = generationHeat - loadHeat
        
        # PCA on mismatch for electricity
        eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(mismatchHeat)
        
        
        if year in safeYear:
            
            # --------------------- Map PC Plot (Elec + Heat) ----------------------#
            # Path to save plots
            path = figurePath + "Mismatch\\Map PC\\"
            
            # Plot map PC for mismatch electricity
            titlePlotElec = "Mismatch for electricity only"
            for i in np.arange(6):
                fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1))#, titlePlotElec, file)
                title = (file[12:-3] + " - Map PC Elec Mismatch (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
            
       
            # Plot map PC for mismatch heat
            titlePlotHeat = "Mismatch for heating only"
            for i in np.arange(6):
                fig = MAP(eigenVectorsHeat, eigenValuesHeat, dataNames, (i + 1))#, titlePlotHeat, file)
                title = (file[12:-3] + " - Map PC Heat Mismatch (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
            
            
            
            # ----------------------- FFT Plot (Elec + Heat) -----------------------#
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
            
            
            
            # -------------------- Seasonal Plot (Elec + Heat) ---------------------#
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
            
            
            
            # ----------------- Contribution plot (Elec + Heat) ------------------- #
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
                fig = ConPlot(eigenValuesHeat,lambdaCollectedConHeat,i+1,6,suptitle=("Heating Contribution - " + file[12:-3]),dpi=200)
                title = (file[12:-3] + " - Contribution Plot Heat (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
                
            
            # ------------------- Response plot (Elec + Heat) -------------------- #
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
        
            # ------------------- Covariance plot (Elec + Heat) -------------------- #
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
        
            # ------------------- Combined Projection plot (Elec + Heat) -------------------- #
            # Path to save contribution plots
            path = figurePath + "Mismatch\\Projection\\"
            
            # --- Elec ---
            for i in range(6):
                fig = CombConPlot(eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec, i+1, depth = 8, suptitle=("Electricity Projection - " + file[12:-3]),dpi=200)
                title = (file[12:-3] + " - Projection Plot Elec (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
                
            
            # --- Heat ---
            for i in range(6):
                fig = CombConPlot(eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat, i+1, depth = 8, suptitle=("Heating Projection - " + file[12:-3]),dpi=200)
                title = (file[12:-3] + " - Projection Plot Heat (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
        
        
            # ------------------- PC1 and PC2 combined plot (Elec + Heat + Transport) -------------------- #
            # Path to save contribution plots
            path = figurePath + "Mismatch\\Combined Plot\\"
            
            # --- Elec --- 
            fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withProjection",dpi=200) # ,suptitle=("Electricity Mismatch - " + file[12:-3])
            title = (file[12:-3] + " - Combined Plot Elec (lambda 1 & 2)")
            SavePlot(fig, path, title)
            
            # --- Heat --- 
            fig = PC1and2Plotter(THeat, timeIndex, [1,2], eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat,PCType="withProjection",dpi=200,depth=3) # ,suptitle=("Heating Mismatch - " + file[12:-3])
            title = (file[12:-3] + " - Combined Plot Heat (lambda 1 & 2)")
            SavePlot(fig, path, title)
            
            fig = PC1and2Plotter(THeat, timeIndex, [1,2], eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat,PCType="onlyProjection",dpi=200,depth=3) # ,suptitle=("Heating Mismatch - " + file[12:-3])
            title = (file[12:-3] + " - Comb Plot Heat only Proj (lambda 1 & 2)")
            SavePlot(fig, path, title)
        
        
        
        
        # ------------------- append to Bar plot (Elec + Heat) -------------------- #
        # --- Elec ---
        # Append value to matrix
        barMatrixGoElecMismatch.append(varianceExplainedElec)            
        
        # --- Heat ---
        # Append value to matrix
        barMatrixGoHeatMismatch.append(varianceExplainedHeat)        
        
        # ------------------- append to contribution plot (Elec + Heat) -------------------- #
        # --- Elec ---
        dircConElec = Contribution(network, "elec")
        lambdaCollectedConElec = ConValueGenerator(normConstElec, dircConElec, eigenVectorsElec)
        # Append value to matrix
        lambdaContributionElec.append(lambdaCollectedConElec)        
        
        # --- Heat ---
        dircConHeat = Contribution(network, "heat")
        lambdaCollectedConHeat = ConValueGenerator(normConstHeat, dircConHeat, eigenVectorsHeat)
        # Append value to matrix
        lambdaContributionHeat.append(lambdaCollectedConHeat)
        
        # ------------------- append to response plot (Elec + Heat) -------------------- #
        # --- Elec ---
        dircResElec = ElecResponse(network,True)
        lambdaCollectedResElec = ConValueGenerator(normConstElec, dircResElec, eigenVectorsElec)
        # Append value to matrix
        lambdaResponseElec.append(lambdaCollectedResElec)  
        
        # --- Heat ---
        dircResHeat = HeatResponse(network,True)
        lambdaCollectedResHeat = ConValueGenerator(normConstHeat, dircResHeat, eigenVectorsHeat)
        # Append value to matrix
        lambdaResponseHeat.append(lambdaCollectedResHeat)
        
        
        # ------------------- append to response plot (Elec + Heat) -------------------- #
        # --- Elec ---
        covMatrix = CovValueGenerator(dircConElec, dircResElec , True, normConstElec,eigenVectorsElec)
        lambdaCovarianceElec.append(covMatrix.T)  
        
        # --- Heat ---
        covMatrix = CovValueGenerator(dircConHeat, dircResHeat , True, normConstHeat,eigenVectorsHeat)
        lambdaCovarianceHeat.append(covMatrix.T)      
        
        # ------------------- append to coherence between network (Elec + Heat) -------------------- #
        # --- Elec ---
        collectedMismatchElec.append(mismatchElec)
        
        #--- Heat ---
        collectedMismatchHeat.append(mismatchHeat)
        
        
        # ------------------- curtailment calculation (Elec + Heat) -------------------- #
        # Determine dispatchable energy for all generators at every hour
        dispatchable = network.generators_t.p
        
        # Determine non-dispatchable energy for all generators at every hour
        nonDispatchable = network.generators_t.p_max_pu * network.generators.p_nom_opt
        
        # Difference between dispatchable and non-dispatchable
        difference = nonDispatchable - dispatchable
        
        # Break into components and sum up the mean
        windDispatch = difference[[x for x in difference.columns if "wind" in x]].mean(axis=0).sum()
        solarDispatch = difference[[x for x in difference.columns if "solar" in x]].mean(axis=0).sum()
        rorDispatch = difference[[x for x in difference.columns if "ror" in x]].mean(axis=0).sum()
    
        # Load of network
        load = network.loads_t.p.mean(axis=0).sum()
        
        # Generation (wind, solar, ror)
        wind = network.generators_t.p[[x for x in network.generators_t.p.columns if "wind" in x]]
        wind = wind.groupby(wind.columns.str.slice(0,2), axis=1).sum().mean(axis=0).sum()
        solar = network.generators_t.p[[x for x in network.generators_t.p.columns if "solar" in x]].mean(axis=0).sum()
        ror = network.generators_t.p[[x for x in network.generators_t.p.columns if "ror" in x]].mean(axis=0).sum()
        
        # Append to array - Generator relative
        windDis.append((windDispatch/wind)*100)
        solarDis.append((solarDispatch/solar)*100)
        rorDis.append((rorDispatch/ror)*100)
        
        
        ##############################################################################   
        ################################# NODAL PRICE ################################    
        ##############################################################################     
        
        # ----------------------- NP PCA (Elec + Heat) --------------------#
        # --- Elec ---
        # Prices for electricity for each country (restricted to 1000 €/MWh)
        priceElec = FilterPrice(network.buses_t.marginal_price[dataNames],465)
        # PCA on nodal prices for electricity
        eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(priceElec)
        
        # --- Heat ---
        # Prices for heat for each country (restricted to 1000 €/MWh)
        priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
        priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
        priceHeat.columns = priceHeat.columns + " heat"
        priceHeat = FilterPrice(priceHeat,465)
        # PCA on nodal prices for heating
        eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(priceHeat)
        
        
    
        if year in safeYear:
        
            # ----------------------- Map PC Plot (Elec + Heat) --------------------#
            # Path to save plots
            path = figurePath + "Nodal Price\\Map PC\\"
            
            # --- Elec ---
            # Plot map PC for electricity nodal prices
            titlePlotElec = "Nodal price for electricity only"
            for i in np.arange(6):
                fig = MAP(eigenVectorsElec, eigenValuesElec, dataNames, (i + 1))#, titlePlotElec, file)
                title = (file[12:-3] + " - Map PC Elec NP (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
            
            # --- Heat ---
            # Plot map PC for heating nodal prices
            titlePlotHeat = "Nodal price for heating only"
            for i in np.arange(6):
                fig = MAP(eigenVectorsHeat, eigenValuesHeat, dataNames, (i + 1))#, titlePlotHeat, file)
                title = (file[12:-3] + " - Map PC Heat NP (lambda " + str(i+1) + ")")
                SavePlot(fig, path, title)
            
            
            
            # ------------------------ FFT Plot (Elec + Heat) -----------------------#
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
            
            
            
            # ----------------------- Seasonal Plot (Elec + Heat) ------------------------#
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
    
            # ------------------- PC1 and PC2 combined plot (Elec + Heat + Transport) -------------------- #
            # Path to save contribution plots
            path = figurePath + "Nodal Price\\Combined Plot\\"
            
            # --- Elec --- 
            fig = PC1and2Plotter(TElec, timeIndex, [1,2], eigenValuesElec, lambdaCollectedConElec, lambdaCollectedResElec, covMatrixElec,PCType="withoutProjection",dpi=200) # ,suptitle=("Electricity Mismatch - " + file[12:-3])
            title = (file[12:-3] + " - Combined Plot ENP (lambda 1 & 2)")
            SavePlot(fig, path, title)
            
            # --- Heat --- 
            fig = PC1and2Plotter(THeat, timeIndex, [1,2], eigenValuesHeat, lambdaCollectedConHeat, lambdaCollectedResHeat, covMatrixHeat,PCType="withoutProjection",dpi=200,depth=3) # ,suptitle=("Heating Mismatch - " + file[12:-3])
            title = (file[12:-3] + " - Combined Plot HNP (lambda 1 & 2)")
            SavePlot(fig, path, title)
            
    
    
        # ----------------------- Append to NP (Elec + Heat) ------------------------#
        # --- Elec ---
        # Append value to matrix
        barMatrixGoElecNP.append(varianceExplainedElec)
    
        # --- Heat ---
        # Append value to matrix
        barMatrixGoHeatNP.append(varianceExplainedHeat)
        
        # ----------------------- NP Mean (Elec + Heat) --------------------#
        # --- Elec ---
        # Mean price for country
        minPrice = priceElec.min().mean()
        meanPrice = priceElec.mean().mean()  
        # append min, max and mean to matrix
        meanPriceElec.append([minPrice, meanPrice])
        collectedPriceElec.append(priceElec)
        
        
        # --- Heat ---
        # Mean price for country
        minPrice = priceHeat.min().mean()
        meanPrice = priceHeat.mean().mean()
        maxPrice = priceHeat.max().mean()
        # append min, max and mean to matrix
        meanPriceHeat.append([minPrice, meanPrice])
        collectedPriceHeat.append(priceHeat)

        # ----------------------- NP Quantile (Elec+Heat) --------------------#
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
        
        ##############################################################################   
        ################################## COHERENCE #################################    
        ############################################################################## 
        
        # Figure path
        figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_heat\\" + networkType + "\\" + year + "\\"
        
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
        
        # Prices for electricity for each country (restricted to 1000 €/MWh)
        priceElec = FilterPrice(network.buses_t.marginal_price[dataNames],465)
        
        # Coherence between prices and mismatch
        c1Elec, c2Elec, c3Elec = Coherence(mismatchElec, priceElec)
        
        # --- Heat ---
        # Heating load
        heat = network.loads_t.p[[x for x in network.loads_t.p.columns if "heat" in x]]
        cooling = network.loads_t.p[[x for x in network.loads_t.p.columns if "cooling" in x]]
        loadHeat = pd.concat([heat, cooling], axis=1).groupby(pd.concat([heat, cooling], axis=1).columns.str.slice(0,2), axis=1).sum()
        
        # Combine generation for heat
        generationHeat = pd.DataFrame(data=np.zeros(loadHeat.shape), index=timeIndex, columns=dataNames)
        
        # Mismatch electricity
        mismatchHeat = generationHeat - loadHeat
        
        # Prices for heat for each country (restricted to 1000 €/MWh)
        priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
        priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
        priceHeat.columns = priceHeat.columns + " heat"
        priceHeat = FilterPrice(priceHeat,465)
        
        # Coherence between prices and mismatch
        c1Heat, c2Heat, c3Heat = Coherence(mismatchHeat, priceHeat)
        
        # --- Elec/Heat Prices ---
        # Coherence between elec prices and heat prices 
        c1Price, c2Price, c3Price = Coherence(priceElec, priceHeat)
        
        
        # ---------------------- Coherence for a single year --------------------------- #   
        if year in safeYear:
            # --- Elec ---
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
            
            
            
            # --- Heat ---       
            # Plot properties
            title1 = "Coherence 1: Heating mismatch and nodal price"
            title2 = "Coherence 2: Heating mismatch and nodal price"
            title3 = "Coherence 3: Heating mismatch and nodal price"
            xlabel = "Heat Mismatch"
            ylabel="Heating Nodal Prices"
            noX = 6
            noY = 6
            fig1 = CoherencePlot(dataMatrix=c1Heat.T, übertitle=file, title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
            fig2 = CoherencePlot(dataMatrix=c2Heat.T, übertitle=file, title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
            fig3 = CoherencePlot(dataMatrix=c3Heat.T, übertitle=file, title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
            SavePlot(fig1, path, title = (file[12:-3] + " - C1 heat mismatch and HNP"))
            SavePlot(fig2, path, title = (file[12:-3] + " - C2 heat mismatch and HNP"))
            SavePlot(fig3, path, title = (file[12:-3] + " - C3 heat mismatch and HNP"))
            
            
            # --- Elec/Heat Prices ---
            # Plot properties
            title1 = "Coherence 1: Electricity and heating nodal prices"
            title2 = "Coherence 2: Electricity and heating nodal prices"
            title3 = "Coherence 3: Electricity and heating nodal prices"
            xlabel = "Electricity Nodal Prices"
            ylabel = "Heating Nodal Prices"
            noX = 6
            noY = 6
            fig1 = CoherencePlot(dataMatrix=c1Price.T, übertitle=file, title=title1, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
            fig2 = CoherencePlot(dataMatrix=c2Price.T, übertitle=file, title=title2, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
            fig3 = CoherencePlot(dataMatrix=c3Price.T, übertitle=file, title=title3, xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
            SavePlot(fig1, path, title = (file[12:-3] + " - C1 ENP and HNP"))
            SavePlot(fig2, path, title = (file[12:-3] + " - C2 ENP and HNP"))
            SavePlot(fig3, path, title = (file[12:-3] + " - C3 ENP and HNP"))
        
        # ---------------------- Coherence Transition Path --------------------------- #   
        # --- Elec ---
        # Append value to matrix
        coherenceElecC1.append(c1Elec)
        coherenceElecC2.append(c2Elec)
        coherenceElecC3.append(c3Elec)
    
        # --- Heat ---
        # Append value to matrix
        coherenceHeatC1.append(c1Heat)
        coherenceHeatC2.append(c2Heat)
        coherenceHeatC3.append(c3Heat)
    
        # --- Elec/Heat Prices ---
        # Append value to matrix
        coherencePriceC1.append(c1Price)
        coherencePriceC2.append(c2Price)
        coherencePriceC3.append(c3Price)
    
        
    #%% Plot "change in..." plot
    
    ##############################################################################   
    ################################### MISMATCH #################################    
    ############################################################################## 
    
    figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_heat\\" + networkType + "\\"
    
    
    # ---------------------- Bar plot Transition Path --------------------------- #    
    # Figure path for change in decarbonization plots
    pathBAR = figurePath + "Bar\\"
    
    #title = "Number of PC describing variance of network as a function of decarbonization (" + networkType + ")"
    #xlabel = "Decarbonization"
    #suptitleElec = ("Electricity Mismatch - " + file[12:-8])
    title=""
    xlabel=""
    suptitleElec=""
    
    
    fig = BAR(barMatrixGoElecMismatch, 7, filenamesGo, constraints, title, xlabel, suptitleElec, figsize=[9,2.5], bbox=(0.5,-0.15))
    SavePlot(fig, pathBAR, (file[12:-8] + " - Bar Year Elec Mismatch"))
    
    #suptitleHeat = ("Heating  Mismatch - " + file[12:-8])
    suptitleHeat=""
    
    fig = BAR(barMatrixGoHeatMismatch, 7, filenamesGo, constraints, title, xlabel, suptitleHeat, figsize=[9,2.5], bbox=(0.5,-0.15))    
    SavePlot(fig, pathBAR, (file[12:-8] + " - Bar Year Heat Mismatch"))    
    
    # ---------------------- Contribution Transition Path --------------------------- # 
    # path
    pathContibution = figurePath + "Change in Contribution\\"
      
    # Plot change in elec contribution
    figtitle = "Change in electricity contribution as a function of decarbonization (" + networkType + ")"
    fig = ChangeContributionElec(lambdaContributionElec, networktype="brown", rotate=True, PC=2) #figtitle
    saveTitle = file[12:-8] + " - Change in elec cont (year)"
    SavePlot(fig, pathContibution, saveTitle)
    
    figtitle = "Change in electricity contribution as a function of decarbonization (" + networkType + ")"
    fig = ChangeContributionElec(lambdaContributionElec, networktype="brown", rotate=False, PC=6) #figtitle
    saveTitle = file[12:-8] + " - Change in elec cont app (year)"
    SavePlot(fig, pathContibution, saveTitle)
    
    # Plot change in heat contribution
    figtitle = "Change in heating contribution as a function of decarbonization (" + networkType + ")"
    fig = ChangeContributionHeat(lambdaContributionHeat, networktype="brown", rotate=True, PC=2) #figtitle
    saveTitle = file[12:-8] + " - Change in heat cont (year)"
    SavePlot(fig, pathContibution, saveTitle)
    
    figtitle = "Change in heating contribution as a function of decarbonization (" + networkType + ")"
    fig = ChangeContributionHeat(lambdaContributionHeat, networktype="brown", rotate=False, PC=6) #figtitle
    saveTitle = file[12:-8] + " - Change in heat cont app (year)"
    SavePlot(fig, pathContibution, saveTitle)
    
    
    # ---------------------- Response Transition Path --------------------------- #   
    # path
    pathResponse = figurePath + "Change in Response\\"
    
    # Plot change in elec response
    figtitle = "Change in electricity response as a function of decarbonization (" + networkType + ")"
    fig = ChangeResponseElec(lambdaResponseElec, networktype="brown", rotate=True, PC=2) #figtitle
    saveTitle = file[12:-8] + " - Change in elec response (year)"
    SavePlot(fig, pathResponse, saveTitle)
    
    figtitle = "Change in electricity response as a function of decarbonization (" + networkType + ")"
    fig = ChangeResponseElec(lambdaResponseElec, networktype="brown", rotate=False, PC=6) #figtitle
    saveTitle = file[12:-8] + " - Change in elec response app (year)"
    SavePlot(fig, pathResponse, saveTitle)
    
    # Plot change in heat response
    figtitle = "Change in heating response as a function of decarbonization (" + networkType + ")"
    fig = ChangeResponseHeat(lambdaResponseHeat, networktype="brown", rotate=True, PC=2) #figtitle
    saveTitle = file[12:-8] + " - Change in heat response (year)"
    SavePlot(fig, pathResponse, saveTitle)

    figtitle = "Change in heating response as a function of decarbonization (" + networkType + ")"
    fig = ChangeResponseHeat(lambdaResponseHeat, networktype="brown", rotate=False, PC=6) #figtitle
    saveTitle = file[12:-8] + " - Change in heat response app (year)"
    SavePlot(fig, pathResponse, saveTitle)    
    
    # Plot change in elec covariance response
    figtitle = "Change in electricity covariance response as a function of decarbonization (" + networkType + ")"
    fig = ChangeResponseCov(lambdaResponseElec, networktype="brown", rotate=True, PC=2) #figtitle
    saveTitle = file[12:-8] + " - Change in elec cov response (year)"
    SavePlot(fig, pathResponse, saveTitle)

    figtitle = "Change in electricity covariance response as a function of decarbonization (" + networkType + ")"
    fig = ChangeResponseCov(lambdaResponseElec, networktype="brown", rotate=False, PC=6) #figtitle
    saveTitle = file[12:-8] + " - Change in elec cov response app (year)"
    SavePlot(fig, pathResponse, saveTitle)
    
    # Plot change in heat covariance response
    figtitle = "Change in heating covariance response as a function of CO2 decarbonization (" + networkType + ")"
    fig = ChangeResponseCov(lambdaResponseHeat, networktype="brown", rotate=True, PC=2) #figtitle
    saveTitle = file[12:-8] + " - Change in heat cov response (year)"
    SavePlot(fig, pathResponse, saveTitle)
    
    figtitle = "Change in heating covariance response as a function of CO2 decarbonization (" + networkType + ")"
    fig = ChangeResponseCov(lambdaResponseHeat, networktype="brown", rotate=False, PC=6) #figtitle
    saveTitle = file[12:-8] + " - Change in heat cov response app (year)"
    SavePlot(fig, pathResponse, saveTitle)
    
    # ---------------------- Covariance Transition Path --------------------------- #   
    # path
    pathCovariance = figurePath + "Change in Covariance\\"
    
    # Plot change in elec covariance
    figtitle = "Change in electricity covariance as a function of CO2 decarbonization (" + networkType + ")"
    fig = ChangeCovariance(lambdaCovarianceElec, networktype="brown", collectTerms=True, rotate=True, PC=2) #figtitle
    saveTitle = file[12:-14] + " - Change in elec covariance (year)"
    SavePlot(fig, pathCovariance, saveTitle)
    
    figtitle = "Change in electricity covariance as a function of CO2 decarbonization (" + networkType + ")"
    fig = ChangeCovariance(lambdaCovarianceElec, networktype="brown", collectTerms=True, rotate=False, PC=6) #figtitle
    saveTitle = file[12:-14] + " - Change in elec covariance app (year)"
    SavePlot(fig, pathCovariance, saveTitle)
    
    # Plot change in heat covariance
    figtitle = "Change in heating covariance as a function of CO2 decarbonization (" + networkType + ")"
    fig = ChangeCovariance(lambdaCovarianceHeat, networktype="brown", collectTerms=False, rotate=True, PC=2) #figtitle
    saveTitle = file[12:-14] + " - Change in heat covariance (year)"
    SavePlot(fig, pathCovariance, saveTitle)
    
    figtitle = "Change in heating covariance as a function of CO2 decarbonization (" + networkType + ")"
    fig = ChangeCovariance(lambdaCovarianceHeat, networktype="brown", collectTerms=False, rotate=False, PC=6) #figtitle
    saveTitle = file[12:-14] + " - Change in heat covariance app (year)"
    SavePlot(fig, pathCovariance, saveTitle)
    
    
    # ---------------------- Curtailment for transition path --------------------------- #  
    # Combine all the generators to a list - Relative to generator
    totalCurtailment = list(np.array(windDis) + np.array(solarDis) + np.array(rorDis))
    curtailment.append(totalCurtailment)
    
    # ---------------------- Curtailment for transition path --------------------------- #  
    # Plot figure
    fig = plt.figure(figsize=(5,5), dpi=200)
    plt.suptitle("Summed average curtailment of electricity generation \n relative non-dispatchable generation", y=0.95)
    
    plt.xticks(np.arange(len(constraints)), constraints)
    plt.plot(totalCurtailment, marker='o', markersize=5, label = ("Brownfield (" + networkType + ")"))
    plt.ylabel("Curtailment [%]")
    plt.ylim([-1,35])
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    
    # save curtailment figure
    pathCurtailment = figurePath + "Pre Analysis\\"
    saveTitle = file[12:-8] + " - Curtailment " + "Brownfield (" + networkType + ")"
    SavePlot(fig, pathCurtailment, saveTitle)
    

    ##############################################################################   
    ################################# NODAL PRICE ################################    
    ##############################################################################  
    
    # ---------------------- Bar plot Transition Path --------------------------- #
    
    pathBAR = figurePath + "Bar\\"
    
    constraints = ["2020", "2025", "2030", "2035", "2040", "2045", "2050"]
    #title = "Number of PC describing variance of network as a function of decarbonization (" + networkType + ")"
    #xlabel = "Decarbonization"
    #suptitleElec = ("Electricity Nodal Price - " + file[12:-8])
    title=""
    xlabel=""
    suptitleElec=""
    
    
    fig = BAR(barMatrixGoElecNP, 7, filenamesGo, constraints, title, xlabel, suptitleElec, figsize=[9,2.5], bbox=(0.5,-0.15))
    SavePlot(fig, pathBAR, (file[12:-8] + " - Bar Year Elec NP"))
    
    #suptitleHeat = ("Heating Nodal Price - " + file[12:-8])
    suptitleHeat=""
    
    fig = BAR(barMatrixGoHeatNP, 7, filenamesGo, constraints, title, xlabel, suptitleHeat, figsize=[9,2.5], bbox=(0.5,-0.15))    
    SavePlot(fig, pathBAR, (file[12:-8] + " - Bar Year Heat NP"))

    
    # ----------------------- Price evalution (Elec) --------------------#
    path = figurePath + "Price Evolution\\"
    #title =  ("Electricity Nodal Price Evalution - "  + file[12:-14])
    title=""
    fig = PriceEvolution(meanPriceElec,quantileMeanPriceElec,quantileMinPriceElec,networktype="brown",title=title, figsize=[9,3], dpi=300)
    title =  (file[12:-14] + " - Elec NP Transition Evolution")
    SavePlot(fig, path, title)
    
    # ----------------------- Price evalution (Heat) --------------------#
    path = figurePath + "Price Evolution\\"
    #title =  ("Heating Nodal Price Evalution - "  + file[12:-14])
    title=""
    fig = PriceEvolution(meanPriceHeat,quantileMeanPriceHeat,quantileMinPriceHeat,networktype="brown",title=title, figsize=[9,3], dpi=300)
    title =  (file[12:-14] + " - Heat NP Transition Evolution")
    SavePlot(fig, path, title)

    
   #%%
    
    ##############################################################################   
    ################################## COHERENCE #################################    
    ############################################################################## 
    
    # ---------------------- Coherence between network types --------------------------- #
    coherenceTypes.append(collectedMismatchElec)
    
    # ---------------------- Coherence between networks --------------------------- #  
    # --- Elec ---
    pathCoherence = figurePath + "Coherence\\"
    axNames = ["2020","2025","2030","2035","2040","2045","2050"]
    #axTitle = "Year"
    #title="Brownfield Elec (" + networkType + ") coherence"
    axTitle = "Early Transition Path"
    title = ""
    fig = NetworkCoherence(collectedMismatchElec,axNames,axTitle,axRange="min",title=title)
    SavePlot(fig, pathCoherence, (file[12:-8] + " - Brownfield Elec (" + networkType + ") coherence"))

    # NOT RELAVANT AS ITS ALL 1.0  
    # # --- Heat ---
    # pathCoherence = figurePath + "Coherence\\"
    # axNames = ["2020","2025","2030","2035","2040","2045","2050"]
    # axTitle = "Year"
    # fig = NetworkCoherence(collectedMismatchHeat,axNames,axTitle,axRange="min",title="Brownfield Heat (" + networkType + ") coherence")
    # SavePlot(fig, pathCoherence, (file[12:-8] + " - Brownfield Heat (" + networkType + ") coherence"))
    
    
    # ---------------------- Coherence between PCA (method 2 summed) --------------------------- #  
    coherenceC2 = np.zeros([3,7])
    
    for j in range(len(coherenceElecC2)):
        coherenceC2[0,j] = round(np.diagonal(coherenceElecC2[j]).sum(),3)
        coherenceC2[1,j] = round(np.diagonal(coherenceHeatC2[j]).sum(),3)
        coherenceC2[2,j] = round(np.diagonal(coherencePriceC2[j]).sum(),3)
    
    # Plot figure
    fig = plt.figure(figsize=(10,3), dpi=300)
    #plt.suptitle("Summed Coherence method 2 (" + networkType + ")", y=0.93)
    
    plt.xticks(np.arange(len(constraints)), constraints, fontsize=12)
    plt.yticks(ticks=np.linspace(0,1,6), fontsize=12)
    plt.grid(alpha=0.3)
    plt.plot(coherenceC2[0], marker='o', markersize=5, label = "Elec Coherence")
    plt.plot(coherenceC2[1], marker='o', markersize=5, label = "Heat Coherence")
    plt.plot(coherenceC2[2], marker='o', markersize=5, label = "Price Coherence")
    plt.ylabel("Coherence [%]", fontsize=14)
    plt.ylim([0,1])
    plt.legend(loc="upper left", fontsize=12)
    
    # save coherence figure
    pathCoherence = figurePath + "Coherence\\"
    saveTitle = "Coherence method 2 summed (" + networkType + ")"
    SavePlot(fig, pathCoherence, saveTitle) 
    plt.show(all)

    
    # ---------------------- Coherence between PCA (method 1 PC1, PC2, PC3) --------------------------- #
    coherenceC1Elec = np.zeros([3,7])
    coherenceC1Heat = np.zeros([3,7])
    coherenceC1Price = np.zeros([3,7])
    
    for j in range(7):
        for k in range(len(coherenceC1Elec)):
            coherenceC1Elec[k,j] = coherenceElecC1[j][k][k]
            coherenceC1Heat[k,j] = coherenceHeatC1[j][k][k]
            coherenceC1Price[k,j] = coherencePriceC1[j][k][k]

    coherenceC1 = [coherenceC1Elec, coherenceC1Heat, coherenceC1Price]
    coherenceC1Type = ["Elec","Heat","Price"]
    for j in range(3):
        # Plot figure
        fig = plt.figure(figsize=(5,5), dpi=200)
        plt.suptitle("Coherence method 1 " + coherenceC1Type[j] + " (" + networkType + ")", y=0.93)
        
        plt.xticks(np.arange(len(constraints)), constraints)
        for k in range(3):
            plt.plot(coherenceC1[j][k], marker='o', markersize=5, label = "PC "+str(k+1))
        plt.ylabel("Coherence [%]")
        plt.ylim([0,1])
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
    
        # save coherence figure
        pathCoherence = figurePath + "Coherence\\"
        saveTitle = "Coherence method 1 " + coherenceC1Type[j] + " (" + networkType + ")"
        SavePlot(fig, pathCoherence, saveTitle) 
    
    plt.show(all)
    
    # ---------------------- Coherence between PCA (method 3 PC1, PC2, PC3) --------------------------- #
    coherenceC3Elec = np.zeros([3,7])
    coherenceC3Heat = np.zeros([3,7])
    coherenceC3Price = np.zeros([3,7])
    
    for j in range(7):
        for k in range(3):
            coherenceC3Elec[k,j] = coherenceElecC3[j][k][k]
            coherenceC3Heat[k,j] = coherenceHeatC3[j][k][k]
            coherenceC3Price[k,j] = coherencePriceC3[j][k][k]

    coherenceC3 = [coherenceC3Elec, coherenceC3Heat, coherenceC3Price]
    coherenceC3Type = ["Elec","Heat","Price"]
    for j in range(3):
        # Plot figure
        fig = plt.figure(figsize=(5,5), dpi=200)
        plt.suptitle("Coherence method 3 " + coherenceC3Type[j] + " (" + networkType + ")", y=0.93)
        
        plt.xticks(np.arange(len(constraints)), constraints)
        for k in range(3):
            plt.plot(coherenceC3[j][k], marker='o', markersize=5, label = "PC "+str(k+1))
        plt.ylabel("Coherence [%]")
        plt.ylim([-1,1])
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
    
        # save coherence figure
        pathCoherence = figurePath + "Coherence\\"
        saveTitle = "Coherence method 3 " + coherenceC3Type[j] + " (" + networkType + ")"
        SavePlot(fig, pathCoherence, saveTitle) 
    
    coherenceC3Elec = np.zeros([3,7])
    coherenceC3Heat = np.zeros([3,7])
    coherenceC3Price = np.zeros([3,7])
    
    for j in range(7):
        for k in range(3):
            coherenceC3Elec[k,j] = coherenceElecC3[j][k][k]
            coherenceC3Heat[k,j] = coherenceHeatC3[j][k][k]
            coherenceC3Price[k,j] = coherencePriceC3[j][k][k]

    coherenceC3 = [coherenceC3Elec, coherenceC3Heat, coherenceC3Price]
    coherenceC3Type = ["Elec","Heat","Price"]
    for j in range(3):
        # Plot figure
        fig = plt.figure(figsize=(5,5), dpi=200)
        plt.suptitle("Coherence method 3 " + coherenceC3Type[j] + " (" + networkType + ")", y=0.93)
        
        plt.xticks(np.arange(len(constraints)), constraints)
        for k in range(3):
            plt.plot(coherenceC3[j][k], marker='o', markersize=5, label = "PC "+str(k+1))
        plt.ylabel("Coherence [%]")
        plt.ylim([-1,1])
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left")
    
        # save coherence figure
        pathCoherence = figurePath + "Coherence\\"
        saveTitle = "Coherence method 3 " + coherenceC3Type[j] + " (" + networkType + ")"
        SavePlot(fig, pathCoherence, saveTitle) 
    
    
    
    plt.show(all)
    
    
    #%% Total generation and storage
    
    from functions_file import *
    
    # ------- Energy Production by different technologies as function of decarboinzation ----#
    # Path to save plots
    path = figurePath + "Total generation and storage\\"
    
    fig1, fig2, fig3, fig4 = EnergyProductionBrownfield(directory, filenames, figsize=[10,4.5], labelFontsize=13, bboxLoc=(1,1.035))
    
    SavePlot(fig1, path, (file[12:-8] + " - total elec generation (year)"))
    SavePlot(fig2, path, (file[12:-8] + " - total elec storage (year)"))
    SavePlot(fig3, path, (file[12:-8] + " - total heat generation (year)"))
    SavePlot(fig4, path, (file[12:-8] + " - total heat storage (year)"))

    plt.show(all)

    #%% Total generation and storage
    
    from functions_file import *
    
    # ------- Energy Production by different technologies as function of decarboinzation ----#
    # Path to save plots
    path = figurePath + "Total generation and storage\\"
    
    fig1, fig2, fig3, fig4 = EnergyCapacityInstalledBrownfield(directory, filenames, figsize=[10,4.5], labelFontsize=13, bboxLoc=(1,1.035))
    
    SavePlot(fig1, path, (file[12:-8] + " - total elec capacity (year)"))
    SavePlot(fig2, path, (file[12:-8] + " - total elec storage capacity (year)"))
    SavePlot(fig3, path, (file[12:-8] + " - total heat capacity (year)"))
    SavePlot(fig4, path, (file[12:-8] + " - total heat storage capacity (year)"))

    plt.show(all)
    
#%%

# Only plot when its multiple brownfield types
if len(brownfieldType) != 1:
    # ---------------------- Curtailment for transition path --------------------------- #  
    # Plot figure
    fig = plt.figure(figsize=(5,5), dpi=200)
    plt.suptitle("Summed average curtailment of electricity generation \n relative non-dispatchable generation", y=0.95)
    
    plt.xticks(np.arange(len(constraints)), constraints)
    for j in range(len(curtailment)):
        plt.plot(curtailment[j], marker='o', markersize=5, label = ("Brownfield (" + brownfieldType[j] + ")"))
    plt.ylabel("Curtailment [%]")
    plt.ylim([-1,35])
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    
    # save curtailment figure
    figurePath = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Figures\\brownfield_heat\\"
    pathCurtailment = figurePath + ""
    saveTitle = "Curtailment Across Transistion Path"
    SavePlot(fig, pathCurtailment, saveTitle)

    
    
    ##############################################################################   
    ################################## COHERENCE #################################    
    ############################################################################## 
    
    # ---------------------- Coherence between different network types --------------------------- #  
    # --- Elec ---
    pathCoherence = figurePath + ""
    axNames = ["2020","2025","2030","2035","2040","2045","2050"]
    axTitle = ["Late Transition Path","Early Transition Path"]
    fig = differentNetworkCoherence(coherenceTypes,axNames,axTitle,axRange="min")#,title="Brownfield Elec coherence between Go and Wait")
    SavePlot(fig, pathCoherence, (file[12:-8] + " - Brownfield Elec coherence between Go and Wait"))



#%% Timer


# Finish timer
t1 = time.time() # End timer
total_time = round(t1-t0)
total_time_min = math.floor(total_time/60)
total_time_sec = round(total_time-(total_time_min*60))
print("\n \nThe code is now done running. It took %s min and %s sec." %(total_time_min,total_time_sec))







