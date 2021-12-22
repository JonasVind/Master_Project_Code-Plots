# Import libraries
import os
import sys
import numpy as np
import pandas as pd
import pypsa

import time
import math

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *        

# Figure path
figurePath = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Figures\\realENP\\"


##############################################################################
##############################################################################

####################### Real electrcitiy prices (2019) #######################

##############################################################################
##############################################################################

# Directory of file
directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\Real Electricity prices\\2019\\"

# file name
file2019 = "electricityPrices2019.csv"

# Generic file name
titleFileName = file2019

# Load real electricity prices 2019
realPrices2019 = pd.read_csv((directory + file2019), index_col=0)
realPrices2019 = pd.DataFrame(data=realPrices2019.values, index=pd.to_datetime(realPrices2019.index), columns=pd.Index(realPrices2019.columns))

# Get names of countries included
dataNames = realPrices2019.columns

# Time index
timeIndex = pd.to_datetime(realPrices2019.index)

# Mean price
meanRealPrices2019 = realPrices2019.mean().mean()
# mean price quantiles
quantileMeanRealPrices2019 = np.quantile(realPrices2019.mean(),[0.05,0.25,0.75,0.95])


# PCA on real electricity prices
eigenValuesRealENP, eigenVectorsRealENP, varianceExplainedRealENP, normConstRealENP, TRealENP = PCA(realPrices2019)

# --------------------- Map PC Plot (Real elec prices) ----------------------#
# Path to save plots
path = figurePath + "Real ENP2019\\Map PC\\"

# Plot map PC for real electricity prices
titlePlot = "Real electricity prices (2019)"
for i in np.arange(7):
    fig = MAP(eigenVectorsRealENP, eigenValuesRealENP, dataNames, (i + 1))#, titlePlot, titleFileName)
    title = (file2019[:-4] + " - Map PC real ENP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ---------------------- FFT Plot (Real elec prices) ------------------------#
# Path to save FFT plots
path = figurePath + "Real ENP2019\\FFT\\"

# Elec
file_name = "Real electricity prices (2019) - " + file2019
for i in np.arange(7):
    fig = FFTPlot(TRealENP.T, varianceExplainedRealENP, title=file_name, PC_NO = (i+1))
    title = (file2019[:-4] + " - FFT real ENP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)


# ----------------------- Seasonal Plot (Elec + Heat) ------------------------#
# Path to save seasonal plots
path = figurePath + "Real ENP2019\\Seasonal\\"

# Elec
file_name = "Real electricity prices (2019) - " + file2019
for i in np.arange(7):
    fig = seasonPlot(TRealENP, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=7)
    title = (file2019[:-4] + " - Seasonal plot real ENP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# -------------------- FFT + Seasonal Plot (Elec) ---------------------#
# Path to save seasonal plots
path = figurePath + "Real ENP2019\\Timeseries\\"

# --- Elec ---
file_name = "Real electricity prices (2019) - " + file2019
for i in np.arange(6):
    fig = FFTseasonPlot(TRealENP, timeIndex, varianceExplainedRealENP, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (file2019[:-4] + " - Timeseries Plot real ENP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# ----------------- Combined PC1 & PC 2 Plot (Elec + Heat) -------------------#
# Path to save PC1 & PC 2 Plot
path = figurePath + "Real ENP2019\\Combined\\"

contribution=0
response=0
covariance=0
fig = PC1and2Plotter(TRealENP, pd.to_datetime(realPrices2019.index), [1,2], eigenValuesRealENP, contribution, response, covariance, PCType="withoutProjection")#,suptitle=("Real electricity prices (2019) - " + file2019[:-4]),dpi=200)
title = (file2019[:-4] + " - Combined Plot real ENP (lambda 1 & 2)")
SavePlot(fig, path, title)


##############################################################################
##############################################################################

##################### elec_only nodal prices - 40% CO2 #####################

##############################################################################
##############################################################################


# Directory of file
directoryENP40 = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Data\\elec_only\\"
#directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\elec_central\\"

# File name
fileENP40 = "postnetwork-elec_only_0.125_0.4.h5"

# Generic file name
titleFileNameENP40 = fileENP40

# Import network
networkENP40 = pypsa.Network(directoryENP40+fileENP40)

# Get the names of the data
dataNamesENP40 = networkENP40.buses.index.str.slice(0,2).unique()

# Get time index
timeIndex = networkENP40.loads_t.p_set.index

# ----------------------------- Map PC Plot (Elec) --------------------------#
# Path to save plots
path = figurePath + "ENP40\\Map PC\\"

# --- Elec ---
# Prices for electricity for each country (restricted to 1000 €/MWh)
priceENP40 = networkENP40.buses_t.marginal_price[dataNamesENP40] #.clip(-1000,1000)
priceENP40 = FilterPrice(priceENP40, 465)

# Mean price
meanPriceENP40 = priceENP40.mean().mean()
# mean price quantiles
quantileMeanPriceENP40 = np.quantile(priceENP40.mean(),[0.05,0.25,0.75,0.95])

# PCA on nodal prices for electricity
eigenValuesENP40, eigenVectorsENP40, varianceExplainedENP40, normConstENP40, TENP40 = PCA(priceENP40)

# Plot map PC for electricity nodal prices
titlePlotENP40 = "Nodal price for electricity only"
for i in np.arange(7):
    fig = MAP(eigenVectorsENP40, eigenValuesENP40, dataNamesENP40, (i + 1))#, titlePlotENP40, titleFileNameENP40)
    title = fileENP40[12:-3] + " - Map PC Elec NP (lambda " + str(i+1) + ")"
    SavePlot(fig, path, title)



# ---------------------------- FFT Plot (Elec) ------------------------------#
# Path to save FFT plots
path = figurePath + "ENP40\\FFT\\"

# Elec
file_name = "Electricity Nodal Price - " + fileENP40
for i in np.arange(7):
    fig = FFTPlot(TENP40.T, varianceExplainedENP40, title=file_name, PC_NO = (i+1))
    title = fileENP40[12:-3] + " - FFT Elec NP (lambda " + str(i+1) + ")"
    SavePlot(fig, path, title)



# -------------------------- Seasonal Plot (Elec) ---------------------------#
# Path to save seasonal plots
path = figurePath + "ENP40\\Seasonal\\"

# Elec
file_name = "Electricity Nodal Price - " + fileENP40
for i in np.arange(7):
    fig = seasonPlot(TENP40, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=7)
    title = fileENP40[12:-3] + " - Seasonal Plot Elec NP (lambda " + str(i+1) + ")"
    SavePlot(fig, path, title)

# -------------------- FFT + Seasonal Plot (Elec) ---------------------#
# Path to save seasonal plots
path = figurePath + "ENP40\\Timeseries\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + fileENP40
for i in np.arange(6):
    fig = FFTseasonPlot(TENP40, timeIndex, varianceExplainedENP40, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (fileENP40[12:-3] + " - Timeseries Plot Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)

# ----------------- Combined PC1 & PC 2 Plot (Elec + Heat) -------------------#
# Path to save PC1 & PC 2 Plot
path = figurePath + "ENP40\\Combined\\"

contribution=0
response=0
covariance=0
fig = PC1and2Plotter(TENP40, timeIndex, [1,2], eigenValuesENP40, contribution, response, covariance, PCType="withoutProjection")#,suptitle=("Electricity Nodal Price - " + fileENP40[12:-3]),dpi=200)
title = (fileENP40[12:-3] + " - Combined Plot Elec NP (lambda 1 & 2)")
SavePlot(fig, path, title)



##############################################################################
##############################################################################

############# Electricity nodal prices - Brownfield (elec) ###################

##############################################################################
##############################################################################


# Directory of file
directory = os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0] + "\\Data\\brownfield_elec\\"

# Name of file
fileBFElec = "postnetwork-global_tyndp1.0_0_0.45.nc"

# Generic file name
titleFileNameBFElec = fileBFElec

# Import network
networkBFElec = pypsa.Network()
networkBFElec.import_from_netcdf(directory + fileBFElec)

# Data names
dataNamesBFElec = networkBFElec.buses.index.str.slice(0,2).unique()

# Time index
timeIndex = networkBFElec.buses_t.p.index

# prices
priceBFElec = networkBFElec.buses_t.marginal_price[dataNamesBFElec] #.clip(-1000,1000)
priceBFElec = FilterPrice(priceBFElec, 465)

# Mean price
meanPriceBFElec = priceBFElec.mean().mean()
# mean price quantiles
quantileMeanPriceBFElec = np.quantile(priceBFElec.mean(),[0.05,0.25,0.75,0.95])

# ----------------------------- Map PC Plot (Elec) --------------------------#
# Path to save plots
path = figurePath + "BFElec\\Map PC\\"

# --- Elec ---
# PCA on nodal prices for electricity
eigenValuesBFElec, eigenVectorsBFElec, varianceExplainedBFElec, normConstBFElec, TBFElec = PCA(priceBFElec)

# Plot map PC for electricity nodal prices
titlePlotBFElec = "Nodal price for electricity only"
for i in np.arange(7):
    fig = MAP(eigenVectorsBFElec, eigenValuesBFElec, dataNamesBFElec, (i + 1))#, titlePlotBFElec, titleFileNameBFElec)
    title = fileBFElec[12:-3] + " - Map PC Elec NP (lambda " + str(i+1) + ")"
    SavePlot(fig, path, title)



# ---------------------------- FFT Plot (Elec) ------------------------------#
# Path to save FFT plots
path = figurePath + "BFElec\\FFT\\"

# Elec
file_name = "Electricity Nodal Price - " + fileBFElec
for i in np.arange(7):
    fig = FFTPlot(TBFElec.T, varianceExplainedBFElec, title=file_name, PC_NO = (i+1))
    title = fileBFElec[12:-3] + " - FFT Elec NP (lambda " + str(i+1) + ")"
    SavePlot(fig, path, title)



# -------------------------- Seasonal Plot (Elec) ---------------------------#
# Path to save seasonal plots
path = figurePath + "BFElec\\Seasonal\\"

# Elec
file_name = "Electricity Nodal Price - " + fileBFElec
for i in np.arange(7):
    fig = seasonPlot(TBFElec, timeIndex, title=file_name, PC_NO=(i+1), PC_amount=7)
    title = fileBFElec[12:-3] + " - Seasonal Plot Elec NP (lambda " + str(i+1) + ")"
    SavePlot(fig, path, title)

# -------------------- FFT + Seasonal Plot (Elec) ---------------------#
# Path to save seasonal plots
path = figurePath + "BFElec\\Timeseries\\"

# --- Elec ---
file_name = "Electricity Nodal Price - " + fileBFElec
for i in np.arange(6):
    fig = FFTseasonPlot(TBFElec, timeIndex, varianceExplainedBFElec, PC_NO=(i+1), PC_amount=6,dpi=200)
    title = (fileBFElec[12:-3] + " - Timeseries Plot Elec NP (lambda " + str(i+1) + ")")
    SavePlot(fig, path, title)



# ----------------- Combined PC1 & PC 2 Plot (Elec + Heat) -------------------#
# Path to save PC1 & PC 2 Plot
path = figurePath + "BFElec\\Combined\\"

contribution=0
response=0
covariance=0
fig = PC1and2Plotter(TBFElec, timeIndex, [1,2], eigenValuesBFElec, contribution, response, covariance, PCType="withoutProjection")#,suptitle=("Electricity Nodal Price - " + fileBFElec[12:-3]),dpi=200)
title = (fileBFElec[12:-3] + " - Combined Plot Elec NP (lambda 1 & 2)")
SavePlot(fig, path, title)






#%%




##############################################################################
##############################################################################

################################# Coherence ##################################

##############################################################################
##############################################################################


# --- Coherence with 40% allowed CO2 + 2019 ---
# Path to save coherence plots
path = figurePath + "Coherence40ENP2019\\"

# Coherence between prices and mismatch
c1ENP40, c2ENP40, c3ENP40 = Coherence(realPrices2019, priceENP40.drop('BA', axis=1))

# Plot properties
title1 = "Coherence 1: Real ENP 2019 and " + fileENP40[12:-3] + " ENP"
title2 = "Coherence 2: Real ENP 2019 and " + fileENP40[12:-3] + " ENP"
title3 = "Coherence 3: Real ENP 2019 and " + fileENP40[12:-3] + " ENP"
xlabel = "Real Prices (2019)"
ylabel = "Overnight Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1ENP40.T, übertitle="", title="", xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2ENP40.T, übertitle="", title="", xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3ENP40.T, übertitle="", title="", xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file2019[:-4] + " - C1 real ENP 2019 and ENP40"))
SavePlot(fig2, path, title = (file2019[:-4] + " - C2 real ENP 2019 and ENP40"))
SavePlot(fig3, path, title = (file2019[:-4] + " - C3 real ENP 2019 and ENP40"))


# Combined Plot
fig = CoherencePlotCombined(c1ENP40.T, c2ENP40.T, c3ENP40.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file2019[:-4] + " - C123 combined real ENP 2019 and ENP40"))


# --- Coherence with Brownfield (elec) + 2019 ---
# Path to save coherence plots
path = figurePath + "CoherenceBFElecENP2019\\"

# Coherence between prices and mismatch
c1BFElec, c2BFElec, c3BFElec = Coherence(realPrices2019, priceBFElec.drop('BA', axis=1))

# Plot properties
title1 = "Coherence 1: Real ENP 2019 and " + fileBFElec[12:-3] + " ENP"
title2 = "Coherence 2: Real ENP 2019 and " + fileBFElec[12:-3] + " ENP"
title3 = "Coherence 3: Real ENP 2019 and " + fileBFElec[12:-3] + " ENP"
xlabel = "Real Prices (2019)"
ylabel = "Transition Path Prices"
noX = 6
noY = 6
fig1 = CoherencePlot(dataMatrix=c1BFElec.T, übertitle="", title="", xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig2 = CoherencePlot(dataMatrix=c2BFElec.T, übertitle="", title="", xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[0,1])
fig3 = CoherencePlot(dataMatrix=c3BFElec.T, übertitle="", title="", xlabel=xlabel, ylabel=ylabel, noX=noX, noY=noY, dataRange=[-1,1])
SavePlot(fig1, path, title = (file2019[:-4] + " - C1 real ENP 2019 and BFElec"))
SavePlot(fig2, path, title = (file2019[:-4] + " - C2 real ENP 2019 and BFElec"))
SavePlot(fig3, path, title = (file2019[:-4] + " - C3 real ENP 2019 and BFElec"))

# Combined Plot
fig = CoherencePlotCombined(c1BFElec.T, c2BFElec.T, c3BFElec.T, xlabel=xlabel, ylabel=ylabel)
SavePlot(fig, path, title = (file2019[:-4] + " - C123 combined real ENP 2019 and BFElec"))


# Finish timer
t1 = time.time() # End timer
total_time = round(t1-t0)
total_time_min = math.floor(total_time/60)
total_time_sec = round(total_time-(total_time_min*60))
print("\n \nThe code is now done running. It took %s min and %s sec." %(total_time_min,total_time_sec))



# 3 min and 5 sec.


#%%

dpi = 200
figsize = [6,2.5]

fig = plt.figure(figsize=figsize, dpi=dpi)

plt.boxplot([realPrices2019.mean(),priceENP40.mean(),priceBFElec.mean()],
            showfliers=False,
            whis = [5,95],
            medianprops = dict(color="red",linewidth=1,alpha=0),
            meanline =True,
            showmeans=True,
            meanprops = dict(color="red",linewidth=1,linestyle="solid"),
            
            )
plt.ylabel("Price [€/MWh]")
plt.ylim(ymin=-2)
plt.grid(axis="y",alpha=0.15)
plt.xticks([1,2,3],['Real Prices\n(2019)', 'Overnight\nPrices', 'Transition Path\nPrices'],rotation=0)

plt.hlines(meanRealPrices2019,color="red",xmin=0.9,xmax=1.1,alpha=1,linewidth=1,label="Mean")
plt.legend()

path = figurePath + "\\"
SavePlot(fig, path, title = ("Electricity Mean Prices with quantiles"))

