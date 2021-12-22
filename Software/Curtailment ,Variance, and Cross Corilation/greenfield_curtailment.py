# Import libraries
import os
import pypsa
import numpy as np
import matplotlib.pyplot as plt


# Directory of file
directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\elec_heat\\"


##############################################################################
#########---------------------- elec_heat --------------------------##########
##############################################################################

# ------------------------------ CO2 constraint -----------------------------#
# File name - CO2 constraint
files = ["postnetwork-elec_heat_0.125_0.6.h5",
         "postnetwork-elec_heat_0.125_0.5.h5",
         "postnetwork-elec_heat_0.125_0.4.h5",
         "postnetwork-elec_heat_0.125_0.3.h5",
         "postnetwork-elec_heat_0.125_0.2.h5",
         "postnetwork-elec_heat_0.125_0.1.h5",
         "postnetwork-elec_heat_0.125_0.05.h5"]

# Preallocate lists
windDis = []
solarDis = []
rorDis = []

for file in files:
    # Load Netowork
    network = pypsa.Network(directory + file)
    
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

# Combine all the generators to a list - Relative to generator
totalCO2_heat = list(np.array(windDis) + np.array(solarDis) + np.array(rorDis))


# -------------------------- Transmission constraint -------------------------#

# File name - Transmission constraint
files = ["postnetwork-elec_heat_0_0.05.h5",
         "postnetwork-elec_heat_0.0625_0.05.h5",
         "postnetwork-elec_heat_0.125_0.05.h5",
         "postnetwork-elec_heat_0.25_0.05.h5",
         "postnetwork-elec_heat_0.375_0.05.h5"]

# Preallocate lists
windDis = []
solarDis = []
rorDis = []

for file in files:
    # Load Netowork
    network = pypsa.Network(directory + file)
    
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

# Combine all the generators to a list - Relative to generator
totalTrans_heat = list(np.array(windDis) + np.array(solarDis) + np.array(rorDis))




##############################################################################
#########--------------------- elec_v2g50 --------------------------##########
##############################################################################

# Directory of file
directory = directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\elec_v2g50\\"

# ------------------------------ CO2 constraint -----------------------------#
# File name - CO2 constraint
files = ["postnetwork-elec_v2g50_0.125_0.6.h5",
         "postnetwork-elec_v2g50_0.125_0.5.h5",
         "postnetwork-elec_v2g50_0.125_0.4.h5",
         "postnetwork-elec_v2g50_0.125_0.3.h5",
         "postnetwork-elec_v2g50_0.125_0.2.h5",
         "postnetwork-elec_v2g50_0.125_0.1.h5",
         "postnetwork-elec_v2g50_0.125_0.05.h5"]

# Preallocate lists
windDis = []
solarDis = []
rorDis = []

for file in files:
    # Load Netowork
    network = pypsa.Network(directory + file)
    
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

# Combine all the generators to a list - Relative to generator
totalCO2_v2g50 = list(np.array(windDis) + np.array(solarDis) + np.array(rorDis))


##############################################################################
#########------------------- elec_heat_v2g50 -----------------------##########
##############################################################################

# Directory of file
directory = directory = os.path.split(os.path.split(os.getcwd())[0])[0] + "\\Data\\elec_heat_v2g50\\"

# ------------------------------ CO2 constraint -----------------------------#
# File name - CO2 constraint
files = ["postnetwork-elec_heat_v2g50_0.125_0.6.h5",
         "postnetwork-elec_heat_v2g50_0.125_0.5.h5",
         "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
         "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
         "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
         "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
         "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

# Preallocate lists
windDis = []
solarDis = []
rorDis = []

for file in files:
    # Load Netowork
    network = pypsa.Network(directory + file)
    
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

# Combine all the generators to a list - Relative to generator
totalCO2_heat_v2g50 = list(np.array(windDis) + np.array(solarDis) + np.array(rorDis))






##############################################################################
######### ---------------------- plotting ------------------------- ##########
##############################################################################

# ------------------------------ Plot figures -------------------------------#

# --- Combined plots ---
# Plot figure
plt.figure(figsize=(10,5), dpi=200)
plt.suptitle("Summed average curtailment of electricity generation relative non-dispatchable generation", y=0.95)

# Constraint names of system (CO2)
constraintsCO2 = ['40%', '50%', '60%', '70%', '80%', '90%', '95%']

plt.subplot(121)
plt.xticks(np.arange(len(constraintsCO2)), constraintsCO2)
plt.plot(totalCO2_heat, color="darkorange", marker='o', markerfacecolor="darkorange", markersize=5)
plt.plot(totalCO2_v2g50, color="dodgerblue", marker='o', markerfacecolor="dodgerblue", markersize=5)
plt.plot(totalCO2_heat_v2g50, color="indigo", marker='o', markerfacecolor="indigo", markersize=5)
plt.ylabel("Curtailment [%]")
plt.ylim([-1,35])
plt.grid(alpha=0.3)
plt.legend(["Heat coupling", "EV coupling", "Heat + EV coupling"], loc="upper left")


# Constraint names of system (transmission)
constraintsTransmission = ['Zero', 'Current', '2x Current', '4x Current', '6x Current']

plt.subplot(122)
plt.xticks(np.arange(len(constraintsTransmission)), constraintsTransmission)
plt.plot(totalTrans_heat, color="darkorange", marker='o', markerfacecolor="darkorange", markersize=5)
plt.ylim([-1,35])
plt.grid(alpha=0.3)
plt.legend(["Heat coupling"], loc="upper right")

