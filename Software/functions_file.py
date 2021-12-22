#%% Libraries 

import os.path
import pypsa
import math
import numpy as np
from numpy.linalg import eig
import pandas as pd

import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.ticker as tick
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

#%% PCA
def PCA(X):
    """
    Input:
        - X: Matrix for performing PCA on
    
    Output:
        - eigen_values:         Eigen values for the input
        - eigen_vectors:        Eigen vectors for the input
        - variance_explained:   How much of the variance is explained by each principal component
        - norm_const:           Normalization constant
        - T:                    Principle component amplitudes (a_k from report)
        
    """
    # Average value of input matrix
    X_avg = np.mean(X, axis=0)
    
    # Mean center data (subtract the average from original data) 
    B = X.values - X_avg.values
    
    # Normalisation constant
    normConst = (1 / (np.sqrt( np.sum( np.mean( ( (B)**2 ), axis=0 ) ) ) ) )
        
    # Covariance matrix (A measure of how much each of the dimensions varies from the mean with respect to each other)
    C = np.cov((B*normConst).T, bias=True)
    
    # Stops if C is larger than [30 x 30] 
    assert np.size(C) <= 900, "C is too big"
    
    # Eigen vector and values
    eigenValues, eigenVectors = eig(C)
        
    # Variance described by each eigen_value
    varianceExplained = (eigenValues * 100 ) / eigenValues.sum()
    
    # Principle component amplitudes
    T = np.dot((B*normConst), eigenVectors)
         
    # Cumulative variance explained
    #variance_explained_cumulative = np.cumsum(variance_explained)
    
    return (eigenValues, eigenVectors, varianceExplained, normConst, T)

#%% SeasonPlot

def seasonPlot(T, time_index,PC_NO=-1, PC_amount=6, title="none",dpi=200):
    """
    
    Parameters
    ----------
    T : Matrix
        Principle component amplitudes. Given by: B*eig_val (so the centered and scaled data dotted with the eigen values)
    data_index : panda index information
        index for a year (used by panda's dataframe')
    file_name: array of strings
        Name of the datafile there is worked with

    Returns
    -------
    fig: Figure
        Saves the figure as a variable to be used by savePlot

    """
    # Define as dataframe
    T = pd.DataFrame(data=T,index=time_index)

    # Average hour and day
    T_avg_hour = T.groupby(time_index.hour).mean() # Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    if PC_NO == -1:
        
        # Upper left figure
        fig = plt.figure(figsize=(16,10),dpi=dpi)
        plt.subplot(2,2,1)
        plt.plot(T_avg_hour[0],label='$\lambda_1$',marker='.',color=color[0])
        plt.plot(T_avg_hour[1],label='$\lambda_2$',marker='.',color=color[1])
        plt.plot(T_avg_hour[2],label='$\lambda_3$',marker='.',color=color[2])
        plt.xticks(ticks=range(0,24,2))
        #plt.legend(loc='upper right',bbox_to_anchor=(1,1))
        plt.xlabel("Hours")
        plt.ylabel("a_k interday")
        plt.title("Hourly average for k-values for 2015 ")
        
        # Upper right figure
        x_ax = range(len(T_avg_day[0])) # X for year plot
        plt.subplot(2,2,2)
        plt.plot(x_ax,T_avg_day[0],label='$\lambda_1$',color=color[0])
        plt.plot(x_ax,T_avg_day[1],label='$\lambda_2$',color=color[1])
        plt.plot(x_ax,T_avg_day[2],label='$\lambda_3$',color=color[2])
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.xlabel("day")
        plt.ylabel("a_k")
        plt.title("daily average for k-values for 2015 ")
        
        # Lower left figure
        plt.subplot(2,2,3)
        plt.plot(T_avg_hour[3],label='$\lambda_1$',marker='.',color=color[3])
        plt.plot(T_avg_hour[4],label='$\lambda_1$',marker='.',color=color[4])
        plt.plot(T_avg_hour[5],label='$\lambda_1$',marker='.',color=color[5])
        plt.xticks(ticks=range(0,24,2))
        #plt.legend(loc='upper right',bbox_to_anchor=(1,1))
        plt.xlabel("Hours")
        plt.ylabel("a_k interday")
        plt.title("Hourly average for k-values for 2015 ")
        
        # Lower right figure
        x_ax = range(len(T_avg_day[0])) # X for year plot
        plt.subplot(2,2,4)
        plt.plot(x_ax,T_avg_day[3],label='$\lambda_4$',color=color[3])
        plt.plot(x_ax,T_avg_day[4],label='$\lambda_5$',color=color[4])
        plt.plot(x_ax,T_avg_day[5],label='$\lambda_6$',color=color[5])
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.xlabel("day")
        plt.ylabel("a_k seasonal")
        plt.title("daily average for k-values for 2015 ")
        
        # Figure title
        if title != "none":
            plt.suptitle(title,fontsize=20,x=.51,y=0.932) #,x=.51,y=1.07
    
    else:
        alpha = np.zeros(PC_amount)+0.2 # Generate alpha values 
        alpha[PC_NO-1] = +1 # give full alpha value to the intersting PC
        
        fig = plt.figure(figsize=(9,3.5),dpi=dpi) # Create figure
        
        # Daily plot
        plt.subplot(1,2,1)
        for j in range(PC_amount):
            label = "$\lambda_" + str(j+1) + "$"
            plt.plot(T_avg_hour[j],label=label,marker='.',alpha=alpha[j],color=color[j])
        
        #subplot setup
        plt.xticks(ticks=range(0,24,2))
        #plt.legend(loc='upper right',bbox_to_anchor=(1,1))
        plt.ylim([-1.25,1.4])
        plt.xlabel("Hours", fontsize=16)
        plt.ylabel("$a_k$", fontsize=18)
        plt.title("Daily average for 2015", fontsize=16)
        plt.xticks(fontsize=14)
        plt.text(-3.7,1.45,"(b)", fontsize=16)

        x_ax = range(len(T_avg_day[0]))
        
        # Year plot
        plt.subplot(1,2,2)
        for j in range(PC_amount):
            if alpha[j] >= 1:
                label = "$\lambda_" + str(j+1) + "$"
                plt.plot(x_ax,T_avg_day[j],label=label,alpha=alpha[j],color=color[j])
            else:
                plt.plot(x_ax,T_avg_day[j],alpha=alpha[j],color=color[j])
        
        #subplot setup
        plt.legend(loc='upper right', fontsize=16)#,bbox_to_anchor=(1,1))
        plt.ylim([-1.99,1.99])
        #plt.xlabel("day")
        #plt.ylabel("$a_k$ seasonal")
        plt.title("Seasonal average for 2015", fontsize=16)
        plt.text(-55,2.1,"(c)", fontsize=16)
        
        # Plot setup
        #title = 'Season plot for '+'$\lambda_{'+str(PC_NO)+'}$'
        plt.tight_layout()
        if title != "none":
            plt.suptitle(title,fontsize=18,x=.5,y=1.02) #,x=.51,y=1.07)
    
        plt.xticks(np.array([0,31,59,90,120,151,181,212,243,273,304,334])+14,["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=14, rotation=-90)
        
    # plt.set_xticks(np.array([0,31,59,90,120,151,181,212,243,273,304,334])+14) # y-axis label (at 14th in the month)
    # plt.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) # y-axis label
    
    
    # Shows the generated plot
    plt.show(all)
    
    return fig
#%% EigValContributionPlot

def EigValContributionPlot(lambda_collect_wmin, lambda_tot, file_name):
    """

    Parameters
    ----------
    lambda_collect_wmin : array
        Lambda array collected with those that are minus already included
    lambda_tot : value
        Sum of all lambda values
    file_name : array of strings
        Name of the datafile there is worked with

    Returns
    -------
    fig: Figure
        Saves the figure as a variable to be used by savePlot

    """
    
    
    fig = plt.figure(figsize=[14,16])
    for n in range(6):
        lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # percentage
        #lambda_collect_procent = lambda_collect_wmin[n:n+1]/lambda_tot[n]*100 # relative
        plt.subplot(3,2,n+1)
        plt.bar(lambda_collect_wmin.columns,lambda_collect_procent.values[0])
        plt.title('PC'+str(n+1)+': '+str(round(lambda_tot[n],3)))
        plt.ylabel('Influance [%]')
        plt.ylim([-50,125])
        plt.grid(axis='y',alpha=0.5)
        for k in range(10):
            if lambda_collect_procent.values[:,k] < 0:
                v = lambda_collect_procent.values[:,k] - 6.5
            else:
                v = lambda_collect_procent.values[:,k] + 2.5
            plt.text(x=k,y=v,s=str(round(float(lambda_collect_procent.values[:,k]),2))+'%',ha='center',size='small')
    plt.suptitle(file_name,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07  
    
    # Show plot
    plt.show(all)
    
    return  fig

#%% PCContributionPlot

def PCContributionPlot(PC_con, type_of_contribution):
    """

    Parameters
    ----------
    PC_con : list
        list of with sorted principle components contributions 
    type_of_contribution : string
        "mismatch" - used when wanting to plot for the mismatch case
        "respone" - used when wanting to plot for the response case

    Returns
    -------
    fig: Figure
        Saves the figure as a variable to be used by savePlot

    """
    
    fig = plt.figure(figsize=(14,16))#,dpi=500)
    # Print out for 6 principle components
    for i in range(6):
        
        # Principle components for nodal mismatch
        if type_of_contribution == "mismatch":
            # y functions comulated
            wind_con_data  = PC_con[i][:,:1].sum(axis=1)
            solar_con_data = PC_con[i][:,:2].sum(axis=1)
            hydro_con_data = PC_con[i][:,:3].sum(axis=1)
            load_con_data  = PC_con[i][:,:4].sum(axis=1)
            gen_cov_data   = PC_con[i][:,:7].sum(axis=1)
            load_cov_data  = PC_con[i][:,8:10].sum(axis=1)
            # plot function
            plt.subplot(3,2,i+1)
            # Plot lines
            plt.plot(wind_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(solar_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(hydro_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(load_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(gen_cov_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(load_cov_data,color='k',alpha=1,linewidth=0.5)
            # Plot fill inbetween lines
            plt.fill_between(range(7), np.zeros(7), wind_con_data,
                             label='Wind',
                             color='cornflowerblue') # Because it is a beutiful color
            plt.fill_between(range(7), wind_con_data, solar_con_data,
                             label='Solar PV',
                             color='yellow')
            plt.fill_between(range(7), solar_con_data, hydro_con_data,
                             label='Hydro',
                             color='darkslateblue')
            plt.fill_between(range(7), hydro_con_data, load_con_data,
                             label='Load',
                             color='slategray')
            plt.fill_between(range(7), load_con_data, gen_cov_data,
                             label='Generator\ncovariance',
                             color='brown',
                             alpha=0.5)
            plt.fill_between(range(7), load_cov_data, np.zeros(7),
                             label='Load\ncovariance',
                             color='orange',
                             alpha=0.5)
            # y/x-axis and title
            #plt.legend(bbox_to_anchor = (1,1))
            plt.ylabel('$\lambda_k$')
            plt.xticks(np.arange(0,7),['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
            plt.title('Principle component '+str(i+1))
            if i == 4: # Create legend of figure 4 (lower left)
                plt.legend(loc = 'center', # How the label should be places according to the placement
                           bbox_to_anchor = (1.1,-0.17), # placement relative to the graph
                           ncol = 6, # Amount of columns
                           fontsize = 'large', # Size of text
                           framealpha = 1, # Box edge alpha
                           columnspacing = 2.5 # Horizontal spacing between labels
                           )
        
        # Principle components for nodal response
        elif type_of_contribution == "response": 
            # y functions comulated
            backup_con_data  = PC_con[i][:,:1].sum(axis=1)
            inport_export_con_con_data = PC_con[i][:,:2].sum(axis=1)
            storage_con_data = PC_con[i][:,:3].sum(axis=1)
            #backup_inport_cov_data  = PC_con[i][:,:4].sum(axis=1)
            #backup_store_cov_data   = PC_con[i][:,:5].sum(axis=1)
            inport_store_cov_data   = PC_con[i][:,:6].sum(axis=1)
            # plot function
            plt.subplot(3,2,i+1)
            # Plot lines
            plt.plot(backup_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(inport_export_con_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(storage_con_data,color='k',alpha=1,linewidth=0.5)
            plt.plot(inport_store_cov_data,color='k',alpha=1,linewidth=0.5)
            plt.fill_between(range(7), np.zeros(7), backup_con_data,
                             label='backup',
                             color='cornflowerblue') # Because it is a beutiful color
            plt.fill_between(range(7), backup_con_data, inport_export_con_con_data,
                             label='import & export',
                             color='yellow')
            plt.fill_between(range(7), inport_export_con_con_data, storage_con_data,
                             label='storage',
                             color='darkslateblue')
            plt.fill_between(range(7), storage_con_data, inport_store_cov_data,
                             label='covariance',
                             color='orange',
                             alpha=0.5)
            plt.ylabel('$\lambda_k$')
            plt.xticks(np.arange(0,7),['40%', '50%', '60%', '70%', '80%', '90%', '95%'])
            plt.title('Principle component '+str(i+1))
            if i == 4: # Create legend of figure 4 (lower left)
                plt.legend(loc = 'center', # How the label should be places according to the placement
                           bbox_to_anchor = (1.1,-0.17), # placement relative to the graph
                           ncol = 6, # Amount of columns
                           fontsize = 'large', # Size of text
                           framealpha = 1, # Box edge alpha
                           columnspacing = 2.5 # Horizontal spacing between labels
                           )     
            else:
                assert True, "type_of_contribution not used correct"
    # Show plot
    plt.show(all)
    
    return fig 

#%% MAP
def MAP(eigen_vectors, eigen_values, data_names, PC_NO, title_plot="none", filename_plot="none", size="medium"):
    """
    Input:
        - eigen_vectors: Eigen vectors [N x N]
        - eigen_values: Eigen values [N x 1]
        - data_names: Name of countries ('alpha-2-code' / ISO_A2 format) [N x 1]
        - PC_NO: Principal component number to plot (starts from 1)
        - title_plot: Title of plot
        - filename_plot: Subplot title
    
    Returns
    -------
    fig: Figure
        Saves the figure as a variable to be used by savePlot
        
    """    
    
    # Define the eigen vectors in a new variable with names
    VT = pd.DataFrame(data=eigen_vectors, index=data_names)
    
    # Variance described by each eigen_value
    variance_explained = (eigen_values * 100 ) / eigen_values.sum()
    
    # Create figure
    if size  == "small":
        fig = plt.figure(figsize=(4,4), dpi=400)
    elif size == "medium":
        fig = plt.figure(figsize=(7,7), dpi=200)
    elif size == "large":
        fig = plt.figure(figsize=(9,9), dpi=200)
    else:
        assert False, "choose size to be small, medium or large"
    ax = plt.axes(projection=cartopy.crs.TransverseMercator(20))
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
    ax.coastlines(resolution='50m')
    ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.6,0.8,1), alpha=0.30)
    ax.set_extent ((-9.5, 32, 35, 71), cartopy.crs.PlateCarree())
    ax.gridlines()
    
    
    # List of european countries not included in the data (ISO_A2 format)
    europe_not_included = {'AD', 'AL','AX','BY', 'FO', 'GG', 'GI', 'IM', 'IS', 
                           'JE', 'LI', 'MC', 'MD', 'ME', 'MK', 'MT', 'RU', 'SM', 
                           'UA', 'VA', 'XK'}
    
    # Create shapereader file name
    shpfilename = shpreader.natural_earth(resolution='50m',
                                          category='cultural',
                                          name='admin_0_countries')
    
    # Read the shapereader file
    reader = shpreader.Reader(shpfilename)
    
    # Record the reader
    countries = reader.records()
        
    # Determine name_loop variable
    name_loop = 'start'
    
    # Start for-loop
    for country in countries:
        
        # If the country is in the list of the european countries, but not 
        # part of the included european countries: color it gray
        if country.attributes['ISO_A2'] in europe_not_included:
            ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
        
        # If the country is in the region Europe
        elif country.attributes['REGION_UN'] == 'Europe':
            
            # Account for Norway and France bug of having no ISO_A2 name
            if country.attributes['NAME'] == 'Norway':
                name_loop = 'NO'
                
            elif country.attributes['NAME'] == 'France':
                name_loop = 'FR'
                
            else:
                name_loop = country.attributes['ISO_A2']
            
            # Color country
            for country_PSA in VT.index.values:
                
                # When the current position in the for loop correspond to the same name: color it
                if country_PSA == name_loop:
                    
                    # Determine the value of the eigen vector
                    color_value = VT.loc[country_PSA][PC_NO-1]
                    
                    # If negative: color red
                    if color_value <= 0:
                        color_value = np.absolute(color_value)*1.5
                        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(1, 0, 0), alpha=(np.min([color_value, 1])), linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
                        
                    
                    # If positive: # Color green
                    else:
                        
                        color_value = np.absolute(color_value)*1.5
                        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(0, 1, 0), alpha=(np.min([color_value, 1])), linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])

                
        # Color any country outside of Europe gray        
        else:
            ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
                    
    if (variance_explained[PC_NO-1] < 0.1):
        plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],2)) + '%'], loc='upper left', fontsize=14, framealpha=1)
    
    elif (variance_explained[PC_NO-1] < 0.01):
        plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],3)) + '%'], loc='upper left', fontsize=14, framealpha=1)

    else:
        plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%'], loc='upper left', fontsize=14, framealpha=1)

    if title_plot != "none":
        plt.title(title_plot)
        plt.suptitle(filename_plot, fontsize=16, x=.51, y=0.94)
    
    # Matrix to determine maximum and minimum values for color bar
    color_matrix = np.zeros([2,2])
    color_matrix[0,0]=-1
    color_matrix[-1,-1]=1
    
    cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0,0),(1,0.333,0.333),(1,0.666,0.666), 'white',(0.666,1,0.666),(0.333,1,0.333),(0,1,0),(0,1,0)])
    
    cax = fig.add_axes([0.86, 0.13, 0.02, 0.745])
    im = ax.imshow(color_matrix,cmap=cmap)                
    plt.colorbar(im,cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])

    
    # Show plot
    plt.show()
    
    return fig

#%% BAR
def BAR(matrix, PC_max, filename, constraints, title, xlabel, suptitle, fontsize=14, figsize=[6.4, 4.8], dpi=200, bbox=(0.5,-0.125), ncol=7, rotation=0):
    """
    Parameters
    ----------
    matrix : list [len(filename) x 1] of lists of float32 [30 x 1]
        Each sub-list contains the eigen values of each case/scenario
        
    PC_max : integer
        Number of PC showed on barplot, where the last is summed up i.e. 1, 2, 3, (4-30) which is 4 in total (PC_max=4)
        
    filename : list of strings
        List of filenames. Used to determine how many bars plotted
        
    constraints : list of strings
        Used as descriptor for x-axis.
        
    title : string
        Title of the plot.
        
    xlabel : string
        label for the x-axis beside the constraint i.e. CO2 or Transmission size.
        
    suptitle : string (optional)
        Subtitle for plot  (above title). Type 'none' for no subtitle.

    Returns
    -------
    fig1: Figure
        Saves the figure as a variable to be used by savePlot

    """
    
    fig1 = plt.figure(figsize=figsize, dpi=dpi)
    # fig.add_axes([x1,y1,x2,y2])
    ax = fig1.add_axes([0,0,1,1.5])    
    
    # List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = plt.get_cmap("tab10")
    
    # Number of "color-bars" / PC to show (k1, k2, k3...)
    j_max = PC_max
    colour_map = cmap(np.arange(j_max)) # Must be equal to the number of bars plotted
    
    # Label variable
    lns_fig1 = []
    
    # Number of scenarios / System configurations (i.e. 40%, 50%, ... CO2 reduction)
    for i in np.arange(0,len(filename)):
        
        # Number of components each bar comprises of (equal to j_max above)
        for j in np.arange(0,j_max):
            
            # Bottom part of bar
            if j == 0:
                lns_plot = ax.bar(constraints[i], 
                                  matrix[i][j], 
                                  color=colour_map[j], 
                                  edgecolor='black', 
                                  linewidth=1.2, 
                                  label=('$k_{' + str(j+1) + '}$'))
              
            # Middle part of bar
            elif (j > 0 and j < (j_max-1)):
                lns_plot = ax.bar(constraints[i], 
                                  matrix[i][j], 
                                  bottom = sum(matrix[i][0:j]), 
                                  color=colour_map[j], 
                                  edgecolor='black', 
                                  label=('$k_{' + str(j+1) + '}$'))
            
            # Last part of bar
            else:
                lns_plot = ax.bar(constraints[i], 
                                  sum(matrix[i][j:29]), 
                                  bottom = sum(matrix[i][0:j]), 
                                  color=colour_map[j], 
                                  edgecolor='black', 
                                  label=('$k_{' + str(j+1) + '}$ - $k_{30}$'))
                
            
            # Create lable only for the first iteration of for-loop 
            if (i==0):
                lns_fig1.append(lns_plot)
    
    
    # Add labels
    labs = [l.get_label() for l in lns_fig1]
    #ax.legend(lns_fig1, labs, fontsize=14, bbox_to_anchor = (1,1))
    ax.legend(lns_fig1, labs, fontsize=fontsize, loc="center", bbox_to_anchor = bbox, ncol=ncol)
    
    
    plt.yticks(range(0, 120, 25), fontsize=fontsize)
    plt.xticks(fontsize=fontsize, rotation=rotation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Variance for each PC [%]', fontsize=fontsize)
    plt.grid(axis='y')
    
    # Include label unless is given variable value 'none'
    if suptitle != 'none':
        plt.suptitle(suptitle, fontsize=20,x=.5,y=1.68)

    # Show plot
    plt.show()    

    return fig1

#%% FFTPlot

def FFTPlot(T,variance_explained, PC_NO=-1,dpi=200,title="none"):
    """

    Parameters
    ----------
    T : Matrix
        Principle component amplitudes. Given by: B*eig_val (so the centered and scaled data dotted with the eigen values)
    file_name: array of strings
        Name of the datafile there is worked with
    PC_NO : 
        PC number, if default generate for 6 together
    dpi : 
        quality of picture

    Returns
    -------
    fig: Figure
        Saves the figure as a variable to be used by savePlot

    """
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # Determins if 6 subplots or just 1 plot
    if PC_NO ==-1: # Generate subplot with the first 6 PC
    
        fig = plt.figure(figsize=(18,12),dpi=dpi) # Generate figure
        
        for j in range(6):
        
            plt.subplot(3,2,j+1)
            freq=np.fft.fftfreq(len(T[j]))  
            FFT=np.fft.fft(T[j])
            FFT[j]=0
            FFT=abs(FFT)/max(abs(FFT))
            plt.plot(1/freq,FFT,color=color[j])
            plt.xscale('log')
            plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2) # 1/2 day
            plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2) # day
            plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2) # week
            plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2) # month
            plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2) # year
            plt.text(12,0.9,"1/2 Day",ha='right')
            plt.text(22,0.9,"Day",ha='right')
            plt.text(22*7,0.9,"Week",ha='right')
            plt.text(22*7*4,0.9,"Month",ha='right')
            plt.text(22*365,0.9,"Year",ha='right')
            plt.xlabel('Hours')
            plt.title('Fourier Power Spectra for PC'+ str(j+1))
            plt.legend([r'$\lambda_{'+ str(j+1) + '}$ = '+str(round(variance_explained[j],1))+"%"], loc=[0.0069,0.69],handlelength=0)
            
        plt.subplots_adjust(wspace=0, hspace=0.28)
        if title != "none":
            plt.suptitle(title,fontsize=20,x=.51,y=0.93) #,x=.51,y=1.07
            
        plt.show()
    
    else: # Generate plot from the given PC number
        
        j = PC_NO-1 # Uses the same varible as the subplot setup
        
        fig = plt.figure(figsize=(5,3.4),dpi=dpi)
        freq=np.fft.fftfreq(len(T[j]))  
        FFT=np.fft.fft(T[j])
        FFT[j]=0
        FFT=abs(FFT)/max(abs(FFT))
        plt.plot(1/freq,FFT,color=color[j])
        plt.xscale('log')
        plt.vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2) # 1/2 day
        plt.vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2) # day
        plt.vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2) # week
        plt.vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2) # month
        plt.vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2) # year
        plt.text(9.6,1,"1/2 Day",     ha='center', va="top", fontsize=14, rotation=90)
        plt.text(19,1,"Day",        ha='center', va="top", fontsize=14, rotation=90)
        plt.text(20*7,1,"Week",     ha='center', va="top", fontsize=14, rotation=90)
        plt.text(20*7*4,1,"Month",  ha='center', va="top", fontsize=14, rotation=90)
        plt.text(20*365,1,"Year",   ha='center', va="top", fontsize=14, rotation=90)
        plt.xlabel('Hours', fontsize=16)
        if (variance_explained[j] < 0.1):
            plt.title("Fourier Power Spectra", fontsize=16)# : " + "$\lambda_{" + str(j+1) + '}$ = '+str(round(variance_explained[j],2))+"%")
        elif (variance_explained[j] < 0.01):
            plt.title("Fourier Power Spectra", fontsize=16)#: " + "$\lambda_{" + str(j+1) + '}$ = '+str(round(variance_explained[j],3))+"%")
        else:
            plt.title("Fourier Power Spectra", fontsize=16)#: " + "$\lambda_{" + str(j+1) + '}$ = '+str(round(variance_explained[j],1))+"%")
        #plt.legend([r'$\lambda_{'+ str(j+1) + '}$ = '+str(round(eigen_values[j]*100,1))+"%"], loc=[0.0069,0.69],handlelength=0)
        
        plt.subplots_adjust(wspace=0, hspace=0.28)
        if title != "none":
            plt.suptitle(title,fontsize=12,x=.51,y=1) #,x=.51,y=1.07
        plt.text(0.73,1.08,"(a)", fontsize=16)    
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.show() # Generate plot
        
    return fig

#%% SavePlot

def SavePlot(fig,path,title):
    """

    Parameters
    ----------
    path : string
        folder path in which the figures needs to be saved in. This needs to be created before
    title: string
        Name of the figure, this is also what the figures is saved as. This does not need to be created

    Returns
    -------
    Nothing

    """
    
    # Check if path exist
    assert os.path.exists(path), "Path does not exist"
    
    # Check if figure is already existing there
    fig.savefig(path+title+".png", bbox_inches='tight')
    
    return fig
    
#%% GeneratorSplit

def GeneratorSplit(network, generatorType):
    """

    Parameters
    ----------
    network : PyPSA network
        Index of all the countries, for this project its 30 countries
    generatorType: string
        Name of the generator 
    
    Returns
    -------
    generator : Panda Dataframe
        ...

    """
    
    # Country names
    country_column = network.loads.index[:30]
    
    # every generator type
    generatorIndex = network.generators.index
    
    # Time index
    generatorsTimeData = network.generators_t.p
    
    # Create empty matrix
    generator = pd.DataFrame(np.zeros([8760, 30]), columns=country_column) # Create empty dataframe
    
    
    
    count = 0 # Add a counter
    
    for j in generatorIndex: # Search through all generator types
        value = np.array(generatorsTimeData)[:,count] # Save the current generator value
        if len(j) <= len(generatorType)+7:
            if (j[-(len(generatorType)):] == generatorType): # If the generator type is equal to the current generator
                generator[j[0:2]] = generator[j[0:2]] + value # Save the value for the country
            
        count += 1 # Add one to the counter
    
    return generator 

#%% MeanAndCenter
def MeanAndCenter(value):
    """

    Parameters
    ----------
    Value : Panda Dataframe
        The given value, either a load or generator to be centered around the mean value

    Returns
    -------
    valueCenter : Panda Dataframe
        The calulated value

    """    
    
    # Mean
    valueMean = np.mean(value, axis=0)
    
    # Center
    valueCenter = np.subtract(value,valueMean.T)
    
    return valueCenter

#%% MapCapacityOriginal
def MapCapacityOriginal(network, filename, renewablesOnly=False, scaleno=5, ncol=3):
    """
    Parameters
    ----------
    network : PyPSA network object
        The network file (.h5) loaded into Python containing the network data
        
    filename : String
        The top-most title name of the plot (often the file name)
        
    renewablesOnly : Boolean (False)
        Determines if renewable technologies are plottet alone or include (some) backups as well.
        When plotting Brownfield networks, its adviced to only plot renewables 
        as they have a specific plotting function to include backup generators
        
    scaleno : Int (scaleno 5)
        Scaling factor for figures
        

    Returns
    -------
    fig1 : matplotlib figure
        Figure containing installed generating capacity (MW) for "elec only" components.
        
    fig2 : matplotlib figure
        Figure containing storage energy capacity (MWh) for "elec only" components.
        
    fig3 : matplotlib figure
        Figure containing storage power capacity (MW) for "elec only" components.

    """
    
    # Figure size
    figureSize = 15.0
    
    # Adjust Norway, Sweden and Finland coordinates
    network.buses.x['NO'] = 6.69
    network.buses.y['NO'] = 60.6969
    
    network.buses.x['SE'] = 14.69
    network.buses.y['SE'] = 59.6969
    
    network.buses.x['FI'] = 24.69
    network.buses.y['FI'] = 62.69
    
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get locations for each country node
    countryLocations = pd.DataFrame(data = [network.buses.x[dataNames], network.buses.y[dataNames]])
    countryLocations = countryLocations.T
    
    # Set coordinates (x,y) for remaining carriers without coordinates
    for i in np.arange(network.buses.shape[0]):
        if network.buses.x[i] == 0 and network.buses.y[i] == 0:
            network.buses.x[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].x
            network.buses.y[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].y
    
    # Create lists to store technologies used for storage + dict for colors + list for legend item
    elec = pd.Series()
    generatorColors = {}
    legendElements = []
    
    # Data: Wind
    if any([x for x in network.generators.p_nom_opt.index.str.slice(3).unique() if "wind" in x]):
        wind = network.generators.p_nom_opt[[technology for technology in network.generators.p_nom_opt.index if "onwind" in technology]]
        wind.index = wind.index.str.slice(0,2)
        wind = wind.groupby(wind.index).sum()
        wind.index = (wind.index + " wind")
    
        elec = pd.concat([elec, wind])
        generatorColors["wind"] = "dodgerblue"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Wind', markerfacecolor=generatorColors["wind"], markersize=24))
    
    
    # Data: Solar PV
    if "solar" in network.generators.p_nom_opt.index.str.slice(3).unique():
        solar = network.generators.p_nom_opt[dataNames + " solar"]
    
        elec = pd.concat([elec, solar])
        generatorColors["solar"] = "gold"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Solar PV', markerfacecolor=generatorColors["solar"], markersize=24))

    # Data: ror
    if "ror" in network.generators.p_nom_opt.index.str.slice(3).unique():
        ror = network.generators.p_nom_opt[[technology for technology in network.generators.p_nom_opt.index if "ror" in technology]]
        ror.index = ror.index.str.slice(0,2)
        ror = ror.groupby(ror.index).sum()
        ror.index = (ror.index + " ror")
        
        elec = pd.concat([elec, ror])
        generatorColors["ror"] = "limegreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='ROR', markerfacecolor=generatorColors["ror"], markersize=24))

    
    # If only want to plot renewable technologies
    if not renewablesOnly:
        # Backup technologies
        
        # Data matrix for backup technologies including efficiency
        dataMatrix = (network.links.p_nom_opt * network.links.efficiency.values).rename("Power")
        
        # Unique technologies included in backup
        backupTech = dataMatrix.index.str.slice(3).unique()
        
        # Data: hydro
        if "hydro" in network.storage_units.index.str.slice(3).unique():
            hydro = network.storage_units.p_nom_opt[[technology for technology in network.storage_units.p_nom_opt.index if "hydro" in technology]]
            elec = pd.concat([elec, hydro])
            generatorColors["hydro"] = "aquamarine"
            legendElements.append(Line2D([0], [0], marker='o', color='white', label='Hydro', markerfacecolor=generatorColors["hydro"], markersize=24))
        
        
        # Data: OCGT
        if "OCGT" in backupTech:
            OCGT = dataMatrix[[technology for technology in dataMatrix.index if "OCGT" in technology]]
            elec = pd.concat([elec, OCGT])
            generatorColors["OCGT"] = "firebrick"
            legendElements.append(Line2D([0], [0], marker='o', color='white', label='OCGT', markerfacecolor=generatorColors["OCGT"], markersize=24))
    
        # Data: OCGT
        if "central CHP electric" in backupTech:
            CHPElectric = dataMatrix[[technology for technology in dataMatrix.index if "CHP electric" in technology]]
            elec = pd.concat([elec, CHPElectric])
            generatorColors["central CHP electric"] = "deeppink"
            legendElements.append(Line2D([0], [0], marker='o', color='white', label='CHP Electric', markerfacecolor=generatorColors["central CHP electric"], markersize=24))
    
        
    

    # Create array of country code and technology type
    elecNames = [item[0:2] for item in elec.index]
    elecTech = [item[3:] for item in elec.index]
    
    # Turn into a series with indexes ['bus', 'carrier']
    dataElec = pd.Series(data=elec.values, index=[pd.Index(elecNames, name="bus"), pd.Index(elecTech, name="carrier")])
    
    # Scale data
    scale = scaleno / dataElec.groupby(level=[0]).sum().max()
    dataElec *= scale
    
    # Plot figure
    fig1 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataElec, bus_colors=generatorColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    #plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    #plt.title("Installed electricity generator capacity [MW]", fontsize=16)
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataElec["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize,0))[0:-2] + " MW)"))
    
    elif len(str(round(DESize,0))) < 9:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,1))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GW)"))
    
    elif len(str(round(DESize,0))) < 12:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e6,1)) + " TW)"))
    
    elif len(str(round(DESize,0))) < 15:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0)) + " PW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e9,1)) + " PW)"))
    
    
    #plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    plt.legend(handles=legendElements, fontsize=28, loc='upper center', bbox_to_anchor = (0.5, 0.0), framealpha=0.8, ncol=ncol)
    
    
    #----------------- STORAGE ENERGY CAPACITY (MWh) ------------------------#
    
    # Create lists to store technologies used for storage + dict for colors + list for legend item
    storageEnergy = pd.Series()
    storageColors = {}
    legendElements = []
    
    # Data: Battery
    if any("battery" == network.stores.e_nom_opt.index.str.slice(3).unique()):
        # Position in array of countries with "battery" storage
        pos = ("battery" == network.stores.e_nom_opt.index.str.slice(3))
        
        # Index name of those positions
        posIndex = network.stores.e_nom_opt.index[pos]
        
        # Index in array with those index names
        battery = network.stores.e_nom_opt[posIndex]
        battery.index = battery.index.str.slice(0,2)
        battery = battery.groupby(battery.index).sum()
        battery.index = (battery.index + " battery")
        
        storageEnergy = pd.concat([storageEnergy, battery])
        storageColors["battery"] = "springgreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Battery', markerfacecolor=storageColors["battery"], markersize=24))
    

    # Data: H2
    if "H2 Store" in network.stores.e_nom_opt.index.str.slice(3).unique():
        H2 = network.stores.e_nom_opt[[technology for technology in network.stores.e_nom_opt.index if "H2" in technology]]
        H2.index = H2.index.str.slice(0,2)
        H2 = H2.groupby(H2.index).sum()
        H2.index = (H2.index + " H2")
        
        storageEnergy = pd.concat([storageEnergy, H2])
        storageColors["H2"] = "purple"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='H2', markerfacecolor=storageColors["H2"], markersize=24))
    
    # Data: PHS
    if "PHS" in network.storage_units.p_nom_opt.index.str.slice(3).unique():
        PHS = network.storage_units_t.state_of_charge.max()[[technology for technology in network.storage_units_t.state_of_charge.max().index if "PHS" in technology]]
        PHS.index = PHS.index.str.slice(0,2)
        PHS = PHS.groupby(PHS.index).sum()
        PHS.index = (PHS.index + " PHS")
        
        storageEnergy = pd.concat([storageEnergy, PHS])
        storageColors["PHS"] = "aqua"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='PHS', markerfacecolor=storageColors["PHS"], markersize=24))
    
    # Include transportation
    if "battery storage" in network.stores.e_nom_opt.index.str.slice(3).unique():
        EVBattery = network.stores.e_nom_opt[[technology for technology in network.stores.e_nom_opt.index if "battery storage" in technology]]
        EVBattery.index = EVBattery.index.str.slice(0,2)
        EVBattery = EVBattery.groupby(EVBattery.index).sum()
        EVBattery.index = (EVBattery.index + " battery storage")
        
        storageEnergy = pd.concat([storageEnergy, EVBattery])
        storageColors["battery storage"] = "darkorange"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='EV Battery', markerfacecolor=storageColors["battery storage"], markersize=24))
        
    
    # Create array of country code and technology type
    storageNames = [item[0:2] for item in storageEnergy.index]
    storageTech = [item[3:] for item in storageEnergy.index]

    # Turn into a series with indexes ['bus', 'carrier']
    dataStorageEnergy = pd.Series(data=storageEnergy.values, index=[pd.Index(storageNames, name="bus"), pd.Index(storageTech, name="carrier")])
    
    # Scale data
    scale = scaleno / dataStorageEnergy.groupby(level=[0]).sum().max()
    dataStorageEnergy *= scale
      
    # Plot figure
    fig2 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataStorageEnergy, bus_colors=storageColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    
    #plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    #plt.title("Installed electricity storage energy capacity [MWh]", fontsize=16)            
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataStorageEnergy["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize,0))[0:-2] + " MWh)"))
    
    elif len(str(round(DESize,0))) < 9:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,1))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GWh)"))
    
    elif len(str(round(DESize,0))) < 12:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e6,1)) + " TWh)"))
    
    elif len(str(round(DESize,0))) < 15:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0)) + " PW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e9,1)) + " PWh)"))
    
    
    #plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    plt.legend(handles=legendElements, fontsize=28, loc='upper center', bbox_to_anchor = (0.5, 0.0), framealpha=0.8, ncol=ncol)
    
    
    #------------------- STORAGE POWER CAPACITY (MW) ------------------------#
    
    # Create lists to store technologies used for storage + dict for colors + list for legend item
    storagePower = pd.Series()
    storageColors = {}
    legendElements = []
    
    # Data: Battery
    if any("battery" == network.stores_t.p.max().index.str.slice(3).unique()):
        # Position in array of countries with "battery" storage
        pos = ("battery" == network.stores_t.p.max().index.str.slice(3))
        
        # Index name of those positions
        posIndex = network.stores_t.p.max().index[pos]
        
        # Index in array with those index names
        battery = network.stores_t.p.max()[posIndex]
        battery.index = battery.index.str.slice(0,2)
        battery = battery.groupby(battery.index).sum()
        battery.index = (battery.index + " battery")
        
        storagePower = pd.concat([storagePower, battery])
        storageColors["battery"] = "springgreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Battery', markerfacecolor=storageColors["battery"], markersize=24))
    

    # Data: H2
    if "H2 Store" in network.stores_t.p.max().index.str.slice(3).unique():
        H2 = network.stores_t.p.max()[[technology for technology in network.stores_t.p.max().index if "H2" in technology]]
        H2.index = H2.index.str.slice(0,2)
        H2 = H2.groupby(H2.index).sum()
        H2.index = (H2.index + " H2")
        
        storagePower = pd.concat([storagePower, H2])
        storageColors["H2"] = "purple"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='H2', markerfacecolor=storageColors["H2"], markersize=24))
    
    # Data: PHS
    if "PHS" in network.storage_units_t.p.max().index.str.slice(3).unique():
        PHS = network.storage_units_t.p.max()[[technology for technology in network.storage_units_t.p.max().index if "PHS" in technology]]
        PHS.index = PHS.index.str.slice(0,2)
        PHS = PHS.groupby(PHS.index).sum()
        PHS.index = (PHS.index + " PHS")
        
        storagePower = pd.concat([storagePower, PHS])
        storageColors["PHS"] = "aqua"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='PHS', markerfacecolor=storageColors["PHS"], markersize=24))
    
    # Include transportation
    if "battery storage" in network.stores_t.p.max().index.str.slice(3).unique():
        EVBattery = network.stores_t.p.max()[[technology for technology in network.stores_t.p.max().index if "battery storage" in technology]]
        EVBattery.index = EVBattery.index.str.slice(0,2)
        EVBattery = EVBattery.groupby(EVBattery.index).sum()
        EVBattery.index = (EVBattery.index + " battery storage")
        
        storagePower = pd.concat([storagePower, EVBattery])
        storageColors["battery storage"] = "darkorange"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='EV Battery', markerfacecolor=storageColors["battery storage"], markersize=24))
        
    
    # Create array of country code and technology type
    storageNames = [item[0:2] for item in storagePower.index]
    storageTech = [item[3:] for item in storagePower.index]

    # Turn into a series with indexes ['bus', 'carrier']
    dataStoragePower = pd.Series(data=storagePower.values, index=[pd.Index(storageNames, name="bus"), pd.Index(storageTech, name="carrier")])
       
    # Down-scale values
    scale = scaleno / dataStoragePower.groupby(level=[0]).sum().max()
    dataStoragePower *= scale
    
    # Plot figure
    fig3 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataStoragePower, bus_colors=storageColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))   
    
    #plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    #plt.title("Installed electricity storage power capacity [MW]", fontsize=16)

    # Create label for expressing relative size of Germany:
    DESize = round((dataStoragePower["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize,0))[0:-2] + " MW)"))
    
    elif len(str(round(DESize,0))) < 9:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,1))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GW)"))
    
    elif len(str(round(DESize,0))) < 12:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e6,1)) + " TW)"))
    
    elif len(str(round(DESize,0))) < 15:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0)) + " PW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e9,1)) + " PW)"))
    
    
    #plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    plt.legend(handles=legendElements, fontsize=28, loc='upper center', bbox_to_anchor = (0.5, 0.0), framealpha=0.8, ncol=ncol)
    
    
    
    
    
    
    plt.show(all)
    return (fig1, fig2, fig3)
    
#%% MapCapacityHeat

def MapCapacityHeat(network, filename, scaleno=5, ncol=3):

    
    """
    

    Parameters
    ----------
    network : PyPSA network object
        The network file (.h5) loaded into Python containing the network data
        
    filename : String
        The top-most title name of the plot (often the file name)
        
    scaleno : Int (scaleno 5)
        Scaling factor for figures

    Returns
    -------
    fig1 : matplotlib figure
        Figure containing installed generating capacity (MW) for heating components.
        
    fig2 : matplotlib figure
        Figure containing storage energy capacity (MWh) for heating components.
        
    fig3 : matplotlib figure
        Figure containing storage power capacity (MW) for heating components.

    """
    
    # Figure size
    figureSize = 15.0
    
    # Adjust Norway, Sweden and Finland coordinates
    network.buses.x['NO'] = 6.69
    network.buses.y['NO'] = 60.6969
    
    network.buses.x['SE'] = 14.69
    network.buses.y['SE'] = 59.6969
    
    network.buses.x['FI'] = 24.69
    network.buses.y['FI'] = 62.69
    
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get locations for each country node
    countryLocations = pd.DataFrame(data = [network.buses.x[dataNames], network.buses.y[dataNames]])
    countryLocations = countryLocations.T
    
    # Set coordinates (x,y) for remaining carriers without coordinates
    for i in np.arange(network.buses.shape[0]):
        if network.buses.x[i] == 0 and network.buses.y[i] == 0:
            network.buses.x[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].x
            network.buses.y[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].y
    
    
    
    # Create lists to store technologies used for storage + dict for colors + list for legend item
    heat = pd.Series()
    generatorColors = {}
    legendElements = []
    
    
    # Data: Solar Collectors
    if any([x for x in network.generators.p_nom_opt.index if "thermal" in x]):
        
        singleCollector = network.generators.p_nom_opt["solar thermal collector" == network.generators.p_nom_opt.index.str.slice(3)]
        heat = pd.concat([heat, singleCollector])
        generatorColors["solar thermal collector"] = "seagreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Solar collector', markerfacecolor=generatorColors["solar thermal collector"], markersize=24))

    
        centralCollector = network.generators.p_nom_opt["central solar thermal collector" == network.generators.p_nom_opt.index.str.slice(3)]
        heat = pd.concat([heat, centralCollector])
        generatorColors["central solar thermal collector"] = "orange"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Central solar collector', markerfacecolor=generatorColors["central solar thermal collector"], markersize=24))

        urbanCollector = network.generators.p_nom_opt["urban solar thermal collector" == network.generators.p_nom_opt.index.str.slice(3)]
        heat = pd.concat([heat, urbanCollector])
        generatorColors["urban solar thermal collector"] = "plum"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Urban solar collector', markerfacecolor=generatorColors["urban solar thermal collector"], markersize=24))


    # Data: Heat pump
    if any([x for x in network.links.p_nom_opt.index if "heat pump" in x]):
        heatPump = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if "heat pump" in x]]
        heatPump.index = heatPump.index.str.slice(0,2)
        heatPump = heatPump.groupby(heatPump.index).sum()
        heatPump.index = (heatPump.index + " heat pump")
    
        heat = pd.concat([heat, heatPump])
        generatorColors["heat pump"] = "coral"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Heat pump', markerfacecolor=generatorColors["heat pump"], markersize=24))

    # Data: Resistive heater
    if any([x for x in network.links.p_nom_opt.index if "resistive heater" in x]):
        resistiveHeater = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if "resistive heater" in x]]
        resistiveHeater.index = resistiveHeater.index.str.slice(0,2)
        resistiveHeater = resistiveHeater.groupby(resistiveHeater.index).sum()
        resistiveHeater.index = (resistiveHeater.index + " resistive heater")
    
        heat = pd.concat([heat, resistiveHeater])
        generatorColors["resistive heater"] = "maroon"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Resistive heater', markerfacecolor=generatorColors["resistive heater"], markersize=24))


    # Data: Gas boiler
    if any([x for x in network.links.p_nom_opt.index if "gas boiler" in x]):
        gasBoiler = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if "gas boiler" in x]]
        gasBoiler.index = gasBoiler.index.str.slice(0,2)
        gasBoiler = gasBoiler.groupby(gasBoiler.index).sum()
        gasBoiler.index = (gasBoiler.index + " gas boiler")
    
        heat = pd.concat([heat, gasBoiler])
        generatorColors["gas boiler"] = "chartreuse"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Gas boiler', markerfacecolor=generatorColors["gas boiler"], markersize=24))

    # Data: CHP heat
    if any([x for x in network.links.p_nom_opt.index if "CHP heat" in x]):
        CHPHeat = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if "CHP heat" in x]]
        CHPHeat.index = CHPHeat.index.str.slice(0,2)
        CHPHeat = CHPHeat.groupby(CHPHeat.index).sum()
        CHPHeat.index = (CHPHeat.index + " CHP heat")
    
        heat = pd.concat([heat, CHPHeat])
        generatorColors["CHP heat"] = "indigo"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='CHP heat', markerfacecolor=generatorColors["CHP heat"], markersize=24))



    # Create array of country code and technology type
    heatNames = [item[0:2] for item in heat.index]
    heatTech = [item[3:] for item in heat.index]
    
    # Turn into a series with indexes ['bus', 'carrier']
    data = pd.Series(data=heat.values, index=[pd.Index(heatNames, name="bus"), pd.Index(heatTech, name="carrier")])
    
    # Scale data to fit plot
    scale = scaleno / data.groupby(level=[0]).sum().max()
    data *= scale
    
    # Plot figure
    fig1 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=data, bus_colors=generatorColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    
    #plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    #plt.title("Installed heating generator capacity [MW]", fontsize=16)

    # Create label for expressing relative size of Germany:
    DESize = round((data["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize,0))[0:-2] + " MW)"))
    
    elif len(str(round(DESize,0))) < 9:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,1))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GW)"))
    
    elif len(str(round(DESize,0))) < 12:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e6,1)) + " TW)"))
    
    elif len(str(round(DESize,0))) < 15:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0)) + " PW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e9,1)) + " PW)"))
    
    
    #plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    plt.legend(handles=legendElements, fontsize=28, loc='upper center', bbox_to_anchor = (0.5, 0.0), framealpha=0.8, ncol=ncol)
    

    #------------------ STORAGE ENERGY CAPACITY (MWh) -----------------------#    
    # List of technolgies available to current network
    heatStorageTechnologies = network.stores.e_nom_opt.index.str.slice(3).unique()
    
    # Create dataframe to store technologies used for heat storage + dict for colors + legend item
    heatStorage = pd.Series()
    heatStorageColors = {}
    legendElements = []
    
    # Checks for water tank specific in technologies
    if any('water tank' == heatStorageTechnologies):
        waterTank = network.stores.e_nom_opt[dataNames + " water tank"]
        heatStorage = pd.concat([heatStorage, waterTank])
        heatStorageColors["water tank"] = "teal"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Water tank', markerfacecolor=heatStorageColors["water tank"], markersize=24))

    # Checks for central water tank specific in technologies
    if any('central water tank' == heatStorageTechnologies):
        cenWaterTank = network.stores.e_nom_opt[[x for x in network.stores.e_nom_opt.index if "central water tank" in x]]
        heatStorage = pd.concat([heatStorage, cenWaterTank])
        heatStorageColors["central water tank"] = "tomato"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Central water tank', markerfacecolor=heatStorageColors["central water tank"], markersize=24))


    # Checks for water tank specific in technologies
    if any('urban water tank' == heatStorageTechnologies):
        urbanWaterTank = network.stores.e_nom_opt[[x for x in network.stores.e_nom_opt.index if "urban water tank" in x]]
        heatStorage = pd.concat([heatStorage, urbanWaterTank])
        heatStorageColors["urban water tank"] = "rebeccapurple"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Urban water tank', markerfacecolor=heatStorageColors["urban water tank"], markersize=24))
    
    
    # Create array of country code and technology type from storageEnergy (network.stores.e_nom_opt)
    storageEnergyNames = [item[0:2] for item in heatStorage.index]
    storageEnergyTech = [item[3:] for item in heatStorage.index]
    
    # Turn into a series with indexes ['bus', 'carrier']
    dataStorageEnergy = pd.Series(data=heatStorage.values, index=[pd.Index(storageEnergyNames, name="bus"), pd.Index(storageEnergyTech, name="carrier")])

    # Down-scale values
    scale = scaleno / dataStorageEnergy.groupby(level=[0]).sum().max()
    dataStorageEnergy *= scale
    
    # Plot figure
    fig2 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataStorageEnergy, bus_colors=heatStorageColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))   
    
    #plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    #plt.title("Installed heating storage energy capacity [MWh]", fontsize=16)
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataStorageEnergy["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize,0))[0:-2] + " MWh)"))
    
    elif len(str(round(DESize,0))) < 9:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,1))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GWh)"))
    
    elif len(str(round(DESize,0))) < 12:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e6,1)) + " TWh)"))
    
    elif len(str(round(DESize,0))) < 15:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0)) + " PW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e9,1)) + " PWh)"))
    
    
    #plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    plt.legend(handles=legendElements, fontsize=28, loc='upper center', bbox_to_anchor = (0.5, 0.0), framealpha=0.8, ncol=ncol)
    

    #------------------- STORAGE POWER CAPACITY (MW) ------------------------#
    # Create dataframe to store technologies used for heat storage + dict for colors + legend item
    heatStorage = pd.Series()
    heatStorageColors = {}
    legendElements = []
    
    # Checks for water tank specific in technologies
    if any('water tank' == heatStorageTechnologies):
        waterTank = network.stores_t.p.max()[dataNames + " water tank"]
        heatStorage = pd.concat([heatStorage, waterTank])
        heatStorageColors["water tank"] = "teal"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Water tank', markerfacecolor=heatStorageColors["water tank"], markersize=24))

    # Checks for central water tank specific in technologies
    if any('central water tank' == heatStorageTechnologies):
        cenWaterTank = network.stores_t.p.max()[[x for x in network.stores_t.p.max().index if "central water tank" in x]]
        heatStorage = pd.concat([heatStorage, cenWaterTank])
        heatStorageColors["central water tank"] = "tomato"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Central water tank', markerfacecolor=heatStorageColors["central water tank"], markersize=24))


    # Checks for water tank specific in technologies
    if any('urban water tank' == heatStorageTechnologies):
        urbanWaterTank = network.stores_t.p.max()[[x for x in network.stores_t.p.max().index if "urban water tank" in x]]
        heatStorage = pd.concat([heatStorage, urbanWaterTank])
        heatStorageColors["urban water tank"] = "rebeccapurple"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Urban water tank', markerfacecolor=heatStorageColors["urban water tank"], markersize=24))
    
    
        
    # Create array of country code and technology type from storagePower (network.stores_t.p)
    storagePowerNames = [item[0:2] for item in heatStorage.index]
    storagePowerTech = [item[3:] for item in heatStorage.index]
    
    
    # Turn into a series with indexes ['bus', 'carrier']
    dataStoragePower = pd.Series(data=heatStorage.values, index=[pd.Index(storagePowerNames, name="bus"), pd.Index(storagePowerTech, name="carrier")])

    # Down-scale values
    scale = scaleno / dataStoragePower.groupby(level=[0]).sum().max()
    dataStoragePower *= scale
    
    # Plot figure
    fig3 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataStoragePower, bus_colors=heatStorageColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    
    #plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    #plt.title("Installed heating storage power capacity [MW]", fontsize=16)
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataStoragePower["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize,0))[0:-2] + " MW)"))
    
    elif len(str(round(DESize,0))) < 9:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,1))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GW)"))
    
    elif len(str(round(DESize,0))) < 12:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e6,1)) + " TW)"))
    
    elif len(str(round(DESize,0))) < 15:
        #legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0)) + " PW"), markerfacecolor="white", markersize=13))
        plt.text(x=-9.5, y=63, fontsize=28, fontstyle="italic", s=("(Size of Germany: " + str(round(DESize/1e9,1)) + " PW)"))
    
    
    #plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    plt.legend(handles=legendElements, fontsize=28, loc='upper center', bbox_to_anchor = (0.5, 0.0), framealpha=0.8, ncol=ncol)
    
    plt.show(all)
    
    return (fig1, fig2, fig3)
    

#%% MapBackupElec
def MapBackupElec(network, filename, scaleno=5):
    """
    Parameters
    ----------
    network : PyPSA network object
        The network file (.h5) loaded into Python containing the network data
        
    filename : String
        The top-most title name of the plot (often the file name)
        
    scaleno : Int (scaleno 5)
        Scaling factor for figures
        
    Returns
    -------
    fig1 : matplotlib figure
        Figure containing installed backup electricity capacity (MW)
        
    """
    
    # Figure size
    figureSize = 15.0
    
    # Adjust Norway, Sweden and Finland coordinates
    network.buses.x['NO'] = 6.69
    network.buses.y['NO'] = 60.6969
    
    network.buses.x['SE'] = 14.69
    network.buses.y['SE'] = 59.6969
    
    network.buses.x['FI'] = 24.69
    network.buses.y['FI'] = 62.69
    
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get locations for each country node
    countryLocations = pd.DataFrame(data = [network.buses.x[dataNames], network.buses.y[dataNames]])
    countryLocations = countryLocations.T
    
    # Set coordinates (x,y) for remaining carriers without coordinates
    for i in np.arange(network.buses.shape[0]):
        if network.buses.x[i] == 0 and network.buses.y[i] == 0:
            network.buses.x[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].x
            network.buses.y[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].y
    
    
    # Data matrix for backup technologies including efficiency
    dataMatrix = (network.links.p_nom_opt * network.links.efficiency.values).rename("Power")
    
    # Unique technologies included in backup
    backupTech = dataMatrix.index.str.slice(3).unique()
    
    # Create dataframe to store technologies used for backup + dict for colors + legend item
    backup = pd.Series()
    backupColors = {}
    legendElements = []
    
    
    # Data: hydro
    if "hydro" in network.storage_units.index.str.slice(3).unique():
        hydro = network.storage_units.p_nom_opt[[technology for technology in network.storage_units.p_nom_opt.index if "hydro" in technology]]
        backup = pd.concat([backup, hydro])
        backupColors["hydro"] = "darkturquoise"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Hydro', markerfacecolor=backupColors["hydro"], markersize=13))
    
    
    # Data: OCGT
    if "OCGT" in backupTech:
        OCGT = dataMatrix[[technology for technology in dataMatrix.index if "OCGT" in technology]]
        backup = pd.concat([backup, OCGT])
        backupColors["OCGT"] = "firebrick"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='OCGT', markerfacecolor=backupColors["OCGT"], markersize=13))


    # Data: CCGT
    if "CCGT" in backupTech:
        CCGT = dataMatrix[[technology for technology in dataMatrix.index if "CCGT" in technology]]
        backup = pd.concat([backup, CCGT])
        backupColors["CCGT"] = "darkgoldenrod"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='CCGT', markerfacecolor=backupColors["CCGT"], markersize=13))


    # Data: coal
    if "coal" in backupTech:
        coal = dataMatrix[[technology for technology in dataMatrix.index if "coal" in technology]]
        backup = pd.concat([backup, coal])
        backupColors["coal"] = "dimgray"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Coal', markerfacecolor=backupColors["coal"], markersize=13))


    # Data: nuclear
    if "nuclear" in backupTech:
        nuclear = dataMatrix[[technology for technology in dataMatrix.index if "nuclear" in technology]]
        backup = pd.concat([backup, nuclear])
        backupColors["nuclear"] = "bisque"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Nuclear', markerfacecolor=backupColors["nuclear"], markersize=13))


    # Data: lignite    
    if "lignite" in backupTech:
        lignite = dataMatrix[[technology for technology in dataMatrix.index if "lignite" in technology]]
        backup = pd.concat([backup, lignite])
        backupColors["lignite"] = "darkseagreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Lignite', markerfacecolor=backupColors["lignite"], markersize=13))


    # Data: oil 
    if "oil" in backupTech:
        oil = dataMatrix["oil" == dataMatrix.index.str.slice(3)]
        backup = pd.concat([backup, oil])
        backupColors["oil"] = "darkviolet"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Oil', markerfacecolor=backupColors["oil"], markersize=13))


    # Data: biomass EOP 
    if "biomass EOP" in backupTech:
        biomassEOP = dataMatrix[[technology for technology in dataMatrix.index if "biomass EOP" in technology]]
        backup = pd.concat([backup, biomassEOP])
        backupColors["biomass EOP"] = "limegreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Biomass EOP', markerfacecolor=backupColors["biomass EOP"], markersize=13))


    # Data: central gas CHP electric
    if "central gas CHP electric" in backupTech:
        gasCHPElectric = dataMatrix[[technology for technology in dataMatrix.index if "central gas CHP electric" in technology]]
        backup = pd.concat([backup, gasCHPElectric])
        backupColors["central gas CHP electric"] = "steelblue"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Gas CHP electric', markerfacecolor=backupColors["central gas CHP electric"], markersize=13))


    # Data: central biomass CHP electric
    if "central biomass CHP electric" in backupTech:
        biomassCHPElectric = dataMatrix[[technology for technology in dataMatrix.index if "central biomass CHP electric" in technology]]
        backup = pd.concat([backup, biomassCHPElectric])
        backupColors["central biomass CHP electric"] = "chocolate"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Biomass CHP electric', markerfacecolor=backupColors["central biomass CHP electric"], markersize=13))

    # Create array of country code and technology type
    backupNames = [item[0:2] for item in backup.index]
    backupTech = [item[3:] for item in backup.index]
    
    # Turn into a series with indexes ['bus', 'carrier']
    dataBackup = pd.Series(data=backup.values, index=[pd.Index(backupNames, name="bus"), pd.Index(backupTech, name="carrier")])
    
    # Scale data
    scale = scaleno / dataBackup.groupby(level=[0]).sum().max()
    dataBackup *= scale
    
    # Plot figure
    fig1 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataBackup, bus_colors=backupColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    plt.title("Installed electricity backup capacity [MW]", fontsize=16)     
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataBackup["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 9:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 12:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 15:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0))[0:-2] + " PW"), markerfacecolor="white", markersize=13))
    
    
    plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    
   
    plt.show(all)
    
    return (fig1)


#%% MapBackupHeat
def MapBackupHeat(network, filename, scaleno=5):
    """
    Parameters
    ----------
    network : PyPSA network object
        The network file (.h5) loaded into Python containing the network data
        
    filename : String
        The top-most title name of the plot (often the file name)
        
    scaleno : Int (scaleno 5)
        Scaling factor for figures

    Returns
    -------
    fig1 : matplotlib figure
        Figure containing installed backup heating capacity (MW)
        
    """
    
    # Figure size
    figureSize = 15.0
    
    # Adjust Norway, Sweden and Finland coordinates
    network.buses.x['NO'] = 6.69
    network.buses.y['NO'] = 60.6969
    
    network.buses.x['SE'] = 14.69
    network.buses.y['SE'] = 59.6969
    
    network.buses.x['FI'] = 24.69
    network.buses.y['FI'] = 62.69
    
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get locations for each country node
    countryLocations = pd.DataFrame(data = [network.buses.x[dataNames], network.buses.y[dataNames]])
    countryLocations = countryLocations.T
    
    # Set coordinates (x,y) for remaining carriers without coordinates
    for i in np.arange(network.buses.shape[0]):
        if network.buses.x[i] == 0 and network.buses.y[i] == 0:
            network.buses.x[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].x
            network.buses.y[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].y
    
    
    # Data matrix for backup technologies including efficiency
    dataMatrix = (network.links.p_nom_opt * network.links.efficiency.values).rename("Power")
    
    # Unique technologies included in backup
    backupTech = dataMatrix.index.str.slice(3).unique()
    
    # Create dataframe to store technologies used for backup + dict for colors + legend item
    backup = pd.Series()
    backupColors = {}
    legendElements = []
    
    # Data: Heat pump
    if any("heat pump" in x for x in backupTech):
        heatPump = dataMatrix[[technology for technology in dataMatrix.index if "heat pump" in technology]]
        heatPump.index = heatPump.index.str.slice(0,2)
        heatPump = heatPump.groupby(heatPump.index).sum()
        heatPump.index = (heatPump.index + " heat pump")
        backup = pd.concat([backup, heatPump])
        backupColors["heat pump"] = "coral"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Heat pump', markerfacecolor=backupColors["heat pump"], markersize=13))

    # Data: Resistive heater
    if any("resistive heater" in x for x in backupTech):
        resistiveHeater = dataMatrix[[technology for technology in dataMatrix.index if "resistive heater" in technology]]
        resistiveHeater.index = resistiveHeater.index.str.slice(0,2)
        resistiveHeater = resistiveHeater.groupby(resistiveHeater.index).sum()
        resistiveHeater.index = (resistiveHeater.index + " resistive heater")
        backup = pd.concat([backup, resistiveHeater])
        backupColors["resistive heater"] = "indigo"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Resistive heater', markerfacecolor=backupColors["resistive heater"], markersize=13))


    if any("gas boiler" in x for x in backupTech):
        gasBoiler = dataMatrix[[technology for technology in dataMatrix.index if "gas boiler" in technology]]
        gasBoiler.index = gasBoiler.index.str.slice(0,2)
        gasBoiler = gasBoiler.groupby(gasBoiler.index).sum()
        gasBoiler.index = (gasBoiler.index + " gas boiler")
        backup = pd.concat([backup, gasBoiler])
        backupColors["gas boiler"] = "chartreuse"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Gas boiler', markerfacecolor=backupColors["gas boiler"], markersize=13))


     
    if any("gas CHP heat" in x for x in backupTech):
        gasCHPHeat = dataMatrix[[technology for technology in dataMatrix.index if "gas CHP heat" in technology]]
        gasCHPHeat.index = gasCHPHeat.index.str.slice(0,2)
        gasCHPHeat = gasCHPHeat.groupby(gasCHPHeat.index).sum()
        gasCHPHeat.index = (gasCHPHeat.index + " gas CHP heat")
        backup = pd.concat([backup, gasCHPHeat])
        backupColors["gas CHP heat"] = "lightseagreen"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Gas CHP heat', markerfacecolor=backupColors["gas CHP heat"], markersize=13))

        
    if any("biomass CHP heat" in x for x in backupTech):
        biomassCHPHeat = dataMatrix[[technology for technology in dataMatrix.index if "biomass CHP heat" in technology]]
        biomassCHPHeat.index = biomassCHPHeat.index.str.slice(0,2)
        biomassCHPHeat = biomassCHPHeat.groupby(biomassCHPHeat.index).sum()
        biomassCHPHeat.index = (biomassCHPHeat.index + " biomass CHP heat")
        backup = pd.concat([backup, biomassCHPHeat])
        backupColors["biomass CHP heat"] = "navy"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Biomass CHP heat', markerfacecolor=backupColors["biomass CHP heat"], markersize=13))


    if any("biomass HOP" in x for x in backupTech):
        biomassHOP = dataMatrix[[technology for technology in dataMatrix.index if "biomass HOP" in technology]]
        biomassHOP.index = biomassHOP.index.str.slice(0,2)
        biomassHOP = biomassHOP.groupby(biomassHOP.index).sum()
        biomassHOP.index = (biomassHOP.index + " biomass HOP")
        backup = pd.concat([backup, biomassHOP])
        backupColors["biomass HOP"] = "rosybrown"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Biomass HOP', markerfacecolor=backupColors["biomass HOP"], markersize=13))
    
    
    if any("cooling pump" in x for x in backupTech):
        coolingPump = dataMatrix[[technology for technology in dataMatrix.index if "cooling pump" in technology]]
        coolingPump.index = coolingPump.index.str.slice(0,2)
        coolingPump = coolingPump.groupby(coolingPump.index).sum()
        coolingPump.index = (coolingPump.index + " cooling pump")
        backup = pd.concat([backup, coolingPump])
        backupColors["cooling pump"] = "silver"
        legendElements.append(Line2D([0], [0], marker='o', color='white', label='Cooling pump', markerfacecolor=backupColors["cooling pump"], markersize=13))



    # Create array of country code and technology type
    backupNames = [item[0:2] for item in backup.index]
    backupTech = [item[3:] for item in backup.index]
    
    # Turn into a series with indexes ['bus', 'carrier']
    dataBackup = pd.Series(data=backup.values, index=[pd.Index(backupNames, name="bus"), pd.Index(backupTech, name="carrier")])
    
    # Scale data
    scale = scaleno / dataBackup.groupby(level=[0]).sum().max()
    dataBackup *= scale
    
    # Plot figure
    fig1 = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataBackup, bus_colors=backupColors, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    plt.title("Installed heating backup capacity [MW]", fontsize=16)     
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataBackup["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MW"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 9:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GW"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 12:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TW"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 15:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0))[0:-2] + " PW"), markerfacecolor="white", markersize=13))
    
    
    plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    
   
    plt.show(all)
    
    return (fig1)


#%% MapCapacityElectricityEnergy

def MapCapacityElectricityEnergy(network, filename, scaleno=5):

    """

    Parameters
    ----------
    network : PyPSA network object
        The network file (.h5) loaded into Python containing the network data
        
    filename : String
        The top-most title name of the plot (often the file name)
        
    scaleno : Int (scaleno 5)
        Scaling factor for figures
    

    Returns
    -------
    fig : matplotlib figure
        Figure containing electrical energy from different technologies for each country [MWh].

    """
    
    # Figure size
    figureSize = 15.0
    
    # Adjust Norway, Sweden and Finland coordinates
    network.buses.x['NO'] = 6.69
    network.buses.y['NO'] = 60.6969
    
    network.buses.x['SE'] = 14.69
    network.buses.y['SE'] = 59.6969
    
    network.buses.x['FI'] = 24.69
    network.buses.y['FI'] = 62.69
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get locations for each country node
    countryLocations = pd.DataFrame(data = [network.buses.x[dataNames], network.buses.y[dataNames]])
    countryLocations = countryLocations.T
    
    # Set coordinates (x,y) for remaining carriers
    for i in np.arange(network.buses.shape[0]):
        if network.buses.x[i] == 0 and network.buses.y[i] == 0:
            network.buses.x[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].x
            network.buses.y[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].y
    
    # Data: Wind
    wind = network.generators_t.p[[technology for technology in network.generators_t.p.columns if "wind" in technology]]
    wind.columns = wind.columns.str.slice(0,2)
    wind = wind.groupby(wind.columns, axis=1).sum()
    wind.columns = (wind.columns + " wind")
    
    # Data: Solar PV
    solar = network.generators_t.p[dataNames + ' solar']
    solar = solar.groupby(solar.columns, axis=1).sum()
    
    # Data: ror
    ror = network.generators_t.p[[technology for technology in network.generators_t.p.columns if "ror" in technology]]
    ror.columns = ror.columns.str.slice(0,2)
    ror = ror.groupby(ror.columns, axis=1).sum()
    ror.columns = (ror.columns + " ror")
    
    # Data: Hydro
    hydro = network.storage_units_t.p[[technology for technology in network.storage_units_t.p.columns if "hydro" in technology]]
    hydro.columns = hydro.columns.str.slice(0,2)
    hydro = hydro.groupby(hydro.columns, axis=1).sum()
    hydro.columns = (hydro.columns + " hydro")
    
    # Data: OCGT (gas)
    OCGT = np.abs(network.links_t.p1[[technology for technology in network.links_t.p1.columns if "OCGT" in technology]])
    OCGT.columns = OCGT.columns.str.slice(0,2)
    OCGT = OCGT.groupby(OCGT.columns, axis=1).sum()
    OCGT.columns = (OCGT.columns + " OCGT")
    
    # Check if file contains "CHP Electric" (not present in elec_only files)
    if (len([check for check in network.links_t.p1.columns.str.slice(3).unique() if "CHP" in check]) > 0):
        # Data: CHP electricity
        CHPElec = np.abs(network.links_t.p1[[technology for technology in network.links_t.p1.columns if "CHP electric" in technology]])
        CHPElec.columns = CHPElec.columns.str.slice(0,2)
        CHPElec = CHPElec.groupby(CHPElec.columns, axis=1).sum()
        CHPElec.columns = (CHPElec.columns + " CHP electric")
        
        # Combine all the heating technologies, sum them and renme them to "energy"
        elec = pd.concat([wind, solar, ror, hydro, OCGT, CHPElec], axis=1).sum().rename("energy")
        
        # Create array of country code and technology type
        elecNames = [item[0:2] for item in elec.index]
        elecTech = [item[3:] for item in elec.index]
        
        # Turn into a series with indexes ['bus', 'carrier']
        dataElec = pd.Series(data=elec.values, index=[pd.Index(elecNames, name="bus"), pd.Index(elecTech, name="carrier")])
        
        # Scale data
        scale = scaleno / dataElec.groupby(level=[0]).sum().max()
        dataElec *= scale
        
        # Plot figure
        fig = plt.figure(figsize=[figureSize, figureSize], dpi=200)
        network.plot(bus_sizes=dataElec, bus_colors={'wind' : 'dodgerblue',
                                                     'solar' : 'gold',
                                                     'ror': 'orange',
                                                     'hydro' : 'mediumseagreen',
                                                     'OCGT' : 'firebrick',
                                                     'CHP electric' : 'darkslategray'}, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
        plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
        plt.title("Electrical energy from different technologies [MWh]", fontsize=16)
        legendElements = [Line2D([0], [0], marker='o', color='white', label='Wind', markerfacecolor='dodgerblue', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='Solar PV', markerfacecolor='gold', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='ror', markerfacecolor='orange', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='Hydro', markerfacecolor='mediumseagreen', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='OCGT', markerfacecolor='firebrick', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='CHP electric', markerfacecolor='darkslategray', markersize=13)]              
        
        # Create label for expressing relative size of Germany:
        DESize = round((dataElec["DE"]/scale).sum(),0)
        if len(str(round(DESize,0))) < 6:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MWh"), markerfacecolor="white", markersize=13))
        
        elif len(str(round(DESize,0))) < 9:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GWh"), markerfacecolor="white", markersize=13))
        
        elif len(str(round(DESize,0))) < 12:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TWh"), markerfacecolor="white", markersize=13))
        
        elif len(str(round(DESize,0))) < 15:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0))[0:-2] + " PWh"), markerfacecolor="white", markersize=13))
    
        
        
        plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
            
    else:
        # Combine all the heating technologies, sum them and renme them to "energy"
        elec = pd.concat([wind, solar, ror, hydro, OCGT], axis=1).sum().rename("energy")
        
        # Create array of country code and technology type
        elecNames = [item[0:2] for item in elec.index]
        elecTech = [item[3:] for item in elec.index]
        
        # Turn into a series with indexes ['bus', 'carrier']
        dataElec = pd.Series(data=elec.values, index=[pd.Index(elecNames, name="bus"), pd.Index(elecTech, name="carrier")])
        
        # Scale data
        scale = scaleno / dataElec.groupby(level=[0]).sum().max()
        dataElec *= scale
        
        # Plot figure
        fig = plt.figure(figsize=[figureSize, figureSize], dpi=200)
        network.plot(bus_sizes=dataElec, bus_colors={'wind' : 'dodgerblue',
                                                     'solar' : 'gold',
                                                     'ror': 'orange',
                                                     'hydro' : 'mediumseagreen',
                                                     'OCGT' : 'firebrick'}, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
        plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
        plt.title("Electrical energy from different technologies [MWh]", fontsize=16)
        legendElements = [Line2D([0], [0], marker='o', color='white', label='Wind', markerfacecolor='dodgerblue', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='Solar PV', markerfacecolor='gold', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='ror', markerfacecolor='orange', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='Hydro', markerfacecolor='mediumseagreen', markersize=13),
                          Line2D([0], [0], marker='o', color='white', label='OCGT', markerfacecolor='firebrick', markersize=13)]              
        
        # Create label for expressing relative size of Germany:
        DESize = round((dataElec["DE"]/scale).sum(),0)
        if len(str(round(DESize,0))) < 6:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MWh"), markerfacecolor="white", markersize=13))
        
        elif len(str(round(DESize,0))) < 9:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GWh"), markerfacecolor="white", markersize=13))
        
        elif len(str(round(DESize,0))) < 12:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TWh"), markerfacecolor="white", markersize=13))
        
        elif len(str(round(DESize,0))) < 15:
            legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0))[0:-2] + " PWh"), markerfacecolor="white", markersize=13))
    
        
        plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=1)
        
    return fig

#%% MapCapacityHeatEnergy

def MapCapacityHeatEnergy(network, filename, scaleno=5):

    """

    Parameters
    ----------
    network : PyPSA network object
        The network file (.h5) loaded into Python containing the network data
        
    filename : String
        The top-most title name of the plot (often the file name)
        
    scaleno : Int (scaleno 5)
        Scaling factor for figures
    

    Returns
    -------
    fig : matplotlib figure
        Figure containing heating energy from different technologies for each country [MWh].

    """
    
    # Figure size
    figureSize = 15.0
    
    # Adjust Norway, Sweden and Finland coordinates
    network.buses.x['NO'] = 6.69
    network.buses.y['NO'] = 60.6969
    
    network.buses.x['SE'] = 14.69
    network.buses.y['SE'] = 59.6969
    
    network.buses.x['FI'] = 24.69
    network.buses.y['FI'] = 62.69
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get locations for each country node
    countryLocations = pd.DataFrame(data = [network.buses.x[dataNames], network.buses.y[dataNames]])
    countryLocations = countryLocations.T
    
    # Set coordinates (x,y) for remaining carriers
    for i in np.arange(network.buses.shape[0]):
        if network.buses.x[i] == 0 and network.buses.y[i] == 0:
            network.buses.x[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].x
            network.buses.y[network.buses.index[i]] = countryLocations.loc[network.buses.index[i][0:2]].y
    
    # Data: Solar collectors
    solarCollector = network.generators_t.p[[technology for technology in network.generators_t.p.columns if "collector" in technology]]
    solarCollector.columns = solarCollector.columns.str.slice(0,2)
    solarCollector = solarCollector.groupby(solarCollector.columns, axis=1).sum()
    solarCollector.columns = (solarCollector.columns + " solar collector")
    
    # Data: Heat pumps
    heatPump = np.abs(network.links_t.p1[[technology for technology in network.links_t.p1.columns if "heat pump" in technology]])
    heatPump.columns = heatPump.columns.str.slice(0,2)
    heatPump = heatPump.groupby(heatPump.columns, axis=1).sum()
    heatPump.columns = (heatPump.columns + " heat pump")
    
    # Data: Resistive heaters
    resistiveHeat = np.abs(network.links_t.p1[[technology for technology in network.links_t.p1.columns if "resistive" in technology]])
    resistiveHeat.columns = resistiveHeat.columns.str.slice(0,2)
    resistiveHeat = resistiveHeat.groupby(resistiveHeat.columns, axis=1).sum()
    resistiveHeat.columns = (resistiveHeat.columns + " resistive heat")
    
    # Data: CHP heating plants
    CHPHeat = np.abs(network.links_t.p1[[technology for technology in network.links_t.p1.columns if "CHP heat" in technology]])
    CHPHeat.columns = CHPHeat.columns.str.slice(0,2)
    CHPHeat = CHPHeat.groupby(CHPHeat.columns, axis=1).sum()
    CHPHeat.columns = (CHPHeat.columns + " CHP heat")
    
    # Data: Gas boilers
    gasBoilerHeat = np.abs(network.links_t.p1[[technology for technology in network.links_t.p1.columns if "boiler" in technology]])
    gasBoilerHeat.columns = gasBoilerHeat.columns.str.slice(0,2)
    gasBoilerHeat = gasBoilerHeat.groupby(gasBoilerHeat.columns, axis=1).sum()
    gasBoilerHeat.columns = (gasBoilerHeat.columns + " gas boiler")
    
    # Combine all the heating technologies, sum them and renme them to "energy"
    heat = pd.concat([solarCollector, heatPump, resistiveHeat, CHPHeat, gasBoilerHeat], axis=1).sum().rename("energy")
    
    # Create array of country code and technology type
    heatNames = [item[0:2] for item in heat.index]
    heatTech = [item[3:] for item in heat.index]
    
    # Turn into a series with indexes ['bus', 'carrier']
    dataHeat = pd.Series(data=heat.values, index=[pd.Index(heatNames, name="bus"), pd.Index(heatTech, name="carrier")])
    
    # Scale data
    scale = scaleno / dataHeat.groupby(level=[0]).sum().max()
    dataHeat *= scale
    
    # Plot figure
    fig = plt.figure(figsize=[figureSize, figureSize], dpi=200)
    network.plot(bus_sizes=dataHeat, bus_colors={'solar collector' : 'darkorange',
                                                 'heat pump' : 'powderblue',
                                                 'resistive heat' : 'crimson',
                                                 'CHP heat' : 'khaki',
                                                 'gas boiler' : 'maroon'}, projection=(ccrs.PlateCarree()), color_geomap=({"ocean" : "azure", "land" : "whitesmoke"}))
    plt.suptitle(filename, fontsize=20, x=0.52, y=0.81)
    plt.title("Heating energy from different technologies [MWh]", fontsize=16)
    legendElements = [Line2D([0], [0], marker='o', color='white', label='Solar collector', markerfacecolor='darkorange', markersize=13),
                       Line2D([0], [0], marker='o', color='white', label='Heat pump', markerfacecolor='powderblue', markersize=13),
                       Line2D([0], [0], marker='o', color='white', label='Resistive heat', markerfacecolor='crimson', markersize=13),
                       Line2D([0], [0], marker='o', color='white', label='CHP heat', markerfacecolor='khaki', markersize=13),
                       Line2D([0], [0], marker='o', color='white', label='Gas boiler', markerfacecolor='maroon', markersize=13)]              
    
    # Create label for expressing relative size of Germany:
    DESize = round((dataHeat["DE"]/scale).sum(),0)
    if len(str(round(DESize,0))) < 6:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize,0))[0:-2] + " MWh"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 9:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e3,0))[0:-2] + " GWh"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 12:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e6,1)) + " TWh"), markerfacecolor="white", markersize=13))
    
    elif len(str(round(DESize,0))) < 15:
        legendElements.append(Line2D([0], [0], marker='o', color='white', label=("\nSize of Germany: " + str(round(DESize/1e9,0))[0:-2] + " PWh"), markerfacecolor="white", markersize=13))

    
    plt.legend(handles=legendElements, fontsize=14, loc='upper left', bbox_to_anchor = (0.0,1.0), framealpha=0.8)
    
    return fig

#%% CovNameGenerator

def CovNameGenerator(conNames):
    """
    

    Parameters
    ----------
    conNames : array
        array of strings with names of either generaters, load or reponsetypes

    Returns
    -------
    con : array
        output an larger array than original where the the covariance versions of the original are attached

    """
    
    amount = int(len((conNames)*(len(conNames)+1))/2)
    
    con = [""]*amount
    
    for j in range( amount ):
        
        if j < len(conNames):
            con[j] = conNames[j]
        if j < len(conNames)-1:
            con[j+len(conNames)] = conNames[0] + "/ \n" + conNames[j+1]
        elif j < len(conNames)-1 + len(conNames)-2:
            con[j+len(conNames)] = conNames[1] + "/ \n" + conNames[j-len(conNames)+1+2]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3:
            con[j+len(conNames)] = conNames[2] + "/ \n" + conNames[j-len(conNames)*2 + 1+2+3]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4:
            con[j+len(conNames)] = conNames[3] + "/ \n" + conNames[j-len(conNames)*3 + 1+2+3+4]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5:
            con[j+len(conNames)] = conNames[4] + "/ \n" + conNames[j-len(conNames)*4 + 1+2+3+4+5]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6:
            con[j+len(conNames)] = conNames[5] + "/ \n" + conNames[j-len(conNames)*5 + 1+2+3+4+5+6]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7:
            con[j+len(conNames)] = conNames[6] + "/ \n" + conNames[j-len(conNames)*6 + 1+2+3+4+5+6+7]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8:
            con[j+len(conNames)] = conNames[7] + "/ \n" + conNames[j-len(conNames)*7 + 1+2+3+4+5+6+7+8]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8 + len(conNames)-9:
            con[j+len(conNames)] = conNames[8] + "/ \n" + conNames[j-len(conNames)*8 + 1+2+3+4+5+6+7+8+9]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8 + len(conNames)-9 + len(conNames)-10:
            con[j+len(conNames)] = conNames[9] + "/ \n" + conNames[j-len(conNames)*9 + 1+2+3+4+5+6+7+8+9+10]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8 + len(conNames)-9 + len(conNames)-10 + len(conNames)-11:
            con[j+len(conNames)] = conNames[10] + "/ \n" + conNames[j-len(conNames)*10 + 1+2+3+4+5+6+7+8+9+10+11]
        else:
            assert True, "something is wrong"
            
    return con

#%% ConValueGenerator

def ConValueGenerator(norm_const, dirc, eigen_vectors):
    """
    

    Parameters
    ----------
    norm_const : float64
        normilazation constant from PCA
    dirc : dirt
        a dictionary with the different generators in
    eigen_vectors : Array of float64
        eigen vectors from PCA

    Returns
    -------
    lambdaCollected : DataFrame
        Colelcted lambda values for each component

    """
    
    dictTypes = list(dirc.keys())
    
    moveToEnd = []
    types = []
    
    for j in range(len(dictTypes)):
        if dictTypes[j].split()[0] == "Load":
            moveToEnd.append(dictTypes[j])
        else:
            types.append(dictTypes[j])
  
    types.extend(moveToEnd)    
  
    for j in types:
    
        # Mean and centered
        centered = MeanAndCenter(dirc[j])
        
        # Projection
        projection = np.dot(centered,eigen_vectors)
    
        dirc[j] = projection  
    
    conNames = CovNameGenerator(types)
    
    amount = len(conNames)
    
    lambdaCollected =  pd.DataFrame( columns = conNames)
    
    for j in range(amount):
        if j < len(types):
            lambdaCollected[conNames[j]] = (norm_const**2)*(np.mean((dirc[types[j]]**2),axis=0))
        elif j < len(types)*2 -1:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[0]]*dirc[types[j+0-(len(types)*1-1)]]),axis=0))
        elif j < len(types)*3 -3:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[1]]*dirc[types[j+1-(len(types)*2-2)]]),axis=0))
        elif j < len(types)*4 -6:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[2]]*dirc[types[j+3-(len(types)*3-3)]]),axis=0))
        elif j < len(types)*5 -10:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[3]]*dirc[types[j+6-(len(types)*4-4)]]),axis=0))  
        elif j < len(types)*6 -15:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[4]]*dirc[types[j+10-(len(types)*5-5)]]),axis=0))  
        elif j < len(types)*7 -21:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[5]]*dirc[types[j+15-(len(types)*6-6)]]),axis=0))  
        elif j < len(types)*8 -28:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[6]]*dirc[types[j+21-(len(types)*7-7)]]),axis=0))  
        elif j < len(types)*9 -36:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[7]]*dirc[types[j+28-(len(types)*8-8)]]),axis=0))  
        elif j < len(types)*10 -45:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[8]]*dirc[types[j+36-(len(types)*9-9)]]),axis=0))  
        elif j < len(types)*11 -55:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[9]]*dirc[types[j+45-(len(types)*10-10)]]),axis=0))     
            
        else:
            assert True, "something is wrong"
    
    return lambdaCollected

#%% CovNames

def CovNames(dirc1, dirc2):
    
    # Convert to list
    covNames1 = list(dirc1.keys())
    covNames2 = list(dirc2.keys())
    
    # Determin size
    covSize = len(covNames1) * len(covNames2)
    
    covNames = pd.DataFrame(np.empty((covSize, 3), dtype = np.str), columns = ["Contribution","Response","Covariance"])
    
    # index counter
    i = 0
    
    for j in covNames1: # List of dirc1 names
        for k in covNames2: # List of dirc2 names
            
            # Save names
            covNames["Contribution"][i] = j
            covNames["Response"][i] = k
            covNames["Covariance"][i] = j + "/\n" + k
    
            # add one to index counter
            i += 1

    return covNames

#%% CovValueGenerator

def CovValueGenerator(dirc1, dirc2, meanAndCenter,normConst, eigenVectors):

    # Generate name list
    covNames = CovNames(dirc1, dirc2)
    
    # Empty dataframe
    covMatrix = pd.DataFrame(np.zeros([len(covNames),30]), index = covNames["Covariance"])
    
    if meanAndCenter == False: # If values have not been meanséd and cented before
        # Mean, center and projection of 1. dirc
        for j in list(dirc1.keys()):
        
            # Mean and centered
            centered = MeanAndCenter(dirc1[j])
            
            # Projection
            projection = np.dot(centered,eigenVectors)
        
            # Save value agian
            dirc1[j] = projection
        
        # Mean, center and projection of 2. dirc
        for j in list(dirc2.keys()):
        
            # Mean and centered
            centered = MeanAndCenter(dirc2[j])
            
            # Projection
            projection = np.dot(centered,eigenVectors)
        
            # Save value agian
            dirc2[j] = projection
            
    # Main calulation
    for i in range(len(covNames)):
        
        # Calculate covariance
        covMatrix.loc[covNames["Covariance"][i]] = - (normConst**2)*(np.mean((dirc1[covNames["Contribution"][i]]*dirc2[covNames["Response"][i]]),axis=0))

    return covMatrix

#%% ConPlot
    
def ConPlot(eigen_values,lambdaCollected, PC_NO, depth, suptitle='none',dpi=100):
    """
    

    Parameters
    ----------
    eigen_values : Array of float64
        DESCRIPTION.
    lambdaCollected : DataFrame
        Colelcted lambda values for each component
    PC_NO : float64
        Number of PC that is plotted
    depth : float64
        how many contribution should be plotted
    suptitle : TYPE, optional
        string of substitle
    dpi : TYPE, optional
        quality. The default is 100.

    Returns
    -------
    fig : plt.figure
        figure

    """
    
    # incase the depth is higher than the contribution
    if depth > len(lambdaCollected.T):
        depth = len(lambdaCollected.T)
    
    if depth < 10:
        size = 10
    else:
        size = depth
    
    PC_NO -= 1
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # Plot figure
    fig = plt.figure(figsize=[size/1.5,size/3],dpi=dpi)
    
    # Find which absolute values are the highest 
    highest = abs(lambdaCollected.iloc[PC_NO,:]).sort_values(ascending=False)[0:depth]
    highest = lambdaCollected[highest.index].iloc[PC_NO].sort_values(ascending=False)[0:depth] # Sort by value
    
    # Create a zero matrix for the percentage values
    percent = np.zeros([depth])
    
    # Loop to calculate and plot the different values
    for j in range(depth):
        # absolute percentage
        percent[j] = lambdaCollected[highest.index[j]][PC_NO]/eigen_values[PC_NO]*100
        
        # Plot
        plt.bar(highest.index[j],percent[j],color=color[PC_NO]) 
    
    # General plot settings    
    plt.xticks(rotation='vertical')
    plt.ylabel('Influance [%]')    
    plt.grid(axis='y',alpha=0.5)
    plt.ylim([-160,160])
    plt.title('$\lambda_{'+str(PC_NO+1)+'}$: '+str(round(lambdaCollected.iloc[PC_NO,:].sum()*100,2))+'%')
    
    # Inserting percentage numbers on the bars
    for k in range(depth):
        if percent[k] > 0:
            if percent[k] > 150:
                v = 135
            else:
                v = percent[k] + 3
        else:
            if percent[k] < -150:
                v = -150
            else:
                v = percent[k] - 12
        plt.text(x=k,y=v,s=str(round(float(percent[k]),1))+'%',ha='center',size='small')
    
    if suptitle != 'none':
        plt.suptitle(suptitle, fontsize=20,x=.5,y=1.05)
    

        
    plt.show()
    
    return fig

#%% MultiConPlot

def MultiConPlot(eigen_values,lambdaCollected, PC_NO, depth, suptitle='none',dpi=100):
    """
    

    Parameters
    ----------
    eigen_values : Array of float64
        DESCRIPTION.
    lambdaCollected : DataFrame
        Colelcted lambda values for each component
    PC_NO : float64
        Number of PC that is plotted
    depth : float64
        how many contribution should be plotted
    suptitle : TYPE, optional
        string of substitle
    dpi : TYPE, optional
        quality. The default is 100.

    Returns
    -------
    fig : plt.figure
        figure

    """
    
    # Subtrack with 1
    PC_NO = [x - 1 for x in PC_NO]
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # Plot figure
    fig = plt.figure(figsize=[depth*len(PC_NO)/1.5,depth*len(PC_NO)/3],dpi=dpi)
    
    # Find the highest values
    highestCollected = []
    
    for j in PC_NO:
        highest = abs(lambdaCollected.iloc[j,:]).sort_values(ascending=False)[0:depth]
        highest = lambdaCollected[highest.index].iloc[j].sort_values(ascending=False)[0:depth] # Sort by value
        
        highestCollected.append(highest)
    
    # Counter for x axis
    counter = 0
    
    for j in PC_NO:
        # Create a zero matrix for the percentage values
        percent = np.zeros([depth])
        
        # Loop to calculate and plot the different values
        for i in range(depth):
            # absolute percentage
            percent[i] = lambdaCollected[highestCollected[j].index[i]][j]/eigen_values[j]*100
            
            # Plot without label
            if i == 0:
                plt.bar(counter,percent[i],color=color[j],label = '$\lambda_{'+str(j+1)+'}$: '+str(round(lambdaCollected.iloc[PC_NO[j],:].sum()*100,2))+'%') 
            else:
                plt.bar(counter,percent[i],color=color[j]) 
            
            # Insert text into bar
            if percent[i] > 0:
                if percent[i] > 150:
                    v = 135
                else:
                    v = percent[i] + 3
            else:
                if percent[i] < -110:
                    v = -110
                else:
                    v = percent[i] - 12
            plt.text(x=counter,y=v,s=str(round(float(percent[i]),1))+'%',ha='center',size='small')
            
            # Count up
            counter += 1
    
    # x axis label
    xLabel = []
    for j in PC_NO:
        xLabel += list(highestCollected[j].index)
    
    # General plot settings    
    plt.xticks(np.arange(0,depth*len(PC_NO)),xLabel,rotation='vertical')
    plt.ylabel('Influance [%]')    
    plt.grid(axis='y',alpha=0.5)
    plt.ylim([-160,160])
    #plt.title('$\lambda_{'+str(PC_NO+1)+'}$: '+str(round(lambdaCollected.iloc[PC_NO,:].sum()*100,2))+'%')
    
    # legend
    plt.legend(loc = "upper right")
    
    # Title
    if suptitle != 'none':
        plt.suptitle(suptitle, fontsize=20,x=.5,y=1.05)

    # Show plot
    plt.show()
    
    return fig

#%% CombConPlot

def CombConPlot(eigen_values, lambdaContribution, lambdaResponse, lambdaCovariance, PC_NO, depth = 8, suptitle='none', dpi=100):

    # Setup PC counter
    PC_NO -= 1
    
    # Setup subplot title
    subtitle = ["Contribution","Response","Covariance"]
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # letters
    letter = ["(a)","(b)","(c)"]
    
    # Plot figure
    fig = plt.figure(figsize=[12,4.5],dpi=dpi)
    
    for i in range(3):
        
        if i == 0:
            lambdaCollected = lambdaContribution
        elif i == 1:
            lambdaCollected = lambdaResponse
        elif i == 2:
            lambdaCollected = lambdaCovariance
        else:
            assert False, "Something went wrong"
    
        # incase the depth is higher than the contribution
        if depth > len(lambdaCollected.T):
            depth = len(lambdaCollected.T)
    
        # subplot
        plt.subplot(1,3,i+1)    
    
        # Find which absolute values are the highest 
        highest = abs(lambdaCollected.iloc[PC_NO,:]).sort_values(ascending=False)[0:depth]
        highest = lambdaCollected[highest.index].iloc[PC_NO].sort_values(ascending=False)[0:depth] # Sort by value
        
        # Create a zero matrix for the percentage values
        percent = np.zeros([depth])
        
        # Loop to calculate and plot the different values
        for j in range(depth):
            # absolute percentage
            percent[j] = lambdaCollected[highest.index[j]][PC_NO]/eigen_values[PC_NO]*100
            
            # Plot
            plt.bar(highest.index[j],percent[j],color=color[PC_NO]) 
        
        # General plot settings    
        plt.xticks(rotation=90, fontsize=14)
         
        plt.grid(axis='y',alpha=0.5)
        plt.ylim([-110,160])
        plt.title(subtitle[i], fontsize=16)#,font="Lucida Sans Unicode")
        
        # Inserting percentage numbers on the bars
        for k in range(depth):
            if percent[k] > 0:
                if percent[k] > 100:
                    v = -80
                elif percent[k] > 10:
                    v = -70
                else:
                    v = -55
            else:
                    v = 8
            plt.text(x=k,y=v,s=str(round(float(percent[k]),1))+'%',ha='center',rotation='vertical', fontsize=12)
    
        # Eigenvalue as legend
        if i == 0:
            plt.ylabel('Influance [%]',labelpad=-5, fontsize=14)   
        elif i == 2:
                plt.legend(['$\lambda_{'+str(PC_NO+1)+'}$ = '+str(round(lambdaCollected.iloc[PC_NO,:].sum()*100,1))+'%'], loc = "upper right", fontsize=14)
                plt.tick_params('y', labelleft=False)
        else:
            plt.tick_params('y', labelleft=False)
        
    
    # subplot setup
    plt.tight_layout()    
    
    # letter
    if depth == 6:
        plt.text(-14.5,170,"(a)", fontsize=14, fontweight='bold')
        plt.text(-7.7,170,"(b)", fontsize=14, fontweight='bold')
        plt.text(-1,170,"(c)", fontsize=14, fontweight='bold')
    elif depth == 5:
        plt.text(-12.15,170,"(a)", fontsize=14, fontweight='bold')
        plt.text(-6.6,170,"(b)", fontsize=14, fontweight='bold')
        plt.text(-1,170,"(c)", fontsize=14, fontweight='bold')
    
    # Title
    if suptitle != 'none':
        plt.suptitle(suptitle, fontsize=20,x=.5,y=1.05)
        
    # Show plot
    plt.show()
    
    return fig

#%% MultiCombConPlot

def MultiCombConPlot(eigen_values, lambdaContribution, lambdaResponse, lambdaCovariance, PC_NO, depth, suptitle='none', dpi=200):

    # Subtrack with 1
    PC_NO = [x - 1 for x in PC_NO]
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # Setup subplot title
    subtitle = ["Contribution","Response","Covariance"]
    
    # Plot figure
    fig = plt.figure(figsize=[15,3],dpi=dpi)
    
    # Which plot
    for k in range(3):
        
        if k == 0:
            lambdaCollected = lambdaContribution
        elif k == 1:
            lambdaCollected = lambdaResponse
        elif k == 2:
            lambdaCollected = lambdaCovariance
        else:
            assert False, "Something went wrong"
    
        # incase the depth is higher than the contribution
        if depth > len(lambdaCollected.T):
            depth = len(lambdaCollected.T)
            
        # subplot
        plt.subplot(1,3,k+1)  
    
        # Find the highest values
        highestCollected = []
        
        for j in PC_NO:
            highest = abs(lambdaCollected.iloc[j,:]).sort_values(ascending=False)[0:depth]
            highest = lambdaCollected[highest.index].iloc[j].sort_values(ascending=False)[0:depth] # Sort by value
            
            highestCollected.append(highest)
            
        # Counter for x axis
        counter = 0
        
        for j in PC_NO:
            # Create a zero matrix for the percentage values
            percent = np.zeros([depth])
            
            # Loop to calculate and plot the different values
            for i in range(depth):
                # absolute percentage
                percent[i] = lambdaCollected[highestCollected[j].index[i]][j]/eigen_values[j]*100
                
                # Plot without label
                if i == 0:
                    plt.bar(counter,percent[i],color=color[j],label = '$\lambda_{'+str(j+1)+'}$: '+str(round(lambdaCollected.iloc[PC_NO[j],:].sum()*100,2))+'%') 
                else:
                    plt.bar(counter,percent[i],color=color[j]) 
                
                # Insert text into bar
                if percent[i] > 0:
                    if percent[i] >= 100:
                        v = -35
                    else:
                        v = -31
                else:
                        v = 5
                plt.text(x=counter,y=v,s=str(round(float(percent[i]),2))+'%',ha='center',size='small',rotation='vertical')
                
                # Count up
                counter += 1
    
        # x axis label
        xLabel = []
        for j in PC_NO:
            xLabel += list(highestCollected[j].index)
    
        # General plot settings    
        plt.xticks(np.arange(0,depth*len(PC_NO)),xLabel,rotation='vertical')
        plt.ylabel('Influance [%]')    
        plt.grid(axis='y',alpha=0.5)
        plt.ylim([-50,120])
        plt.title(subtitle[k])
    
        # Legend
        if k == 0:
            plt.legend(loc = "upper right")
    
    # Change space horizontal
    plt.subplots_adjust(hspace=0.5)
    
    # Title
    if suptitle != 'none':
        plt.suptitle(suptitle, fontsize=20,x=.5,y=1.05)
    
    # Show plot
    plt.show()
    
    return fig

#%% Coherence

def Coherence(mismatch, electricityNodalPrices, decimals=3):
    """
    Parameters
    ----------
    mismatch : matrix [8760 x 30] floating point
        mismatch between generation and load for each country [MWh].
        
    electricityNodalPrices : matrix [8760 x 30] floating point
        prices of electricity for each country [€/MWh]

    Returns
    -------
    c1 : matrix [30 x 30]
        Coherence method 1 - orthogonality between eigen vectors.
        
    c2 : matrix [30 x 30]
        Coherence method 2 - weighed (eigen values) orthogonality between eigen vectors.
        
    c3 : matrix [30 x 30]
        Coherence method 3 - mean time-varying (principle component amplitudes) orthognoality between eigen vectors.

    """
    
    # Stops if mismatch have the wrong format
    #assert mismatch.shape == (8760,30), "'mismatch' matrix has wrong dimensions. It must be [8760 x 30] !"
    
    # Stops if electricity prices have the wrong format
    #assert electricityNodalPrices.shape == (8760,30), "'electricityNodalPrices' matrix has wrong dimensions. It must be [8760 x 30] !"
    
    # PCA for mismatch
    eigenValuesMismatch, eigenVectorsMismatch, varianceExplainedMismatch, normConstMismatch, akMismatch = PCA(mismatch)
    
    # PCA for electricity nodal prices
    eigenValuesPrice, eigenVectorsPrice, varianceExplainedPrice, normConstPrice, akPrice = PCA(electricityNodalPrices)
    
    matrixSize = np.zeros(eigenVectorsMismatch.shape)
    c1 = np.zeros(matrixSize.shape)
    c2 = np.zeros(matrixSize.shape)
    c3 = np.zeros(matrixSize.shape)
    
    for n in range(matrixSize.shape[0]):
        for k in range(matrixSize.shape[1]):
            
            # Eigen vectors
            p_k1 = eigenVectorsMismatch[:,n]
            p_k2 = eigenVectorsPrice[:,k]
            
            # Principal component weights (because some of the smalles values can become negative, it is chosen to use the aboslute value)
            lambda1 = np.abs(eigenValuesMismatch[n])
            lambda2 = np.abs(eigenValuesPrice[k])
    
            # Time amplitude
            a_k1 = akMismatch[:,n]
            a_k2 = akPrice[:,k]
            
            # Coherence calculations
            c1[n,k] = np.abs(np.dot(p_k1, p_k2))
            c2[n,k] = np.sqrt(lambda1 * lambda2) * np.abs(np.dot(p_k1, p_k2))
            # OLD METHOD! c3[n,k] = (a_k1 * a_k2).mean() * np.abs(np.dot(p_k1, p_k2))
            c3[n,k] = ( (a_k1 * a_k2).mean() / np.sqrt(lambda1 * lambda2) )
            
    # Round off coherence matrices
    c1 = np.around(c1, decimals)
    c2 = np.around(c2, decimals)
    c3 = np.around(c3, decimals)
    
    return (c1, c2, c3)

#%% Contribution

def Contribution(network,contributionType):

    # Save index with only countries
    country_column = network.loads.index[:30]
    
    # For contribution of the electric sector
    if contributionType == "elec":
        # Load
        load     = network.loads_t.p_set.filter(items=country_column)
        # Generators
        wind     = network.generators_t.p.filter(regex="wind").groupby(network.generators.bus,axis=1).sum()
        solar    = network.generators_t.p.filter(items=country_column+" solar").groupby(network.generators.bus,axis=1).sum()
        ror = network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]]
        ror = pd.DataFrame(np.zeros([8760,30]), index=network.loads_t.p.index, columns=(network.buses.index.str.slice(0,2).unique() + ' ror')).add(ror, fill_value=0)
        
        # Collect terms
        collectedContribution = {"Load Electric":               - load, 
                                 "Wind":                        + wind,
                                 "Solar PV":                    + solar,
                                 "RoR":                         + ror
                                 }
     
    # For contribution of the heating sector    
    elif contributionType == "heat":
        # Load
        load        = pd.DataFrame(data=network.loads_t.p_set.filter(items=country_column+" heat").values,columns = country_column, index = network.generators_t.p.index)
        loadUrban = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "urban heat" in country]]
        loadUrban = pd.DataFrame(np.zeros([8760,30]), index=network.loads_t.p_set.index, columns=(network.buses.index.str.slice(0,2).unique() + ' urban heat')).add(loadUrban, fill_value=0)
        #loadUrban   = pd.DataFrame(data=network.loads_t.p_set.filter(items=country_column+" urban heat").values,columns = country_column, index = network.generators_t.p.index)
        
        # Check if brownfield network
        isBrownfield = len([x for x in network.links_t.p0.columns if "nuclear" in x]) > 0
        if isBrownfield == False: 
        
            # Generators
            SolarCol = pd.DataFrame(data=network.generators_t.p.filter(items=country_column+" solar thermal collector").groupby(network.generators.bus,axis=1).sum().values,columns = country_column, index = network.generators_t.p.index)
            centralUrbanSolarCol = pd.DataFrame(data=network.generators_t.p.filter(regex=" solar").groupby(network.generators.bus,axis=1).sum().filter(regex=" urban").values,columns = country_column, index = network.generators_t.p.index)
            
            # Collect terms
            collectedContribution = {"Load Heat":                   - load,
                                     "Load Urban Heat":             - loadUrban,
                                     "Solar Collector":             + SolarCol,
                                     "Central-Urban\nSolar Collector": + centralUrbanSolarCol
                                     }
            
        elif isBrownfield == True: 
        
            loadCooling = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "cooling" in country]]
            loadCooling = pd.DataFrame(np.zeros([8760,30]), index=network.loads_t.p_set.index, columns=(network.buses.index.str.slice(0,2).unique() + ' cooling')).add(loadCooling, fill_value=0)    
        
            # Collect terms
            collectedContribution = {"Load Heat":                   - load,
                                     "Load Urban Heat":             - loadUrban,
                                     "Load Cooling":                - loadCooling
                                     }
            
    else: # Error code if wrong input
        assert False, "choose either elec or heat as a contributionType"
    
    return collectedContribution

#%% LinksEff
    
def LinksEff(network):
    """
    

    Parameters
    ----------
    network : PyPSA network type
        input network

    Returns
    -------
    linkseff : array
        an array of the types of links that have an efficency. Used to calculate the response values

    """
    linkseff = network.links # Save link data
    linkseff = linkseff.drop(linkseff.index[np.where(linkseff["bus1"].str.contains("H2"))]) # Delete bus1 = H2
    linkseff = linkseff.drop(linkseff.index[np.where(linkseff["bus1"].str.contains("battery"))]) # Delete bus1 = battery
    linkseff = linkseff.drop(linkseff.index[np.where(linkseff["bus1"].str.contains("water"))]) # Delete bus1 = water tanks
    linkseff = pd.DataFrame(data={"eff": linkseff.efficiency.values}, index=linkseff.index.str.slice(3))
    linkseff = linkseff[~linkseff.index.duplicated(keep='first')]
    linkseff = linkseff[:np.where(linkseff.index.str.len()==2)[0][0]]
    
    return linkseff


#%% ElecResponse

def ElecResponse(network, collectTerms=True):
    
    # Save index with only countries
    country_column = network.loads.index[:30]
    
    # Determin size of loads to know if the system includes heat and/or transport
    size = network.loads.index.size
    
    # Find efficency of links
    busEff = LinksEff(network)
    
    # Check if brownfield network
    isBrownfield = len([x for x in network.links_t.p0.columns if "nuclear" in x]) > 0
    if isBrownfield == True: 
        size = 0 # Makes use that the sector coupled links are correct
        
    # General Reponse
    H2                          = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    battery                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    PHS                         = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    importExport                = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    OCGT                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    hydro                       = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    if isBrownfield == False:
        # Heating related response:
        if size >= 90:
            groundHeatPump              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            centralUrbanHeatPump        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            ResistiveHeater             = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            centralUrbanResistiveHeater = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            CHPElec                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        
        # Transportation related response
        if size == 60 or size == 120:
            EVBattery                   = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    elif isBrownfield == True:
        
        # new types of backup
        CCGT                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        coal                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        lignite                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        nuclear                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        oil                         = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        biomassEOP                  = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        
        # Heat related
        groundHeatPump              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        centralUrbanHeatPump        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        ResistiveHeater             = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        centralUrbanResistiveHeater = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        coolingPump                 = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        gasCHPElec                  = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        biomassCHPElec              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    for country in country_column:
        # Storage
        H2[country] = (- network.links_t.p0.filter(regex=country).filter(regex="Electrolysis")
            + network.links_t.p0.filter(regex=country).filter(regex="H2 Fuel Cell").values
            * busEff.loc["H2 Fuel Cell"].values[0]
            ).values 
        
        battery[country] = (- network.links_t.p0.filter(regex=country).filter(regex="battery charger")
                  + network.links_t.p0.filter(regex=country).filter(regex="battery discharger").values
                  * busEff.loc["battery discharger"].values[0]
                  ).values
        
        if (+ network.storage_units_t.p.filter(regex=country).filter(regex="PHS")).size > 0:
            PHS[country] = (+ network.storage_units_t.p.filter(regex=country).filter(regex="PHS")).values
        
        # Import/export
        if network.links_t.p0.filter(regex=country).filter(regex=str(country)+"-").groupby(network.links.bus0.str.slice(0,2), axis=1).sum().size > 0:
            Import = (+ network.links_t.p0.filter(regex=country).filter(regex=str(country)+"-").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()).values
        else:
            Import = np.zeros([8760,1])
        
        if network.links_t.p0.filter(regex=country).filter(regex="-"+str(country)).groupby(network.links.bus0.str.slice(0,2), axis=1).sum().size > 0:
            Export = (- network.links_t.p0.filter(regex=country).filter(regex="-"+str(country)).groupby(network.links.bus1.str.slice(0,2), axis=1).sum()).values
        else:
            Export = np.zeros([8760,1])  
            
        importExport[country] = Import + Export
    
        # Backup generator
        OCGT[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="OCGT").values
                * busEff.loc["OCGT"].values[0]
                )
        # dispathable backup generator
        if (+ network.storage_units_t.p.filter(regex=country).filter(regex="hydro")).size > 0:
            hydro[country] = (+ network.storage_units_t.p.filter(regex=country).filter(regex="hydro")).values
        
        
        # Heating related response:
        if size >= 90:
            # Sector links to Heat
            groundHeatPump[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="ground heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values
            
            if (network.links_t.p1.filter(regex=country).filter(regex="central heat pump")).size > 0:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="central heat pump")).values
            else:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban heat pump")).values
            
            ResistiveHeater[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" resistive heater") ).values
           
            if (network.links_t.p1.filter(regex=country).filter(regex="central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="central resistive heater")).values
            else:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban resistive heater")).values

            # CHP
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="CHP electric")).size > 0:
                CHPElec[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central CHP electric").values
                       * busEff.loc["central CHP electric"].values[0]
                       )        
        
        
        # Transportation related response
        if size == 60 or size == 120:
        
            # Sector links to vehicle
            EVBattery[country] = (- network.links_t.p0.filter(regex=str(country)+" ").filter(regex="BEV")
                - network.links_t.p1.filter(regex=country).filter(regex="V2G").values
                #* busEff.loc["V2G"].values[0]
                ).values      

        # Brownfield related response
        if isBrownfield == True: 
            # Backup generator
            CCGT[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="CCGT").values
                    * busEff.loc["CCGT"].values[0]
                    )
            oil[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=" oil").values
                    * busEff.loc["oil"].values[0]
                    )
            coal[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="coal").values
                    * busEff.loc["coal"].values[0]
                    )
            lignite[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="lignite").values
                    * busEff.loc["lignite"].values[0]
                    )
            nuclear[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="nuclear").values
                    * busEff.loc["nuclear"].values[0]
                    )
            biomassEOP[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="EOP").values
                    * busEff.loc["biomass EOP"].values[0]
                    )
            # Sector links to Heat
            groundHeatPump[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="decentral heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values
            
            if (network.links_t.p1.filter(regex=country).filter(regex="central heat pump")).size > 0:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="central heat pump")).values
            else:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban heat pump")).values
            
            ResistiveHeater[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" decentral resistive heater") ).values
           
            if (network.links_t.p1.filter(regex=country).filter(regex="central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="central resistive heater")).values
            else:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban resistive heater")).values

            coolingPump[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="cooling pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values
                
            # CHP
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="gas CHP electric")).size > 0:
                gasCHPElec[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central gas CHP electric").values
                       * busEff.loc["central gas CHP electric"].values[0]
                       )        
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="biomass CHP electric")).size > 0:
                biomassCHPElec[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central biomass CHP electric").values
                       * busEff.loc["central biomass CHP electric"].values[0]
                       )  
        
    # Collecting terms        
    storage = + H2 + battery.values + PHS.values
    importExport = - importExport
    backupGenerator = OCGT
    dispBackupGenerator = hydro
    
    if size >= 90:
        heatCouple = - groundHeatPump - centralUrbanHeatPump.values - ResistiveHeater.values - centralUrbanResistiveHeater.values
        CHP = CHPElec
    if size == 60 or size == 120:
        transCouple = EVBattery
    
    if isBrownfield == True:  
        backupGenerator = OCGT + CCGT + coal.values + lignite.values + nuclear.values + oil.values + biomassEOP.values
        heatCouple = - groundHeatPump - centralUrbanHeatPump.values - ResistiveHeater.values - centralUrbanResistiveHeater.values - coolingPump.values
        CHP = gasCHPElec + biomassCHPElec.values
    
    if isBrownfield == False:
        if collectTerms == True: # Collecting into general terms
            if size == 60: # elec + transport
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator,
                                     "Transport Couple":                + transCouple
                                     }
            elif size == 90: # elec + heating
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator,
                                     "Heat Couple":                     + heatCouple,
                                     "CHP Electric":                    + CHP
                                     }
            elif size == 120: # elec + heating + transport
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator,
                                     "Heat Couple":                     + heatCouple,
                                     "Transport Couple":                + transCouple,
                                     "CHP Electric":                    + CHP
                                     }   
            else: # elec only
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator
                                     }
    
        elif collectTerms == False: # Collecting into general terms
            if size == 60: # elec + transport
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro,
                                     "EV Battery":                      + EVBattery
                                     }
            elif size == 90: # elec + heating
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro,
                                     "Ground Heat Pump":                - groundHeatPump,
                                     "Central-Urban Heat Pump":         - centralUrbanHeatPump,
                                     "Resistive Heater":                - ResistiveHeater,
                                     "Central-Urban Resistive Heater":  - centralUrbanResistiveHeater,
                                     "CHP Electric":                    + CHPElec
                                     }
            elif size == 120: # elec + heating + transport
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro,
                                     "Ground Heat Pump":                - groundHeatPump,
                                     "Central-Urban Heat Pump":         - centralUrbanHeatPump,
                                     "Resistive Heater":                - ResistiveHeater,
                                     "Central-Urban Resistive Heater":  - centralUrbanResistiveHeater,
                                     "EV Battery":                      + EVBattery,
                                     "CHP Electric":                    + CHPElec
                                    }   
            else: # elec only
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro
                                    }
    if isBrownfield == True:
        if collectTerms == True: # Collecting into general terms
            collectedResponse = {"Storage":                         + storage, 
                                 "Import-Export":                   + importExport,
                                 "Backup Generator":                + backupGenerator,
                                 "Hydro Reservoir":                 + dispBackupGenerator,
                                 "Heat Couple":                     + heatCouple,
                                 "CHP Electric":                    + CHP
                                 }
        
        elif collectTerms == False: # Collecting into general terms
            collectedResponse = {"H2":                              + H2, 
                                 "Battery":                         + battery,
                                 "PHS":                             + PHS,
                                 "Import-Export":                   + importExport,
                                 "OCGT":                            + OCGT,
                                 "CCGT":                            + CCGT,
                                 "Coal":                            + coal,
                                 "Lignite":                         + lignite,
                                 "Nuclear":                         + nuclear,
                                 "Oil":                             + oil,
                                 "Biomass EOP":                     + biomassEOP,
                                 "Hydro Reservoir":                 + hydro,
                                 "Ground Heat Pump":                - groundHeatPump,
                                 "Central-Urban Heat Pump":         - centralUrbanHeatPump,
                                 "Resistive Heater":                - ResistiveHeater,
                                 "Central-Urban Resistive Heater":  - centralUrbanResistiveHeater,
                                 "Cooling Pump":                    - coolingPump,
                                 "Gas CHP Electric":                + gasCHPElec,
                                 "Biomass CHP Electric":            + biomassCHPElec,
                         }

    return collectedResponse

#%% HeatResponse

def HeatResponse(network, collectTerms=True):
    
    # Save index with only countries
    country_column = network.loads.index[:30]
    
    # Find efficency of links
    busEff = LinksEff(network)
    
    # Check if brownfield network
    isBrownfield = len([x for x in network.links_t.p0.columns if "nuclear" in x]) > 0
    
    # General Reponse
    waterTanks                  = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanWaterTanks      = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    GasBoiler                   = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanGasBoiler       = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    groundHeatPump              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanHeatPump        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    ResistiveHeater             = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanResistiveHeater = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    CHPHeat                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    
    if isBrownfield == True:
        biomassCHPHeat                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        HOP                                = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        coolingPump                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    for country in country_column:
        
        # Storage
        waterTanks[country] = (- network.links_t.p0.filter(regex=country).filter(regex=str(country)+" water tanks charger")
            + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" water tanks discharger").values
            * busEff.loc["water tanks discharger"].values[0]
            ).values 
        
        if network.links_t.p0.filter(regex=country).filter(regex="central water tanks charger").size > 0:
            centralUrbanWaterTanks[country] = (- network.links_t.p0.filter(regex=country).filter(regex="central water tanks charger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            + network.links_t.p0.filter(regex=country).filter(regex="central water tanks discharger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            * busEff.loc["water tanks discharger"].values[0]
                            ).values
        else:
            centralUrbanWaterTanks[country] = (- network.links_t.p0.filter(regex=country).filter(regex="urban water tanks charger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            + network.links_t.p0.filter(regex=country).filter(regex="urban water tanks discharger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            * busEff.loc["water tanks discharger"].values[0]
                            ).values
        
        # Backup generator
        if isBrownfield == False:
            GasBoiler[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" gas").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                       * busEff.loc["central gas boiler"].values[0]
                      ).values
        elif isBrownfield == True:
            GasBoiler[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" decentral gas")
                       * busEff.loc["central gas boiler"].values[0]
                      ).values
            if network.links_t.p0.filter(regex=country).filter(regex="HOP").size > 0:
                HOP[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="HOP")
                                * busEff.loc["central biomass HOP"].values[0]
                                ).values
    
        if network.links_t.p0.filter(regex=country).filter(regex=" central gas").size > 0:
            centralUrbanGasBoiler[country] = (+ network.links_t.p1.filter(regex=country).filter(regex=" central gas boiler").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            #* busEff.loc["central gas boiler"].values[0]
                            * -1
                         ).values
        else:
            centralUrbanGasBoiler[country] = (+ network.links_t.p0.filter(regex=country).filter(regex="urban gas").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            * busEff.loc["central gas boiler"].values[0]
                         ).values
    
        # Sector links
        if isBrownfield == False:
            groundHeatPump[country] = ( - network.links_t.p1.filter(regex=country).filter(regex="ground heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values

        elif isBrownfield == True:
            groundHeatPump[country] = ( - network.links_t.p1.filter(regex=country).filter(regex="decentral heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values

        if (network.links_t.p1.filter(regex=country).filter(regex="central heat pump")).size > 0:
            centralUrbanHeatPump[country] = (- network.links_t.p1.filter(regex=country).filter(regex=" central heat pump")).values
        else:
            centralUrbanHeatPump[country] = (- network.links_t.p1.filter(regex=country).filter(regex="urban heat pump")).values
        
        if isBrownfield == False:
            ResistiveHeater[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" resistive heater") 
                                    * (busEff.loc["central resistive heater"].values[0])
                                    ).values
        elif isBrownfield == True:
           ResistiveHeater[country] = ( + network.links_t.p1.filter(regex=country).filter(regex=str(country)+" decentral resistive heater") 
                                    #* (busEff.loc["decentral resistive heater"].values[0])
                                    * -1
                                    ).values
        if isBrownfield == False:
            if (network.links_t.p1.filter(regex=country).filter(regex=" central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex=" central resistive heater")
                                                        * (busEff.loc["central resistive heater"].values[0])
                                                        ).values
            else:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban resistive heater")
                                                        * (busEff.loc["central resistive heater"].values[0])
                                                        ).values
        elif isBrownfield == True:
            if (network.links_t.p1.filter(regex=country).filter(regex=" central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p1.filter(regex=country).filter(regex=" central resistive heater")
                                                        #* (busEff.loc["central resistive heater"].values[0])
                                                        * -1
                                                        ).values
        if isBrownfield == True:
            coolingPump[country] = ( - network.links_t.p1.filter(regex=country).filter(regex="cooling pump") ).values
                
                
                
        # CHP
        if isBrownfield == False:
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="CHP heat")).size > 0:
                CHPHeat[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central CHP heat").values
                       * busEff.loc["central CHP heat"].values[0]
                       )      
        elif isBrownfield == True:
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="gas CHP heat")).size > 0:
                CHPHeat[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central gas CHP heat").values
                       * busEff.loc["central gas CHP heat"].values[0]
                       )     
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="biomass CHP heat")).size > 0:
                biomassCHPHeat[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central biomass CHP heat").values
                       * busEff.loc["central biomass CHP heat"].values[0]
                       )   

    if isBrownfield == False:
        storage = + waterTanks + centralUrbanWaterTanks.values
        backupGenerator = GasBoiler + centralUrbanGasBoiler.values
        elecCouple = + groundHeatPump + centralUrbanHeatPump.values + ResistiveHeater.values + centralUrbanResistiveHeater.values 
        CHP = CHPHeat
        
        if collectTerms == True: # Collecting into general terms
            collectedResponse = {"Storage":                         + storage,
                                 "Backup Generator":                + backupGenerator,
                                 "Electricity Couple":              + elecCouple,
                                 "CHP Heat":                        + CHP
                                 }
        
        elif collectTerms == False: # Collecting into general terms
            collectedResponse = {"Water Tanks":                     + waterTanks, 
                                 "Central-Urban Water Tanks":       + centralUrbanWaterTanks,
                                 "Gas Boiler":                      + GasBoiler,
                                 "Central-Urban Gas Boiler":        + centralUrbanGasBoiler,
                                 "Ground Heat Pump":                + groundHeatPump,
                                 "Central-Urban Heat Pump":         + centralUrbanHeatPump,
                                 "Resistive Heater":                + ResistiveHeater,
                                 "Central-Urban Resistive Heater":  + centralUrbanResistiveHeater,
                                 "CHP Heat":                        + CHPHeat
                                 }
            
    elif isBrownfield == True:
        storage = waterTanks + centralUrbanWaterTanks.values
        backupGenerator = GasBoiler + centralUrbanGasBoiler.values + HOP.values
        elecCouple = + groundHeatPump + centralUrbanHeatPump.values + ResistiveHeater.values + centralUrbanResistiveHeater.values + coolingPump.values
        CHP = CHPHeat + biomassCHPHeat.values
        
        if collectTerms == True: # Collecting into general terms
            collectedResponse = {"Storage":                         + storage,
                                 "Backup Generator":                + backupGenerator,
                                 "Electricity Couple":              + elecCouple,
                                 "CHP Heat":                        + CHP
                                 }
        
        elif collectTerms == False: # Collecting into general terms
            collectedResponse = {"Water Tanks":                     + waterTanks,
                                 "Central Water Tanks":             + centralUrbanWaterTanks, 
                                 "Decentral Gas Boiler":            + GasBoiler,
                                 "Central Gas Boiler":              + centralUrbanGasBoiler,
                                 "Biomass HOP":                     + HOP,
                                 "Decentral Heat Pump":             + groundHeatPump,
                                 "Central Heat Pump":               + centralUrbanHeatPump,
                                 "Decentral Resistive Heater":      + ResistiveHeater,
                                 "Central Resistive Heater":        + centralUrbanResistiveHeater,
                                 "Cooling Hump":                    + coolingPump,
                                 "Gas CHP Heat":                    + CHPHeat,
                                 "Biomass CHP Heat":                + biomassCHPHeat
                                 }
            
                   

    return collectedResponse

#%% SortLambdaMatrix

def SortLambdaMatrix(lambdaMatrix):
    """
    

    Parameters
    ----------
    lambdaMatrix : list of DataFrames
        Takes in a list mostly for 7 different cases and including the different contributions

    Returns
    -------
    lambdaSorted : list of DataFrames
        outputs a list of contribution where the Dataframes are the different cases

    """
    
    # Print different types
    names = lambdaMatrix[0].columns
    # empty list
    lambdaSorted = []
    
    for j in range(len(names)): # For loop for each name
        
        # Create empty DataFrame
        lambdaAlike = pd.DataFrame(data=np.zeros([30,len(lambdaMatrix)]))
    
        for i in range(len(lambdaMatrix)): # For loop for each instance
            
            # Save into dataframe
            lambdaAlike[i] = lambdaMatrix[i][names[j]]
    
        # Save into list
        lambdaSorted.append(lambdaAlike)    
    
    return lambdaSorted

#%% ChangePlotSetup

def ChangePlotSetup(i,rotate,PCLegend,length,title, networktype,PC,PCLegendAmount=8):
    
    # y-axis label:
    if rotate == True:
        if i == 0 or i == 3:
            plt.ylabel('$\lambda$',fontsize=14,rotation=0, labelpad=10)
    else:
        if i == 0 or i == 2 or i == 4:
            plt.ylabel('$\lambda$',fontsize=14,rotation=0, labelpad=10)
    
    # xticks
    if networktype == "brown":
        plt.xticks(np.arange(0,length),['2020', '2025', '2030', '2035', '2040', '2045', '2050'],fontsize=12)
    else:
        if length == 7:
            plt.xticks(np.arange(0,length),['40%', '50%', '60%', '70%', '80%', '90%', '95%'],fontsize=12)
        if length == 5:
             plt.xticks(np.arange(0,length),['Zero', 'Current', '2x Current', '4x Current', '6x Current'],fontsize=12,rotation=-12.5)
    plt.yticks(fontsize=12)
    
    # Subplot title
    plt.title('Principle component '+str(i+1),fontsize=13,fontweight="bold")
    
    # legend
    if i == PCLegend: # Create legend of figure 4 (lower left)
        if rotate == True:
            if PC == 2:
                if length == 5:
                    plt.legend(loc = 'upper center', # How the label should be places according to the placement
                               bbox_to_anchor = (-0.08,-0.3), # placement relative to the graph
                               ncol = PCLegendAmount, # Amount of columns
                               fontsize = 12, # Size of text
                               framealpha = 1, # Box edge alpha
                               columnspacing = 2, # Horizontal spacing between labels
                               )
                else:
                    plt.legend(loc = 'upper center', # How the label should be places according to the placement
                               bbox_to_anchor = (-0.08,-0.15), # placement relative to the graph
                               ncol = PCLegendAmount, # Amount of columns
                               fontsize = 12, # Size of text
                               framealpha = 1, # Box edge alpha
                               columnspacing = 2, # Horizontal spacing between labels
                               )
                #plt.tight_layout()
            else:
                if length == 5:
                    plt.legend(loc = 'upper center', # How the label should be places according to the placement
                               bbox_to_anchor = (0.45,-0.125), # placement relative to the graph
                               ncol = PCLegendAmount, # Amount of columns
                               fontsize = 'medium', # Size of text
                               framealpha = 1, # Box edge alpha
                               columnspacing = 2, # Horizontal spacing between labels
                               )
                else:
                    plt.legend(loc = 'upper center', # How the label should be places according to the placement
                               bbox_to_anchor = (0.5,-0.15), # placement relative to the graph
                               ncol = PCLegendAmount, # Amount of columns
                               fontsize = 12, # Size of text
                               framealpha = 1, # Box edge alpha
                               columnspacing = 2, # Horizontal spacing between labels
                               )
        else:
            if length == 5:
                plt.legend(loc = 'upper center', # How the label should be places according to the placement
                               bbox_to_anchor = (-0.08,-0.3), # placement relative to the graph
                               ncol = PCLegendAmount, # Amount of columns
                               fontsize = 12, # Size of text
                               framealpha = 1, # Box edge alpha
                               columnspacing = 2, # Horizontal spacing between labels
                               )
            else:
                plt.legend(loc = 'upper center', # How the label should be places according to the placement
                               bbox_to_anchor = (-0.08,-0.15), # placement relative to the graph
                               ncol = PCLegendAmount, # Amount of columns
                               fontsize = 12, # Size of text
                               framealpha = 1, # Box edge alpha
                               columnspacing = 2, # Horizontal spacing between labels
                               )
    return

#%% ChangeContributionElec

def ChangeContributionElec(lambdaMatrix,title="none",dpi=200, networktype="green",rotate=False,PC=6):
    
    # Calculate lambda over constraints
    lambdaMatrixSum = []
    for j in range(6):
        temp = []
        for k in range(len(lambdaMatrix)):
            temp.append(lambdaMatrix[k].sum(axis=1)[j])
        lambdaMatrixSum.append(temp)
    
    # sort
    lambdaSorted = SortLambdaMatrix(lambdaMatrix)
    
    length = len(lambdaMatrix)
    
    # subplot setup
    if rotate == True:
        col = 3
        row = 2
    else:
        col = 2
        row = 3
    
    # legend setup
    if rotate == True:
        if PC == 2 or 3: PCLegend = 1
        if PC == 6:  PCLegend = 4
    else:
        PCLegend = 5
    
    # Plotting
    if rotate == True:
        if PC == 2: fig = plt.figure(figsize=(15,6),dpi=dpi)
        else: fig = plt.figure(figsize=(10,6),dpi=dpi)
    else:
        fig = plt.figure(figsize=(11,8),dpi=dpi)
        
    # Print out for 6 principle components
    for i in range(PC):
        
        # Data for each PC
        windConData =   lambdaSorted[0].loc[i,:]
        SolarConData =  lambdaSorted[1].loc[i,:]+windConData
        rorConData =    lambdaSorted[2].loc[i,:]+SolarConData
        loadConData =   lambdaSorted[3].loc[i,:]+rorConData
        genCovData =  (+lambdaSorted[4].loc[i,:]
                       +lambdaSorted[5].loc[i,:]
                       +lambdaSorted[6].loc[i,:])+loadConData
        loadCovData = (+lambdaSorted[7].loc[i,:]
                       +lambdaSorted[8].loc[i,:]
                       +lambdaSorted[9].loc[i,:])
        
        # Plot
        ax = plt.subplot(row,col,i+1)
        if i > 1:
            plt.ticklabel_format(axis='y', style='sci')
            ax.yaxis.major.formatter.set_powerlimits((0,0))
            ax.yaxis.offsetText.set_fontsize(12)

        # Plot lines
        plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
        plt.plot(windConData,color='k',alpha=1,linewidth=0.5)
        plt.plot(SolarConData,color='k',alpha=1,linewidth=0.5)
        plt.plot(rorConData,color='k',alpha=1,linewidth=0.5)
        plt.plot(loadConData,color='k',alpha=1,linewidth=0.5)
        plt.plot(genCovData,color='k',alpha=1,linewidth=0.5)
        plt.plot(loadCovData,color='k',alpha=1,linewidth=0.5)
        # Plot fill inbetween lines
        plt.fill_between(range(length), np.zeros(length), windConData,
                         label='Wind',
                         color='cornflowerblue') # Because it is a beutiful color
        plt.fill_between(range(length), windConData, SolarConData,
                         label='Solar PV',
                         color='yellow')
        plt.fill_between(range(length), SolarConData, rorConData,
                         label='RoR',
                         color='darkslateblue')
        plt.fill_between(range(length), rorConData, loadConData,
                         label='Load',
                         color='slategray')
        plt.fill_between(range(length), loadConData, genCovData,
                         label='Generator\ncovariance',
                         color='brown',
                         alpha=0.5)
        plt.fill_between(range(length), loadCovData, np.zeros(length),
                         label='Load\ncovariance',
                         color='orange',
                         alpha=0.5)
        
        # Axis and legend setup
        ChangePlotSetup(i,rotate,PCLegend,length,title,networktype,PC)
    
    # Setup tight layout
    plt.tight_layout()
    if rotate == True:
        if PC == 2:
            plt.subplots_adjust(wspace = 0.15)
        else:
            plt.subplots_adjust(wspace = 0.25)
    else:
        plt.subplots_adjust(wspace = 0.15)
    
    # Write title
    if title != "none":
        if rotate == True:
            plt.suptitle(title,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07  
        else:
            plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07
        
    # Show plot
    plt.show()
    
    return fig
    
#%% ChangeContributionHeat

def ChangeContributionHeat(lambdaMatrix,title="none",dpi=200, networktype="green",rotate=False,PC=6):
    
    # Calculate lambda over constraints
    lambdaMatrixSum = []
    for j in range(6):
        temp = []
        for k in range(len(lambdaMatrix)):
            temp.append(lambdaMatrix[k].sum(axis=1)[j])
        lambdaMatrixSum.append(temp)
    
    # sort
    lambdaSorted = SortLambdaMatrix(lambdaMatrix)
    
    length = len(lambdaMatrix)
    
    # subplot setup
    if rotate == True:
        col = 3
        row = 2
    else:
        col = 2
        row = 3
    
    # legend setup
    if rotate == True:
        if PC == 2 or 3: PCLegend = 1
        if PC == 6:  PCLegend = 4
    else:
        PCLegend = 5
    
    # Plotting
    if rotate == True:
        if PC == 2: fig = plt.figure(figsize=(15,6),dpi=dpi)
        else: fig = plt.figure(figsize=(10,6),dpi=dpi)
    else:
        fig = plt.figure(figsize=(10.5,8),dpi=dpi)
        
    # Print out for 6 principle components
    for i in range(PC):
        
        # For greenfield network:
        if networktype == "green":
            # Data for each PC
            solarColConData      =   lambdaSorted[0].loc[i,:]
            urbanSolarColConData =   lambdaSorted[1].loc[i,:] + solarColConData
            loadConData          =   lambdaSorted[2].loc[i,:] + urbanSolarColConData
            urbanLoadConData     =   lambdaSorted[3].loc[i,:] + loadConData
            genCovData           =   lambdaSorted[4].loc[i,:] + urbanLoadConData
            loadCovData          = (+lambdaSorted[5].loc[i,:]
                                    +lambdaSorted[6].loc[i,:]
                                    +lambdaSorted[7].loc[i,:]
                                    +lambdaSorted[8].loc[i,:])
            loadOnlyCovData      =   lambdaSorted[9].loc[i,:] + genCovData
            
            # Plot
            ax = plt.subplot(row,col,i+1)
            if i > 1:
                plt.ticklabel_format(axis='y', style='sci')
                ax.yaxis.major.formatter.set_powerlimits((0,0))
                ax.yaxis.offsetText.set_fontsize(12)
            # Plot lines
            plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
            plt.plot(solarColConData,       color='k',alpha=1,linewidth=0.5)
            plt.plot(urbanSolarColConData,  color='k',alpha=1,linewidth=0.5)
            plt.plot(loadConData,           color='k',alpha=1,linewidth=0.5)
            plt.plot(urbanLoadConData,      color='k',alpha=1,linewidth=0.5)
            plt.plot(genCovData,            color='k',alpha=1,linewidth=0.5)
            plt.plot(loadCovData,           color='k',alpha=1,linewidth=0.5)
            plt.plot(loadOnlyCovData,       color='k',alpha=1,linewidth=0.5)
            # Plot fill inbetween lines
            plt.fill_between(range(length), np.zeros(length), solarColConData,
                             label='Solar\nCollector',
                             color='orange') # Because it is a beutiful color
            plt.fill_between(range(length), solarColConData, urbanSolarColConData,
                             label='Central-Urban\nSolar Collector',
                             color='gold')
            plt.fill_between(range(length), urbanSolarColConData, loadConData,
                             label='Heat Load',
                             color='forestgreen')
            plt.fill_between(range(length), loadConData, urbanLoadConData,
                             label='Urban\nHeat Load',
                             color='lime')
            plt.fill_between(range(length), urbanLoadConData, genCovData,
                             label='Generator\novariance',
                             color='darkorange',
                             alpha=0.5)
            plt.fill_between(range(length), genCovData, loadOnlyCovData,
                             label='Load\ncovariance',
                             color='darkgreen',
                             alpha=0.5)
            plt.fill_between(range(length), loadCovData, np.zeros(length),
                             label='Generator/Load\ncovariance',
                             color='lightsteelblue',
                             alpha=0.5)
        # For greenfield network:
        elif networktype == "brown":
            # Data for each PC
            loadConData          =   lambdaSorted[0].loc[i,:]
            urbanLoadConData     =   lambdaSorted[1].loc[i,:] + loadConData
            loadCoolingConData   =   lambdaSorted[2].loc[i,:] + urbanLoadConData
            CovData              = (+lambdaSorted[3].loc[i,:]
                                    +lambdaSorted[4].loc[i,:]
                                    +lambdaSorted[5].loc[i,:])
            
            # Plot
            ax = plt.subplot(row,col,i+1)
            if i > 1:
                plt.ticklabel_format(axis='y', style='sci')
                ax.yaxis.major.formatter.set_powerlimits((0,0))
                ax.yaxis.offsetText.set_fontsize(12)
            # Plot lines
            plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
            plt.plot(loadConData,           color='k',alpha=1,linewidth=0.5)
            plt.plot(urbanLoadConData,      color='k',alpha=1,linewidth=0.5)
            plt.plot(loadCoolingConData,    color='k',alpha=1,linewidth=0.5)
            plt.plot(CovData,               color='k',alpha=1,linewidth=0.5)
            # Plot fill inbetween lines
            plt.fill_between(range(length), np.zeros(length), loadConData,
                             label='Heat Load',
                             color='forestgreen')
            plt.fill_between(range(length), loadConData, urbanLoadConData,
                             label='Urban\nHeat Load',
                             color='lime')
            plt.fill_between(range(length), urbanLoadConData, loadCoolingConData,
                             label='Cooling load',
                             color='royalblue',
                             alpha=0.5)             
            plt.fill_between(range(length), loadCoolingConData, CovData,
                             label='Load\ncovariance',
                             color='lightsteelblue',
                             alpha=0.5)  
            
                # Axis and legend setup
        ChangePlotSetup(i,rotate,PCLegend,length,title,networktype,PC,PCLegendAmount=5)
    
    # Setup tight layout
    plt.tight_layout()
    if rotate == True:
        if PC == 2:
            plt.subplots_adjust(wspace = 0.15)
        else:
            plt.subplots_adjust(wspace = 0.25)
    else:
        plt.subplots_adjust(wspace = 0.15)
    
    # Write title
    if title != "none":
        if rotate == True:
            plt.suptitle(title,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07  
        else:
            plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07
        
    # Show plot
    plt.show()
    
    return fig

#%% ChangeResponseElec

def ChangeResponseElec(lambdaMatrix,title="none",dpi=200, networktype="green",rotate=False,PC=6):
    
    # Calculate lambda over constraints
    lambdaMatrixSum = []
    for j in range(6):
        temp = []
        for k in range(len(lambdaMatrix)):
            temp.append(lambdaMatrix[k].sum(axis=1)[j])
        lambdaMatrixSum.append(temp)
    
    # Sort
    lambdaSorted = SortLambdaMatrix(lambdaMatrix)

    length = len(lambdaMatrix)
    
    names = lambdaMatrix[0].columns
    
    includeHeat = len([x for x in names if "Heat" in x]) > 0
    includeTrans = len([x for x in names if "Transport" in x]) > 0
    
    # subplot setup
    if rotate == True:
        col = 3
        row = 2
    else:
        col = 2
        row = 3
    
    # legend setup
    if rotate == True:
        if PC == 2 or 3: PCLegend = 1
        if PC == 6:  PCLegend = 4
    else:
        PCLegend = 5
    
    # Plotting
    if rotate == True:
        if PC == 2: fig = plt.figure(figsize=(15,6),dpi=dpi)
        else: fig = plt.figure(figsize=(10,6),dpi=dpi)
    else:
        fig = plt.figure(figsize=(10.5,8),dpi=dpi)
        
    # Print out for 6 principle components
    for i in range(PC):
        
        
        if includeHeat: # Check if include heat
            
            if includeTrans: # Case with elec+heat+trans
                # Data for each PC
                storageConData      =   lambdaSorted[0].loc[i,:]
                importConData       =   lambdaSorted[1].loc[i,:] + storageConData
                backupConData       =   lambdaSorted[2].loc[i,:] + importConData
                dispbackupConData   =   lambdaSorted[3].loc[i,:] + backupConData
                heatCoupleConData   =   lambdaSorted[4].loc[i,:] + dispbackupConData
                transCoupleConData  =   lambdaSorted[5].loc[i,:] + heatCoupleConData
                CHPCovData          =   lambdaSorted[6].loc[i,:] + transCoupleConData
                CovData             = ( lambdaSorted[7].loc[i,:]
                                       +lambdaSorted[8].loc[i,:]
                                       +lambdaSorted[9].loc[i,:]
                                       +lambdaSorted[10].loc[i,:]
                                       +lambdaSorted[11].loc[i,:]
                                       +lambdaSorted[12].loc[i,:]
                                       +lambdaSorted[13].loc[i,:]
                                       +lambdaSorted[14].loc[i,:]
                                       +lambdaSorted[15].loc[i,:]
                                       +lambdaSorted[16].loc[i,:]
                                       +lambdaSorted[17].loc[i,:]
                                       +lambdaSorted[18].loc[i,:]
                                       +lambdaSorted[19].loc[i,:]
                                       +lambdaSorted[20].loc[i,:]
                                       +lambdaSorted[21].loc[i,:]
                                       +lambdaSorted[22].loc[i,:]
                                       +lambdaSorted[23].loc[i,:]
                                       +lambdaSorted[24].loc[i,:]
                                       +lambdaSorted[25].loc[i,:]
                                       +lambdaSorted[26].loc[i,:]
                                       +lambdaSorted[27].loc[i,:]
                                        ) + CHPCovData
                
                # Plot
                ax = plt.subplot(row,col,i+1)
                if i > 1:
                    plt.ticklabel_format(axis='y', style='sci')
                    ax.yaxis.major.formatter.set_powerlimits((0,0))
                    ax.yaxis.offsetText.set_fontsize(12)
                # Plot lines
                plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
                plt.plot(storageConData,       color='k',alpha=1,linewidth=0.5)
                plt.plot(importConData,        color='k',alpha=1,linewidth=0.5)
                plt.plot(backupConData,        color='k',alpha=1,linewidth=0.5)
                plt.plot(dispbackupConData,    color='k',alpha=1,linewidth=0.5)
                plt.plot(heatCoupleConData,    color='k',alpha=1,linewidth=0.5)
                plt.plot(transCoupleConData,   color='k',alpha=1,linewidth=0.5)
                plt.plot(CHPCovData,           color='k',alpha=1,linewidth=0.5)
                plt.plot(CovData,              color='k',alpha=1,linewidth=0.5)
                # Plot fill inbetween lines
                plt.fill_between(range(length), np.zeros(length), storageConData,
                                 label='Storage',
                                 color='skyblue') # Because it is a beutiful color
                plt.fill_between(range(length), storageConData, importConData,
                                 label='Import-\nExport',
                                 color='limegreen')
                plt.fill_between(range(length), importConData, backupConData,
                                 label='Backup\nGenerator',
                                 color='firebrick')
                plt.fill_between(range(length), backupConData, dispbackupConData,
                                 label='Hydro\nReservoir',
                                 color='dodgerblue')
                plt.fill_between(range(length), dispbackupConData, heatCoupleConData,
                                 label='Heating\nCouple',
                                 color='darkorange')
                plt.fill_between(range(length), heatCoupleConData, transCoupleConData,
                                 label='Transport\nCouple',
                                 color='tan')
                plt.fill_between(range(length), transCoupleConData, CHPCovData,
                                 label='CHP Electric',
                                 color='slategrey')
                plt.fill_between(range(length), CHPCovData, CovData,
                                 label='Collected\ncovariances',
                                 color='gray',
                                 alpha=0.5)
                PCLegendAmount = 5
            else: # Case with elec+heat
                # Data for each PC
                storageConData      =   lambdaSorted[0].loc[i,:]
                importConData       =   lambdaSorted[1].loc[i,:] + storageConData
                backupConData       =   lambdaSorted[2].loc[i,:] + importConData
                dispbackupConData   =   lambdaSorted[3].loc[i,:] + backupConData
                heatCoupleConData   =   lambdaSorted[4].loc[i,:] + dispbackupConData
                CHPCovData          =   lambdaSorted[6].loc[i,:] + heatCoupleConData
                CovData             = (+lambdaSorted[7].loc[i,:]
                                       +lambdaSorted[8].loc[i,:]
                                       +lambdaSorted[9].loc[i,:]
                                       +lambdaSorted[10].loc[i,:]
                                       +lambdaSorted[11].loc[i,:]
                                       +lambdaSorted[12].loc[i,:]
                                       +lambdaSorted[13].loc[i,:]
                                       +lambdaSorted[14].loc[i,:]
                                       +lambdaSorted[15].loc[i,:]
                                       +lambdaSorted[16].loc[i,:]
                                       +lambdaSorted[17].loc[i,:]
                                       +lambdaSorted[18].loc[i,:]
                                       +lambdaSorted[19].loc[i,:]
                                       +lambdaSorted[20].loc[i,:]
                                       ) + CHPCovData
                
                
                # Plot
                ax = plt.subplot(row,col,i+1)
                if i > 1:
                    plt.ticklabel_format(axis='y', style='sci')
                    ax.yaxis.major.formatter.set_powerlimits((0,0))
                    ax.yaxis.offsetText.set_fontsize(12)
                # Plot lines
                plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
                plt.plot(storageConData,       color='k',alpha=1,linewidth=0.5)
                plt.plot(importConData,        color='k',alpha=1,linewidth=0.5)
                plt.plot(backupConData,        color='k',alpha=1,linewidth=0.5)
                plt.plot(dispbackupConData,    color='k',alpha=1,linewidth=0.5)
                plt.plot(heatCoupleConData,    color='k',alpha=1,linewidth=0.5)
                plt.plot(CHPCovData,           color='k',alpha=1,linewidth=0.5)
                plt.plot(CovData,              color='k',alpha=1,linewidth=0.5)
                # Plot fill inbetween lines
                plt.fill_between(range(length), np.zeros(length), storageConData,
                                 label='Storage',
                                 color='skyblue') # Because it is a beutiful color
                plt.fill_between(range(length), storageConData, importConData,
                                 label='Import-\nExport',
                                 color='limegreen')
                plt.fill_between(range(length), importConData, backupConData,
                                 label='Backup\nGenerator',
                                 color='firebrick')
                plt.fill_between(range(length), backupConData, dispbackupConData,
                                 label='Hydro\nReservoir',
                                 color='dodgerblue')
                plt.fill_between(range(length), dispbackupConData, heatCoupleConData,
                                 label='Heating\nCouple',
                                 color='darkorange')
                plt.fill_between(range(length), heatCoupleConData, CHPCovData,
                                 label='CHP Electric',
                                 color='slategrey')
                plt.fill_between(range(length), CHPCovData, CovData,
                                 label='Collected\ncovariances',
                                 color='gray',
                                 alpha=0.5)     
                PCLegendAmount = 5
        elif includeTrans: # Case with elec+trans
            # Data for each PC
            storageConData      =   lambdaSorted[0].loc[i,:]
            importConData       =   lambdaSorted[1].loc[i,:] + storageConData
            backupConData       =   lambdaSorted[2].loc[i,:] + importConData
            dispbackupConData   =   lambdaSorted[3].loc[i,:] + backupConData
            transCoupleConData  =   lambdaSorted[4].loc[i,:] + dispbackupConData
            CovData             = (+lambdaSorted[5].loc[i,:]
                                   +lambdaSorted[6].loc[i,:]
                                   +lambdaSorted[7].loc[i,:]
                                   +lambdaSorted[8].loc[i,:]
                                   +lambdaSorted[9].loc[i,:]
                                   +lambdaSorted[10].loc[i,:]
                                   +lambdaSorted[11].loc[i,:]
                                   +lambdaSorted[12].loc[i,:]
                                   +lambdaSorted[13].loc[i,:]
                                   +lambdaSorted[14].loc[i,:]   
                                   ) + transCoupleConData
            
            # Plot
            ax = plt.subplot(row,col,i+1)
            if i > 1:
                plt.ticklabel_format(axis='y', style='sci')
                ax.yaxis.major.formatter.set_powerlimits((0,0))
                ax.yaxis.offsetText.set_fontsize(12)
            # Plot lines
            plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
            plt.plot(storageConData,       color='k',alpha=1,linewidth=0.5)
            plt.plot(importConData,        color='k',alpha=1,linewidth=0.5)
            plt.plot(backupConData,        color='k',alpha=1,linewidth=0.5)
            plt.plot(dispbackupConData,    color='k',alpha=1,linewidth=0.5)
            plt.plot(transCoupleConData,   color='k',alpha=1,linewidth=0.5)
            plt.plot(CovData,              color='k',alpha=1,linewidth=0.5)
            # Plot fill inbetween lines
            plt.fill_between(range(length), np.zeros(length), storageConData,
                             label='Storage',
                             color='skyblue') # Because it is a beutiful color
            plt.fill_between(range(length), storageConData, importConData,
                             label='Import-\nExport',
                             color='limegreen')
            plt.fill_between(range(length), importConData, backupConData,
                             label='Backup\nGenerator',
                             color='firebrick')
            plt.fill_between(range(length), backupConData, dispbackupConData,
                             label='Hydro\nReservoir',
                             color='dodgerblue')
            plt.fill_between(range(length), dispbackupConData, transCoupleConData,
                             label='Transport\nCouple',
                             color='coral')
            plt.fill_between(range(length), transCoupleConData, CovData,
                             label='Collected\ncovariances',
                             color='gray',
                             alpha=0.5)
            PCLegendAmount = 6
        else: # Case with elec only
            # Data for each PC
            storageConData      =   lambdaSorted[0].loc[i,:]
            importConData       =   lambdaSorted[1].loc[i,:] + storageConData
            backupConData       =   lambdaSorted[2].loc[i,:] + importConData
            dispbackupConData   =   lambdaSorted[3].loc[i,:] + backupConData
            CovData             = (+lambdaSorted[4].loc[i,:]
                                   +lambdaSorted[5].loc[i,:]
                                   +lambdaSorted[6].loc[i,:]
                                   +lambdaSorted[7].loc[i,:]
                                   +lambdaSorted[8].loc[i,:]
                                   +lambdaSorted[9].loc[i,:]
                                   ) + dispbackupConData
            
            # Plot
            ax = plt.subplot(row,col,i+1)
            if i > 1:
                plt.ticklabel_format(axis='y', style='sci')
                ax.yaxis.major.formatter.set_powerlimits((0,0))
                ax.yaxis.offsetText.set_fontsize(12)
            # Plot lines
            plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
            plt.plot(storageConData,       color='k',alpha=1,linewidth=0.5)
            plt.plot(importConData,        color='k',alpha=1,linewidth=0.5)
            plt.plot(backupConData,        color='k',alpha=1,linewidth=0.5)
            plt.plot(dispbackupConData,    color='k',alpha=1,linewidth=0.5)
            plt.plot(CovData,              color='k',alpha=1,linewidth=0.5)
            # Plot fill inbetween lines
            plt.fill_between(range(length), np.zeros(length), storageConData,
                             label='Storage',
                             color='skyblue') # Because it is a beutiful color
            plt.fill_between(range(length), storageConData, importConData,
                             label='Import-\nExport',
                             color='limegreen')
            plt.fill_between(range(length), importConData, backupConData,
                             label='Backup\nGenerator',
                             color='firebrick')
            plt.fill_between(range(length), backupConData, dispbackupConData,
                             label='Hydro\nReservoir',
                             color='dodgerblue')
            plt.fill_between(range(length), dispbackupConData, CovData,
                             label='Collected\ncovariances',
                             color='gray',
                             alpha=0.5)        
            PCLegendAmount = 6    
        # Axis and legend setup
        ChangePlotSetup(i,rotate,PCLegend,length,title,networktype,PC,PCLegendAmount=PCLegendAmount)
    
    # Setup tight layout
    plt.tight_layout()
    if rotate == True:
        if PC == 2:
            plt.subplots_adjust(wspace = 0.15)
        else:
            plt.subplots_adjust(wspace = 0.25)
    else:
        plt.subplots_adjust(wspace = 0.15)
    
    # Write title
    if title != "none":
        if rotate == True:
            plt.suptitle(title,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07  
        else:
            plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07
        
    # Show plot
    plt.show()
    
    return fig
    
#%% ChangeResponseHeat
    
def ChangeResponseHeat(lambdaMatrix,title="none", dpi=200, networktype="green",rotate=False,PC=6):
    
    # Calculate lambda over constraints
    lambdaMatrixSum = []
    for j in range(6):
        temp = []
        for k in range(len(lambdaMatrix)):
            temp.append(lambdaMatrix[k].sum(axis=1)[j])
        lambdaMatrixSum.append(temp)
    
    # Sort
    lambdaSorted = SortLambdaMatrix(lambdaMatrix)
    
    length = len(lambdaMatrix)
    
    # subplot setup
    if rotate == True:
        col = 3
        row = 2
    else:
        col = 2
        row = 3
    
    # legend setup
    if rotate == True:
        if PC == 2 or 3: PCLegend = 1
        if PC == 6:  PCLegend = 4
    else:
        PCLegend = 5
    
    # Plotting
    if rotate == True:
        if PC == 2: fig = plt.figure(figsize=(15,6),dpi=dpi)
        else: fig = plt.figure(figsize=(10,6),dpi=dpi)
    else:
        fig = plt.figure(figsize=(10.5,8),dpi=dpi)
        
    # Print out for 6 principle components
    for i in range(PC):
        
        # Data for each PC
        storageConData      =   lambdaSorted[0].loc[i,:]
        backupConData       =   lambdaSorted[1].loc[i,:] + storageConData
        elecCoupleConData   =   lambdaSorted[2].loc[i,:] + backupConData
        CHPCovData          =   lambdaSorted[3].loc[i,:] + elecCoupleConData
        CovData             = (+lambdaSorted[4].loc[i,:]
                               +lambdaSorted[5].loc[i,:]
                               +lambdaSorted[6].loc[i,:]
                               +lambdaSorted[7].loc[i,:]
                               +lambdaSorted[8].loc[i,:]
                               +lambdaSorted[9].loc[i,:]
                               ) + CHPCovData
        
        
        # Plot
        ax = plt.subplot(row,col,i+1)
        if i > 1:
            plt.ticklabel_format(axis='y', style='sci')
            ax.yaxis.major.formatter.set_powerlimits((0,0))
            ax.yaxis.offsetText.set_fontsize(12)
        # Plot lines
        plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
        plt.plot(storageConData,       color='k',alpha=1,linewidth=0.5)
        plt.plot(backupConData,        color='k',alpha=1,linewidth=0.5)
        plt.plot(elecCoupleConData,    color='k',alpha=1,linewidth=0.5)
        plt.plot(CHPCovData,           color='k',alpha=1,linewidth=0.5)
        plt.plot(CovData,              color='k',alpha=1,linewidth=0.5)
        # Plot fill inbetween lines
        plt.fill_between(range(length), np.zeros(length), storageConData,
                         label='Storage',
                         color='skyblue') # Because it is a beutiful color
        plt.fill_between(range(length), storageConData, backupConData,
                         label='Backup\nGenerator',
                         color='firebrick')
        plt.fill_between(range(length), backupConData, elecCoupleConData,
                         label='Electricity\nCouple',
                         color='darkorange')
        plt.fill_between(range(length), elecCoupleConData, CHPCovData,
                         label='CHP Heat',
                         color='slategrey')
        plt.fill_between(range(length), CHPCovData, CovData,
                         label='Collected\ncovariances',
                         color='gray',
                         alpha=0.5) 
        # Axis and legend setup
        ChangePlotSetup(i,rotate,PCLegend,length,title,networktype,PC)
    
    # Setup tight layout
    plt.tight_layout()
    if rotate == True:
        if PC == 2:
            plt.subplots_adjust(wspace = 0.15)
        else:
            plt.subplots_adjust(wspace = 0.25)
    else:
        plt.subplots_adjust(wspace = 0.15)
    
    # Write title
    if title != "none":
        if rotate == True:
            plt.suptitle(title,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07  
        else:
            plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07
        
    # Show plot
    plt.show()


    return fig

#%% ChangeResponseCov

def ChangeResponseCov(lambdaMatrix,title="none", dpi=200, networktype="green", rotate=False, PC=6):
    
    lambdaSorted = SortLambdaMatrix(lambdaMatrix)

    length = len(lambdaMatrix)
    
    names = lambdaMatrix[0].columns
    
    namesLength = len(names)
    
    # subplot setup
    if rotate == True:
        col = 3
        row = 2
    else:
        col = 2
        row = 3
    
    # legend setup
    if rotate == True:
        if PC == 2 or 3: PCLegend = 1
        if PC == 6:  PCLegend = 4
    else:
        PCLegend = 5
    
    # Plotting
    if rotate == True:
        if PC == 2: fig = plt.figure(figsize=(15,6),dpi=dpi)
        else: fig = plt.figure(figsize=(10,6),dpi=dpi)
    else:
        fig = plt.figure(figsize=(10.5,8),dpi=dpi)
    
    # Color choice
    if namesLength > 21:
        color = list(mcolors._colors_full_map.values())
    else:
        color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
                 'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','tan',
                 'darkgreen','brown','fuchsia','yellow','purple','black',
                 'olivedrab','teal','gainsboro']
        
    # Print out for 6 principle components
    for i in range(PC):
        
        covData = []
        k = 0
        
        if namesLength == 28: # with 21 covariances
        
            lengthCon = 7
            
            # Data for each PC
            for j in np.arange(7,28):
                if k == 0:
                    covData.append(lambdaSorted[j].loc[i,:])
                else:
                    covData.append(lambdaSorted[j].loc[i,:]+covData[k])
                    k += 1
       
        elif namesLength == 21:  # with 15 covariances
            
            lengthCon = 6
        
            # Data for each PC
            for j in np.arange(6,21):
                if k == 0:
                    covData.append(lambdaSorted[j].loc[i,:])
                else:
                    covData.append(lambdaSorted[j].loc[i,:]+covData[k])
                    k += 1
                
    
        elif namesLength == 15:  # with 15 covariances
        
            lengthCon = 5    
        
            # Data for each PC
            for j in np.arange(5,15):
                if k == 0:
                    covData.append(lambdaSorted[j].loc[i,:])
                else:
                    covData.append(lambdaSorted[j].loc[i,:]+covData[k])
                    k += 1
            
        elif namesLength == 10:  # with 6 covariances
            
            lengthCon = 4
        
            # Data for each PC
            for j in np.arange(4,10):
                if k == 0:
                    covData.append(lambdaSorted[j].loc[i,:])
                else:
                    covData.append(lambdaSorted[j].loc[i,:]+covData[k])
                    k += 1
        
        else:
            
            assert False, "Something went very wrong"
        
        # Plot
        ax = plt.subplot(row,col,i+1)
        if i > 1:
            plt.ticklabel_format(axis='y', style='sci')
            ax.yaxis.major.formatter.set_powerlimits((0,0))
            ax.yaxis.offsetText.set_fontsize(12)
        # Plot lines
        for j in range(len(covData)):
            # Plot lines
            plt.plot(covData[j],       color='k',alpha=1,linewidth=0.5)
        
            # Plot fill inbetween lines
            if j == 0:
                plt.fill_between(range(length), np.zeros(length), covData[j],
                                 label=lambdaMatrix[0].columns[j+lengthCon],
                                 color=color[j],
                                 alpha=0.5)
            else:
               plt.fill_between(range(length), covData[j-1], covData[j],
                                 label=lambdaMatrix[0].columns[j+lengthCon],
                                 color=color[j],
                                 alpha=0.5)         
    
            
        # Axis and legend setup
        ChangePlotSetup(i,rotate,PCLegend,length,title,networktype,PC,PCLegendAmount=4)
    
    # Setup tight layout
    plt.tight_layout()
    if rotate == True:
        if PC == 2:
            plt.subplots_adjust(wspace = 0.15)
        else:
            plt.subplots_adjust(wspace = 0.25)
    else:
        plt.subplots_adjust(wspace = 0.15)
    
    # Write title
    if title != "none":
        if rotate == True:
            plt.suptitle(title,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07  
        else:
            plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07
        
    # Show plot
    plt.show()
    
    return fig

#%% CollectCovarianceTerms

def CollectCovarianceTerms(lambdaCovariance):

    # Setup forloop
    lambdaCovarianceCollected = []
    
    for j in range(len(lambdaCovariance)):
        
        # DataFrame
        df = lambdaCovariance[j]
    
        # Setup forloop
        sumLoad = np.zeros(30) # empty array
        sumRor = np.zeros(30) # empty array
        L = False # checker
        R = False # checker
        
        for i in df.columns: # Loop with all names
            
            if i[:4] == "Load": # if name has Load in it
                # add value
                sumLoad += df[i].values 
                
                # delete column from dataframe
                df.drop(i, axis=1, inplace=True)
        
                # Confirm if there was any Load
                L = True
        
            if i[:3] == "RoR": # if name has RoR in it
                # add value
                sumRor += df[i].values 
                
                # delete column from dataframe
                df.drop(i, axis=1, inplace=True)
                
                # Confirm if there was any Ror 
                R = True
        
        # add back into dataframe
        if L == True:
            df.insert(0,"Load covariance",sumLoad)
        if R == True:
            df["RoR covariance"] = sumRor
        
        # Collect terms
        lambdaCovarianceCollected.append(df)
        
    return lambdaCovarianceCollected

#%% ChangeCovariance

def ChangeCovariance(lambdaMatrix,title="none", collectTerms=True,dpi=200,networktype="green",rotate=False,PC=6):

    # Calculate lambda over constraints
    lambdaMatrixSum = []
    for j in range(6):
        temp = []
        for k in range(len(lambdaMatrix)):
            temp.append(lambdaMatrix[k].sum(axis=1)[j])
        lambdaMatrixSum.append(temp)    

    if collectTerms == True:
        lambdaMatrix = CollectCovarianceTerms(lambdaMatrix)

    # sort matrix
    lambdaSorted = SortLambdaMatrix(lambdaMatrix)
    
    # determin length (used to determin constrain type)
    length = len(lambdaMatrix)
    
    # as there are no contributer set it to zero
    lengthCon = 0
    
    # List of covariance names
    names = lambdaMatrix[0].columns
    namesLength = len(names)
    
    # subplot setup
    if rotate == True:
        col = 3
        row = 2
    else:
        col = 2
        row = 3
    
    # legend setup
    if rotate == True:
        if PC == 2 or 3: PCLegend = 1
        if PC == 6:  PCLegend = 4
    else:
        PCLegend = 5
    
    # Plotting
    if rotate == True:
        if PC == 2: fig = plt.figure(figsize=(15,6),dpi=dpi)
        else: fig = plt.figure(figsize=(10,6),dpi=dpi)
    else:
        fig = plt.figure(figsize=(10.5,8),dpi=dpi)
    
    # Color choice
    if namesLength > 21:
        color = list(mcolors._colors_full_map.values())
    else:
        color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
                 'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','tan',
                 'darkgreen','brown','fuchsia','yellow','purple','black',
                 'olivedrab','teal','gainsboro']
    
    # Print out for 6 principle components
    for i in range(PC):
        
        # create data matrix
        covData = []
        
        # add data
        for j in range(len(lambdaMatrix[0].T)):
        
            if j == 0:
                covData.append(lambdaSorted[j].loc[i,:])
            else:
                covData.append(lambdaSorted[j].loc[i,:]+covData[j-1])
        
        # Plot
        ax = plt.subplot(row,col,i+1)
        if i > 1:
            plt.ticklabel_format(axis='y', style='sci')
            ax.yaxis.major.formatter.set_powerlimits((0,0))
            ax.yaxis.offsetText.set_fontsize(12)
        # Plot lines
        plt.plot(lambdaMatrixSum[i],color='k',alpha=1,linewidth=2,linestyle="dashed",label="$\lambda_{k}$")
        for j in range(len(covData)):
            # Plot lines
            plt.plot(covData[j],       color='k',alpha=1,linewidth=0.5)
        
            # Plot fill inbetween lines
            if j == 0:
                plt.fill_between(range(length), np.zeros(length), covData[j],
                                 label=lambdaMatrix[0].columns[j+lengthCon],
                                 #color=color[list(color.keys())[j*11]],
                                 color=color[j],
                                 alpha=0.5)
            else:
               plt.fill_between(range(length), covData[j-1], covData[j],
                                 label=lambdaMatrix[0].columns[j+lengthCon],
                                 #color=color[list(color.keys())[j*11]],
                                 color=color[j],
                                 alpha=0.5)         
    
        
        # Axis and legend setup
        ChangePlotSetup(i,rotate,PCLegend,length,title,networktype,PC,PCLegendAmount=4)
    
    # Setup tight layout
    plt.tight_layout()
    if rotate == True:
        if PC == 2:
            plt.subplots_adjust(wspace = 0.15)
        else:
            plt.subplots_adjust(wspace = 0.25)
    else:
        plt.subplots_adjust(wspace = 0.15)
    
    # Write title
    if title != "none":
        if rotate == True:
            plt.suptitle(title,fontsize=20,x=.51,y=0.95) #,x=.51,y=1.07  
        else:
            plt.suptitle(title,fontsize=20,x=.51,y=0.92) #,x=.51,y=1.07
        
    # Show plot
    plt.show()

    return fig


#%% MatrixPlot

def CoherencePlot(dataMatrix, übertitle, title, xlabel, ylabel, noX=6, noY=6, dataRange=[0,1]):
    """
    

    Parameters
    ----------
    dataMatrix : Matrix of floating points
        matrix of choerence values to plot.
    
    übertitle : String
        Top most title of plot.
    
    title : String
        Title of plot.
   
    xlabel : String
        Label for x-axis (placed at top).
    
    ylabel : String
        Label for y-axis.
        
    noX : Integer, optional
        Number of elements to plot (x-axis). The default is 6.
    
    noY : Integer, optional
        Number of elements to plot (y-axis). The default is 6.
    
    dataRange : Array of integers, optional
        Range of values. Either 0 to 1 or -1 to 1. The default is [0,1].

    Returns
    -------
    fig : figure
        figure of coherence between two variables.

    """
    fig, ax = plt.subplots(figsize=(noX, noY-1), dpi=200)
    
    # URL to colormap: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    if dataRange[0] < 0:
        plt.imshow(dataMatrix[0:noY,0:noX], cmap=plt.cm.PRGn)
    else:
        plt.imshow(dataMatrix[0:noY,0:noX], cmap=plt.cm.Greens)
    
    
    PCX = [("PC" + str(number+1)) for number in np.arange(noX)]
    PCY = [("PC" + str(number+1)) for number in np.arange(noY)]
    ax.xaxis.tick_top()
    plt.xticks(np.arange(len(PCX)), PCX,fontsize=12)
    plt.yticks(np.arange(len(PCY)), PCY,fontsize=12)
    
    # Labels
    ax.xaxis.set_label_position('top')
    plt.xlabel(xlabel,fontsize=14)
    plt.ylabel(ylabel,fontsize=14)
    
    
    if noX == 6:
        # Works well with noX=noY=6
        plt.title(title, y=1.125)
        plt.suptitle(übertitle, fontsize=16, x=0.44, y=1.085)
    
    elif noX == 8:
        # Works well with noX=noY=8
        plt.title(title, y=1.09)
        plt.suptitle(übertitle, fontsize=16, x=0.44, y=1)
    
    else:
        plt.title(title)
        plt.suptitle(übertitle, fontsize=14, x=0.44, y=1)
    
    
    # Add text
    for i in np.arange(noY):
        for j in np.arange(noX):
            string = str(round(dataMatrix[i][j],2))
            plt.text(x=j, y=i, s=string, color='red', fontweight='bold', horizontalalignment='center', verticalalignment='center',fontsize=12)
            
    
    # Set colorbar limit
    plt.clim(dataRange[0], dataRange[1])
    
    if dataRange[0] < 0:
        plt.colorbar(ticks=[-1, 0, 1], format=tick.FormatStrFormatter('%.0f'))
    else:
        plt.colorbar(ticks=[0, 1], format=tick.FormatStrFormatter('%.0f'))
    
    
    
    plt.show()
    
    return fig

#%% CoherencePlotCombined
def CoherencePlotCombined(c1, c2, c3, xlabel, ylabel, dpi=200):
    
    """
    Parameters
    ----------
    c1 : numpy array of float
        Numpy array of values for coherence method 1.
    
    c2 : numpy array of float
        Numpy array of values for coherence method 2.
    
    c3 : numpy array of floatPE
        Numpy array of values for coherence method 3.
    
    xlabel : string
        Text for the label on the x-axis.
    
    ylabel : string
        Text for the label on the y-axis.
    
    dpi : int, optional
        DESCRIPTION. The default is 200.

    Returns
    -------
    fig : maplotlib figure
        Combined figure for the 3 coherence methods.

    """
    
    # Number of components shown across plot
    noX = 4
    noY = noX
    
    # Create the list of strings to use as x- and y-axis labels
    PCX = [("PC " + str(number+1)) for number in np.arange(noX)]
    PCY = [("PC " + str(number+1)) for number in np.arange(noY)]
    
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5), dpi=dpi)
    
    # Add coherence method 1
    ax1.imshow(c1[0:noX,0:noY], vmin=0, vmax=1, cmap=plt.cm.Greens)
    
    # Modify the ticks/labels/axis
    ax1.xaxis.tick_top()
    ax1.set_xticks(np.arange(len(PCX)))
    ax1.set_xticklabels(PCX)
    ax1.set_yticks(np.arange(len(PCY)))
    ax1.set_yticklabels(PCY)
    ax1.tick_params(axis='both', labelsize=14)
    
    # Add the values of the image with text
    for i in np.arange(noX):
        for j in np.arange(noY):
            string = str(round(c1[i][j],2))
            ax1.text(x=j, y=i, s=string, color='red', fontweight='bold', horizontalalignment='center', verticalalignment='center', size=15)
                
    
    # Add coherence method 2
    ax2.imshow(c2[0:noX,0:noY], vmin=0, vmax=1, cmap=plt.cm.Greens)
    
    # Modify the ticks/labels/axis
    ax2.xaxis.tick_top()
    ax2.xaxis.tick_top()
    ax2.set_xticks(np.arange(len(PCX)))
    ax2.set_xticklabels(PCX)
    ax2.set_yticks(np.arange(len(PCY)))
    ax2.set_yticklabels(PCY)
    ax2.tick_params(axis='both', labelsize=14)
    
    # Add the values of the image with text
    for i in np.arange(noX):
        for j in np.arange(noY):
            string = str(round(c2[i][j],2))
            ax2.text(x=j, y=i, s=string, color='red', fontweight='bold', horizontalalignment='center', verticalalignment='center', size=15)
               
    
    # Add a plot to axis 3 BEFORE the correct plot. This gives the variable "im" 
    # which have values changed to -1 and 1 to match the colorbar. 
    im = ax3.imshow(c3[0:noX,0:noY], vmin=-1, vmax=1, cmap=plt.cm.PRGn.reversed())
    
    
    # Add coherence method 3
    ax3.imshow(c3[0:noX,0:noY], vmin=-1, vmax=1, cmap=plt.cm.PRGn)
    
    # Modify the ticks/labels/axis
    ax3.xaxis.tick_top()
    ax3.xaxis.tick_top()
    ax3.set_xticks(np.arange(len(PCX)))
    ax3.set_xticklabels(PCX)
    ax3.set_yticks(np.arange(len(PCY)))
    ax3.set_yticklabels(PCY)
    ax3.tick_params(axis='both', labelsize=14)
    
    # Add the values of the image with text
    for i in np.arange(noX):
        for j in np.arange(noY):
            string = str(round(c3[i][j],2))
            ax3.text(x=j, y=i, s=string, color='red', fontweight='bold', horizontalalignment='center', verticalalignment='center', size=15)
               
    
    # Get position of each axis (plot)
    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    
    # Add axis from coordinates of the position of the figure
    cbar_ax = fig.add_axes([p0[0], p0[1]-0.08, p2[2]-p0[0], 0.05])
    
    # Turn the new axis into a colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[1, 0, -1], orientation="horizontal")
    
    # Correct the labels according to the colorscheme
    cbar.ax.set_xticklabels(['-1', '0', '1'])
    cbar.ax.tick_params(labelsize=14)
    #cbar.ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add titles to all the plots
    ax1.set_title("Coherence: $c^{(1)}$", y=1.20, fontweight='bold', size=16)
    ax2.set_title("Coherence: $c^{(2)}$", y=1.20, fontweight='bold', size=16)
    ax3.set_title("Coherence: $c^{(3)}$", y=1.20, fontweight='bold', size=16)
    
    # Set label for each axis
    ax1.set_xlabel(xlabel, size=16)
    ax1.xaxis.set_label_position('top')
    ax1.set_ylabel(ylabel, size=16)
    ax2.set_xlabel(xlabel, size=16)
    ax2.xaxis.set_label_position('top')
    ax2.set_ylabel(ylabel, size=16)
    ax3.set_xlabel(xlabel, size=16)
    ax3.xaxis.set_label_position('top')
    ax3.set_ylabel(ylabel, size=16)
    
    # Add a/b/c indicators to plots
    ax1.text(x=-1.1, y=-0.55, s="(a)", fontweight='bold', size=16)
    ax2.text(x=-1.1, y=-0.55, s="(b)", fontweight='bold', size=16)
    ax3.text(x=-1.1, y=-0.55, s="(c)", fontweight='bold', size=16)
    
    # Adjust the size between plots
    plt.subplots_adjust(wspace=0.4)

    # Show figure
    plt.show(fig)
    
    return (fig)


#%% NetworkCoherence

def NetworkCoherence(collectedMismatch,axNames,axTitle, axRange="min", title="None"):
    """
    
    
    Parameters
    ----------
    collectedMismatch : DataFrame
        either a mismatch or a NP timeseries
    axNames : array og strings
        The names of the different networks in short format
    axTitle : string
        ax names (the same for x and y)
    subTitle : string
        what type of network it is and what its coupled with
    title : string, optional
        DESCRIPTION. The default is "Coherence between networks".

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    
    # Amount of networks being analysed
    networkAmount = len(collectedMismatch)
    
    # empty mateix
    coherenceMatrix = np.zeros([networkAmount, networkAmount])

    # Coherence calculation
    for i in range(networkAmount):
        for j in range(networkAmount):
        
            # Coherence between prices and mismatch
            c1, c2, c3 = Coherence(collectedMismatch[i], collectedMismatch[j],10)
            
            coherenceMatrix[i,j] = round(np.diagonal(c2).sum(),3)
            

    #% Plotting
    fig, ax = plt.subplots(figsize=(networkAmount, networkAmount-1), dpi=200)

    # URL to colormap: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.imshow(coherenceMatrix[0:networkAmount,0:networkAmount], cmap=plt.cm.Greens)
    
    # X and Y axis ticks labels
    ax.xaxis.tick_top()
    plt.xticks(np.arange(networkAmount), axNames, fontsize=12)
    plt.yticks(np.arange(networkAmount), axNames, fontsize=12)

    # X and Y Labels
    ax.xaxis.set_label_position('top')
    plt.xlabel(axTitle, fontsize=14,fontweight='bold')
    plt.ylabel(axTitle, fontsize=14,fontweight='bold')

    # Title
    if title != "None":
        plt.title(title, fontsize=18)
        #plt.suptitle(title, fontsize=20, x=0.45, y=1)

    # Add text inside boxes
    for i in np.arange(networkAmount):
        for j in np.arange(networkAmount):
            string = str(round(coherenceMatrix[i][j],2))
            plt.text(x=j, y=i, s=string, color='red', fontsize=12, fontweight='bold', horizontalalignment='center', verticalalignment='center')
            

    # Set colorbar limit
    if axRange == "min": # Set colorbar limit relative to the lowest value
        dataRange=[(np.floor(coherenceMatrix.min()*10)/10),1]
    elif axRange == "full": # Set colorbar limit between 0 and 1
        dataRange= [0,1]
    else:
        assert False, "please choose a valid range"
        
    plt.clim(dataRange[0], dataRange[1])
    plt.colorbar(ticks=dataRange, format=tick.FormatStrFormatter('%.1f'))
    
    # Show figure
    plt.show()
    
    return fig

#%% differentNetworkCoherence

def differentNetworkCoherence(collectedMismatch,axNames,axTitle, axRange="min", title="None"):
    """
    
    
    Parameters
    ----------
    collectedMismatch : DataFrame
        either a mismatch or a NP timeseries
    axNames : array og strings
        The names of the different networks in short format
    axTitle : string
        ax names (the same for x and y)
    subTitle : string
        what type of network it is and what its coupled with
    title : string, optional
        DESCRIPTION. The default is "Coherence between networks".

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    
    # Amount of networks being analysed
    networkAmount = len(collectedMismatch[0])
    
    # empty mateix
    coherenceMatrix = np.zeros([networkAmount, networkAmount])

    # Coherence calculation
    for i in range(networkAmount):
        for j in range(networkAmount):
        
            # Coherence between prices and mismatch
            c1, c2, c3 = Coherence(collectedMismatch[0][i], collectedMismatch[1][j],10)
            
            coherenceMatrix[i,j] = round(np.diagonal(c2).sum(),3)
            

    #% Plotting
    fig, ax = plt.subplots(figsize=(networkAmount, networkAmount-1), dpi=200)

    # URL to colormap: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.imshow(coherenceMatrix[0:networkAmount,0:networkAmount], cmap=plt.cm.Greens)
    
    # X and Y axis ticks labels
    ax.xaxis.tick_top()
    plt.xticks(np.arange(networkAmount), axNames)
    plt.yticks(np.arange(networkAmount), axNames)

    # X and Y Labels
    ax.xaxis.set_label_position('top')
    plt.xlabel(axTitle[0])
    plt.ylabel(axTitle[1])

    # Title
    if title != "None":
        plt.title(title, fontsize=18)
        #plt.suptitle(title, fontsize=20, x=0.45, y=1)

    # Add text inside boxes
    for i in np.arange(networkAmount):
        for j in np.arange(networkAmount):
            string = str(round(coherenceMatrix[i][j],3))
            plt.text(x=j, y=i, s=string, color='red', fontweight='bold', horizontalalignment='center', verticalalignment='center')
            

    # Set colorbar limit
    if axRange == "min": # Set colorbar limit relative to the lowest value
        dataRange=[coherenceMatrix.min(),1]
    elif axRange == "full": # Set colorbar limit between 0 and 1
        dataRange= [0,1]
    else:
        assert False, "please choose a valid range"
        
    plt.clim(dataRange[0], dataRange[1])
    plt.colorbar(ticks=dataRange, format=tick.FormatStrFormatter('%.1f'))
    
    # Show figure
    plt.show()
    
    return fig

#%% TimeSeriesPlot

def TimeSeriesPlot(timeSeries,title,time=[0,8760],dpi=200):
    """
    

    Parameters
    ----------
    timeSeries : DataFrame
        timeseries of either mismatch or NP
    title : String
        DESCRIPTION.
    time : array with 2 values, optional
        Choose which time value to plot between. The default is [0,8760].
    dpi : float64, optional
        quality of figure. The default is 200.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """

    # Tjeck in title has Mismatch in it
    if any(x in title for x in ["mismatch", "Mismatch"]):
        label = "mismatch"
        yLabel = "MWh"
        
    # Tjeck in title has NP in it
    if any(x in title for x in ["NP", "Nodal", "nodal", "Price", "price"]):
        label = "Nodal Price"  
        yLabel = "€/MWh"

    # Line width
    if time[1]-time[0] > 1000:
        lineWidth = 0.1
    else:
        lineWidth = 1

    # Plot figure
    fig = plt.figure(figsize=(12,8),dpi=dpi)
    
    # general input
    x_fill = timeSeries.index[time[0]:time[1]]
    y1_fill = 0 

    # Subtitles
    subTitle = ["Average Europa with Max price at each hour","Spain","Denmark"]

    for i in range(3):
        # Plot
        plt.subplot(3,1,i+1)
        
        # different cases
        if i == 0: # Sum of EU
            y2 = timeSeries.mean(axis=1)[time[0]:time[1]]
            y3 = timeSeries.max(axis=1)[time[0]:time[1]]
        elif i == 1: # Spain
            y2 = timeSeries[time[0]:time[1]].filter(regex="ES").sum(axis=1)
        elif i == 2: # Denmark
            y2 = timeSeries[time[0]:time[1]].filter(regex="DK").sum(axis=1)
        
        # Save y2 value
        y2_fill = y2.values
        # plot line    
        plt.plot(y2,color='k',alpha=1,linewidth=lineWidth)
    
        # color between mismatch or NP and 0
        if label == "mismatch":
            plt.fill_between(x_fill, y1_fill, y2_fill,
                             label='Positive\nmismatch',
                             where= y2_fill >= y1_fill,
                             color='g')
            plt.fill_between(x_fill, y1_fill, y2_fill,
                             label='Negative\nmismatch',
                             where= y2_fill <= y1_fill,
                             color='r')
        elif label == "Nodal Price":
            plt.fill_between(x_fill, y1_fill, y2_fill,
                             label='Nodal Price',
                             where= y2_fill >= y1_fill,
                             color='g')
            if i == 0:
                plt.plot(y3, label="Max NP",color='k',alpha=1,linewidth=1)
        
        # Plot legend
        if i == 0:
            plt.legend(loc='upper right',bbox_to_anchor=(1,1))  
        
        # Settings
        plt.ylabel(yLabel) # Y label
        plt.grid(axis='y',alpha=0.5)
        plt.title(subTitle[i]) # Title for subplot
    
    # Add more space between plots
    fig.tight_layout()

    # Title
    plt.suptitle(title,fontsize=20,x=.51,y=1.02) #,x=.51,y=1.07  

    # Plot figures
    plt.show()

    return fig


#%% EnergyProductionBrownfield

def EnergyProductionBrownfield(path, filenames, figsize=[9,9], labelFontsize=14, bboxLoc=(1,1), ncol=1):
    """
    Parameters
    ----------
    path : String
        file path for files.
        
    filenames : List of Strings
        List of filenames for the brownfield optimization.

    Returns
    -------
    fig1 : matplotlib figure
        Figure of total electricity generation [MWh].
        
    fig2 : matplotlib figure
        Figure of total electricity storage [MWh].
        
    fig3 : matplotlib figure
        Figure of total heating generation [MWh].
        
    fig4 : matplotlib figure
        Figure of total heating storage [MWh].

    """
    
    # Electricity generating technologies:
    elecGeneration = ["wind",
                      "solar",
                      "ror",
                      "hydro",
                      "CCGT",
                      "OCGT",
                      "coal",
                      "oil",
                      "lignite",
                      "nuclear",
                      "gas CHP electric",
                      "biomass CHP electric",
                      "biomass EOP"]
    
    # Electricity storing technologies:
    elecStorage = ["H2 Fuel Cell",
                   "battery discharger",
                   "PHS",
                   "Sabatier"]
    
    
    # Heating generating technologies:
    heatGeneration = ["heat pump",
                      "cooling pump",
                      "resistive heater",
                      "gas boiler",
                      "gas CHP heat",
                      "biomass CHP heat",
                      "biomass HOP"]
    
    # Heating storing technologies:
    heatStorage = ["central water tanks discharge", 
                   "water tanks discharge"]
    
    # Create index for matrix with correct spaced years
    matrixIndex = [str(int(filenames[0][-7:-3])+i*5) for i in np.arange(len(filenames))]
    
    # Variable to store mismatch PC componentns for each network
    totalElecGen = pd.DataFrame(data=np.zeros([len(filenames), len(elecGeneration)]), index=pd.Index(matrixIndex), columns=elecGeneration)
    totalElecStore = pd.DataFrame(data=np.zeros([len(filenames), len(elecStorage)]), index=pd.Index(matrixIndex), columns=elecStorage)
    
    totalHeatGen = pd.DataFrame(data=np.zeros([len(filenames), len(heatGeneration)]), index=pd.Index(matrixIndex), columns=heatGeneration)
    totalHeatStore = pd.DataFrame(data=np.zeros([len(filenames), len(heatStorage)]), index=pd.Index(matrixIndex), columns=heatStorage)
    
    
    for i in np.arange(len(filenames)):
        
        # Import network
        network = pypsa.Network()
        network.import_from_netcdf(path + filenames[i])
        
        # Electricity generators
        for comp in elecGeneration:
            if comp in ["wind", "solar", "ror"]:
                # code
                totalElecGen[comp][i] = network.generators_t.p[[x for x in network.generators_t.p if comp in x]].sum().sum()
                
            elif comp == "hydro":
                # code
                totalElecGen[comp][i] = network.storage_units_t.p[[x for x in network.storage_units_t.p if comp in x]].sum().sum()
    
            else:
                # if comp == oil
                if comp == "oil":
                    totalElecGen[comp][i] = np.abs(network.links_t.p1[network.links_t.p1.columns[network.links_t.p1.columns.str.slice(3) == comp]]).sum().sum()
                
                # If comp != oil
                else: 
                    totalElecGen[comp][i] = np.abs(network.links_t.p1[[x for x in network.links_t.p1 if comp in x]]).sum().sum()
        
    
        # Electricity storage
        for comp in elecStorage:
             
            # Get Pumped-Hydro-Storage from network.storage_units (special case)
            if comp == "PHS":
                # Get all the values from PHS
                PHS = network.storage_units_t.p[[x for x in network.storage_units_t.p if "PHS" in x]]
                
                # Convert negative values to 0
                PHS.values[PHS.values < 0] = 0
                
                # Insert sum into the array            
                totalElecStore[comp][i] = PHS.sum().sum()
            
            else:
                # Get rest of components from network.links_t.p1
                totalElecStore[comp][i] = np.abs(network.links_t.p1[[x for x in network.links_t.p1 if comp in x]]).sum().sum()
             
        
        
        # Heating generators
        for comp in heatGeneration:
            totalHeatGen[comp][i] = np.abs(network.links_t.p1[[x for x in network.links_t.p1 if comp in x]]).sum().sum()
            
        # Heating storage
        for comp in heatStorage:
            totalHeatStore[comp][i] = np.abs(network.links_t.p1[[x for x in network.links_t.p1 if comp in x]]).sum().sum()
        
    
    # Correct for central water tanks also being included in water tanks
    totalHeatStore['water tanks discharge'] -= totalHeatStore['central water tanks discharge']
    
    # Turn M;Wh into TWh
    scale = 1e6
    totalElecGen /= scale
    totalElecStore /= scale
    totalHeatGen /= scale
    totalHeatStore /= scale
    
    
    # -------------- Plot electricity generation and storage ---------------- #
    # Edit labels to include a start capital letter
    
    # Electricity generators
    elecGenerationLabel = ["Wind",
                           "Solar PV",
                           "RoR",
                           "Hydro Resevoir",
                           "CCGT",
                           "OCGT",
                           "Coal",
                           "Oil",
                           "Lignite",
                           "Nuclear",
                           "Gas CHP Electric",
                           "Bio CHP Electric",
                           "Bio EOP"]
    
    # Electricity storages
    elecStorageLabel = ["H2 Fuel Cell",
                        "Battery",
                        "PHS",
                        "Sabatier"]
    
    # Heating generators
    heatGenerationLabel = ["Heat Pump",
                           "Cooling Pump",
                           "Resistive Heater",
                           "Gas Boiler",
                           "Gas CHP Heat",
                           "Bio CHP Heat",
                           "Bio HOP"]
    
    # Heating storages
    heatStorageLabel = ["Central Water Tanks", 
                        "Water Tanks"]
    
    # Color
    # List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    #cmap = plt.get_cmap("tab20")
    #colors = cmap(np.arange(len(elecGeneration))) 
    #        ["wind",       "solar", "ror",       "hydro",      "CCGT",          "OCGT",      "coal",    "oil",   "lignite",      "nuclear", "gas CHP electric", "biomass CHP electric", "biomass EOP"]
    colors = ["dodgerblue", "gold",  "limegreen", "aquamarine", "darkgoldenrod", "firebrick", "dimgray", "black", "darkseagreen", "bisque",  "steelblue",        "chocolate",            "mediumpurple"]
    
    fig1 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalElecGen.values.T, colors=colors, lw=2)
    plt.legend(labels=elecGenerationLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize, bbox_to_anchor=bboxLoc, ncol=ncol)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in electricity production as function of decarbonization", fontsize=14)
    plt.ylabel("Energy production [TWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    
    # --- Elec Storage ---
    #cmap = plt.get_cmap("tab10")
    #colors = cmap(np.arange(len(elecStorage))) 
    #        ["H2 Store", "Battery",    "PHS",  "Sabatier"]
    colors = ["purple",   "darkorange", "aqua", "navy"]
    
    
    fig2 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalElecStore.values.T, colors=colors, lw=2)
    plt.legend(labels=elecStorageLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in electricity supplied by storage as function of decarbonization", fontsize=14)
    plt.ylabel("Energy supplied [TWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    
    # ----------------- Plot heating generation and storage ----------------- #
    
    # Colors
    # List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    #cmap = plt.get_cmap("Set3")
    #colors = cmap(np.arange(len(heatGeneration))) # Must be equal to the number of bars plotted
    #        ["Heat Pump", "Cooling Pump", "Resistive Heater", "Gas Boiler", "Gas CHP Heat", "Bio CHP Heat", "Bio HOP"]
    colors = ["coral",     "darkkhaki",    "maroon",           "chartreuse", "indigo",       "crimson",   "mediumaquamarine"]
    
    fig3 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalHeatGen.values.T, colors=colors, lw=2)
    plt.legend(labels=heatGenerationLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize, bbox_to_anchor=bboxLoc, ncol=ncol)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in heating production as function of decarbonization", fontsize=14)
    plt.ylabel("Energy production [TWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    # --- Heat Storage ---
    #cmap = plt.get_cmap("Set2")
    #colors = cmap(np.arange(len(heatStorage))) 
    #        ["Central Water Tank", "Water Tank"]
    colors = ["tomato" , "teal"]
    
    fig4 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalHeatStore.values.T, colors=colors, lw=2)
    plt.legend(labels=heatStorageLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in heating supplied by storage as function of decarbonization", fontsize=14)
    plt.ylabel("Energy supplied [TWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    return (fig1, fig2, fig3, fig4)

#%% EnergyCapacityInstalledBrownfield

def EnergyCapacityInstalledBrownfield(path, filenames, figsize=[9,9], labelFontsize=14, bboxLoc=(1,1)):
    """
    Parameters
    ----------
    path : string
        destination of "filenames".
    
    filenames : list of strings
        List containing the filenames for the Brownfield networks.

    Returns
    -------
    fig1 : matplotlib figure
        Figure of total electricity capacity installed [MW].
    
    fig2 : matplotlib figure
        Figure of total electricity storage energy capacity [MWh].
    
    fig3 : matplotlib figure
        Figure of total heating capacity installed [MW].
    
    fig4 : matplotlib figure
        Figure of total heating storage energy capacity [MWh].

    """
    # Electricity generating technologies:
    elecGeneration = ["wind",
                      "solar",
                      "ror",
                      "hydro",
                      "CCGT",
                      "OCGT",
                      "coal",
                      "oil",
                      "lignite",
                      "nuclear",
                      "gas CHP electric",
                      "biomass CHP electric",
                      "biomass EOP"]
    
    # Electricity storing technologies:
    elecStorage = ["H2 Store",
                   "battery",
                   "PHS"]
    
    
    # Heating generating technologies:
    heatGeneration = ["heat pump",
                      "cooling pump",
                      "resistive heater",
                      "gas boiler",
                      "gas CHP heat",
                      "biomass CHP heat",
                      "biomass HOP"]
    
    # Heating storing technologies:
    heatStorage = ["central water tank", 
                   "water tank"]
    
    # Create index for matrix with correct spaced years
    matrixIndex = [str(int(filenames[0][-7:-3])+i*5) for i in np.arange(len(filenames))]
    
    # Variable to store mismatch PC componentns for each network
    totalElecGen = pd.DataFrame(data=np.zeros([len(filenames), len(elecGeneration)]), index=pd.Index(matrixIndex), columns=elecGeneration)
    totalElecStore = pd.DataFrame(data=np.zeros([len(filenames), len(elecStorage)]), index=pd.Index(matrixIndex), columns=elecStorage)
    
    totalHeatGen = pd.DataFrame(data=np.zeros([len(filenames), len(heatGeneration)]), index=pd.Index(matrixIndex), columns=heatGeneration)
    totalHeatStore = pd.DataFrame(data=np.zeros([len(filenames), len(heatStorage)]), index=pd.Index(matrixIndex), columns=heatStorage)
    
    
    for i in np.arange(len(filenames)):
        
        # Import network
        network = pypsa.Network()
        network.import_from_netcdf(path + filenames[i])
        
        # Electricity generators
        for comp in elecGeneration:
            if comp in ["wind", "solar", "ror"]:
                # code
                totalElecGen[comp][i] = network.generators.p_nom_opt[[x for x in network.generators.p_nom_opt.index if comp in x]].sum().sum()
                
            elif comp == "hydro":
                # code
                totalElecGen[comp][i] = network.storage_units.p_nom_opt[[x for x in network.storage_units.p_nom_opt.index if comp in x]].sum().sum()
    
            # code
            else:
                # if comp == oil
                if comp == "oil":
                    totalElecGen[comp][i] = network.links.p_nom_opt[network.links.p_nom_opt.index.str.slice(3) == comp].sum()
                
                # If comp != oil
                else: 
                    totalElecGen[comp][i] = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if comp in x]].sum()
    
        # Electricity storage
        for comp in elecStorage:
             
            # Get Pumped-Hydro-Storage from network.storage_units (special case)
            if comp == "PHS":
                # Get all the values from PHS
                temporary = network.storage_units.p_nom_opt * network.storage_units.max_hours.values
                PHS = temporary[[x for x in temporary.index if comp in x]]
                
                # Insert sum into the array            
                totalElecStore[comp][i] = PHS.sum().sum()
            
            else:
                # Get rest of components from network.links_t.p1
                totalElecStore[comp][i] = network.stores.e_nom_opt[[x for x in network.stores.e_nom_opt.index if comp in x]].sum().sum()
             
        
        
        # Heating generators
        for comp in heatGeneration:
            totalHeatGen[comp][i] = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if comp in x]].sum().sum()
            
        # Heating storage
        for comp in heatStorage:
            totalHeatStore[comp][i] =  network.stores.e_nom_opt[[x for x in network.stores.e_nom_opt.index if comp in x]].sum().sum()
        
    
    # Correct for central water tanks also being included in water tanks
    totalHeatStore['water tank'] -= totalHeatStore['central water tank']
    
    # Turn MW into GW
    scale = 1e3
    totalElecGen /= scale
    totalElecStore /= scale
    totalHeatGen /= scale
    totalHeatStore /= scale
    
    
    # -------------- Plot electricity generation and storage ---------------- #
    # Convert the first letter of each technology to capital letter
    
    # Elec generators
    elecGenerationLabel = ["Wind",
                           "Solar PV",
                           "RoR",
                           "Hydro Resevoir",
                           "CCGT",
                           "OCGT",
                           "Coal",
                           "Oil",
                           "Lignite",
                           "Nuclear",
                           "Gas CHP Electric",
                           "Bio CHP Electric",
                           "Bio EOP"]
    
    
    # Elec storages
    elecStorageLabel = ["H2 Store",
                        "Battery",
                        "PHS"]
        
        
    # Heat generators
    heatGenerationLabel = ["Heat Pump",
                           "Cooling Pump",
                           "Resistive Heater",
                           "Gas Boiler",
                           "Gas CHP Heat",
                           "Bio CHP Heat",
                           "Bio HOP"]
    
    # Heat storages
    heatStorageLabel = ["Central Water Tank", 
                        "Water Tank"]
    
    
    
    # Color
    # List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    #cmap = plt.get_cmap("tab20")
    #colors = cmap(np.arange(len(elecGeneration))) 
    #        ["wind",       "solar", "ror",       "hydro",      "CCGT",          "OCGT",      "coal",    "oil",   "lignite",      "nuclear", "gas CHP electric", "biomass CHP electric", "biomass EOP"]
    colors = ["dodgerblue", "gold",  "limegreen", "aquamarine", "darkgoldenrod", "firebrick", "dimgray", "black", "darkseagreen", "bisque",  "steelblue",        "chocolate",            "mediumpurple"]
    
    fig1 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalElecGen.values.T, colors=colors, lw=2)
    plt.legend(labels=elecGenerationLabel, framealpha=0.6, fontsize=labelFontsize, loc='upper left', bbox_to_anchor=bboxLoc)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in installed electricity capacity as function of decarbonization", fontsize=14)
    plt.ylabel("Generator capacity [GW]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    
    # --- Elec Storage ---
    #cmap = plt.get_cmap("tab10")
    #colors = cmap(np.arange(len(elecStorage))) 
    #        ["H2 Store", "Battery",    "PHS"]
    colors = ["purple",   "darkorange", "aqua"]
    
    fig2 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalElecStore.values.T, colors=colors, lw=2)
    plt.legend(labels=elecStorageLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in electricity storage energy capacity as function of decarbonization", fontsize=14)
    plt.ylabel("Storage capacity [GWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    
    # ----------------- Plot heating generation and storage ----------------- #
    
    # Colors
    # List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    #cmap = plt.get_cmap("Set3")
    #colors = cmap(np.arange(len(heatGeneration))) # Must be equal to the number of bars plotted
    #        ["Heat Pump", "Cooling Pump", "Resistive Heater", "Gas Boiler", "Gas CHP Heat", "Bio CHP Heat", "Bio HOP"]
    colors = ["coral",     "darkkhaki",    "maroon",           "chartreuse", "indigo",       "crimson",   "mediumaquamarine"]
    
    fig3 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalHeatGen.values.T, colors=colors, lw=2)
    plt.legend(labels=heatGenerationLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize, bbox_to_anchor=bboxLoc)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in installed heating capacity as function of decarbonization", fontsize=14)
    plt.ylabel("Generator capacity [GW]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    # --- Heat Storage ---
    #cmap = plt.get_cmap("Set2")
    #colors = cmap(np.arange(len(heatStorage))) 
    #        ["Central Water Tank", "Water Tank"]
    colors = ["tomato" , "teal"]
    
    fig4 = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(matrixIndex)), matrixIndex, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(matrixIndex)), totalHeatStore.values.T, colors=colors, lw=2)
    plt.legend(labels=heatStorageLabel, loc='upper left', framealpha=0.6, fontsize=labelFontsize)
    #plt.suptitle(filenames[0][:-8])
    #plt.title("Change in heating storage energy capacity as function of decarbonization", fontsize=14)
    plt.ylabel("Storage capacity [GWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    return (fig1, fig2, fig3, fig4)


#%% OvernightHeatCapacityInstalled
def OvernightHeatCapacityInstalled(path, filenames, constraints, loc='upper left', figsize=[9,9], rotation=0):
    """
    

    Parameters
    ----------
    path : String
        destination of "filenames".
    
    filenames : list of strings
        List containing the filenames for the Overnight networks.
    
    constraints : list of strings
        List of string containing the constraints of each Overnight network.
        len(filenames) must be equal to len(constraints).

    Returns
    -------
    fig : matplotlib figure
        Matplotlib figure of the installed heating generator capacity.

    """
    
    heatGeneration = ["solar thermal collector",
                      "heat pump",
                      "resistive heater",
                      "gas boiler",
                      "CHP heat"]
    
    # Variable to store mismatch PC componentns for each network
    totalHeatGen = pd.DataFrame(data=np.zeros([len(constraints), len(heatGeneration)]), index=pd.Index(constraints), columns=heatGeneration)

    for i in np.arange(len(filenames)):
        
        # Import network
        network = pypsa.Network(path + filenames[i])
        
        # Heating generators
        for comp in heatGeneration:
            
            if "thermal collector" in comp:
                totalHeatGen[comp][i] = network.generators.p_nom_opt[[x for x in network.generators.p_nom_opt.index if "thermal collector" in x]].sum()
            
            else:
                totalHeatGen[comp][i] = network.links.p_nom_opt[[x for x in network.links.p_nom_opt.index if comp in x]].sum()
            

    # Turn MW into GW
    scale = 1e3
    totalHeatGen /= scale


    # ----------------- Plot heating generation and storage ----------------- #
    
    heatGenerationLabel = ["Solar Thermal Collectors",
                           "Heat Pumps",
                           "Resistive Heaters",
                           "Gas Boilers",
                           "CHP Heat"]
    
    # Colors
    # List of colors: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = plt.get_cmap("Set3")
    colors = cmap(np.arange(len(heatGeneration))) # Must be equal to the number of bars plotted
       
    fig = plt.figure(figsize=figsize, dpi=200.0)
    plt.xticks(np.arange(len(constraints)), constraints, rotation=rotation, fontsize=14)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(constraints)), totalHeatGen.values.T, colors=colors, lw=2)
    plt.legend(labels=heatGenerationLabel, loc=loc, framealpha=0.9, fontsize=12)
    #plt.suptitle(filenames[0][:-13])
    #plt.title("Change in installed heating capacity as function of constraints", fontsize=14)
    #plt.xlabel("Years", fontsize=14)
    plt.ylabel("Installed capacity [GW]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')


    plt.show(all)
    
    return (fig)


#%% ElecProductionOvernight
def ElecProductionOvernight(directory, filenames, constraints, rotation=0, fontsize=12, ncol=1, figsize=[9,9], dpi=200):
    
    """
    Parameters
    ----------
    path : string
        String of the directory for the files.
        
    filenames : list of strings
        Lsit of string containing the filenames for all the included networks.
        
    constraints : list of strings
        Lsit of string containing the corresponding constraints for the included networks.
        
    figsize : list of float, optional
        figure size / aspect ratio. The default is [9,9].
        
    dpi : int, optional
        figure resolution / quality. The default is 200.

    Returns
    -------
    fig : matplotlib figure
        Figure of electricity produced by each technology for the Overnight networks.

    """
    
    # Load first network to determine what is included
    network = pypsa.Network(directory + filenames[0])
    
    # Determine if the network includes CHP or not
    if "central CHP electric" in network.links_t.p1.columns.str.slice(3).unique():
        elecGenerators = ["offwind",
                          "solar",
                          "ror",
                          "hydro",
                          "OCGT",
                          "central CHP electric"]
    
    else:
        elecGenerators = ["offwind",
                          "solar",
                          "ror",
                          "hydro",
                          "OCGT"]

    # Variable to store mismatch PC componentns for each network
    totalElecGen = pd.DataFrame(data=np.zeros([len(constraints), len(elecGenerators)]), index=pd.Index(constraints), columns=elecGenerators)
    
    # lists to store labels and colors
    labels = []
    Colors = []
    
    # Run a forloop for each network included
    for i in np.arange(len(filenames)):
        
        # Import network
        network = pypsa.Network(directory + filenames[i])
        
        # Heating generators
        for comp in elecGenerators:
            
            # Check if the curret technology is wind, solar or ROR
            if comp in network.generators_t.p.columns.str.slice(3).unique():
                
                # If the technology is wind
                if comp == "offwind":
                    totalElecGen[comp][i] = network.generators_t.p[[x for x in network.generators_t.p.columns if "wind" in x]].sum().sum()
                    labels.append("Wind")
                    Colors.append("dodgerblue")
                
                # If the technology is solar or ror
                else:
                    totalElecGen[comp][i] = network.generators_t.p[[x for x in network.generators_t.p.columns if comp in x]].sum().sum()
                    
                    if comp == "solar":
                        labels.append("Solar PV")
                        Colors.append("gold")
                        
                    if comp == "ror":
                        labels.append("RoR")
                        Colors.append("limegreen")
                    
    
        
            # Check if the technology is hydro
            elif comp in network.storage_units_t.p.columns.str.slice(3).unique():
                totalElecGen[comp][i] = network.storage_units_t.p[[x for x in network.storage_units_t.p.columns if comp in x]].sum().sum()
                labels.append(comp[0].upper() + comp[1:])
                Colors.append("aquamarine")
            
            
            # Check if the technology is OCGT or CHP
            elif comp in network.links_t.p1.columns.str.slice(3).unique():
                
                # If the technology is OCGT:
                if comp == "OCGT":
                    totalElecGen[comp][i] = np.abs(network.links_t.p1[[x for x in network.links_t.p1.columns if comp in x]]).sum().sum()
                    labels.append(comp[0].upper() + comp[1:])
                    Colors.append("firebrick")
                
                # If the technology is CHP:
                elif comp == "central CHP electric":
                    totalElecGen[comp][i] = np.abs(network.links_t.p1[[x for x in network.links_t.p1.columns if comp in x]]).sum().sum()
                    labels.append("CHP Electric")
                    Colors.append("deeppink")
    
    
    # Turn MW into TWh
    scale = 1e6
    totalElecGen /= scale
    
    
    # ------------------------------ Plot  --------------------------------- #
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.xticks(np.arange(len(constraints)), constraints, fontsize=14, rotation=rotation)
    plt.yticks(fontsize=14)
    plt.stackplot(np.arange(len(constraints)), totalElecGen.values.T, colors=Colors, lw=2)
    plt.legend(labels=labels, loc="lower right", framealpha=0.6, ncol=ncol, fontsize=fontsize)
    #plt.title("Change in electricity production as function of constraints", fontsize=14)
    #plt.xlabel("Constraint", fontsize=14)
    plt.ylabel("Electricity Production [TWh]", fontsize=14)
    plt.grid(axis='y', alpha=0.75, color='white', linestyle='-.')
    
    
    plt.show(all)
    
    return (fig)


#%% FilterPrices

def FilterPrice(prices, maxVal):
    """
    Parameters
    ----------
    prices : DataFrame
        Pandas dataframe containing .

    maxVal : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    pricesCopy = prices.copy()
    
    for name in pricesCopy.columns:
        for i in np.arange(len(pricesCopy.index)):
            
            # Check if the current value is larger than the maxVal
            if pricesCopy[name][i] > maxVal:
                
                # If i is the first element
                if i == 0:
                    position = 0
                    value = maxVal + 1
                    
                    # Replace the current value with the first value that is less than the max allowable value
                    while value > maxVal:
                        value = pricesCopy[name][i + position]
                        pricesCopy[name][i] = value
                        position +=1
                
                # If i is the last element
                elif i == (len(pricesCopy.index)-1):
                    pricesCopy[name][i] = pricesCopy[name][i-1]
            
                # Average the current element with its closest neighbouring elements
                else:
                    
                    # Forward looking
                    position = 0
                    valueForward = maxVal + 1
                    while valueForward > maxVal:
                        valueForward = pricesCopy[name][i + position]
                        position +=1
                        
                        # If the end of the array is reached
                        if i + position == (len(pricesCopy.index)-1):
                            valueForward = np.inf
                            break
                    
                    # Backward looking
                    position = 0
                    valueBackward = maxVal + 1
                    while valueBackward > maxVal:
                        valueBackward = pricesCopy[name][i - position]
                        position +=1
                        
                        # If the beginning of the array is reached
                        if i - position == 0:
                            valueBackward = np.inf
                            break
                    
                    
                    # Determine the value to insert into the array
                    value = 0
                    
                    # If the position of the array resulted in being out of bound, the value to insert is determined on only a one of them or the maxVal
                    if valueForward == np.inf and valueBackward == np.inf:
                        value = maxVal
                    
                    # If only one of the val
                    elif valueForward == np.inf:
                        value = valueBackward
                    
                    elif valueBackward == np.inf:
                        value = valueForward
                    
                    else:
                        value = (valueForward + valueBackward) / 2
                    
                    pricesCopy[name][i] = value
    return(pricesCopy)

#%% Curtailment

def Curtailment(directory, files, title, constraints = ['40%', '50%', '60%', '70%', '80%', '90%', '95%'], figsize=[6.4, 4.8], rotation=0, fontsize=12, ylim=[-0.5,20], legendLoc="upper left", dpi=200):
    """
    Parameters
    ----------
    directory : String
        Directory of files location.
        
    files : list of strings
        Name of pypsa files (.h5) to read.
        
    title : String
        Title for plot.
        
    constraints : list of strings, optional
        Name of each constraint used along the x-axis. The default is ['40%', '50%', '60%', '70%', '80%', '90%', '95%'].
        
    figsize : List of float, optional
        Size of the figure (width, height). The default is [6.4, 4.8].
    
    
    ylim : List of float, optional
        Upper and lower bound for the y-axis limitations. The default is [-2,42].
        
    legendLoc : String, optional
        Location of legend. The default is "upper left".

    Returns
    -------
    fig : matplotlib figure
        Figure displaying the curtailment.

    """
    # Preallocate lists
    
    windDispatchGenerator = []
    solarDispatchGenerator = []
    rorDispatchGenerator = []
    
    for file in files:
        
        # Load Netowork
        if file[12:16] == "elec":
            network = pypsa.Network(directory + file)
            
        else:
            network = pypsa.Network()
            network.import_from_netcdf(directory + file)
        
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

        # Generation (wind, solar, ror)
        wind = network.generators_t.p[[x for x in network.generators_t.p.columns if "wind" in x]]
        wind = wind.groupby(wind.columns.str.slice(0,2), axis=1).sum().mean(axis=0).sum()
        solar = network.generators_t.p[[x for x in network.generators_t.p.columns if "solar" in x]].mean(axis=0).sum()
        ror = network.generators_t.p[[x for x in network.generators_t.p.columns if "ror" in x]].mean(axis=0).sum()
        
        # Append to array - Generator relative
        windDispatchGenerator.append((windDispatch/wind)*100)
        solarDispatchGenerator.append((solarDispatch/solar)*100)
        rorDispatchGenerator.append((rorDispatch/ror)*100)
    
    
    # Combine all the generators to a list - Relative to generator
    totalDispatchGenerator = np.array(windDispatchGenerator) + np.array(solarDispatchGenerator) + np.array(rorDispatchGenerator)
    totalDispatchGenerator = list(totalDispatchGenerator)

    # Plot figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #plt.figure()

    plt.title(title, fontweight="bold")
    plt.yticks(ticks=np.linspace(0, ylim[1], 5).astype(int), fontsize=14)
    plt.xticks(np.arange(len(constraints)), constraints, fontsize=14, rotation=rotation)
    plt.plot(windDispatchGenerator, color="dodgerblue", marker='o', markerfacecolor="dodgerblue", markersize=5)
    plt.plot(solarDispatchGenerator, color="gold", marker='o', markerfacecolor="gold", markersize=5)
    plt.plot(rorDispatchGenerator, color="limegreen", marker='o', markerfacecolor="limegreen", markersize=5)
    plt.plot(totalDispatchGenerator, color="black", linestyle="--", marker='o', markerfacecolor="black", markersize=5)
    plt.ylabel("Curtailment [%]", fontsize=14)
    plt.ylim(ylim)
    plt.grid(alpha=0.3)
    plt.legend(["Wind", "Solar PV", "RoR", "Combined"], loc=legendLoc, fontsize=fontsize)
    
    # Restrict the number of digits on the y-axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    return (fig)

#%% CurtailmentHeat
def CurtailmentHeat(directory, files, title, constraints = ['40%', '50%', '60%', '70%', '80%', '90%', '95%'], figsize=[6.4, 4.8], dpi=200, ylim=[-0.5,35], fontsize=12, legendLoc="upper left", rotation=0):
    """
    Parameters
    ----------
    directory : String
        Directory of files location.
        
    files : list of strings
        Name of pypsa files (.h5) to read.
        
    title : String
        Title for plot.
        
    constraints : list of strings, optional
        Name of each constraint used along the x-axis. The default is ['40%', '50%', '60%', '70%', '80%', '90%', '95%'].
        
    figsize : List of float, optional
        Size of the figure (width, height). The default is [6.4, 4.8].
    
    ylim : List of float, optional
        Upper and lower bound for the y-axis limitations. The default is [-2,42].
        
    legendLoc : String, optional
        Location of legend. The default is "upper left".

    Returns
    -------
    fig : matplotlib figure
        Figure displaying the curtailment.

    """
    # Preallocate lists
    
    solColDispatchGenerator = []
    solColCentDispatchGenerator = []
    solColUrbDispatchGenerator = []
    
    for file in files:
        
        # Load Netowork
        if file[12:16] == "elec":
            network = pypsa.Network(directory + file)
            
        else:
            network = pypsa.Network()
            network.import_from_netcdf(directory + file)
        
        # Determine dispatchable energy for all generators at every hour
        dispatchable = network.generators_t.p
        
        # Determine non-dispatchable energy for all generators at every hour
        nonDispatchable = network.generators_t.p_max_pu * network.generators.p_nom_opt
        
        # Difference between dispatchable and non-dispatchable
        difference = nonDispatchable - dispatchable
        
        # Break into components and sum up the mean
        solColDispatch = difference.T["solar thermal collector" == difference.columns.str.slice(3)].mean(axis=1).sum()
        solColCentDispatch = difference.T["central solar thermal collector" == difference.columns.str.slice(3)].mean(axis=1).sum()
        solColUrbDispatch = difference.T["urban solar thermal collector" == difference.columns.str.slice(3)].mean(axis=1).sum()

        # Generation (Solar Collector, Central Solar Collector, Urban Solar Collector)
        solCol = network.generators_t.p.T["solar thermal collector" == network.generators_t.p.columns.str.slice(3)].mean(axis=1).sum()
        solColCent = network.generators_t.p.T["central solar thermal collector" == network.generators_t.p.columns.str.slice(3)].mean(axis=1).sum()
        solColUrb = network.generators_t.p.T["urban solar thermal collector" == network.generators_t.p.columns.str.slice(3)].mean(axis=1).sum()

        # Append to array - Generator relative
        # If either of the generators have a combined generation of less than 
        # 10 MWh, then the value is reduced to 0 to remove "noise"
        limit=10
        if solCol < limit:
            solColDispatchGenerator.append(0)
        else:
            solColDispatchGenerator.append((solColDispatch/solCol)*100)
        
            
        if solColCent < limit:
            solColCentDispatchGenerator.append(0)
        else:
            solColCentDispatchGenerator.append((solColCentDispatch/solColCent)*100)
        
            
        if solColUrb < limit:
            solColUrbDispatchGenerator.append(0)
        else:
            solColUrbDispatchGenerator.append((solColUrbDispatch/solColUrb)*100)
    
    # Combine all the generators to a list - Relative to generator
    totalDispatchGenerator = list(np.array(solColDispatchGenerator) + np.array(solColCentDispatchGenerator) + np.array(solColUrbDispatchGenerator))

    # Plot figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #plt.figure()

    plt.title(title, fontweight="bold")
    plt.xticks(np.arange(len(constraints)), constraints, fontsize=14, rotation=rotation)
    plt.yticks(ticks=np.linspace(0, ylim[1], 5).astype(int), fontsize=14)
    plt.plot(solColDispatchGenerator, color="seagreen", marker='o', markerfacecolor="seagreen", markersize=5)
    plt.plot(solColCentDispatchGenerator, color="orange", marker='o', markerfacecolor="orange", markersize=5)
    plt.plot(solColUrbDispatchGenerator, color="plum", marker='o', markerfacecolor="plum", markersize=5)
    plt.plot(totalDispatchGenerator, color="black", linestyle="--", marker='o', markerfacecolor="black", markersize=5)
    plt.ylabel("Curtailment [%]", fontsize=14)
    plt.ylim(ylim)
    plt.grid(alpha=0.3)
    plt.legend(["Solar Thermal Collector", "Central Solar Thermal Collector", "Urban Solar Thermal Collector", "Combined"], loc=legendLoc, fontsize=fontsize)
    
    # Restrict the number of digits on the y-axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.show(all)
    
    return (fig)




#%% PC1and2Plotter

def PC1and2Plotter(T, time_index, PC_NO, eigenValues, lambdaContribution, lambdaResponse, lambdaCovariance, depth=3, PCType="withProjection", suptitle='none', dpi=200):
    
    # Subtrack with 1
    PC_NO = [x - 1 for x in PC_NO]
    
    # Define as dataframe
    T = pd.DataFrame(data=T,index=time_index)
    
    # Average hour and day
    T_avg_hour = T.groupby(time_index.hour).mean() # Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Day
    
    # Setup subplot title
    subtitle = ["Contribution","Response","Covariance"]
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

    # plot add grid specifications
    if PCType == "withProjection":
        fig = plt.figure(figsize=(15,9),dpi=dpi)
        gs = fig.add_gridspec(8, 6)
        letter = ["(e)","(f)","(g)"]
    elif PCType == "withoutProjection":
        fig = plt.figure(figsize=(15,5),dpi=dpi)
        gs = fig.add_gridspec(5, 6)
    elif PCType == "onlyProjection":
        fig = plt.figure(figsize=(15,3),dpi=dpi)
        gs = fig.add_gridspec(3, 6)
        letter = ["(a)","(b)","(c)"]
    else:
        assert False, "wrong type given, choose from withProjection or withoutProjection"
    # open axes/subplots
    axs = []
    if PCType == "onlyProjection":
        axs.append( fig.add_subplot(gs[0:3,0:2]) )   # Contribution
        axs.append( fig.add_subplot(gs[0:3,2:4]) )   # Response
        axs.append( fig.add_subplot(gs[0:3,4:6]) )   # Covariance
        FFTCount = 0
    else:
        axs.append( fig.add_subplot(gs[0:3,0:3]) )   # Daily plot
        axs.append( fig.add_subplot(gs[0:3,3:6]) )   # Year plot
        if len(PC_NO) == 3:
            axs.append( fig.add_subplot(gs[3:5,0:2]) )   # FFT PC1
            axs.append( fig.add_subplot(gs[3:5,2:4]) )   # FFT PC2
            axs.append( fig.add_subplot(gs[3:5,4:6]) )   # FFT PC3
            FFTCount=3+2
        else:
            axs.append( fig.add_subplot(gs[3:5,0:3]) )   # FFT PC1
            axs.append( fig.add_subplot(gs[3:5,3:6]) )   # FFT PC2
            FFTCount=2+2
        if PCType == "withProjection":
            axs.append( fig.add_subplot(gs[5:8,0:2]) )   # Contribution
            axs.append( fig.add_subplot(gs[5:8,2:4]) )   # Response
            axs.append( fig.add_subplot(gs[5:8,4:6]) )   # Covariance
    
    if PCType != "onlyProjection":
        # Daily plot
        axs[0].hlines(0,-2,25 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
        for j in PC_NO: # Visible PC
            axs[0].plot(T_avg_hour[j],marker='.',color=color[j], label = '$\lambda_{'+str(j+1)+'}$ = '+str(round(eigenValues[j].sum()*100,1))+'%') # Plot
        # for j in range(6-len(PC_NO)):
        #     axs[0].plot(T_avg_hour[j+len(PC_NO)],color="k",alpha=0.1) # Plot
        axs[0].set(xlim = [-0.5,23.5],
                   xticks= range(0,24,2),
                   ylim = [-1.5,1.5])
        axs[0].set_title(label="Daily Average", fontweight="bold", size=13) # Title
        axs[0].set_xlabel('Hours', fontsize = 13) # X label
        axs[0].set_ylabel("$a_k$", fontsize = 14,rotation=0) # Y label
        axs[0].legend(loc="upper left",fontsize = 12)
        axs[0].text(-2,1.6,"(a)",fontsize=13, fontweight="bold")
        axs[0].text(-2,-2.45,"(c)",fontsize=13, fontweight="bold")
        axs[0].text(24.6,1.6,"(b)",fontsize=13, fontweight="bold")
        axs[0].text(24.6,-2.45,"(d)",fontsize=13, fontweight="bold")
        axs[0].set_yticks([-1,-0.5,0,0.5,1])
        axs[0].tick_params(axis='both',
                           labelsize=12)
        
        # Year plot
        x_ax = range(len(T_avg_day[0])) # X for year plot
        maxOffset = 0 # find max space between values
        
        for j in range(2):
            offset = T_avg_day[j].max() + abs(T_avg_day[j].min())
            if maxOffset < offset: 
                maxOffset = offset
        offsetValue = math.ceil(maxOffset)/2
        # add zero value
        axs[1].hlines(offsetValue,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
        axs[1].hlines(-offsetValue,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
        # plot PC1
        T_avg_day[0] = T_avg_day[0]+offsetValue # add offset
        T_avg_day[0].values[T_avg_day[0].values < 0] = 0 # Remove values lower than 0
        axs[1].plot(x_ax,T_avg_day[0],color=color[PC_NO[0]]) # Plot
        # plot PC2
        T_avg_day[1] = T_avg_day[1]-offsetValue
        T_avg_day[1].values[T_avg_day[1].values > 0] = 0 # Remove values higher than 0
        axs[1].plot(x_ax,T_avg_day[1],color=color[PC_NO[1]]) # Plot
        # Plot setting
        axs[1].set(#xlabel = "Day",
                   #ylabel = "$a_k$ seasonal",
                   xticks = range(0,370,50),
                   ylim = [-offsetValue*2,offsetValue*2],
                   xlim = [-10,370])
        axs[1].set_title(label="Seasonal", fontweight="bold", size=13) # Title
        axs[1].tick_params(axis='both',
                           labelsize=12)
        axs[1].hlines(0,-50,400 ,colors="k",linewidth=1,)
        axs[1].set_yticks([-(offsetValue*2)*0.75,-offsetValue,-(offsetValue*2)*0.25,0,(offsetValue*2)*0.25,offsetValue,(offsetValue*2)*0.75]) # y-axis label
        axs[1].set_yticklabels([str(-(offsetValue*2)*0.75+offsetValue),"0",str(-(offsetValue*2)*0.25+offsetValue),"",str((offsetValue*2)*0.25-offsetValue),"0",str((offsetValue*2)*0.75-offsetValue)]) # y-axis label
        #axs[1].set_xticks([0,31,59,90,120,151,181,212,243,273,304,334]) # y-axis label (at 1st in the month)
        axs[1].set_xticks(np.array([0,31,59,90,120,151,181,212,243,273,304,334])+14) # y-axis label (at 14th in the month)
        axs[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]) # y-axis label
        
        
        # FFT (both PC1 and PC2)
        for j in range(len(PC_NO)):
            # Calculate FFT
            freq=np.fft.fftfreq(len(T[j]))  
            FFT=np.fft.fft(T[j])
            FFT[0]=0
            FFT=abs(FFT)/max(abs(FFT))
            # Plot graph
            axs[j+2].plot(1/freq,FFT,color=color[PC_NO[j]])
            axs[j+2].set(xscale = 'log')
            axs[j+2].tick_params(axis='both',
                               labelsize=12)
            axs[j+2].set_title(label="Fourier Power Spectra", fontweight="bold", size=13) # Title
            axs[j+2].set_xlabel('Hours', fontsize = 13) # X label
            # Lines and text
            axs[j+2].vlines(12,0,1 ,colors="k", linestyles="dotted",linewidth=2) # 1/2 day
            axs[j+2].vlines(24,0,1 ,colors="k", linestyles="dotted",linewidth=2) # day
            axs[j+2].vlines(24*7,0,1 ,colors="k", linestyles="dotted",linewidth=2) # week
            axs[j+2].vlines(24*30,0,1 ,colors="k", linestyles="dotted",linewidth=2) # month
            axs[j+2].vlines(24*365,0,1 ,colors="k", linestyles="dotted",linewidth=2) # year
            axs[j+2].text(9.8,1,"1/2 Day",   ha='center', va="top", fontsize=12, rotation=90)
            axs[j+2].text(20,1,"Day",        ha='center', va="top", fontsize=12, rotation=90)
            axs[j+2].text(20*7,1,"Week",     ha='center', va="top", fontsize=12, rotation=90)
            axs[j+2].text(20*7*4,1,"Month",  ha='center', va="top", fontsize=12, rotation=90)
            axs[j+2].text(20*365,1,"Year",   ha='center', va="top", fontsize=12, rotation=90)
        
    if PCType != "withoutProjection":
        # projection
        for k in range(3):
            # Choose which type of plot
            if k == 0:
                lambdaCollected = lambdaContribution
                ylabel = 'Influance [%]'
            elif k == 1:
                lambdaCollected = lambdaResponse
                ylabel = ""
            elif k == 2:
                lambdaCollected = lambdaCovariance
                ylabel = ""
            # Find the highest values
            highestCollected = []
            for j in range(len(PC_NO)):
                highest = abs(lambdaCollected.iloc[j,:]).sort_values(ascending=False)[0:depth]
                highest = lambdaCollected[highest.index].iloc[j].sort_values(ascending=False)[0:depth] # Sort by value
                highestCollected.append(highest)
            # Counter for loop
            counter = 0
            for j in PC_NO:
                # Create a zero matrix for the percentage values
                percent = np.zeros([depth])
                # Loop to calculate and plot the different values    
                for i in range(depth):
                    # absolute percentage
                    percent[i] = lambdaCollected[highestCollected[j].index[i]][j]/eigenValues[j]*100
                    # Plot
                    if i == 0:
                        axs[k+FFTCount].bar(counter,percent[i],color=color[j], label = '$\lambda_{'+str(j+1)+'}$ = '+str(round(eigenValues[j].sum()*100,1))+'%')
                    else:
                        axs[k+FFTCount].bar(counter,percent[i],color=color[j])
                    # Insert text into bar
                    if percent[i] > 0:
                        if percent[i] >= 100:
                            v = 10
                        else:
                            v = percent[i]+10
                    else:
                            v = 10
                    axs[k+FFTCount].text(x=counter,y=v,s=str(round(float(percent[i]),1))+'%',ha='center',size=12,rotation='vertical')
                    # Count up
                    counter += 1
            # x axis label
            xLabel = []
            for j in PC_NO:
                xLabel += list(highestCollected[j].index)
            
            
            
            # General plot settings   
            axs[k+FFTCount].set_xticks(np.arange(0,depth*len(PC_NO)))
            axs[k+FFTCount].set_xticklabels(xLabel,rotation=90,fontsize=12)
            axs[k+FFTCount].set(ylabel=ylabel,
                         ylim = [-60,160],
                         title = subtitle[k])
            axs[k+FFTCount].tick_params(axis='both',
                               labelsize=12)
            axs[k+FFTCount].set_title(label=subtitle[k], fontweight="bold", size=13) # Title
            axs[k+FFTCount].set_ylabel(ylabel, fontsize = 12) # Y label
            axs[k+FFTCount].grid(axis='y',alpha=0.5)
            axs[k+FFTCount].text(-1.6,165,letter[k],fontsize=13, fontweight="bold")
    if PCType == "onlyProjection":
        axs[k+FFTCount].legend()
  
    # Space between subplot
    plt.subplots_adjust(wspace=0.4, hspace=5)
    #plt.tight_layout()

    # Title
    if suptitle != 'none':
        if PCType == "withProjection":
            plt.suptitle(suptitle, fontsize=20,x=.5,y=0.95)
        elif PCType == "onlyProjection":
            plt.suptitle(suptitle, fontsize=20,x=.5,y=1.05)
        else:
            plt.suptitle(suptitle, fontsize=20,x=.5,y=1.0)

    plt.show(all)
    
    return fig

#%% PriceEvolution

def PriceEvolution(meanPrice, quantileMeanPrice, quantileMinPrice, networktype="green", figsize=[5,4], fontsize=12, dpi=200, title="none"):
    
    # Amount of data
    N = len(meanPrice)
    
    # Colors
    color = ["tab:blue","tab:green"]
    
    # alpha
    alpha = [0.2,0.4,0.2]
    
    # empty data
    datas = []
       
    # create dataframe
    meanPrice = pd.DataFrame(meanPrice, columns=["min","mean"])
    quantileMeanPrice = pd.DataFrame(quantileMeanPrice, columns=["0.05","0.25","0.75","0.95"])
    quantileMinPrice = pd.DataFrame(quantileMinPrice, columns=["0.05","0.25","0.75","0.95"])
    
    # label
    if meanPrice["min"].sum() < 1:
        label = ["Mean","Min"]
        meanPrice = meanPrice.drop("min",axis=1)
    else:
        label = label = ["Mean","Min"]
    
    # create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # plot data
    for i, data in enumerate(reversed(list(meanPrice.columns))):

        # Plot quantiles
        plt.plot(meanPrice[data],color[i],linewidth=2,label=label[i],marker='o', markersize=3)
        for j, k in enumerate(list(quantileMeanPrice.columns)):
            plt.plot(quantileMeanPrice[k],color=color[i],alpha=0) # only there to fix legend "best"
        
            if j != 0:
                if i == 0:
                    quantile1 = quantileMeanPrice[quantileMeanPrice.columns[j-1]]
                    quantile2 = quantileMeanPrice[quantileMeanPrice.columns[j]]
                elif i == 1:
                    quantile1 = quantileMinPrice[quantileMinPrice.columns[j-1]]
                    quantile2 = quantileMinPrice[quantileMinPrice.columns[j]]
                plt.fill_between(range(N), quantile1, quantile2,
                                 color=color[i],
                                 alpha=alpha[j-1])
        for l in range(N): # quantile lines
            for k in range(3):
                if i == 0:
                    plt.plot([l,l],[quantileMeanPrice.iloc[l][k],quantileMeanPrice.iloc[l][k+1]],
                              color=color[i],
                              alpha=alpha[k]+0.1,
                              linestyle=(0,(2,2)))
                elif i == 1:
                    plt.plot([l,l],[quantileMinPrice.iloc[l][k],quantileMinPrice.iloc[l][k+1]],
                              color=color[i],
                              alpha=alpha[k]+0.1,
                              linestyle=(0,(2,2)))
    
    # plot setup
    plt.ylabel("Price [€/MWh]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(ymin=-5)
    #plt.ylim([20,120])
    plt.grid(alpha=0.15)
    plt.legend(loc="best", fontsize=fontsize)
    
    if networktype == "green":
        if N == 5:
            plt.xticks(np.arange(N),['Zero', 'Current', '2x Current', '4x Current', '6x Current'], fontsize=fontsize, rotation=-17.5)
        else:
            plt.xticks(np.arange(N),['40%', '50%', '60%', '70%', '80%', '90%', '95%'], fontsize=fontsize)
    elif networktype == "brown":
        plt.xticks(np.arange(N),["2020","2025","2030","2035","2040","2045","2050"], fontsize=fontsize)
    else: 
        assert False, "choose either green og brown as type"
    
    # title
    if title != "none":
        plt.title(title)
    
    # Show plot
    plt.show(all)
    
    return fig

#%%

def FFTseasonPlot(T, time_index, varianceExplained, PC_NO=-1, PC_amount=6, title="none",dpi=200):
    
    # Setup for timeseries
    T = pd.DataFrame(data=T,index=time_index) # Define as dataframe
    T_avg_hour = T.groupby(time_index.hour).mean() # Average Hour
    T_avg_day = T.groupby([time_index.month,time_index.day]).mean() # Average Day
    
    # Setup alpha for seasonal plot
    alpha = np.zeros(PC_amount)+0.1 # Generate alpha values 
    alpha[PC_NO-1] = +1 # give full alpha value to the intersting PC
    
    # Colorpallet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # Create plot
    fig = plt.figure(figsize=(12,3),dpi=dpi) # Create figure
    
    # FFT plot
    plt.subplot(1,3,1)
    freq=np.fft.fftfreq(len(T[PC_NO-1]))
    FFT=np.fft.fft(T[PC_NO-1])
    FFT[0]=0
    FFT=abs(FFT)/max(abs(FFT))
    # Plot graph
    plt.plot(1/freq,FFT,color=color[PC_NO-1])
    plt.xscale("log")
    plt.xlabel('Hours', fontsize=16)
    plt.title("Fourier Power Spectra", fontsize=16)
    # Lines and text
    plt.vlines(12,0,1 ,     colors="k", linestyles="dotted",linewidth=2) # 1/2 day
    plt.vlines(24,0,1 ,     colors="k", linestyles="dotted",linewidth=2) # day
    plt.vlines(24*7,0,1 ,   colors="k", linestyles="dotted",linewidth=2) # week
    plt.vlines(24*30,0,1 ,  colors="k", linestyles="dotted",linewidth=2) # month
    plt.vlines(24*365,0,1 , colors="k", linestyles="dotted",linewidth=2) # year
    plt.text(9.4,1,"1/2 Day",   ha='center', va="top", fontsize=12, rotation=90)
    plt.text(18,1,"Day",        ha='center', va="top", fontsize=12, rotation=90)
    plt.text(19*7,1,"Week",     ha='center', va="top", fontsize=12, rotation=90)
    plt.text(19*7*4,1,"Month",  ha='center', va="top", fontsize=12, rotation=90)
    plt.text(19*365,1,"Year",   ha='center', va="top", fontsize=12, rotation=90)
    plt.xticks(fontsize=12)
    plt.text(0.52,1.085,"(a)", fontsize=16, fontweight='bold')
    
    # Daily plot
    plt.subplot(1,3,2)
    for j in range(PC_amount):
        label = "$\lambda_" + str(j+1) + "$"
        plt.plot(T_avg_hour[j],label=label,marker='.',alpha=alpha[j],color=color[j])
    #subplot setup
    plt.xticks(ticks=range(0,24,2))
    plt.ylim([-1.25,1.4])
    plt.xlabel("Hours", fontsize=16)
    plt.ylabel("$a_k$", fontsize=16)
    plt.title("Daily Average", fontsize=16)
    plt.xticks(fontsize=12)
    plt.text(-3.7,1.45,"(b)", fontsize=16, fontweight='bold')
    
    x_ax = range(len(T_avg_day[0]))
    
    # Year plot
    plt.subplot(1,3,3)
    for j in range(PC_amount):
        if alpha[j] >= 1:
            label = "$\lambda_" + str(j+1) + "$ = " + str(round(varianceExplained[j],1))+"%"
            plt.plot(x_ax,T_avg_day[j],label=label,alpha=alpha[j],color=color[j])
        else:
            plt.plot(x_ax,T_avg_day[j],alpha=alpha[j],color=color[j])
    #subplot setup
    plt.legend(loc='upper right', fontsize=14)#,bbox_to_anchor=(1,1))
    plt.ylim([-1.99,1.99])
    plt.title("Seasonal Average", fontsize=16)
    plt.ylabel("$a_k$", fontsize=16)
    plt.text(-55,2.1,"(c)", fontsize=16, fontweight='bold')
    plt.xticks(np.array([0,31,59,90,120,151,181,212,243,273,304,334])+14,["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=12, rotation=90)
      
    # subplot setup
    plt.tight_layout(0.3)
    
    # Title
    if title != "none":
        plt.suptitle(title,fontsize=18,x=.5,y=1.02) #,x=.51,y=1.07)
    
    # Shows the generated plot
    plt.show(all)
    
    
    return fig    
    