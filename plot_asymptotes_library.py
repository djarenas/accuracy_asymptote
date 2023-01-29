import matplotlib.pyplot as plt
import numpy as np

import asymptote_functions_library as afl

def plot_asymptote(xarray, yarray, limits, ax_labels,
                   graph_filename = [], fit = "none"):
    """
    Purpose: Plot (x,y) data with the option of fitting linear or asymptotes;
    option to save to file.

    Parameters
    ----------
    xarray : Collection (list, numpy array, ...) of Floats
        x-values
    yarray : Collection (list, numpy array, ...) of Floats
        y-values
    limits : List of Floats
        [x left-limit, x right-limit], [y left-limit, y right-limit]
    ax_labels : List of Strings
        [x-axis label, y-axis label]
    graph_filename : String, optional
        JPG filename. If empty, graph is not saved to file. The default is [].
    fit : String, optional
        . The default is "none".

    Returns
    -------
    None.

    """
    #Check that a valid option was chosen for fit
    if fit not in ["none", "asymptote", "linear"]:
        fit = "none"
    
    #Figure size
    plt.figure(figsize=(20,12), dpi = 400)
    
    #Limits and ticks
    x0 = limits[0][0]
    x1 = limits[0][1]
    y0 = limits[1][0]
    y1 = limits[1][1]
    plt.ylim(y0 - (y1-y0)/20, y1 + (y1-y0)/20)
    plt.xlim(x0, x1)
    plt.yticks(np.arange(y0, y1 + (y1-y0)/10, (y1-y0)/10), fontsize = 40)
    plt.xticks(range(x0,x1,2), fontsize = 40)

    #Labels
    plt.xlabel(ax_labels[0], fontsize = 50)
    plt.ylabel(ax_labels[1], fontsize = 50)
    
    #Add the scatter points to the plot
    plt.scatter(xarray, yarray, s = 200, c = "b")
    
    #Add fit
    
    #Asymptote option for the fit: y = Lambda(1 - exp(-x/gamma))
    if fit == "asymptote":
        
        #Get fit parameters.
        lamb, gamma = afl.fit_to_asymp(xarray, yarray)
        
        #Sort the x-array. Helps the fit line look cleaner.
        xarray2 = np.sort(xarray)
        
        #Get the y-values predicted by the fit at each x
        fitted = np.array(list([afl.calc_asympt_fx(x, lamb, gamma) 
                                for x in xarray2]))
                    
        #Add a vertical line at 2*gamma
        plt.axvline(x = 2*gamma, color = 'r', ls = 'dotted'  )
        
        #Add the fit to the plot
        plt.plot(xarray2, fitted, color = 'r', linewidth = 10) 
    
    if fit == "linear":
        
        #Get fit parameters.
        fp = afl.fit_to_line(xarray, yarray)
        
        #Sort the x-array. Helps the fit line look cleaner.
        xarray2 = np.sort(xarray)

        #Get the y-values predicted by the fit at each x
        fitted = np.array(list([fp[0][0] + fp[1][0]*x for x in xarray2]))
        
        #Add the fit to the plot
        plt.plot(xarray2, fitted, color = 'r', linewidth = 10) 

    #Save to file if there was a provided filename.
    if graph_filename:
        plt.savefig(graph_filename, format = 'jpg', 
                    dpi = 200,  bbox_inches='tight')
        
    #Show the plot. 
    #This must be done after the option to save the file; anything after
    #plt.show may create a new figure. 
    plt.show()
    

def plot_histogram(rlist, alpha, xlabel):
    """
    Purpose: Take a list of values and plot a histogram. Plot vertical 
    lines for percentiles speficied by alpha.

    Parameters
    ----------
    rlist : List of Floats
        Collection of results for which to plot histogram
    alpha : Float
        T1E desired when calculating and plotting the percentile interval.
    xlabel : String
        Label for the x-axis of the histogram.

    Returns
    -------
    None.

    """

    #Calculate median, lower, and upper percentiles
    perc = afl.get_percentiles_from_list(rlist,alpha)
    
    #Prepare plot histogram
    plt.hist(rlist, bins = 40)
    
    #Vertical lines for median, lower, and upper percentile
    plt.axvline(perc[0], color = 'r', ls = 'dotted'  )
    plt.axvline(perc[1], color = 'r', ls = 'dotted'  )
    plt.axvline(perc[2], color = 'r', ls = 'dotted'  )
    
    #Axes labels
    plt.xlabel(xlabel)
    plt.ylabel("counts")
    
    #Save and show
    plt.savefig("histogram_" + xlabel +".jpg", format = 'jpg', 
                dpi = 200,  bbox_inches='tight')
    plt.show()