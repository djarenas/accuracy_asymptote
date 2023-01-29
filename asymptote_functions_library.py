import numpy as np
import pandas as pd
import scipy.stats

from scipy.optimize import curve_fit 

import plot_asymptotes_library as pal

def fit_to_line(xarray, yarray, alpha = 0.05):
    """
    Purpose: Fit (x,y) data using linear regression; return intercept, slope,
    and their confidence intervals. Confidence intervals are based on desired 
    type 1 error. 

    Parameters
    ----------
    xarray : Collection (list, numpy array, ...) of Floats
        x-values
    yarray : Collection (list, numpy array, ...) of Floats
        y-values
    alpha : Float
        Desired Type 1 Error for the confidence intervals, the default is 0.05.

    Returns
    -------
    list
        [[y0, y0_lower, y0_upper],[slope, slope_lower, slope_upper], p_value]

    """

    #For confidence intervals, calculate Z-score based on desired T1E
    z_score = scipy.stats.norm.ppf(1 - alpha/2)

    #Linear regression; returns slope, intercept, and their standard errors
    r = scipy.stats.linregress(xarray,yarray)

    #Get estimated parameters
    y0 = r[1]
    slope = r[0]
    p_value = r.pvalue

    #Calculate the ends of the confidence intervals
    y0_upper = y0 + z_score*r.intercept_stderr
    y0_lower = y0 - z_score*r.intercept_stderr
    slope_lower = slope - 1.96*r.stderr
    slope_upper = slope + 1.96*r.stderr

    #Return information on the [intercept], [slope], and p value    
    return [[y0, y0_lower, y0_upper],
            [slope, slope_lower, slope_upper], 
            p_value]
    

def calc_asympt_fx(x, lamb, gamma):
    """
    Purpose: Calculate f(x) = lambda*(1- exp(-x/gamma))

    Parameters
    ----------
    x : Float
        x-value
    lamb : Float
        "Lambda": Asymptote value as x approaches infinity
    gamma : Float
        Decay parameter; same units as x

    Returns
    -------
    Float
        f(x)

    """
    
    return lamb*(1 - np.exp(-x/gamma))


def fit_to_asymp(xarray, yarray):
    """
    Purpose: Fit (x,y) values to an asymptote function. 
    Return lambda and gamma. 

    Parameters
    ----------
    xarray : Collection (list, numpy array, ...) of Floats
        x values
    yarray : Collection (list, numpy array, ...) of Floats
        y values

    Returns
    -------
    list
        [lambda, gamma]

    """
    
    parameters, covariance = curve_fit(calc_asympt_fx, xarray, yarray)
    
    return [parameters[0], parameters[1]]


def fit_to_asymp_get_CI(xarray, yarray, alpha = 0.05):
    """
    Purpose: Purpose: Fit (x,y) values to an asymptote function. 
    Return lambda, gamma, and their confidence intervals.

    Parameters
    ----------
    xarray : Collection (list, numpy array, ...) of Floats
        x-values
    yarray : Collection (list, numpy array, ...) of Floats
        y-values
    alpha : Float, optional
        Desired Type 1 Error for the confidence intervals. The default is 0.05.

    Returns
    -------
    list of lists
        [lambda, lambda lower, lambda upper], 
        [gamma, gamma lower, gamma upper]

    """
       
    #For confidence intervals, calculate Z-score based on desired T1E
    z_score = scipy.stats.norm.ppf(1 - alpha/2)

    #Curve fit outputs parameters and covariance  
    popt, pcov = curve_fit(calc_asympt_fx, xarray, yarray)    
    
    #Get relative error from covariance
    sigma = np.sqrt(np.diag(pcov))
    
    #Calculate confidence intervals for lambda and gamma
    lambCI = popt[0], popt[0] - z_score*sigma[0], popt[0] + z_score*sigma[0]
    gammaCI = popt[1], popt[1] - z_score*sigma[1], popt[1] + z_score*sigma[1]
    
    return [lambCI, gammaCI]


def fit_asympt_bootstrap(xarray, yarray, simuls, plot_each = False):
    """
    Purpose: Perform bootstrap simulations on (x,y) data; return the 
    asymptote parameters for every simulation.

    Parameters
    ----------
    xarray : Collection (list, numpy array, ...) of Floats
        x-values  
    yarray : Collection (list, numpy array, ...) of Floats
        y-values  
    simuls : Integer  
        Number of bootstrap simulations
    plot_each : Boolean, optional
        Option whether to plot each bootstrap and fit. The default is False.

    Returns
    -------
    list of lists
        Lists of each estimand from each bootstrap simulation    
        [lambda results], [gamma results]

    """
    
    #N: Number of data points
    N = len(xarray)
        
    #Create a Pandas dataframe to take advantage of pandas' sample-function.
    xarray = np.array(xarray)
    yarray = np.array(yarray)
    df = pd.DataFrame({'x': xarray, 'y':yarray})
    
    #Collection to keep the results from each simulation
    lambda_c = []
    gamma_c = []
    
    #For each bootstrap simulation
    for s in range(0,simuls):
        
        #Randomly draw (with replacement) N data points
        sampled = df.sample(n = N, replace = True)
        
        #Fit to asymptote
        r = fit_to_asymp(sampled["x"], sampled["y"])
        
        #Update the collections of results
        lambda_c.append(r[0])
        gamma_c.append(r[1])
        
        #Plot option
        if plot_each:
            pal.plot_asymptote(sampled["x"], sampled["y"], 
                            limits = [[0,18],[0,1]], 
                            ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                            fit="none",
                            graph_filename = "bootstrap.jpg")
            pal.plot_asymptote(sampled["x"], sampled["y"], 
                            limits = [[0,18],[0,1]], 
                            ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                            fit="asymptote",
                            graph_filename = "bootstrap_fitted.jpg")
        
    return [lambda_c, gamma_c]


def get_percentiles_from_list(rlist, alpha = 0.05):
    """
    Purpose: Take a list of values, sort, and get percentile values.

    Parameters
    ----------
    rlist : List of Floats
        
    alpha: Float, optional
        Desired Type 1 Error for the confidence intervals. The default is 0.05.
        

    Returns
    -------
    list of Floats
        [median, lowerbound, upperbound]

    """
    
    #Number of values in the list
    n = len(rlist)
    
    #Sort the list
    rlist = np.sort(np.array(rlist))
    
    #Calculate percentiles
    median = rlist[int(0.5*n)]
    lowerbound = rlist[int(alpha/2*n)]
    upperbound = rlist[int((1-alpha/2)*n)]
    
    return [median, lowerbound, upperbound]
    

def simulate_binomial_error(prob, n):
    """
    Purpose: Take an observed probability, from n observations -> 
    simulate a new probability with error from the binomial distribution.

    Parameters
    ----------
    prob : Float
        Probability (0 to 1)
    n : Integer
        Number of observations from which prob was calculated.

    Returns
    -------
    prob_new : Float
        Probability with error simulated from the binomial distribution.

    """
    prob_new = np.random.normal(loc = prob, scale = (prob*(1-prob)/n)**0.5)
    
    return prob_new 
             

def monteCarlo_asymptotes_with_error(xarray, lamb, gamma, n, 
                                     simuls, plot_each=False):
    """
    Purpose: Obtain a collection of asymptote fit parameters from 
    Monte Carlo simulations. In each simulation, error from the binomial
    distribution is added to a perfect asympote at each of the specified 
    x-values. Return fit parameters from each simulation.

    Parameters
    ----------
    xarray : Collection (list, numpy array, ...) of Floats
        x-values at which you want to simulate each point of the asymptote.
    lamb : Float
        "Lambda": Asymptote value as x approaches infinity
    gamma : Float
        Decay parameter; same units as x
    n : Integer
        Number of observations from which the probabilities were calculated.
    simuls : Integer  
        Number of bootstrap simulations
    plot_each : Boolean, optional
        Option whether to plot each bootstrap and fit. The default is False.

    Returns
    -------
    list of lists (Floats)
        [simulation results for lambda], [simulations results for gamma]

    """
    #Collections for the results of each MC simulation
    lamb_c = []
    gamma_c = []
    
    #Calculate the errorless asymptote, at each x, given the lamb and gamma.
    vectorized =  np.vectorize(calc_asympt_fx)    
    yarray = vectorized(xarray, lamb, gamma)

    #Vectorize the function that adds binomial error.
    vectorized_binomial = np.vectorize(simulate_binomial_error)

    #For each MC simulation...
    for s in range(0,simuls):
        #Simulate binomial-distribution error for each y point.
        yMC = vectorized_binomial(yarray, n)
        
        #Fit the (x,y) data to an asymptote
        r = fit_to_asymp(xarray, yMC)
        
        #Plot option 
        if plot_each:
            pal.plot_asymptote(xarray, yMC, 
                            limits = [[0,18],[0,1]], 
                            ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                            fit="none",
                            graph_filename = "output_perfect.jpg")
            pal.plot_asymptote(xarray, yMC, 
                            limits = [[0,18],[0,1]], 
                            ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                            fit="asymptote",
                            graph_filename = "output_perfect.jpg") 
        
        #Update collections of results.
        lamb_c.append(r[0])
        gamma_c.append(r[1])
        
    return [lamb_c, gamma_c]


def simulate_T1E(simuls, xarray, yarray = [],
                 n = 1, simulation = "binomialMC", 
                 fit = "linear", plot_each = False):
    """
    Purpose: Simulate type 1 error. Data can be simulated by either
    using the binomial distribution ("binomialMC") or random drawing 
    with replacement from experimental data (bootstrap).

    Parameters
    ----------
    simuls : Integer
        Number of simulations to run
    xarray : Collection (list, numpy array, ...) of Floats
        x-values at which to simulate
    yarray : Collection (list, numpy array, ...) of Floats, optional
        If "bootstrap" simulation chosen, y values from which to draw
        randomly. The default is [].
    n : Integer, optional
        If "binomialMC" is chosen, the number of observations to use
        in the simulation of binomial error. The default is 1.
    simulation : String, optional
        Two options: "binomialMC" and "bootstrap". The default is "binomialMC".
    fit : String, optional
        Two options: "linear" and "asymptote" fit. The default is "linear".
    plot_each : Boolean, optional
        If true, plots every simulation and its fit. The default is False.

    Returns
    -------
    Float
        Type 1 Error

    """
    
    #If an invalid option was selected for simulation type...
    if simulation not in ["binomialMC", "bootstrap"]:
        print("Error in simulate_T1E. For simulation, \
              choose either: binomialMC or bootstrap")
        return
    
    #If an invalid option was selected for fit...
    if fit not in ["linear", "asymptote"]:
        print("Error in simulate_T1E. For fit, \
              choose either: linear or asymptote")
        return
    
    #Number of data points and average y-value
    n_points = len(xarray)    
    av_y = np.mean(yarray)
    
    #Keep track of the number of positive results.
    pos = 0
    
    #For each simulation...
    for s in range(0,simuls):
        #----------------------------------------------------------------------
        #Simulate the data using specified method
        #----------------------------------------------------------------------
        if simulation == "binomialMC":
            #Make y values array, all equal to the average-y of the input
            simulated = np.repeat(av_y, n_points)
            
            #Simulate the error for each using the binomial distribution
            vectorized =  np.vectorize(simulate_binomial_error)    
            simulated = vectorized(simulated, n)
            
        if simulation == "bootstrap":
            if len(yarray) == 0:
                print("Error in simulate_T1E. yarray must not be empty")
                return
            
            #Make a dataframe to take advantadge of sample
            df = pd.DataFrame({'x': xarray, 'y':yarray})
            simulated = df.sample(n = n_points, replace = True)
            
            #Control the type by turning it into a numpy array
            simulated = np.array(simulated.y)

        #----------------------------------------------------------------------
        #Fit using specified method
        #----------------------------------------------------------------------
        if fit == "linear":

            fline = fit_to_line(xarray, simulated)

            #A positive result if both lower and upper bound are the same sign
            if fline[1][1]*fline[1][2] > 0:
                pos = pos + 1
        
        if fit == "asymptote":
            fasymp = fit_to_asymp_get_CI(xarray, simulated)
            
            #Positive result if the lower bound is above zero
            if fasymp[1][1] > 0:
                pos = pos + 1
        
        #----------------------------------------------------------------------
        #If plotting option was chosen
        #----------------------------------------------------------------------
        if plot_each:
            if fit == "linear": 
                pal.plot_asymptote(xarray, simulated, 
                                limits = [[0,18],[0,1]], 
                                ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                                fit="none",
                                graph_filename = "output_none.jpg")

                pal.plot_asymptote(xarray, simulated, 
                                limits = [[0,18],[0,1]], 
                                ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                                fit="linear",
                                graph_filename = "output_linear.jpg")
             
            if fit == "asymptote":
                 pal.plot_asymptote(xarray, simulated, 
                                 limits = [[0,18],[0,1]], 
                                 ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                                 fit="none",
                                 graph_filename = "output_none.jpg")

                 pal.plot_asymptote(xarray, simulated, 
                                 limits = [[0,18],[0,1]], 
                                 ax_labels = ["Area (mm$^{2}$)", "Accuracy"],
                                 fit="asymptote",
                                 graph_filename = "output_linear.jpg") 
    #Done with simulations
    
    #Return the ratio of positive results to total simulations
    return pos/simuls