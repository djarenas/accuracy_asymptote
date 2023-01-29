# accuracy_asymptote
Functions to analyze accuracy/effectiveness versus time/size as an asymptote. Asymptote function is f(x) = lambda*(1 - exp(-x/gamma)). Where lambda is the limit as x approaches infinity and gamma is a decay parameter that dictates how the function decreases at smaller x. 

Python-functions included are for fitting the data to either asymptote or straight lines, calculate confidence intervals by residuals or by bootstrap. Simulate asymptote function with error from the binomial distribution. Simulate null-hypothesis data and estimate type 1 error. 

Last updated: 01/28/23

