import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# from lmfit import Model
from lmfit import Parameters, Parameter, report_fit, Minimizer
from lmfit import minimize, fit_report,conf_interval, ci_report


class Fit:
    def __init__(self, model_func, params, data, method_global, method_local):
        self.model_func = model_func
        self.params_population = params
        self.data = data
        self.method_global = method_global
        self.method_local = method_local
        self.result_parameters = None

    def fit_data(self):

        
        minimizer = Minimizer(self.model_func, self.params_population, fcn_args = (self.data,)) 
        if self.method_global == 'leastsq':
            self.result_parameters = minimizer.minimize(method = self.method_global)
        else: 
            self.result_parameters = minimizer.minimize(method = self.method_global, **{'local': self.method_local}) 
        return(self.result_parameters)



        

# Functions to minimize ------------------------------------------------------------------------------------------------------------------

## NTCP - Neg Log Likelihood

def NegLogLikelihood(values, outcomes, TD50, m):

        LL=list()

        for value_cur,curOutcome in zip(values, outcomes):

            ntcp = CalcNTCP(value_cur, TD50, m)
            LL_Cur = (curOutcome*np.log(ntcp))+((1-curOutcome)*np.log(1-ntcp)) # Seppenwoolde et al
          
            if np.isnan(LL_Cur):
                print (ntcp, value_cur,curOutcome)
            LL.append(LL_Cur)

        negLL=-np.sum(LL)

        return negLL

def CalcNTCP(value,TD50,m):

    t=(value-TD50)/(m*TD50)
    NTCP = 0.5 + 0.5 * sp.special.erf(t/np.sqrt(2))
    if NTCP<=0.0:
        NTCP=0.0000001
    elif NTCP>=1.0:
        NTCP=1.0-0.0000001
    return NTCP


def minimizeNLL(params, data):
    TD50 = params['TD50'].value
    m = params['m'].value
    
    values=data['values']
    outcomes=data['outcomes']

    output = NegLogLikelihood(values, outcomes, TD50, m)
    return output

# Plot the data and the fitted curve
def plot_fitNTCP(result, data):

    x=data['values'].to_numpy()
    y=data['outcomes'].to_numpy()
    
    plt.scatter(x, y, label='Data')
    plt.plot(np.sort(x), logistic_regression(np.sort(x), **result.params.valuesdict()), label='Best Fit', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

## Logistic regression 

# Define the logistic function
def logistic_regression(x, b0, b1):
    
    value = 1 / (1 + np.exp(-(b0 + b1 * x)))
    
    if type(value) is np.ndarray:
        # to prevent errors with neg_log_likelihood
        value[value >= 1] = 1 - 0.0000001
        value[value <= -1] = -1 + 0.0000001
        value[value == 0] = 0.0000001
    return value

### - least squares


def logistic_regression_residuals(params, data):
    b0 = params['b0']
    b1 = params['b1']
    
    x=data['values'].to_numpy()
    y=data['outcomes'].to_numpy()
    
    # Calculate predicted probabilities
    y_pred = logistic_regression(x, b0, b1)
    return y_pred - y



### - Neg Log Likelihood


def logistic_regression_NLL(params, data):
    b0 = params['b0']
    b1 = params['b1']
    
    x=data['values'].to_numpy()
    y=data['outcomes'].to_numpy()
    
    # Calculate predicted probabilities
    y_pred = logistic_regression(x, b0, b1)
    
    # Calculate negative log-likelihood
    neg_log_likelihood = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return neg_log_likelihood


# Plot logistic regression

# Plot the data and the fitted curve
def plot_fit(result, data):

    x=data['values'].to_numpy()
    y=data['outcomes'].to_numpy()
    
    plt.scatter(x, y, label='Data')
    plt.plot(np.sort(x), logistic_regression(np.sort(x), **result.params.valuesdict()), label='Best Fit', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    