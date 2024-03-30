import numpy as np
import pandas as pd
from scipy.stats import norm

def vuong(m1, m2, dof1 = 2, dof2 = 2):
    # being m1 and m2 two data frames with the results of the models

    # first check if the patients are in order in the df
    m1_patient = m1['PatNR']
    m2_patient = m2['PatNR']
    if any(m1_patient != m2_patient):
        raise ValueError("Models appear to have rows ordered differently.")

    # Extract predicted probabilities for both models
    # Extract the outcoms for both models
    m1y = m1['y']
    m2y = m2['y']
    m1n = len(m1y)
    m2n = len(m2y)
    
    if m1n == 0 or m2n == 0:
        raise ValueError("Could not extract dependent variables from models.")
    
    if m1n != m2n:
        raise ValueError("Models appear to have different numbers of observations.\n"
                         "Model 1 has {} observations.\n"
                         "Model 2 has {} observations.".format(m1n, m2n))
    
    if np.any(m1y != m2y):
        raise ValueError("Models appear to have different values on dependent variables.")
        
        
    m1['neg_y_pred'] = 1 - m1['y_pred']
    m2['neg_y_pred'] = 1 - m2['y_pred']
  
    p1= m1[['neg_y_pred', 'y_pred']]
    p2= m2[['neg_y_pred', 'y_pred']]
    
    p1.columns = ['0', '1']
    p2.columns = ['0', '1']

    
    if not np.all(p1.columns == p2.columns):
        raise ValueError("Models appear to have different values on dependent variables.")
    
#     which_col = np.where(m1y == p1.columns)[0]
#     which_col2 = np.where(m2y == p2.columns)[0]
    
    which_col = [list(p1.columns).index(str(y))  for y in m1y]
    which_col2 = [list(p1.columns).index(str(y))  for y in m2y]
    
    if not np.all(which_col == which_col2):
        raise ValueError("Models appear to have different values on dependent variables.")
    
    m1p = np.zeros(m1n)
    m2p = np.zeros(m2n)
    
    for i in range(m1n):
        m1p[i] = p1.iloc[i, which_col[i]]  # pick off correct predicted probability for observed y
        m2p[i] = p2.iloc[i, which_col[i]]
    
    # Gather degrees of freedom
    k1 = dof1
    k2 = dof2
    
    lm1p = np.log(m1p)
    lm2p = np.log(m2p)
    m = lm1p - lm2p  # vector of log likelihood ratios (diffs of log probabilities)
    
    bad1 = np.isnan(lm1p) | np.isinf(lm1p)
    bad2 = np.isnan(lm2p) | np.isinf(lm2p)
    bad3 = np.isnan(m) | np.isinf(m)
    bad = np.logical_or(bad1, np.logical_or(bad2, bad3))
    neff = np.sum(~bad)
    
    if np.any(bad):
        print("NA or numerical zeros or ones encountered in fitted probabilities")
        print("dropping these {} cases, but proceed with caution".format(np.sum(bad)))
    
    aic_factor = (k1 - k2) / neff
    bic_factor = (k1 - k2) / (2 * neff) * np.log(neff)
    
    # Compute test statistics
    v = np.zeros(3)
    arg1 = np.column_stack([m[~bad], m[~bad], m[~bad]])
    arg2 = np.column_stack([np.zeros(neff), aic_factor * np.ones(neff), bic_factor * np.ones(neff)])
    
    num = arg1 - arg2
    s = np.std(num, axis=0)
    numsum = np.sum(num, axis=0)
    v = numsum / (s * np.sqrt(neff))  # Vuong
    
#     v = pd.DataFrame(v)
#     v.columns = ["Vuong z-statistic"]
#     v['Test'] = ["Raw","AIC-corrected","BIC-corrected"]
#     v = v[["Test", "Vuong z-statistic"]]
    # Determine p-values
    pval = np.zeros(3)
    msg = [""] * 3
    for j in range(3):
        if v[j] > 0:
            pval[j] = 1 - norm.cdf(v[j])
            msg[j] = "model1 > model2"
        else:
            pval[j] = norm.cdf(v[j])
            msg[j] = "model2 > model1"
    out_data = {
        "Test":["Raw","AIC-corrected","BIC-corrected"],
        "Vuong z-statistic":v,
        "H_A":msg,
        "p-value":pval,
    }        
    out = pd.DataFrame(out_data)
    out = out[['Test', "Vuong z-statistic", "H_A", "p-value"]]
    
    print("Vuong Non-Nested Hypothesis Test-Statistic:")
    print("(test-statistic is asymptotically distributed N(0,1) under the")
    print(" null that the models are indistinguishible)")
    print("-------------------------------------------------------------")
    print(out)
