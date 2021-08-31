#Copyright: Joydeep Munshi
#Date: 07/01/2021
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def predictions(X, gpr):
    mu, sigma = gpr.predict(X, return_std=True)
    mu = np.exp(mu)
    return mu


def expected_improvement(X, X_sample, Y_sample, gpr, xi):
    
    ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model. 
    Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d). Y_sample: Sample values (n x 1). 
    gpr: A GaussianProcessRegressor fitted to samples. xi: Exploitation-exploration trade-off parameter. 
    Returns: Expected improvements at points X. '''
    
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    #sigma = sigma.reshape(-1, 2)
    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.min(mu_sample)
    #print(mu_sample_opt)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei[0][0]


def propose_location(acquisition, X_sample, Y_sample, gpr, space, bounds, xi=0.01, n_restarts=500, method='L-BFGS-B'):
    
    ''' Proposes the next sampling point by optimizing the acquisition function. 
    Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
    Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
    Returns: Location of the acquisition function maximum. '''
    
    dim = X_sample.shape[1]
    min_val = 10000
    min_x = None
    c = 0
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, xi)
    
    # Find the best optimum by starting from n_restart different random points.
    for i in range(n_restarts):
        c = c+1
        params = {}
        for key in space.keys():
            params[key] = space[key]()
        x0 = np.asarray(np.asarray(list(params.values())))
        
        res = minimize(min_obj, x0=x0, bounds=bounds, method=method)
        if res.fun < min_val:
            etol = min_val - res.fun
            min_val = res.fun
            min_x = res.x 
    X_pred = min_x
    muu,sig = gpr.predict(X_pred.reshape(-1, dim),return_std=True)
            
    return X_pred,muu


def best_location(acquisition,X_sample, gpr, space, bounds, n_restarts=50000, method='L-BFGS-B'):
    
    
    dim = X_sample.shape[1]
    min_val = 10000
    min_x = None
    c = 0
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return acquisition(X.reshape(-1, dim), gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for i in range(n_restarts):
        c = c+1
        params = {}
        for key in space.keys():
            params[key] = space[key]()
        x0 = np.asarray(np.asarray(list(params.values())))
        
        res = minimize(min_obj, x0=x0, bounds=bounds, method=method)     
        if res.fun < min_val:
            etol = min_val - res.fun
            min_val = res.fun
            min_x = res.x  
    X_pred = min_x
    muu,sig = gpr.predict(X_pred.reshape(-1, dim),return_std=True)
            
    return X_pred,muu