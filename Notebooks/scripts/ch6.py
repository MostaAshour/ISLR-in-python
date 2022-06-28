import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import itertools
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as MSE


def best_subset(X, y, data):
    """
    Fit all possible models that contain exactly k predictors.
    """
    # get all model results
    model_subsets = []
    for k in range(1, len(X)+1):
        tic = time.time()
        
        # List all available predictors that contain exactly k predictors
        X_combos = itertools.combinations(list(X), k)

        # Fit all models accumulating Residual Sum of Squares (RSS)
        models = []
        for X_label in X_combos:
            # Parse patsy formula
            X_f = ' + '.join(X_label)
            f   = '{} ~ {}'.format(y, X_f)

            # Fit model
            model = smf.ols(formula=f, data=data).fit()
            models.append((f, model))

        model_subsets.append(models)
        toc = time.time()
        print(f'Progess: k = {k}, done, time: {round(toc-tic, 4)} s')
    return model_subsets


def forward_stepwise(X, y, data):
    """Perform forward stepwise selection as described in this book page 207.
    Using BIC for getting the best model obtained.
    inputs:
    X: names of x column in list
    y: name of y in list
    data: the dataframe contains the dataset of X, y
    Returns list with best model (name, statsmodel) for for each n"""
    
    # get all model results
    model_subsets = []
    
    X_f = []
    best_models = []
    p_all  = X

    for i in range(len(p_all)):
        models = []
        for i in p_all:
            f = '{} ~ {}'.format(y, ' + '.join(X_f + [i]))

            # Fit model
            model = smf.ols(formula=f, data=data).fit()
            models.append((f, model))
        
        model_subsets.append(models)
        best_models.append(min_bic(models))
        X_f = [best_models[-1][0].split(' ~ ')[-1]]
        p_all = p_all.drop(X_f[0].split(" + ")[-1])

    return model_subsets, best_models


def backward_stepwise(X, y, data):
    """Perform backward stepwise selection as described in this book page 209.
    Using BIC for getting the best model obtained.
    inputs:
    X: names of x column in list
    y: name of y in list
    data: the dataframe contains the dataset of X, y
    Returns list with best model (name, statsmodel) for for each n"""
    
    # get all model results
    model_subsets = []
    
    X_f = X
    best_models = []
    p_all  = X

    f = '{} ~ {}'.format(y, ' + '.join(X_f))
    model = smf.ols(formula=f, data=data).fit()
    best_models.append((f, model))
    model_subsets.append([(f, model)])
    
    for i in range(len(X_f)-1):
        models = []
        for i in X_f:
            f = '{} ~ {}'.format(y, ' + '.join(X_f.drop(i)))
            # Fit model
            model = smf.ols(formula=f, data=data).fit()
            models.append((f, model))
            
        model_subsets.append(models)
        best_models.append(min_bic(models))
        X_f = pd.Index(best_models[-1][0].split('~')[-1].split(' + '))

    return model_subsets[::-1], best_models


def min_rss(statsmodels):
    """Return model with lowest Residual Sum of Squares (RSS)"""
    return sorted(statsmodels, key=lambda tup: tup[1].ssr)[0]

def max_adjr2(statsmodels):
    """Return model with best R-squared score"""
    return sorted(statsmodels, reverse=True, key=lambda tup: tup[1].rsquared_adj)[0]

def min_bic(statsmodels):
    """Return model with best Bayes Information Criteria score"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].bic)[0]

def min_aic(statsmodels):
    """Return model with best Akaike Information Criteria score"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].aic)[0]

def scores_plot(model_subsets, figsize = (8,5)):
    '''plotting the reasults of the three scores of:
       Adjusted R-squared, 
       Bayes Information Criteria (BIC),
       Akaike's Information Criteria (AIC / C_p)
       With the best num of predictors using the statsmodel subsets'''
    
    k = np.arange(1, len(model_subsets)+1)
    
    # best adj R^2 score for each subset
    adjr2   = [max_adjr2(m)[1].rsquared_adj for m in model_subsets]
    adj_max_idx = adjr2.index(max(adjr2))

    # Plot best adjusted R-squared score for each subset
    plt.figure(figsize=figsize)
    plt.plot(k, adjr2)
    plt.plot(adj_max_idx+1, adjr2[adj_max_idx], 'rx', ms=15)
    plt.title('Adjusted R-squared')
    plt.xlabel('k')
    plt.ylabel('adjusted $R^2$')
    plt.xticks(k)
    plt.show()

    # Select best model from all the subsets
    coefs_r2 = [(max_adjr2(m)[1].rsquared_adj, max_adjr2(m)[1].params) for m in model_subsets]
    print('Model selected coefficients: \n{}'.format(max(coefs_r2)[1]))
    print('----------------------------------------------------\n')


    ### Bayes Information Criteria (BIC)
    # best BIC for each subset
    bic = [min_bic(m)[1].bic for m in model_subsets]
    bic_min_idx = bic.index(min(bic))

    # Plot best Bayes Information Criteria (BIC) score for each subset
    plt.figure(figsize=figsize)
    plt.plot(k, bic)
    plt.plot(bic_min_idx+1, bic[bic_min_idx], 'rx', ms=15)
    plt.title('Bayes Information Criteria (BIC)')
    plt.xlabel('k')
    plt.ylabel('BIC')
    plt.xticks(k)
    plt.show()

    # Select best subset
    coefs_bic = [(min_bic(m)[1].bic, min_bic(m)[1].params) for m in model_subsets]
    print('Model selected coefficients: \n{}'.format(min(coefs_bic)[1]))
    print('----------------------------------------------------\n')


    ### Akaike's Information Criteria (AIC / C_p)
    # best AIC for each subset
    aic = [min_aic(m)[1].aic for m in model_subsets]
    aic_min_idx = aic.index(min(aic))

    # Plot best Akaike Information Criteria (AIC / C_p) score for each subset
    plt.figure(figsize=figsize)
    plt.plot(k, aic)
    plt.plot(aic_min_idx+1, aic[aic_min_idx], 'rx', ms=15)
    plt.title('Akaike Information Criteria (AIC / C_p)')
    plt.xlabel('k')
    plt.ylabel('AIC')
    plt.xticks(k)
    plt.show()

    # Select best subset
    coefs_aic = [(min_aic(m)[1].aic, min_aic(m)[1].params) for m in model_subsets]
    print('Model selected coefficients: \n{}'.format(min(coefs_aic)[1]))
    
def lasso_cv(X, y, lambd, k):
    """Perform the lasso with k-fold cross validation 
    to return mean MSE scores for each fold"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = linear_model.Lasso(alpha=lambd,
                                   fit_intercept=True,
                                   normalize=False,
                                   max_iter=1000000)
        model.fit(X_train, y_train)

        # Measure MSE
        y_hat = model.predict(X_test)
        MSEs += [MSE(y_hat, y_test)]
    return MSEs

def ridge_cv(X, y, lambd, k):
    """Perform ridge regresion with 
    k-fold cross validation to return mean MSE scores for each fold"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = linear_model.Ridge(alpha=lambd,
                                   fit_intercept=True,
                                   normalize=False,
                                   max_iter=1000000,
                                  solver='cholesky')
        m = model.fit(X_train, y_train)
        
        # Measure MSE
        y_hat = m.predict(X_test)
        MSEs += [MSE(y_hat, y_test)]
    return MSEs