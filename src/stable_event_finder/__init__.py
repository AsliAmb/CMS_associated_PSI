#!/usr/bin/env python
# coding: utf-8




import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.random import sample_without_replacement





"""
===================================
FUNCTIONS
===================================
"""


"""
Replaces CMS with integers
"""
def class_code(CLASS):
    if CLASS == 'CMS1':
        return 1
    elif CLASS == 'CMS2':
        return 2
    elif CLASS == 'CMS3':
        return 3
    elif CLASS == 'CMS4':
        return 4
"""
Prepares the input PSI file by first filtering off non-concordant and Unclassified samples. Then, 
replaces predictions with integers and separates the input file into sample labels and PSI.

input: file that contains CMS labels and PSI values for tumor samples
output:
    x= PSI values
    y= CMS labels {1,2,3,4}
"""    
def prep(inputfile):
    df=pd.read_csv(inputfile)
    df = df[df['prediction'] != 'non-concordant']
    df = df[df['prediction'] != 'Unclassified']
    df = df.dropna()
    df['prediction']=df['prediction'].apply(class_code) 
    y = df['prediction'].values
    x = df.drop(['prediction','case'], axis=1)
    return(x,y)

"""
Randomly selects half of the data while preserving the portion of each class within the subsample
input:
    X: PSI values (samples in rows, observations in columns)
    y: CMS labels {1,2,3,4}
    portion: portion of the data to subsampled. 
output:
    indexes for the random selection

"""
def stratified_sampling(X,y,portion=0.5,random_state=123):
    subsample_size = np.floor(X.shape[0] * portion).astype(int)
    classes, class_counts = np.unique(y, return_counts = True)
    per_class_n = (np.round((class_counts / class_counts.sum()) * subsample_size)).astype(int)
    rand_selection = []
    for i in range(len(classes)):
        class_ind = np.array([j for j, x in enumerate(y) if x == classes[i]])
        sampled_ind = sample_without_replacement(class_ind.shape[0], per_class_n[i],random_state=random_state)
        sampled_ind = class_ind[sampled_ind]
        rand_selection.append(sampled_ind)
    return(np.concatenate(rand_selection))

"""
After fitting a model selects features based on importance weights.
input:
    model : Estimator
    X: PSI values (samples in rows, observations in columns)
    y: CMS labels {1,2,3,4}
    C_value: regularization strength
output: 
    Boolean array of selected features

"""            
def _fit_model(model, X, y,C_value):
    model.C = C_value
    model.fit(X, y)
    return(SelectFromModel(estimator=model, threshold=1e-5, prefit=True).get_support())
           
            
"""
For a given number of iterations, for each iteration
fits the estimator for a given array of lasso regularization strengths and calculates for each feature
probability of selection
input:
    X: PSI values (samples in rows, observations in columns)
    y: CMS labels {1,2,3,4}
    model: estimator
    model_C: an array of lasso regularization strengths
    iterations: Number of samplings
output:
    an array of selection probabilites where number of columns equal to number of regularization parameters and 
    number of rows equal to number of features
"""    
def stable_features(X,y,model,model_C,iterations,random_state=123):
    n_samples, n_variables = X.shape
    stability_scores = np.zeros((n_variables,len(model_C)))
    for index, c_value in enumerate(model_C):
        sampled = []
        for ii in range(iterations):
            sampled.append(stratified_sampling(X,y,portion=0.5,random_state=None))
        covariates = Parallel(n_jobs=-1, verbose=False,pre_dispatch=1)(delayed(_fit_model)(model,
                             X=np.array(X)[subsample, :],
                             y=y[subsample],
                             C_value=c_value)for subsample in sampled)
        stability_scores[:, index] = np.vstack(covariates).mean(axis=0)
    return(stability_scores)           
            
"""
Filter features whose maximum selection probability is greater or equal to  a given cutoff [0-1] 
input:
    stability_scores: Output of stable_features
    X: PSI values (samples in rows, observations in columns)
    cutoff : threshold for selection probability
output:
    data frame of feature stability scores

"""
def stability_filter(stability_scores,X,cutoff):
    filtered = stability_scores.max(axis=1) >= cutoff
    scores = stability_scores[filtered].max(axis=1)
    feature_score = pd.DataFrame({"Features": X.columns[filtered],"Scores":scores})
    return(feature_score)


    
"""
Selects stable alternative splicing features that distinguish CMS
input:
    psi_input : File name that contains PSI values and CMS labels
    model_C: an array of lasso regularization strengths
    model: estimator
    iterations: number of samplings
    cutoff : threshold for selection probability
output:
    data frame of feature stability scores.

"""    
def main(psi_input,model_C,model,iterations,cutoff): 
    X,y = prep(psi_input)
    X_arr = X.to_numpy()
    stable_ = stable_features(X_arr,y,model,model_C,iterations,random_state=123)
    filtered = stability_filter(stable_,X,cutoff)
    return(filtered)
    

    
    

