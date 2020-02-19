''' In this file, you will utilize two parameters degree and include_bias.
    Reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PolynomialFeatures():
    
    def __init__(self, degree=2,include_bias=True):
        """
        Inputs:
        param degree : (int) max degree of polynomial features
        param include_bias : (boolean) specifies wheter to include bias term in returned feature array.
        """
        assert(degree>0)
        
        self.degree = degree
        self.include_bias = include_bias

    
    def transform(self,X):
        """
        Transform data to polynomial features
        Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. 
        For example, if an input sample is  np.array([a, b]), the degree-2 polynomial features with "include_bias=True" are [1, a, b, a^2, ab, b^2].
        
        Inputs:
        param X : (np.array) Dataset to be transformed
        
        Outputs:
        returns (np.array) Tranformed dataset.
        """
        X1 = X[:]

        if(len(X1.shape)==1):
            X1 = np.array([X1])
        
        result = X1[:]
        if(self.include_bias):
            intercept = np.ones((X1.shape[0], 1))
            result = np.concatenate((intercept, result), axis=1)

        for d in range(2,self.degree+1):
            result = np.concatenate((result, np.power(X1,d)), axis=1)
        
        return result

    
        
        
        
        
        
        
        
        
    
                
                
