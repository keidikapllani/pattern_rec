# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:13:46 2018

@author: Jacopo
"""
import random as rnd
from scipy import io

def split_load(ratio):
    '''
    Function to generate test and train sets keeping in class ratios
    '''
    data = io.loadmat('face.mat')
    data['X']
    # Images
    # N: number of images
    # D: number of pixels
    X = data['X']  # shape: [D x N]
    y = data['l']  # shape: [1 x N]
    
    test_id = []
    train_id = []
    pool = [i for i in range(0,10)]
    for c in range(0,52):
        _inclass_id = rnd.sample(pool,int(ratio*10))
        
        _train_id = [x + c*10 for x in _inclass_id]
  
        _test_id =[]
        for i in range(0,10):
            if i in set(_inclass_id):
                None
            else:
                _test_id.append(i+ c*10)
            
        
        test_id += _test_id
        train_id += _train_id
   
    print(test_id)    
    x_train = X[:,train_id]
    y_train = y[:,train_id]
    x_test = X[:,test_id]
    y_test = y[:,test_id]
    return x_train,y_train,x_test,y_test