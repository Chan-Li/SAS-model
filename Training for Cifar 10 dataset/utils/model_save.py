#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
def para():
    print("class name: model_save(path)")
class model_save(object):
    def __init__(self,path):
        self.path=path
    def para(self):
        print("main function: model_s(net)")
    def model_l(self):
        with open(self.path, 'rb') as file:
            model=pickle.load(file)
        return model
    def model_s(self,netx):
        with open(self.path, 'wb') as file:
            pickle.dump(netx, file)
     

