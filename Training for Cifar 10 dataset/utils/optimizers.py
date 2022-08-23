#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
def params():
    print("RMS_prop(),Adam(theta),Momentum(theta)")
    
class RMS_prop:
    def __init__(self):
        self.lr=0.1
        self.beta=0.9
        self.epislon=1e-8
        self.s=0
        self.t=0
        
    def initial(self):
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,eta):
        self.lr = eta
        self.t += 1
        g=gradient
        self.decay=1e-4
        self.s = self.beta*self.s + (1-self.beta)*(g*g)
        theta -= self.lr*((g/pow(self.s+self.epislon,0.5))+self.decay*theta)
        return theta

class Adam:
    def __init__(self,theta):
        self.lr=0.01
        self.beta1=0.9
        self.beta2=0.999
        self.epislon=1e-8
        self.m=[np.zeros(ms.shape) for ms in theta]
        self.s=[np.zeros(ms.shape) for ms in theta]
        self.t=0
    
    def New_theta(self,theta,gradient,eta):
        self.t += 1
        if type(eta) == list:
            self.lr = eta*1
        if type(eta) == float:
            self.lr=[eta*np.ones((theta_s.shape)) for theta_s in theta]
        self.decay=1e-4
        g=gradient*1
        theta2 = [np.zeros(ms.shape) for ms in theta]
        for l in range(len(gradient)):
            self.m[l] = self.beta1*self.m[l] + (1-self.beta1)*g[l]
            self.s[l] = self.beta2*self.s[l] + (1-self.beta2)*(g[l]*g[l])
            self.mhat = self.m[l]/(1-self.beta1**self.t)
            self.shat = self.s[l]/(1-self.beta2**self.t)
            theta2[l] = theta[l]-self.lr[l]*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay*theta[l])
        return theta2*1
class Momentum:
    def __init__(self,theta):
        self.lr=0.01
        self.epislon=1e-8
        self.velocity=(np.zeros(shape = theta.shape))
        self.theta2 = (np.zeros(shape = theta.shape))
    def New_theta(self,theta,gradient,lr):
        self.lr = lr*1
        eta = 0.9
        g=gradient*1
        self.velocity = self.velocity*eta+lr*g
        self.theta2 = theta - self.velocity
        return self.theta2

