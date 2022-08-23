#Import
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig

def uni_permu(a,b,direction):
    if direction ==1:
        p = np.random.permutation(len(a.T))
        return np.array((a.T[p]).T), np.array((b.T[p]).T)
    if direction == 0:
        p = np.random.permutation(len(a))
        return np.array((a[p])), np.array((b[p]))
from utils.functions import relu,drelu,softmax,divi_,mini_batch_generate,sigmoid,dsigmoid,turn_2_zero,scale
# from functions import relu,drelu,softmax,divi_,mini_batch_generate,sigmoid,dsigmoid,turn_2_zero,scale
from utils.optimizers import Adam

def sampling(m,p,v):
    m=np.array(m)
    p=np.array(p)
    v=np.array(v)
    b=[np.ones(p1.shape) for p1 in p]
    ran=np.array([random.random(size=(pi.shape)) for pi in p ])
    for i in range(0,p.shape[0]):
        for j in range(0,p.shape[1]):
            if v[i][j]==0:
                b[i][j]=(turn_2_zero((ran[i][j]-p[i][j])))*m[i][j]
            else:
                b[i][j]=(turn_2_zero((ran[i][j]-p[i][j])))*(np.random.normal(m[i][j],np.sqrt(v[i][j])))
    return b

#Import Cifar10 dataset
from PIL import Image
import os
import pickle
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y
def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def _one_hot(labels, num):
    size= labels.shape[0]
    label_one_hot = np.zeros([size, num])
    for i in range(size):
        label_one_hot[i, np.squeeze(labels[i])] = 1
    return label_one_hot
#Two ways to turn dataset into grey style
def _grayscale(a):  
    #print (a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1))
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)
def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    #rst = np.expand_dims(rst, axis=3)
    return rst.reshape(data.shape[0],-1)
#Generate data officially.
def data_generate(root):
    Xtr_ori, Ytr_ori, Xte_ori, Yte_ori=load_CIFAR10(root)
    Xtr=(Xtr_ori).reshape(50000,3072).astype('float32')
    Xte=(Xte_ori).reshape(10000,3072).astype('float32') 
    Xtr = (Xtr.astype('float32') / 255.0).T
    Xte = (Xte.astype('float32') / 255.0).T
    Ytr=_one_hot(Ytr_ori, 10).T
    Yte=_one_hot(Yte_ori, 10).T
    return Xtr,Ytr,Xte,Yte

class NeuralNetwork:
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.sigmas = [(0.01*random.random(size=(y,x))) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.pis = [np.clip(np.random.normal(0.0,0.0,(y,x)),0,1) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.messes=[((np.sqrt(2/784))*(np.random.normal(0.0,1.0,(y,x)))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        self.mu = [mess*(1-pi) for mess,pi in zip(self.messes,self.pis)]
        self.rho = [(1-pi)*(sigma+(mess*mess)) for pi,sigma,mess in zip(self.pis,self.sigmas,self.messes)]
        self.Adam_sigmas = Adam(self.sigmas)
        self.Adam_pis = Adam(self.pis)
        self.Adam_messes = Adam(self.messes)
        self.beta = 1.0
    def update_moment(self):
        self.mu = [mess*(1-pi) for mess,pi in zip(self.messes,self.pis)]
        self.rho = [(1-pi)*(sigma+(mess*mess)) for pi,sigma,mess in zip(self.pis,self.sigmas,self.messes)]    
    def w_feedforward(self,a,activate,back=False):
        process=[]
        flag=0
        zm=[]
        process=[a]
        for mess,pis,sigmas in zip(self.messes,self.pis,self.sigmas):
            ws = np.array(sampling(mess,pis,sigmas))
            flag=flag+1
            z=(np.dot(ws,a))*(1/np.sqrt(ws.shape[1]))
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = softmax_more(z)
            zm.append(z)
            process.append(a)
        if back == False:
            return process[-1]
        if back == True:
            return process,zm    
    def feedforward(self,a,activate,back=False):
        #x为输入的图片，尺寸为784*mini_batch_size
        epsilon=[]
        process=[]
        flag=0
        var=[]
        zm=[]
        process=[a]
        self.update_moment()
        for mu_s,rho_s in zip(self.mu,self.rho):          
            medicine=pow(10,-20)
            flag=flag+1
            ep=np.random.normal(0,1,(mu_s.shape[0],a.shape[1]))
            G=(np.dot(mu_s,a))
            delta=np.sqrt(np.dot((rho_s-mu_s**2),a**2)+medicine)
            z=(G+delta*ep)*(1/(np.sqrt(mu_s.shape[1])))
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = softmax(z)            
            zm.append(z)            
            process.append(a)
            epsilon.append(ep)
            var.append(delta)
        if back == False:
            return process[-1]
        if back == True:
            return process,epsilon,var,zm
        
    def evaluate(self, testdata,testlabel,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        accuracy_all = []
        data1,label1 = mini_batch_generate(500,testdata*1,testlabel*1)
        accuracy=[]
        for j in range(data1.shape[0]):
            a=self.feedforward(data1[j],activate,back=False)
            max0=np.argmax(a,axis=0)
            max1=np.argmax(label1[j],axis=0)
            accuracy.append((np.sum((max0-max1) == 0))/(data1[j].shape[1])*1)
        accuracy_all.append(np.average(accuracy)*1)
        return accuracy_all
    
   
       

    
    
    
    
    def sampling_evaluate(self, testdata,testlabel,activate):
        # 获得预测结果a:10*batch_size
        #testlabel:10*batch_size
        accuracy_all = []
        data1,label1 = mini_batch_generate(500,testdata*1,testlabel*1)
        accuracy=[]
        for j in range(data1.shape[0]):
            a=self.w_feedforward(data1[j],activate,back=False)
            max0=np.argmax(a,axis=0)
            max1=np.argmax(label1[j],axis=0)
            accuracy.append((np.sum((max0-max1) == 0))/(data1[j].shape[1])*1)
        accuracy_all.append(np.average(accuracy)*1)
        return accuracy_all
    

    
    
    def backprop(self,x,y,activate,dactivate,back=True):
        medicine=pow(10,-30)
        #x:输入：784*batch_size
        #y:输入标签：10*batch_size
        tri=[]
        out,epsi,va,zm=self.feedforward(x,activate,back=True)
        var = [(vas+medicine) for vas in va]
        nabla_sigma = [np.zeros(sigma.shape) for sigma in self.sigmas]
        nabla_pi = [np.zeros(pi.shape) for pi in self.pis]
        nabla_mess = [np.zeros(mess.shape) for mess in self.messes]
        self.update_moment()
        for l in range(1, (self.num_layers)):
            if l==1:
                tri_=(out[-1]-y)
                tri.append(tri_)
            else:
                tri_= (1/(np.sqrt(self.sizes[-l])))*dactivate(zm[-l])*(np.dot(self.mu[-l+1].T,tri[-1])\
             +np.dot(((self.rho[-l+1]-self.mu[-l+1]**2).T),(turn_2_zero(va[-l+1])\
            *tri[-1]*epsi[-l+1]/var[-l+1]))*out[-l])
                tri.append(tri_)
            nabla_mess[-l]= (1/(np.sqrt(self.sizes[-l-1])))*(np.dot(tri_,out[-l-1].T)*(1-self.pis[-l])\
            +np.dot(turn_2_zero(va[-l])*tri_*epsi[-l]/var[-l],(out[-l-1]**2).T)*self.mu[-l]*self.pis[-l])/(np.shape(x)[1])
            nabla_sigma[-l]=((1/(np.sqrt(self.sizes[-l-1])))*np.dot(turn_2_zero(va[-l])*tri_*epsi[-l]/(2*var[-l]),(out[-l-1]**2).T)*(1-self.pis[-l]))/(np.shape(x)[1])
            nabla_pi[-l]=-(1/(np.sqrt(self.sizes[-l-1])))*(np.dot(tri_,out[-l-1].T)*(self.messes[-l])\
            +np.dot(turn_2_zero(va[-l])*tri_*epsi[-l]/(2*var[-l]),(out[-l-1]**2).T)*((2*self.pis[-l]-1)*self.messes[-l]*self.messes[-l]+self.sigmas[-l]))/(np.shape(x)[1])
        return nabla_sigma,nabla_pi,nabla_mess
    
    def adam_update(self,lr,mini_batch_size,activate,dactivate,train_data_x,train_label_x):
        self.update_moment()
        data_x=train_data_x*1
        label_x=train_label_x*1
        data,label = mini_batch_generate(mini_batch_size,data_x,label_x)
        for j in range(data.shape[0]):
            self.update_moment()
            delta_nabla_sigma,delta_nabla_pi,delta_nabla_mess = self.backprop(data[j],label[j],activate,dactivate,back=True)
            self.sigmas= self.Adam_sigmas.New_theta(self.sigmas,delta_nabla_sigma,lr)
            self.sigmas = [np.maximum(0,sigma) for sigma in self.sigmas]
            self.messes= self.Adam_messes.New_theta(self.messes,delta_nabla_mess,lr)
            self.pis= self.Adam_pis.New_theta(self.pis,delta_nabla_pi,lr)
            self.pis= [np.clip(pis,0,1) for pis in self.pis]
            self.update_moment()
            print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
    
    def SGD(self,train_data,train_label,test_data,test_label,mini_batch_size,epoch,lr0,activate,dactivate):
        learning_rate=[]
        acc_all = []
        lr_=[]
        data_all=[]
        label_all=[]
        train_datat = train_data*1
        train_labelt = train_label*1
        for i in range(epoch):
            self.update_moment()
            lr = lr0
            self.adam_update(lr,mini_batch_size,activate,dactivate,train_datat,train_labelt)
            acc1 = self.evaluate(test_data,test_label,activate)
            acc_all.append(acc1*1)
            print(acc1)
        return (acc_all)




#Save data
if __name__ == '__main__':
    X_train,Y_train,X_test,Y_test=data_generate('cifar-10-batches-py')
    for i in range(1):
        net1=NeuralNetwork([3072,200,200,10])
        test_error=net1.SGD(X_train,Y_train,X_test,Y_test,mini_batch_size=200,epoch=100,lr0=0.006,relu,drelu)
        np.save('4layer-gBP-'+str(i+1)+'testdata_lr6',test_error, allow_pickle=True)
        np.save('4layer-gBP-'+str(i+1)+'sigmas_lr6',net1.sigmas, allow_pickle=True)
        np.save('4layer-gBP-'+str(i+1)+'pis_lr6',net1.pis, allow_pickle=True)
        np.save('4layer-gBP-'+str(i+1)+'mess_lr6',net1.messes, allow_pickle=True)





