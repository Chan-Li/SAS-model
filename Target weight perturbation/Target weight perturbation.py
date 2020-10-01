
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig
from PIL import Image
import load
#Import MNIST dataset
mnist=np.array(load.load_mnist(one_hot=True))
train_data = mnist[0][0][0:10000].T
train_label = mnist[0][1][0:10000].T
test_data = mnist[1][0][0:10000].T
test_label = mnist[1][1][0:10000].T
print(np.shape(train_data))
print(np.shape(train_label))
#Import Cifar dataset
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
    Xtr = np.concatenate(xs)#使变成行向量
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
def data_generate(root):
    Xtr_ori, Ytr_ori, Xte_ori, Yte_ori=load_CIFAR10(root)
    Xtr=(Xtr_ori).reshape(50000,3072).astype('float32')
    Xte=(Xte_ori).reshape(10000,3072).astype('float32')
    Xtr = (Xtr.astype('float32') / 255.0).T
    Xte = (Xte.astype('float32') / 255.0).T
    Ytr=_one_hot(Ytr_ori, 10).T
    Yte=_one_hot(Yte_ori, 10).T
    return Xtr,Ytr,Xte,Yte
X_train,Y_train,test_data,test_label=data_generate('cifar')


def relu(y):
    tmp = y.copy()
    tmp[tmp < 0] = 0
    return tmp
def turn_2_zero(x):
    b=[np.ones(x1.shape) for x1 in x]
    for i in range(0,x.shape[0]):
        b[i]=np.int64(x[i]>0)
    return b
def sampling(m,p,v):
    m=np.array(m)
    p=np.array(p)
    v=np.array(v)
    b=[np.ones(p1.shape) for p1 in p]
    ran=np.array([random.random(size=(pi.shape)) for pi in p ])
    for i in range(0,p.shape[0]):
        b[i]=(turn_2_zero((ran[i]-p[i])))*(np.random.normal(m[i],np.sqrt(v[i])))
    return b
def softmax(x):
    max1=np.max(x)
    return (np.exp(x-max1))/(np.sum(np.exp(x-max1)))
def softmax_more(x):
    soft=[]
    for i in range(x.shape[1]):
        cut=softmax(x[:,i])
        soft.append(cut)
    return np.array(soft).T

def w_feedforward(a,activate,ws):
    process=[]
    flag=0
    zm=[]
    process=[a]
    l=np.shape(ws)[0]
    for w in ws:
            
        flag=flag+1
        z=(np.dot(w,a))*(1/(np.sqrt(w.shape[1])))
            
        if (flag<(l)):
            a = activate(z)
        if (flag>=(l)):
            a = softmax_more(z)
        zm.append(z)
        process.append(a) 
    return process[-1]
def evaluate(testdata,testlabel,activate,m,p,v):
    
    value=[]
    for i in range(10):
        ws=sampling(m,p,v)
        a=w_feedforward(testdata,activate,ws)
        max1=np.argmax(a,axis=0)
        max2=np.argmax(testlabel,axis=0)
        accuracy=(np.sum((max1-max2) == 0))/(testlabel.shape[1])
        #cost=np.sum(-(testlabel)*ln(a+pow(10,-20)))/testlabel.shape[1]
        value.append((1-accuracy)*100)
    return value
def evaluate_w(testdata,testlabel,activate,w):
    value=[]
    ws=w
    a=w_feedforward(testdata,activate,ws)
    max1=np.argmax(a,axis=0)
    max2=np.argmax(testlabel,axis=0)
    accuracy=(np.sum((max1-max2) == 0))/(testlabel.shape[1])
    value.append((1-accuracy)*100)
    return value
import numpy as np
#Calculate the number of VIP and UIP weights
def VIP_amount(p,v,m,l):
    count=0
    m_amount=0
    for j in range(v[l].shape[0]):
        for k in range(v[l].shape[1]):
            if v[l][j][k]==0 and p[l][j][k]==0:
                count=count+1
                m_amount=m_amount+m[l][j][k]
    return count
def UIP_amount(p,l):
    count=0
    for j in range(p[l].shape[0]):
        for k in range(p[l].shape[1]):
            if p[l][j][k]==1:
                count=count+1
    return count
def VIP_deleting(m,p,v,l,activate,f,location=False):
    VIP_loc=[]
    VI_loc=[]
    for j in range(v[l].shape[0]):
        for k in range(v[l].shape[1]):
            if v[l][j][k]==0 and p[l][j][k]==0:
                vip1=[j,k]
                VIP_loc.append(vip1)
    f_num=int(f*len(VIP_loc)) 
    randomA=[]
    while(len(randomA)<(f_num)):
        xx=random.randint(0,len(VIP_loc))
        if xx not in randomA:
            randomA.append(xx)
            
    for i in range(f_num):
        VI_loc.append(VIP_loc[randomA[i]])
        
    pis=[np.ones(p1.shape) for p1 in p]
    for layer in range(3):
        if layer!=l:
            pis[layer]=p[layer]
        else:
            pis[layer]=p[l]*1.0
            for i in range(len(VI_loc)):
                loc=VI_loc[i]
                pis[l][loc[0]][loc[1]]=1   
    
    VIP_evaluate=evaluate(test_data,test_label,activate,m,pis,v)
    
    if location==True:
        return VIP_loc
    if location==False:
        return (np.average(VIP_evaluate))
#All weights perturbation
def deletrandom(m,p,v,l,activate,f,location=False):
    VIP_loc=[]
    for j in range(v[l].shape[0]):
        for k in range(v[l].shape[1]):
            if v[l][j][k]==0 and p[l][j][k]==0:
                vip1=[j,k]
                VIP_loc.append(vip1)
    print("validation now!")
    validation=0
    for i in range(len(VIP_loc)):
        test=VIP_loc[i]
        if not (v[l][test[0]][test[1]]==0 and p[l][test[0]][test[1]]==0):
            validation=validation+1
    if validation>0:
        print("something wrong happened!")
    else:
        print("nothing wrong, now put out")
  
    
    
    loc_ran=[]
    for i in range(len(VIP_loc)):
        a=random.randint(0,((p[l].shape[0])-1))
        b=random.randint(0,((p[l].shape[1])-1))
        loc_ran.append([a,b])
    w_or=sampling(m,p,v)
    wws=[np.zeros(w.shape) for w in w_or]
    locl=loc_ran
    pro_mat=np.random.rand(len(locl))
    f_mat=f*np.ones((len(locl)))
    f_vs=(turn_2_zero((f_mat-pro_mat)))
    for layer in range(3):
        if layer!=l:
            wws[layer]=w_or[layer]*1
        else:
            wws[layer]=w_or[l]*1
            for i in range(len(locl)):
                if f_vs[i]==1:
                    loc=locl[i]
                    wws[l][loc[0]][loc[1]]=0
    
    rand_evaluate=evaluate_w(test_data,test_label,activate,wws)
    
    if location==True:
        return loc_ran,wws
    else:
        return np.average(rand_evaluate)
def dRandom(m,p,v,l,activate,f,location=False):
    content=[]
    for i in range(10):
        a=deletrandom(m,p,v,l,activate,f,location=False)
        content.append(a)
    return np.average(content)

# Other weights perturbation
def Random2(m,p,v,l,activate,f,location=False):
    random_lo=[]
    random_loc=[]
    VIP_loc=VIP_deleting(m,p,v,l,activate,f,location=True)
    
    for j in range(v[l].shape[0]):
        for k in range(v[l].shape[1]):
            if not ((p[l][j][k]==1) or (v[l][j][k]==0 and p[l][j][k]==0)) :
                vip2=[j,k]
                random_lo.append(vip2)    
    
    f_num=int(f*len(VIP_loc))
    
    randomA=[]
    
    while(len(randomA)<(f_num)):
        xx=random.randint(0,len(random_lo))
        if xx not in randomA:
            randomA.append(xx)
            
    for i in range(f_num):
        random_loc.append(random_lo[randomA[i]])
        
   
    pis=[np.ones(p1.shape) for p1 in p]
    for layer in range(3):
        if layer!=l:
            pis[layer]=p[layer]
        else:
            pis[layer]=p[l]*1.0
            for i in range(len(random_loc)):
                loc=random_loc[i]
                pis[l][loc[0]][loc[1]]=1   
    
    random_evaluate=evaluate(test_data,test_label,activate,m,pis,v)
    
    
    if location==True:
        return random_loc
    if location==False:
        return (np.average(random_evaluate))


    
def UIP_deleting(m,p,v,l,activate,f,location=False):
    UI_loc=[]
    UIP_loc=[]
    for j in range(v[l].shape[0]):
        for k in range(v[l].shape[1]):
            if p[l][j][k]==1:
                vip1=[j,k]
                UIP_loc.append(vip1)
    
    
    
    #UIP_loc产生UIP的权重位置
    f_num=int(f*len(UIP_loc))
    #敲除的UIP个数
    randomA=[]
    while(len(randomA)<(f_num)):
        xx=random.randint(0,len(UIP_loc))
        if xx not in randomA:
            randomA.append(xx)
   
    for i in range(f_num):
        UI_loc.append(UIP_loc[randomA[i]])
    
    w_origin=sampling(m,p,v)
    w_new=[np.ones(w.shape) for w in w_origin]
    for layer in range(3):
        if layer!=l:
            w_new[layer]=w_origin[layer]
        else:
            w_new[layer]=w_origin[l]*1.0
            for i in range(len(UI_loc)):
                loc=UI_loc[i]
                w_new[l][loc[0]][loc[1]]=np.random.normal(0,1)   
    
    UIP_evaluate=evaluate_w(test_data,test_label,relu,w_new)
    
    if location==True:
        return UIP_loc
    if location==False:
        return (np.average(UIP_evaluate))
def UIP_d(m,p,v,l,activate,f,location=False):
    content=[]
    for i in range(10):
        a=UIP_deleting(m,p,v,l,activate,f,location=False)
        content.append(a)
    return np.average(content)



#Save data for the first layer.
VIP00=[]
for i in range(11):
    VIP00.append(VIP_deleting(m,p,v,0,relu,0.1*i,location=False))
print(VIP00)
    np.save("VIP.npy",VIP00,allow_pickle=True)




UIP00=[]
for i in range(11):
    UIP00.append(UIP_d(m,p,v,0,relu,0.1*i,location=False))
print(UIP00)
    np.save("UIP.npy",UIP00,allow_pickle=True)




R00=[]
for i in range(11):
    R00.append(Random2(m,p,v,0,relu,0.1*i,location=False))
print(R00)
    np.save("Ran.npy",R00,allow_pickle=True)

