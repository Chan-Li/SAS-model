#Import
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig

#Possible activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
def tanh(x):
    return np.tanh(x)
def dtanh(y):
    return 1.0 - y ** 2
def relu(y):
    tmp = y.copy()
    tmp[tmp < 0] = 0
    return tmp
def drelu(x):
    tmp = x.copy()
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = 0
    return tmp

#Softmax functions
def softmax(x):
    max1=np.max(x)
    return (np.exp(x-max1))/(np.sum(np.exp(x-max1)))
def softmax_test(x):
    return (np.exp(x))/(np.sum(np.exp(x)))
def softmax_more(x):
    soft=[]
    for i in range(x.shape[1]):
        cut=softmax(x[:,i])
        soft.append(cut)
    return np.array(soft).T
    
#Some defined functions for convenience in the matrix operations
def find_aver(x,k):
    n=x.shape[1]/k
    sp=np.split(x,n,axis=1)
    av=np.sum(sp,axis=0)/n
    return av

def clip_sun(x,k):
    n=x.shape[1]/k
    sp=np.split(x,n,axis=1)
    av=np.sum(sp,axis=0)
    return av
def clip_in_sum(x,k):
    n=int(x.shape[1]/k)
    a=[]
    for i in range(n):
        av=np.sum(x[:,(i*k):(i*k+k)],axis=1).reshape(x.shape[0],1)
        a.append(av)
    return np.squeeze(a,axis=2).T

def upper_bound(x,count):
    return np.repeat(x,count,axis=1)


def lower_bound(x,count):
    return np.repeat((x.T).reshape(1,(x.shape[0])*(x.shape[1])),count,axis=0)

def i_k_func(x,count):
    return np.tile(x,count)
#Some possible rate decay
def decay_rate(learning_rate,global_step,decay_steps,alpha,num_periods):
    global_step = min(global_step, decay_steps)
    l_decay=(decay_steps-global_step)/(decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(((math.pi)*2)*num_periods * (global_step / decay_steps)))
    decayed = (alpha+l_decay) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    return decayed_learning_rate 


def divi_(lr0,global_step,decay_step):
    return lr0*(0.5**((int(global_step/decay_step))))

#Some functions for sampling weights from trained parameters.
def turn_2_zero(x):
    return np.int64(x>0)


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
#Generate mini-batches.
def mini_batch_generate(mini_batch_size,traindata,trainlabel):
    n=traindata.shape[1]
    state = np.random.get_state()
    np.random.shuffle(((traindata).T))
    np.random.set_state(state)
    np.random.shuffle(((trainlabel).T))
    mini_batches = np.array([traindata[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches_labels =np.array([trainlabel[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    return mini_batches,mini_batches_labels

#Network structure
class NeuralNetwork:
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.sigmas = [(0.01*random.random(size=(y,x))) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.pis = [np.clip(np.zeros((y,x)),0,1) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.messes=[((np.random.normal(0.0,0.8,(y,x)))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
    
    
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
        epsilon=[]
        process=[]
        flag=0
        var=[]
        zm=[]
        mean=self.messes*(np.ones_like((self.pis))-self.pis)
        mean2=(self.sigmas+np.array(self.messes)*np.array(self.messes))*np.array((np.ones_like((self.pis))-self.pis))
        process=[a]
        for mea,mea2 in zip(mean,mean2):
            medicine=pow(10,-20)
            flag=flag+1
            ep=np.random.normal(0,1,(mea.shape[0],a.shape[1]))
            G_i=(1/(np.sqrt(mea.shape[1])))*np.dot(mea,a)
            delta=((1/(mea.shape[1]))*np.dot((mea2-mea*mea),a*a)+medicine)**(0.5)
    
            z=G_i+ep*delta
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = softmax_more(z)
            
            zm.append(z)
            process.append(a)
            epsilon.append(ep)
            var.append(delta)
            
        if back == False:
            return process[-1]
        if back == True:
            return process,epsilon,var,zm,mean,mean2
        
    def sampling_evaluate(self,testdata,testlabel,activate):
        a=self.w_feedforward(testdata,activate,back=False)
        max1=np.argmax(a,axis=0)
        max2=np.argmax(testlabel,axis=0)
        accuracy=(np.sum((max1-max2) == 0))/(testlabel.shape[1])
        cost=np.sum(-(testlabel)*ln(a+pow(10,-30)))/testlabel.shape[1]
        return cost, accuracy
   
       

    
    
    
    
    def evaluate(self, testdata,testlabel,activate):
        a=self.feedforward(testdata,activate,back=False)
        max1=np.argmax(a,axis=0)
        max2=np.argmax(testlabel,axis=0)
        accuracy=(np.sum((max1-max2) == 0))/(testlabel.shape[1])
        cost=np.sum(-(testlabel)*ln(a+pow(10,-30)))/testlabel.shape[1]
        return cost, accuracy
    
    def backprop(self,x,y,activate,dactivate,back=True):
        medicine=pow(10,-20)
        tri=[]
        out,epsi,va,zm,mean,mean2=self.feedforward(x,activate,back=True)
        va=va+medicine*np.ones_like((va))
        nabla_sigma = [np.zeros(sigma.shape) for sigma in self.sigmas]
        nabla_pi = [np.zeros(pi.shape) for pi in self.pis]
        nabla_mess = [np.zeros(mess.shape) for mess in self.messes]
        for l in range(1, (self.num_layers)):
            if l==1:
                tri_=(out[-1]-y)
                tri.append(tri_)
            else:
                tri_=((1/(np.sqrt(self.sizes[-l])))*(np.sum(((upper_bound(tri[-1],self.sizes[-l]))*lower_bound(dactivate(zm[-l]),self.sizes[-l+1]))*((i_k_func(mean[-l+1],x.shape[1]))+upper_bound((epsi[-l+1]/(va[-l+1])),self.sizes[-l])*lower_bound(out[-l],self.sizes[-l+1])*i_k_func((mean2[-l+1]-mean[-l+1]*mean[-l+1]),x.shape[1])),axis=0)).reshape(x.shape[1],self.sizes[-l])).T
                tri.append(tri_)
            nabla_pi[-l]=(1/(np.sqrt(self.sizes[-l-1])))*find_aver(upper_bound(tri_,self.sizes[-l-1])*((i_k_func((-1)*self.messes[-l],x.shape[1]))*lower_bound(out[-l-1],self.sizes[-l])+((upper_bound((epsi[-l]/(2*va[-l])),self.sizes[-l-1]))*lower_bound((out[-l-1]*out[-l-1]),self.sizes[-l])*i_k_func((-((self.sigmas[-l])+((self.messes[-l])**(2)))-2*(self.pis[-l]-1)*self.messes[-l]*self.messes[-l]),x.shape[1]))),self.sizes[-l-1])
            nabla_sigma[-l]=(1/(np.sqrt(self.sizes[-l-1])))*find_aver((upper_bound((tri_*epsi[-l]/(2*va[-l])),self.sizes[-l-1])*lower_bound((out[-l-1]*out[-l-1]),self.sizes[-l])*i_k_func(((1-self.pis[-l])),x.shape[1])),self.sizes[-l-1])
            nabla_mess[-l]=(1/(np.sqrt(self.sizes[-l-1])))*find_aver(upper_bound(tri_,self.sizes[-l-1])*(i_k_func((1-self.pis[-l]),x.shape[1])*lower_bound(out[-l-1],self.sizes[-l])+upper_bound((epsi[-l]/(va[-l])),self.sizes[-l-1])*lower_bound((out[-l-1])*(out[-l-1]),self.sizes[-l])*i_k_func(((mean[-l])*self.pis[-l]),x.shape[1])),self.sizes[-l-1])
            
        return nabla_sigma,nabla_pi,nabla_mess 

    

    
    
    
    def adam_mini(self,lr,mini_batch_size,activate,dactivate,traindata,trainlabel):
        data,label = mini_batch_generate(mini_batch_size,traindata,trainlabel)
        beta1=0.9
        beta2=0.999
        eps=[(pow(10,-8)*(np.ones((y,x)))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        m_sig=[(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        m_pi = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        m_me = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        s_sig = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        s_pi =[(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        s_me = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        m_hat_sig=[(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        m_hat_pi = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        m_hat_me = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        s_hat_sig = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        s_hat_pi =[(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        s_hat_me = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        update_sig = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        update_pi = [(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        update_me=[(np.zeros((y,x))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        decay=pow(10,-4)
        medicine=pow(10,-30)
        for j in range(data.shape[0]):
            delta_nabla_sigma, delta_nabla_pi,delta_nabla_mess = self.backprop(data[j],label[j],activate,dactivate,back=True)
            for l in range(0,(self.num_layers-1)):
                m_sig[l]=m_sig[l]*beta1+(1-beta1)*delta_nabla_sigma[l]
                s_sig[l] = beta2*s_sig[l]+(1-beta2)*delta_nabla_sigma[l]*delta_nabla_sigma[l]
                m_hat_sig[l] = m_sig[l]/(1-((beta1)**(j+1)))
                s_hat_sig[l] = s_sig[l]/(1-((beta2)**(j+1)))
                update_sig[l] = m_hat_sig[l]/((np.sqrt(s_hat_sig[l]+medicine*np.ones_like((s_hat_sig[l]))))+eps[l])
                m_pi[l]=m_pi[l]*beta1+(1-beta1)*delta_nabla_pi[l]
                s_pi[l] = beta2*s_pi[l]+(1-beta2)*delta_nabla_pi[l]*delta_nabla_pi[l]
                m_hat_pi[l] = m_pi[l]/(1-((beta1)**(j+1)))
                s_hat_pi[l] = s_pi[l]/(1-((beta2)**(j+1)))
                update_pi[l] = m_hat_pi[l]/((np.sqrt(s_hat_pi[l]+medicine*np.ones_like((s_hat_pi[l]))))+eps[l])
                m_me[l]=m_me[l]*beta1+(1-beta1)*delta_nabla_mess[l]
                s_me[l] = beta2*s_me[l]+(1-beta2)*delta_nabla_mess[l]*delta_nabla_mess[l]
                m_hat_me[l] = m_me[l]/(1-((beta1)**(j+1)))
                s_hat_me[l] = s_me[l]/(1-((beta2)**(j+1)))
                update_me[l] = m_hat_me[l]/((np.sqrt(s_hat_me[l]+medicine*np.ones_like((s_hat_me[l]))))+eps[l])
            self.sigmas = [np.maximum((sigma-lr*(nsigma+decay*sigma)),0) for sigma, nsigma in zip(self.sigmas, update_sig)]
            self.pis = [np.clip((pi-lr*npi),0,1) for pi, npi in zip(self.pis, update_pi)]
            self.messes = [(mess-lr*(nmess+decay*mess)) for mess, nmess in zip(self.messes, update_me)]
            




    def SGD(self,traindata,trainlabel,testdata,testlabel,mini_batch_size,epoch,lr0):
        evaluation_cost, evaluation_error = [], []
        training_cost, training_accuracy = [], []
        learning_rate=[]
        test1,label1=testdata,testlabel
        for i in range(epoch):
            lr = divi_(lr0,i,30)
            print ("Epoch %s training complete" % i)
            self.adam_mini(lr,mini_batch_size,relu,drelu,traindata,trainlabel)
            cost1,accuracy1 = self.sampling_evaluate(test1,label1,relu)
            evaluation_cost.append(cost1)
            evaluation_error.append((1-accuracy1))
            cost2,accuracy2 = self.evaluate(traindata,trainlabel,relu)
            training_cost.append(cost2)
            training_accuracy.append(accuracy2)
            print("the training Accuracy is:{} %".format((accuracy2)*100))
            print("the training cost is ",cost2)
            log=open("name4l1.txt","a")
            print("the test error is:{} %".format((1-accuracy1)*100),file=log)
            log.close()
            print("the cost is ",cost1)
        return evaluation_error            



#Save data
if __name__ == '__main__':
    X_train,Y_train,X_test,Y_test=data_generate('cifar-10-batches-py')
    for i in range(1):
        net1=NeuralNetwork([3072,200,200,10])
        test_error=net1.SGD(X_train,Y_train,X_test,Y_test,mini_batch_size=200,epoch=100,lr0=0.006)
        np.save('4layer-gBP-'+str(i+1)+'testdata_lr6',test_error, allow_pickle=True)
        np.save('4layer-gBP-'+str(i+1)+'sigmas_lr6',net1.sigmas, allow_pickle=True)
        np.save('4layer-gBP-'+str(i+1)+'pis_lr6',net1.pis, allow_pickle=True)
        np.save('4layer-gBP-'+str(i+1)+'mess_lr6',net1.messes, allow_pickle=True)





