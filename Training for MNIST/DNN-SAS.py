#Import
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
import load

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

#softmax functions(preventing explosion)
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
    
#Some defined functions to fit in matrix.
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
def clip_in_sum(x,k):#
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

# Some possible decay methods for learning rate.
def decay_rate(learning_rate,global_step,decay_steps,alpha,num_periods):
    global_step = min(global_step, decay_steps)
    l_decay=(decay_steps-global_step)/(decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(((math.pi)*2)*num_periods * (global_step / decay_steps)))
    decayed = (alpha+l_decay) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    return decayed_learning_rate 
def divi_(lr0,global_step,decay_step):
    return lr0*(0.5**((int(global_step/decay_step))))


#Some defined functions to sample weights from trained paramaters.
def turn_2_zero(x):
    return np.int64(x>0)
def sampling(m,p,v):
    m=np.array(m)
    p=np.array(p)
    v=np.array(v)
    b=[np.ones(p1.shape) for p1 in p]
    ran=np.array([np.random.random(size=(pi.shape)) for pi in p ])
    for i in range(0,p.shape[0]):
        for j in range(0,p.shape[1]):
            if v[i][j]==0:
                b[i][j]=(turn_2_zero((ran[i][j]-p[i][j])))*m[i][j]
            else:
                b[i][j]=(turn_2_zero((ran[i][j]-p[i][j])))*(np.random.normal(m[i][j],np.sqrt(v[i][j])))
    return b   


#import data and confirm the shape of it.
mnist=np.array(load.load_mnist(one_hot=True))
train_data = mnist[0][0][0:10000].T
train_label = mnist[0][1][0:10000].T
test_data = mnist[1][0][:10000].T
test_label = mnist[1][1][:10000].T
print(np.shape(train_data))
print(np.shape(train_label))

# Define the mini-batch generating function
def mini_batch_generate(mini_batch_size):
    n=train_data.shape[1]
    state = np.random.get_state()
    np.random.shuffle(((train_data).T))
    np.random.set_state(state)
    np.random.shuffle(((train_label).T))
    mini_batches = np.array([train_data[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches_labels =np.array([train_label[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    return mini_batches,mini_batches_labels

#Network structure
class NeuralNetwork:
    def __init__(self,sizes):
    #Initialization
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.sigmas = [(0.01*random.random(size=(y,x))) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.pis = [np.clip(np.zeros((y,x)),0,1) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.messes=[((np.sqrt(2/784))*(np.random.normal(0.0,1.0,(y,x)))) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    
    
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
        
        process=[a]
        for sigma,pi,mess in zip(self.sigmas, self.pis, self.messes):
            medicine=pow(10,-20)
            flag=flag+1
            ep=np.random.normal(0,1,(sigma.shape[0],a.shape[1]))
            v=((np.dot(((1-pi)*((sigma)+(mess*mess))-(1-pi)*(1-pi)*mess*mess),a*a)))**(0.5)
            mea=(np.dot((1-pi)*mess,a))
            z=(mea+v*ep)*(1/(np.sqrt(sigma.shape[1])))
            if (flag<(self.num_layers-1)):
                a = activate(z)
            if (flag>=(self.num_layers-1)):
                a = softmax_more(z)
            
            zm.append(z)
            
            process.append(a)
            epsilon.append(ep)
            var.append(v)
        if back == False:
            return process[-1]
        if back == True:
            return process,epsilon,var,zm
        
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
    #If va<=0, the corresponding gradients are forced to zero.
        medicine=pow(10,-30)
        tri=[]
        out,epsi,va,zm=self.feedforward(x,activate,back=True)
        var=va+medicine*np.ones_like((va))
        nabla_sigma = [np.zeros(sigma.shape) for sigma in self.sigmas]
        nabla_pi = [np.zeros(pi.shape) for pi in self.pis]
        nabla_mess = [np.zeros(mess.shape) for mess in self.messes]
        for l in range(1, (self.num_layers)):
            if l==1:
                tri_=(out[-1]-y)
                tri.append(tri_)
            else:
                tri_=((1/(np.sqrt(self.sizes[-l])))*(np.sum(((upper_bound(tri[-1],self.sizes[-l]))*lower_bound(dactivate(zm[-l]),self.sizes[-l+1]))*((i_k_func((1-self.pis[-l+1])*self.messes[-l+1],x.shape[1]))+upper_bound((epsi[-l+1]/(var[-l+1])),self.sizes[-l])*lower_bound(out[-l],self.sizes[-l+1])*i_k_func(((1-self.pis[-l+1])*(((self.sigmas[-l+1])+((self.messes[-l+1])**(2)))-((1-self.pis[-l+1])**(2))*(self.messes[-l+1]**(2)))),x.shape[1])),axis=0)).reshape(x.shape[1],self.sizes[-l])).T
                tri.append(tri_)
            nabla_pi[-l]=(1/(np.sqrt(self.sizes[-l-1])))*find_aver(upper_bound(tri_,self.sizes[-l-1])*((i_k_func((-1)*self.messes[-l],x.shape[1]))*lower_bound(out[-l-1],self.sizes[-l])+((upper_bound((turn_2_zero(va[-l])*epsi[-l]/(2*var[-l])),self.sizes[-l-1]))*lower_bound((out[-l-1]*out[-l-1]),self.sizes[-l])*i_k_func((-((self.sigmas[-l])+((self.messes[-l])**(2)))-2*(self.pis[-l]-1)*self.messes[-l]*self.messes[-l]),x.shape[1]))),self.sizes[-l-1])
            nabla_sigma[-l]=(1/(np.sqrt(self.sizes[-l-1])))*find_aver((upper_bound((turn_2_zero(va[-l])*tri_*epsi[-l]/(2*var[-l])),self.sizes[-l-1])*lower_bound((out[-l-1]*out[-l-1]),self.sizes[-l])*i_k_func(((1-self.pis[-l])),x.shape[1])),self.sizes[-l-1])
            nabla_mess[-l]=(1/(np.sqrt(self.sizes[-l-1])))*find_aver(upper_bound(tri_,self.sizes[-l-1])*(i_k_func((1-self.pis[-l]),x.shape[1])*lower_bound(out[-l-1],self.sizes[-l])+upper_bound((turn_2_zero(va[-l])*epsi[-l]/(var[-l])),self.sizes[-l-1])*lower_bound((out[-l-1])*(out[-l-1]),self.sizes[-l])*i_k_func(((1-self.pis[-l])*self.messes[-l]-((1-self.pis[-l])**(2))*self.messes[-l]),x.shape[1])),self.sizes[-l-1])
        return nabla_sigma,nabla_pi,nabla_mess
    
    def adam_mini(self,lr,mini_batch_size,activate,dactivate):
        data,label = mini_batch_generate(mini_batch_size)
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
        decay=pow(10,-4)#Add l2 decay
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
            print('\r'+str(j)+'/'+str(int(data.shape[0])),end='')
    
    def SGD(self,mini_batch_size,epoch,lr0):
        evaluation_cost, evaluation_error = [], []
        training_cost, training_accuracy = [], []
        learning_rate=[]
        test1,label1=test_data,test_label
        sigma_tra=[]
        pis_tra=[]
        mess_tra=[]
        for i in range(epoch):
            #np.save('decay0/4layer-d0-10000'+str(1)+'sigmas_all',self.sigmas, allow_pickle=True)
            #np.save('decay0/4layer-d0-10000'+str(1)+'pis_all',self.pis, allow_pickle=True)
            #np.save('decay0/4layer-d0-10000'+str(1)+'mess_all',self.messes, allow_pickle=True)
            lr = divi_(lr0,i,20)
            print ("Epoch %s training complete" % i)
            sigma_tra.append(self.sigmas)
            pis_tra.append(self.pis)
            mess_tra.append(self.messes)
            self.adam_mini(lr,mini_batch_size,relu,drelu)
            cost1,accuracy1 = self.sampling_evaluate(test1,label1,relu)
            evaluation_cost.append(cost1)
            evaluation_error.append((1-accuracy1))
            cost2,accuracy2 = self.evaluate(train_data,train_label,relu)
            training_cost.append(cost2)
            training_accuracy.append(accuracy2)
            
            print("the training Accuracy is:{} %".format((accuracy2)*100))
            print("the training cost is ",cost2)
            log=open("accd4.txt","a")
            print("the test error is:{} %".format((1-accuracy1)*100),file=log)
            log.close()
            print("the cost is ",cost1)
            
        return evaluation_error
            
#Save data
if __name__ == '__main__':
    for i in range(5):
        net=NeuralNetwork([784,100,100,10])
        test_error=net.SGD(mini_batch_size=500,epoch=200,lr0=0.1)   
        np.save('4layer-d4-10000'+str(i+1)+'MNIST',test_error, allow_pickle=True)
        np.save('4layer-d4-10000'+str(i+1)+'sigmas_MNIST',net.sigmas, allow_pickle=True)
        np.save('4layer-d4-10000'+str(i+1)+'pis_MNIST',net.pis, allow_pickle=True)
        np.save('4layer-d4-10000'+str(i+1)+'mess_MNIST',net.messes, allow_pickle=True)




