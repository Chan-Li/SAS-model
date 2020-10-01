
import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
import random
from numpy import random
import math
from matplotlib.pyplot import plot,savefig
from PIL import Image


#Define the entropy function for each connection
def delta(x):
    if abs(x)<pow(10,-4):
        return 0.5*pow(10,4)
    else:
        return 0.0
def entropy(m,p,v,noun):
    sam=0
    if p==1:
        return 0
    if v<=pow(10,-8) and m==0:
        return 0
    if v<=pow(10,-8) and p==0:
        return 0
    if v<=pow(10,-8) and 0<p<1:
        return -p*ln(p)-(1-p)*ln(1-p)
    
    else:
        epsilon=pow(10,-6)
        A=-p*ln(p+(1-p)*scipy.stats.norm.pdf(0,m,np.sqrt(v))+epsilon)
        for i in range(noun):
            ran=np.random.normal(0,1)
            sam=sam+ln((p)*delta(ran*np.sqrt(v)+m)+((1-p)/(np.sqrt(2*math.pi*v)))*(math.exp(-ran*ran/2)))
        s=(((-(1-p)*sam)/noun)+A)

        return s
#The average entropy for each layer and all the layers.
def entropy_layer(inpu,outpu,layer_num,m,p,v):
    S=[]
    number=0
    for i in range(outpu):
        for j in range(inpu):
            s=entropy(m[layer_num][i][j],p[layer_num][i][j],v[layer_num][i][j],100)
            if s!=None:
                number=number+1
                S.append(s)
    return np.average(S)
def aver_S(m,p,v):
    av_S=[]
    for kk in range(m.shape[0]):
        xx=np.shape(m[kk])[0]
        yy=np.shape(m[kk])[1]
        S=entropy_layer(yy,xx,kk,m,p,v)
        av_S.append(S)
    return av_S               


#Plot the figures with jupyter notebook with data.
number=7
cm = plt.get_cmap("gist_gray")
a = [cm((i)/number) for i in range(number)]
Sx=['1-2','2-3','3-4','4-5']
plt.plot(Sx, E1w_av, color=a[3],marker = 's',ms='14',mfc='None',mec=a[3],linewidth=1.0,label='10k',markeredgewidth=1)

plt.plot(Sx, E2w_av, color=a[2],linewidth=1.2,marker='o',ms='14',mfc='None',mec=a[2],label='20k',markeredgewidth=1.9)
plt.plot(Sx, E3w_av, color=a[1],marker = 'D',ms='11',mfc='None',mec=a[1],linewidth=1.4,label='30k',markeredgewidth=2.4)

plt.plot(Sx, E4w_av, color=a[0],linewidth=1.8,marker='^',ms='14',mfc='None',mec=a[0],label='40k',markeredgewidth=3.3)

plt.errorbar(Sx,E1w_av,yerr=E1w_st,elinewidth=2,ecolor=a[3],capsize=5,color=a[3],linewidth=0.0)
plt.errorbar(Sx,E2w_av,yerr=E2w_st,elinewidth=2,ecolor=a[2],capsize=5,color=a[2],linewidth=0.0)
plt.errorbar(Sx,E3w_av,yerr=E3w_st,elinewidth=2,ecolor=a[1],capsize=5,color=a[1],linewidth=0.0)
plt.errorbar(Sx,E4w_av,yerr=E4w_st,elinewidth=2,ecolor=a[0],capsize=5,color=a[0],linewidth=0.0)

plt.yticks(size = 25)
plt.xticks(size = 25)
plt.ylim(-0.4,0.65)
plt.xlabel(xlabel='layers',fontsize=25,labelpad=3)
plt.ylabel(ylabel=r'$\frac{\sum_{I}S_{I}} {M}$',rotation='horizontal',fontsize=29,verticalalignment="center",labelpad=15)
plt.legend(fontsize=14,loc='upper left')
plt.savefig('file-name.pdf',dpi=1000, bbox_inches = 'tight')


# In[ ]:




