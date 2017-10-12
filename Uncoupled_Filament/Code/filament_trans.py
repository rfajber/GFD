import numpy as np
from scipy.integrate import odeint 
import sys

####defaults
global cL,alpha
cL=0.1
alpha=4.
V=1.

#define the grid 
L=1.
delta=0.005
N=int(L/delta)-1
y=np.linspace(delta,L-delta,N)

vw=V*y*(y-L) 

Um=np.max(np.abs(vw))
dt=delta/Um
M=10**4
t=np.linspace(dt,M*dt,M)

#functions which get the derivatives
def get_P(c):
    return np.power(c,alpha)

def dcdt_neg(c,t,vw):
    dcdt_A=np.zeros(len(c))
    dcdt_D=np.zeros(len(c))
    for ii in range(0,len(c)-1):
        dcdt_A[ii]=(vw[ii+1]*c[ii+1]-vw[ii]*c[ii])/delta
    dcdt_A[-1]=(-vw[-1]*c[-1])/delta
    for ii in range(1,len(c)-1):
        dcdt_D[ii]=(get_P(c[ii+1])+get_P(c[ii-1])-2.*get_P(c[ii]))/delta**2
    dcdt_D[-1]=(get_P(cL)+get_P(c[-2])-2.*get_P(c[-1]))/delta**2
    dcdt_D[0]=(get_P(c[1])-get_P(c[0]))/delta**2
    return -dcdt_A+dcdt_D

#pass a file name with extra arguements 
if len(sys.argv)>1:
    f=open(sys.argv[1])
    for ff in f.read().split('\n'):
        exec(ff)

#integrate using odeint 
C=odeint(dcdt_neg,np.ones(len(y))*cL,t,args=(vw,))

#save the output
filename='C_CL'+str(cL)+'_a'+str(alpha)+'.npy'
filename0='C0_CL'+str(cL)+'_a'+str(alpha)+'.npy'
filenameE='CE_CL'+str(cL)+'_a'+str(alpha)+'.npy'
np.save(filename,C)
np.save(filename0,C[:,0])
np.save(filenameE,C[-1,:])
