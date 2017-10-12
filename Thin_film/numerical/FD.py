#Packages
#############################################
import numpy as np
import sys 
from scipy.integrate import odeint
#import matplotlib.pyplot as plt

#Methods 
#############################################
def uw(x,l):
    v=-np.exp(-.5*(x**2))*np.sin(np.pi*(x/l))
    z=np.linspace(-l,l,1000)
    vz=-np.exp(-.5*(z**2))*np.sin(np.pi*(z/l))
    A=np.trapz(x=z,y=np.abs(vz))
    return v/A

def advective_term(h,u,dx):
    N=len(h)
    A=np.zeros(N)
    n=int((N)/2)
    A[0]=-u[0]*h[0]
    A[1:n-1]=-u[1:n-1]*h[1:n-1]+u[:n-2]*h[:n-2]
    A[n-1]=u[n-2]*h[n-2]
    A[n]=-u[n+1]*h[n+1]
    A[n+1:N-1]=u[n+1:N-1]*h[n+1:N-1]-u[n+2:N]*h[n+2:N]
    A[N-1]=u[N-1]*h[N-1]
    return A/dx

def diffusive_term(h,alpha,dx):
    N=len(h)
    D=np.zeros(N)
    P=np.power(h,alpha)
    D[1:-1]=(P[2:]+P[:-2]-2*P[1:-1])/(dx**2.)
    D[0]=(-1.*P[0]+1*P[1])/(dx**2)
    D[-1]=(-1.*P[-1]+1.*P[-2])/(dx**2)
    return D

def dhdt(h,t,u,alpha,dx):
    A=advective_term(h,u,dx)
    D=diffusive_term(h,alpha,dx)
    return A+D

###just do the loop in python

L=np.linspace(2,10,5)
D=np.linspace(0.2,0.4,3)
L=[100]
D=[0.2]

for l in L:
    for d in D:

        #Parameters
        #############################################
        T=5*10**2
        alpha=4.
        N=2000
        #Simulation
        #############################################
        dx=(l/(.5*N))
        t=np.linspace(0,T,10*T)
        x=np.linspace(-l+dx,l-dx,N)
        u=uw(x,l)
        h0=np.ones(len(x))*d
        h=odeint(dhdt,h0,t,args=(u,alpha,dx))

        file_name='data_l_'+'{:1.0E}'.format(l)+'_d_'+'{:1.0E}'.format(d)+'.npy'

        np.save(file_name,[x,t,u,h])
