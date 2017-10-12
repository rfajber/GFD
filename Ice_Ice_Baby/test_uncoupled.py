#%%
#modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#%%
#functions
def LF(fun,dt,r,A0,A1):
    A2=fun(y0)*2*dt
    Af=(1-2.*r)*A1+(A0+A1)*2.*r
    return yf

def P(m):
    return np.exp(-k*m)

def func(A):
    m=A[:Ny]
    v=A[Ny:]
    dAdt=np.zeros(2*Ny)

    #mass advection
    dmdt=np.zeros(Ny)
    vp=v[v<=0]=0. #this is a convenient way to do upwinding
    vm=v[v>=0]=0.
    dmdt[1:-1]=-m[1:-1]*(vp[1:-1]+vm[1:-1])/dy
    dmdt[1:-1]+=m[:-1]*vm[:-1]/dy-m[1:]*vp[1:]/dy
    dmdt[-1]=0
    dmdt[0]=vp[1]*m[1]/dy-vm[0]*m[0]/dy

    #momentum evolution
    dvdt=np.zeros(N)
    #drag
    dvdt-=np.abs(v-vw)*(v-vw)
    #pressure term
    p=P(m)
    dvdt[1:-1]-=(p[2:]-p[:-2])/(2*dy)
    #viscous term
    dvdt[1:-1]+=l*(v[2:]+v[:-2]-v[1:-1])/(dy)**2
    #boundary terms
    dvdt[0]-=(p[1]-p[0])/dy+l*(v[1]-v[0])/(dy)**2
    dvdt[-1]-=(P(m[0])-p[-2])/(2*dy)+l*(v[-1]-v[-2])/(dy)**2

    dAdt[:N]=dmdt
    dAdt[N:]=dvdt

    return dAdt

#%%
#constant declration
global y,dy,N,t,dt,M,k,vw,l
Ny=100
y=np.linspace(0,1,Ny)
dy=y[1]-y[0]
dt=0.1
Nt=100
t=np.linspace(0,Nt*dt,int(Nt/dt)+1)
k=5.
M=0.1
vw=y*(1-y)
l=0.01

#%%
#initialization
m0=M*np.ones(Ny)
v0=np.zeros(Ny)
A0=np.zeros(2*Ny)
A0[:Ny]=m0
A0[Ny:]=v0

#%%
#main time loop
A=odeint(func,t,A0)

#%%
#try some small plotting
m=A[:,:Ny]
v=A[:,Ny:]
plt.subplot(211)
plt.contour(y,t,m)
plt.colorbar()
plt.subplot(212)
plt.contour(y,t,v)
plt.colorbar()
plt.show()
