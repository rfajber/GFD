import numpy as np
#import matplotlib.pyplot as plt
import sys 
import os 
from scipy.integrate import solve_bvp
from scipy.integrate import odeint
from scipy import interpolate as interp

#######################################################################################

#set constants
global adv_BCL_no_flux, adv_BCR_no_flux, diff_BCL_no_flux, diff_BCR_no_flux 
global cL,cR,PL,PR
global alpha
global cinterp
global eps
global A1,A2,A3
global u_save
global UW

#######################################################################################

def get_P(c):
    return np.power(c,alpha)

def u2(y,u):
    u1=u[1]
    u0=u[0]
    uw=get_uw(y)
    u2=A3*cinterp(y)*(u0-uw)
    #print(cinterp(y))
    return np.vstack([u1,u2])

def ubc(ua,ub):
    return np.array([ua[0],ub[1]])

def get_uw(y):
    return -UW*y*(y-1)/.25#y*np.exp(-.5*y**2/0.02)####-1./3.*y**3.

def get_vw(y):
    return y*(y-1.)

def get_vE(y,c):
    global cinterp
    global usave
    cinterp=interp.interp1d(y,c) #make interpolation object from the concentrations
    ug=[-y*(y-1.),-.1*(.5*y-1.)] 
    u=solve_bvp(u2,ubc,y,ug,verbose=0)
    u=u.sol(y)[0]
    usave=u
    uw=get_uw(y)
    return A1*c*(u-uw)
    
def dcdt(c,t):
    dcdt_adv=np.zeros(len(c)) #concentration advective part 
    dcdt_diff=np.zeros(len(c)) #concentration diffusive part 
    v=np.zeros(len(c))
    #v+=get_vW(y)
    v+=get_vE(y,c)
    #advection interior
    for ii in range(1,len(c)-1):
        if v[ii]<=0:
            dcdt_adv[ii]=(c[ii+1]*v[ii+1]-c[ii]*v[ii])/delta
        elif v[ii]>0:
            dcdt_adv[ii]=(c[ii]*v[ii]-c[ii-1]*v[ii-1])/delta
    #BCL:
    if v[0]<=0:
        dcdt_adv[0]=(v[1]*c[1]-c[0]*v[0])/delta
    else:
        dcdt_adv[0]=dcdt_adv_BCL(v[0],c[0])
    #BCR:
    if v[-1]>=0:
        dcdt_adv[-1]=(v[-1]*c[-1]-c[-2]*v[-2])/delta
    else:
        dcdt_adv[-1]=dcdt_adv_BCR(v[-1],c[-1])
    #non linear diffusion interior
    P=get_P(c)
    dcdt_diff[1:-1]=(P[:-2]-2.*P[1:-1]+P[2:])/delta**2.
    #BCL
    dcdt_diff[0]=dcdt_diff_BCL(P[1],P[0])
    #BCR
    dcdt_diff[-1]=dcdt_diff_BCR(P[-1],P[-2])

    #linear diffusion interior
    dcdt_diff[1:-1]+=eps*(c[:-2]-2.*c[1:-1]+c[2:])/delta**2.
    #BCL
    dcdt_diff[0]+=eps*dcdt_diff_BCL_lin(c[1],c[0])
    #BCR
    dcdt_diff[-1]+=eps*dcdt_diff_BCR_lin(c[-1],c[-2])
    #return sum 
    return -dcdt_adv+dcdt_diff/A2

def dcdt_adv_BCL(vw0,c0):
    if adv_BCL_no_flux==1:
        #no flux condition
        return vw0*c0/delta
    else:
        #fixed C condition
        return (vw0*c0-vwL*cL)/delta

def dcdt_adv_BCR(vw1,c1):
    if adv_BCR_no_flux==1:
        #no flux condition 
        return -vw1*c1/delta
    else:
        #fixed C condition 
        return (vwR*cR-vw1*c1)/delta

def dcdt_diff_BCL(P1,P0):
    if diff_BCL_no_flux==1:
        return (P1-P0)/delta**2
    else:
        return (PL-2*P1+P0)/delta**2

def dcdt_diff_BCR(P1,P2):
    if diff_BCR_no_flux==1:
        return -(P1-P2)/delta**2
    else:
        return (PR-2*P1+P2)/delta**2
    
def dcdt_diff_BCL_lin(C1,C0):
    if diff_BCL_no_flux==1:
        return (C1-C0)/delta**2
    else:
        return (cL-2*C1+C0)/delta**2

def dcdt_diff_BCR_lin(C1,C2):
    if diff_BCR_no_flux==1:
        return -(C1-C2)/delta**2
    else:
        return (cR-2*C1+C2)/delta**2
    
def odeint_substepper(c0,t,dM):#reduces output
    dt=t[1]-t[0]
    C=np.zeros((len(t),len(c0)))
    U=np.zeros((len(t),len(c0)))
    C[0]=c0
    t_int=np.linspace(0,dM*dt,int(dM))
    for ii in range(1,len(t)):
        temp=odeint(dcdt,C[ii-1],t_int)
        C[ii]=temp[-1]
        U[ii]=usave
    return C,U

#######################################################################################

alpha=4.
eps=0.

A2=1.
A1=1.
A3=1.

cinf=.375 #note this is rescaled c_inf 

UW=1.

F=sys.argv[1]
for ff in open(F).read().split('\n'):
    exec(ff) 

#ygrid 
L=1.
delta=0.01
N=int(L/delta)-1

Um=0.1#np.max(np.abs(vw))
dt=delta/Um#**2/(4*4*cinf**3)

T=50.
dM=2. #substep size 
M=T/dt

#boundary conditions 
adv_BCL_no_flux=1
adv_BCR_no_flux=1
diff_BCL_no_flux=1
diff_BCR_no_flux=0

fname='save_data_U_coupled_cinf'+str(cinf)+'_A3'+str(A3)+'_UW'+str(UW)+'.npy'

#######################################################################################

if not os.path.isfile(fname):
    y=np.linspace(delta,L-delta,N)

    t=np.linspace(0,T+dt,int(M/dM))

    cR=cinf
    PR=get_P(cinf)

    c0=cinf*np.ones(N)

    X=odeint_substepper(c0,t,dM)
    C,U=X

    S=[t,y,C,U]

    np.save(fname,S)

else:
    print('already done: ' + fname)

#######################################################################################



