import numpy as np
from sklearn.neighbors import KDTree
from numba import njit
import time
import os
from tqdm import tqdm
#from knowing import periodic_phi,prep_arrs,image,periodicKD
import matplotlib.pyplot as plt, seaborn as sns
#import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.cm as cm
import csv

#TODO: change sigmadt -> noise

def prep_arrs(posarray):
    """make image arrays for periodic boundary condition"""
    posarray = np.array(posarray,dtype=float)
    x, y = posarray[:,0], posarray[:,1]
    y1 = np.empty_like(x)
    #image1x = x
    #image2y = y
    x2 = np.empty_like(x)
    y3 = np.empty_like(x)
    x3 = np.empty_like(x)
    return x,y,x2,x3,y1,y3

@njit 
def image(x,y,x2,x3,y1,y3, size: float):
    """fill in image arrays"""
    for n in range(len(x)):
        #image1
        if y[n] <= size/2: 
            y1[n] = y[n] + size
        elif y[n] > size/2: 
            y1[n] = y[n] - size
        #image2
        if x[n] <= size/2: 
            x2[n] = x[n] + size
        elif x[n] > size/2: 
            x2[n] = x[n] - size
        #image3
        if x[n] <= size/2 and y[n] <= size/2:
            x3[n] = x[n] + size
            y3[n] = y[n] + size
        elif x[n] <= size/2 and y[n] > size/2:
            x3[n] = x[n] + size
            y3[n] = y[n] - size
        elif x[n] > size/2 and y[n] <= size/2:
            x3[n] = x[n] - size
            y3[n] = y[n] + size
        elif x[n] > size/2 and y[n] > size/2:
            x3[n] = x[n] - size
            y3[n] = y[n] - size
    x = np.concatenate((x, x, x2, x3))
    y = np.concatenate((y, y1, y, y3))
    return x,y

#sklearn is already c-compiled, no need to jit
def periodicKD(k,pos,image):
    """find K-nearest neighbors over periodic boundary"""
    tree = KDTree(image)
    dist,ind=tree.query(pos,k+1)
    return ind[:,1:]%len(pos) #to rid focal agent & wrap image to ID

@njit #optimize with numba
def periodic_phi(pt):
    """normalize heading to [-pi,pi]"""
    pt %= 2*np.pi 
    if pt > np.pi: 
        pt -= 2*np.pi
    elif pt < -np.pi: 
        pt += 2*np.pi
    return pt

@njit
def add_force(N,nbs,phi,k):
    """calculate social force as heading difference with neighbors"""
    force = np.zeros(N)
    for i in range(N):
        for n in nbs[i]:
            force[i] += periodic_phi(phi[n] - phi[i]) 
        force[i] /= k
    return force

@njit
def informing(force, phi, who_correct, w_informed, rightdir, tau_info, tau):
    """calculate personal force as heading difference with right direction"""
    idx = np.where(who_correct==1)[0]
    for id in idx:
        force[id] = (1-w_informed)*force[id] +\
            w_informed * periodic_phi(rightdir - phi[id]) *tau/tau_info
    return force

@njit
def update_phi(phi,force,tau,N,noise,dt):
    """calculate heading from force and noise"""
    dphidt = force/tau + np.random.normal(0,1,N) * noise/np.sqrt(dt)
    phi += dphidt*dt
    phi %= 2*np.pi
    return phi 

@njit
def update_vpos(phi_comp,size:float,speed:float,dt:float,pos):
    """calculate velocity and position"""
    v = speed * phi_comp
    pos += v * dt
    pos %= size #normalize position to [0,size]
    return pos,v

class Simulation:
    """
      Function: 
      2D continuous Viscek Model with k-nearest-neighbors alignment, homogeneous noise, and informed individuals.

      Parameters: 
      - noise: scaling factor for the gaussian noise on the alignment. 
      - tau: relaxation time. 
      - informed_ratio: proportion of individuals who are influenced by a right direction based on a cue. 

      Returns: 
      Position and direction of each agent at each step. 
      """
    def __init__(self, noise: float, tau: float, informed_ratio: float=0.,\
                    k: int=3, N: int=2500, size: float=50, speed: float=0.2,
                    noise_informed: float=0, rightdir: float=0., w_informed: float=0.5,tau_info:float=0.039):
        #parameters for topological vicsek
        self.N = N #number of agents
        self.size = size #size of box
        self.tau = tau #relaxation time (stationary rate)
        self.k = k #number of neighbors
        self.speed=speed #constant magnitude of velocity
        self.dt = tau/10 #persistence length bc v_mag is constant
        self.noise = noise

        #parameters for spatial information
        self.tau_info = tau_info #to fix this rate despite tau_social changes
        self.informed_ratio = informed_ratio
        self.rightdir = rightdir #like a field. default 0rad
        self.noise_informed = noise_informed
        self.w_informed = w_informed #default 0.5
        self.who_correct = np.random.choice(a=[0,1], size=int(self.N), p=[1-self.informed_ratio,self.informed_ratio]) #boolean

        #agents
        self.pos = np.random.random((int(self.N), 2)) * self.size
        self.phi = 2*np.pi*np.random.random(size=int(self.N))
        self.v = self.speed*np.array([np.cos(self.phi),np.sin(self.phi)]).transpose()

    def update_force(self):
        """makes a temporary array of social and global force that is then applied to update phi, v, and pos."""
        #make images for periodic boundary
        x,y,x2,x3,y1,y3 = prep_arrs(self.pos)
        xall,yall = image(x,y,x2,x3,y1,y3,self.size)
        im=np.transpose(np.array([xall,yall]))
        #find K nearest neighbors with KD tree
        nbs = periodicKD(k=self.k,pos=self.pos,\
          image=im)
        #make force from neighbor directions 
        force = add_force(self.N,nbs,self.phi,self.k)
        #make force from informedness
        force = informing(force=force, phi=self.phi, who_correct=self.who_correct, w_informed=self.w_informed, rightdir=self.rightdir,tau_info=self.tau_info,tau=self.tau)
        #use forces to update phi
        self.phi = update_phi(phi=self.phi,force=force,tau=self.tau,N=self.N,noise=self.noise,dt=self.dt)
        #componentize phi for faster calculation
        phi_comp=np.array([np.cos(self.phi),np.sin(self.phi)]).transpose()
        #update v and pos based on force and phi
        self.pos,self.v=update_vpos(phi_comp=phi_comp,size=self.size,speed=self.speed,dt=self.dt,pos=self.pos)
        return

def run(tau,noise,informed_ratio,simtime,logname=None,count=1000,firststep=0,k=3,N=2500,size: float=50, speed: float=0.2,
                    noise_informed: float=0, rightdir: float=0., w_informed: float=0.5,tau_info:float=0.039):
    """the main function to run the simulation with user input parameters and save the data for later analysis"""
    #make simulation
    sim = Simulation(noise=noise,tau=tau,informed_ratio=informed_ratio,N=N,k=k,size=size,speed=speed,noise_informed=noise_informed,rightdir=rightdir,w_informed=w_informed,tau_info=tau_info)
    
    #assign logname
    if not logname: 
        string = f'tau{np.round(tau,3)}_noise{np.round(noise,3)}_r{informed_ratio}_simtime{simtime}_k{k}_N{N}_w{w_informed}_tauinfo_{tau_info}_iter1'
        logname = string.replace('.','d')
        while os.path.exists(logname): #if logname exists, add a number
            #print(f"logname exists {logname}, adding iter")
            logname = logname[:-1] + str(int(logname[-1])+1)
            #print("logname",logname)
    #make directory
    newpath = f'{logname}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    #write parameters to file
    params = pd.DataFrame({'tau': tau, 'noise': noise, 'tau_info':tau_info,'informed_ratio': informed_ratio, 'simtime': simtime, 'k': k, 'N': N,'rightdir':rightdir,'size':size,'speed':speed,'noise_informed':noise_informed,'w_informed':w_informed,'count':count,'firststep':firststep,'who_correct':[sim.who_correct]}, index=[0])
    params.to_csv(f'{logname}/metadata.csv', header=True, index=None)
    #write data
    with open(f"{logname}/data.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        steps = int(simtime / sim.dt)  # determine number of steps approximately from discrete stepsize
        interval = steps // count + 1  # fix count of frames for runs with differing dt. min=1
        print(f"total steps: {steps}")
        pbar = tqdm(desc='for loop', total=steps)  # track progress
        start_time = time.time()  # measuring runtime
        for step in range(steps):
            sim.update_force()  # run simulation
            if step % interval == 0:  # log outputs
                a = np.full(sim.N, firststep + step)  # melt steps
                arr = np.column_stack((a,  # col 1: step
                                       np.copy(sim.pos[:, 0]),  # col 2: posx
                                       np.copy(sim.pos[:, 1]),  # col 3: posy
                                       np.copy(sim.phi)))  # col 4: phi
                writer.writerows(arr)  # Write the data to the file
            pbar.update()
    print(f"finished execution with {np.round(time.time() - start_time, 5)} s")
    pbar.close()
    return
