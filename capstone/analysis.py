#this py module contains functions used for getting the timeseries and descriptive statistics of the topological informed swarm simulation. 


#dependencies
import numpy as np
#from knowing import periodic_phi,prep_arrs,image,periodicKD
import matplotlib.pyplot as plt, seaborn as sns
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def load_data_new(logname):
    """load parameters and position, angle data with informed individual indices"""
    #parameters   
    paramsdf = pd.read_csv(f'{logname}/metadata.csv').loc[0] #get rid of index
    params = {'noise':paramsdf['noise'],'tau':paramsdf['tau'],'tau_info':paramsdf['tau_info'],'informed_ratio':paramsdf['informed_ratio'],'rightdir':paramsdf['rightdir'],\
        'size': paramsdf['size'], 'k': paramsdf['k'], 'N': paramsdf['N'], 'speed':paramsdf['speed'],'simtime':paramsdf['simtime'],'noise_informed':paramsdf['noise_informed'],'w_informed':paramsdf['w_informed'],'count':paramsdf['count'],'firststep':paramsdf['firststep'],'who_correct':paramsdf['who_correct']}
    #params['who_correct'][1]

    #data
    output = pd.read_csv(f'{logname}/data.csv',header=None) #don't read first row as header
    output.columns = ['steps','posx','posy','phi']
    steps = output['steps'].unique()
    #print("steps", len(steps),"N",params['N'])
    output['id']= list(range(params['N']))*len(steps)

    return output, params

#stats
def calculate_polarization(phis):
    """find the mean angle of particles"""
    x_sum = np.sum(np.cos(phis))
    y_sum = np.sum(np.sin(phis))
    magnitude = np.sqrt(x_sum**2 + y_sum**2)/len(phis)
    return magnitude

def calc_polarity_step(output,params,step):
    phis = output[output['steps']==step]['phi']
    mean = calculate_polarization(phis)
    return mean

def calc_polarity_stat(output,params):
    """plot the timeseries and calculate the average stat from every 10th opint after 1/4 time burn-in"""
    means = [] 
    for s in output['steps'].unique():
        mean_s = calc_polarity_step(output,params,step=s)
        means.append(mean_s)
    
    fig,ax=plt.subplots()
    ax.plot(means)
    ax.set_title(f"Global polarization over time\nTau_social:{params['tau']},Tau_info:{params['tau_info']},Noise:{params['noise']},Informed:{params['informed_ratio']}")
    plt.savefig(f"{params['tau']}_{params['noise']}_{params['informed_ratio']}_polarization.png")
    plt.show()
    ax.clear()

    #if params['simtime']>=10000: #no more
    print("global polarization mean:",np.average(means[int(len(means)/4):])) #every 10th TO JUSTIFY

#animation 
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

def ani(logname,output,params):
    """make animation of data from frames of every 10th step"""
    norm = colors.Normalize(vmin=-np.pi,vmax=np.pi)
    mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
    fig,ax=plt.subplots(figsize=(7,7))
    ax.set_xlim(0, params['size']) 
    ax.set_ylim(0, params['size']) 
    ax.set_xticks([])
    ax.set_yticks([])

    fr = list(output['steps'].unique())[::10]

    def update(frame):
        ax.clear()
        ax.set_xlim(0, params['size'])  
        ax.set_ylim(0, params['size'])  
        ax.set_xticks([])
        ax.set_yticks([])
        step_data = output[output['steps'] == frame]
        for idx, row in step_data.iterrows():
            position_x = row['posx']
            position_y = row['posy']
            phi_angle = row['phi']
            color = mapper.to_rgba(phi_angle) 
            ax.arrow(position_x, position_y, 
                     0.05 * np.cos(phi_angle), 0.05 * np.sin(phi_angle), #dx,dy
                     head_width=0.04, head_length=0.04, 
                     color=color)
        ax.set_xlabel(f"Simulation time: {np.round(frame*params['tau']/10,3)}") #count in simulation time 
        ax.set_title(f"{logname}\nTau:{params['tau']},Noise:{params['noise']},Informed:{params['informed_ratio']}")

    ani = FuncAnimation(fig, update, frames=fr, blit=False, repeat=False)
    ani.save(f'{logname}.gif', writer='pillow')


import networkx as nx
from simulation import prep_arrs, image, periodicKD

def recalc_nbs(output,params,step):
    """recalculate interaction from positions"""
    posx = output[output['steps']==step]['posx']
    posy = output[output['steps']==step]['posy']
    pos = np.array([posx,posy]).transpose()
    x,y,x2,x3,y1,y3= prep_arrs(pos)
    xall,yall=image(x,y,x2,x3,y1,y3,params['size'])
    im=np.transpose(np.array([xall,yall]))
    nbs = periodicKD(k=params['k'],pos=pos,image=im)
    return nbs,pos

def build_graph(nbs,N):
    """make graph from interactions"""
    #TODO mark the informed individuals.
    agentlist = np.arange(N)
    graph=nx.DiGraph()
    graph.add_nodes_from(agentlist)
    for agent in range(N):
        for v in nbs[agent]:
            graph.add_edge(agent,v)
            #nbs[v].remove(agent) #remove duplicate links if undirected graph
    return graph

def build_clusters(graph):
    """make clusters based on weakly connected component subgraphs"""
    #TODO constraint: each agent can only belong to at most one subgraph. 
    digraphs = [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]
    return digraphs

def calc_cluster_step(digraphs, output, params, step):
    """given cluster list, calculate polarization and size of cluster"""
    means = [] 
    sizes = []
    for digraph in digraphs:
        members = list(digraph.nodes)
        
        phis = output[output['id'].isin(members)][output['steps']==step]['phi']
        mean = calculate_polarization(phis)
        means.append(mean)
        
        size = digraph.number_of_nodes()
        sizes.append(size)
        
    meansavg = np.mean(means)
    sizesavg = np.mean(sizes)

    return meansavg, sizesavg

import warnings
warnings.filterwarnings("ignore")

def calc_cluster_timeseries(output, params):
    """calculate cluster size and cluster polarization every 10 steps in the last 1/3 of the series"""
    mlist = []
    slist = []
    stepping = list(output['steps'].unique())
    for s in stepping[-len(stepping)/3::10]:
        nbs,pos = recalc_nbs(output=output,params=params,step=s) 
        graph = build_graph(nbs,params['N'])
        digraphs= build_clusters(graph)
        meansavg, sizesavg = calc_cluster_step(digraphs, output, params, s)
        mlist.append(meansavg)
        slist.append(sizesavg)
    return mlist, slist

def calc_cluster_stat(output, params):
    """plot timeseries of cluster polar and cluster size, return point averages"""
    nowm,nows = calc_cluster_timeseries(output, params)
    fig,ax=plt.subplots()
    ax.clear()
    ax.plot(nowm)
    plt.xlabel("time")
    plt.ylabel("average cluster polarization")
    plt.title(f"average cluster polarization over time\nTau:{params['tau']},Noise:{params['noise']},Informed:{params['informed_ratio']}")
    plt.savefig(f"{params['tau']}_{params['noise']}_{params['informed_ratio']}_cluster_polarization.png")
    plt.show()

    ax.clear()
    plt.plot(nows)
    plt.xlabel("time")
    plt.ylabel("average cluster size")
    plt.title(f"average cluster size over time\nTau:{params['tau']},Noise:{params['noise']},Informed:{params['informed_ratio']}")
    plt.savefig(f"{params['tau']}_{params['noise']}_{params['informed_ratio']}_cluster_size.png")
    plt.show()
    return np.mean(nowm), np.mean(nows)

def analyze_pipe(logname):
    """save figures and return point statistics for global polar, cluster polar, mean cluster size, and animation"""
    output,params = load_data_new(logname = logname)
    clusterpolar,clustersize = calc_cluster_stat(output,params)
    globalpolar = calc_polarity_stat(output,params) 
    ani(logname,output,params)
    return params,globalpolar,clusterpolar,clustersize