import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
#from gmm_classifier import *
import qutip as qt
import datetime
import scipy as sp
import pandas as pd
from importlib import reload
import os
from datetime import datetime, date
import glob

###########################
### plot time evolution ###
###########################

def get_population_data(new_path, starting, Nshots, tsteps, clf=None, mesolve=[False, []], correction=[[1,0,0],[0,1,0],[0,0,1]]):
    '''
    TODO: load pulses from data file if files provided instead of lists/arrays
    
    Parameters
    ----------
    new_path : .hdf5
        .hdf5 file from your measurement.
    starting : state you started in
    Nshots : Int
        Number of shots
    tsteps : Numpy array
        Time step list. 'time_evolution' function returns 'tsteps' as the 
        first return arg. 
        np.array(16, length of the pulse, tstep)
    clf : classifier
        Classifier for state classification
    mesolve : [Bool, pulse_info]
        default [False, []]
        If True, calculates and plots mesolve result for applied pulse.
        Here, pulse_info is a list [xpath, ypath, downsamp]
        Compatible formats for xpath, ypath are list, array or filepath
        downsamp should be the same as the 'sampling_rate' used in 
           TFOC_pulse_gen (if this is how you the pulses were calculated).
           units: 1 data point/ns
    correction : Matrix, optional
        Input needs to be inv(cmat). Default is identity -- no correction.

    Returns
    -------
    None.

    '''
    
    if clf == None:
        clf, _ = get_classifier()    
    
    with h5py.File(new_path,'r') as datfile:
        Navg = int(Nshots)
        I = datfile['OPX_data/output_arrays/I/value'][:]
        Q = datfile['OPX_data/output_arrays/Q/value'][:]
        
        IData = I.reshape((-1, len(tsteps), Navg))
        QData = Q.reshape((-1, len(tsteps), Navg))

    ii = IData*1e2
    qq = QData*1e2

    g0 = np.hstack([ii.flatten().reshape(-1,1),qq.flatten().reshape(-1,1)])
    prob = clf.predict_proba(g0)
    
    prob_c = np.matmul(prob.reshape((-1, Navg, 3)), correction)
    
    p_avg = prob.reshape(-1, Navg, 3).mean(axis=1)
    p_avg_c = prob_c.reshape(-1, Navg, 3).mean(axis=1)
    
    
    fig, ax = plt.subplots(1,3, figsize=(14, 4))
    fig.suptitle(new_path)
    
    fig.subplots_adjust(hspace = .2, wspace=.2)

    for i in range(3):
        ax[0].plot(tsteps, p_avg[:,i], label=str(i))
    
    if mesolve[0]:
        try:
            Htot = generate_Hamiltonian(3)
        
            xpath, ypath, downsamp = mesolve[1]
            t = np.arange(0, len(xpath), 1)/downsamp
            
            concat_pulse = np.stack((xpath,ypath), axis=-1) *2*np.pi*1e-3
            
            prob_me_time, prob_me_gate, numgate = qt_mesolve(Htot, 3, starting, concat_pulse, t, cop=True)
            
        except Exception as e:
            print('error in ME solve')
            print(e)
            
        for i in range(3):
            ax[0].plot(t, prob_me_time[i].real, '--', label=str(i))

    ax[0].grid()
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Population')
    ax[0].set_title('As measured')
    ax[0].legend()
    
    for i in range(3):
        ax[1].plot(tsteps, p_avg_c[:,i], label=str(i))
        
    cost = []
    if mesolve[0]:
        try:
            Htot = generate_Hamiltonian(3)
        
            starting, xpath, ypath, downsamp = mesolve[1]
            t = np.arange(0, len(xpath), 1)/downsamp
            
            concat_pulse = np.stack((xpath,ypath), axis=-1) *2*np.pi*1e-3
            
            prob_me_time, prob_me_gate, numgate = qt_mesolve(Htot, 3, starting, concat_pulse, t, cop=True)
            
        except:
            pass
            
        for i in range(3):
            ax[1].plot(t, prob_me_time[i].real, '--', label=str(i))
        
        try:
            for i in range(3):
                cost.append(prob_me_time[i][(tsteps)*downsamp-1] - p_avg[:,i])
        except Exception as e:
            print(e)
            for i in range(3):
                cost.append(prob_me_time[i][(tsteps*downsamp-1)[:-1]] - p_avg[:,i])

    ax[1].set_xlabel('Time (ns)')
    ax[1].set_ylabel('Population')
    ax[1].set_title('Correction matrix')
    ax[1].grid()
    ax[1].legend()
    
    ax[2].hist2d(I, Q, bins=200);
    
    plt.show()
    
    return np.array(cost)
    

# file_path = r"Z://OPX Root//opx-vibranium/Users/Anders"

# fname = ["//p_ctrl.dat", "//q_ctrl.dat"]

# with open(file_path+fname[0], 'r') as f:
#     p = f.readlines()

# with open(file_path+fname[1], 'r') as f:
#     q = f.readlines()

# pp = np.array([float(pval.replace('\n','')) for pval in p])
# qq = np.array([float(qval.replace('\n','')) for qval in q])

# xpath, ypath = pp, qq # sign of q-pulse appears correct
# name_short = 'swap02'
# name_op = name_short + '_op'
# downsamp = 64                  # Pulses p and q are given at 64 points per second, now downsample so that sample rate is ONE point per ns for OPX
# pulse_info = [xpath, ypath, downsamp] ## needed afterwards for ME solve
# #config=upload_pulse(config, 'andrew01', name_short, pulse_info, scale=None, spectral_filter = True, sf_weight=1.6)
# config=upload_pulse(config, 'andrew01', name_short, pulse_info, scale=None, spectral_filter = False)

# NOTE: the file name changes every time the experiment is run
fname = r"/Users/petersson1/src/Quandary/examples/ap/Data_no_spectral_filter/swap02_op_tevol_init0.hdf5"
#from optimal_control_analysis_library import plot_time_evolution

start = 16  # Starting value
stop = 241  # Ending value (inclusive)
step = 4  # Number of elements

tarray = np.arange(start, stop, step)
#plot_time_evolution(fname, starting=0, Nshots=1000, tsteps=np.array([16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]), clf=clf)
diff = my_plot_time_evolution(fname, starting=0, Nshots=1000, tsteps=tarray, clf=clf)
