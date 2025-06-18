# -*- coding: utf-8 -*-
"""
Created on Mon May  9 08:23:53 2022

@author: cho25
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from gmm_classifier import *
import qutip as qt
import datetime
import scipy as sp
import pandas as pd
from importlib import reload
import os
from datetime import datetime, date
import glob
from OC_pulse_generation import qt_mesolve, generate_Hamiltonian
from calibration_analysis_library import getlast_db, analyze_readout

from calibration_analysis_library import plot_calibration, strpath_to_db_table, read_opx_hdf5

##############################################
### call classifier from the last database ###
##############################################
def get_classifier():
    clf, cmat = analyze_readout(getlast_db('qpus.tant.andrew.confusionmatrix')['file_name'][0])
    return clf, cmat

###########################
### plot time evolution ###
###########################

def plot_time_evolution(new_path, starting, Nshots, tsteps, clf=None, mesolve=[False, []], correction=[[1,0,0],[0,1,0],[0,0,1]]):
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
    
############################
### Plot gate repetition ###
############################
def plot_gate_ops(new_path, maxM, Nshots, clf=None, correction=[[1,0,0],[0,1,0],[0,0,1]], mesolve=[False, []], display_num=None, nstates=3):
    '''
    Parameters
    ----------
    path
    maxM : int, optional
        Max number of gates. The default is 200.
    Nshots : int, optional
        Number of shots. The default is 1000.
    clf : classifier
    correction : inverse of the confusion matrix, default is the identity.
    mesolve : [Bool, [pulse_info_With_starting]]
        default [False, []]
        If True, calculates and plots mesolve result for applied pulse.
        Starting is a number 0-4 that specifies the starting state.
        Here, pulse_info_With_starting is a list [starting, xpath, ypath, downsamp]
        Compatible formats for xpath, ypath are list, array or filepath
        downsamp should be the same as the 'sampling_rate' used in 
           TFOC_pulse_gen (if this is how you the pulses were calculated).
           units: 1 data point/ns    
    display_num : integer
        Default None.
        if this number is specified, sets the x limit to [0, display_num]
        
    Returns
    -------
    cost: difference between the calculated value with mesolve and the measurement
    '''

    gates = np.linspace(0,maxM-1,maxM)
    
    if clf == None:
        clf, _ = get_classifier()

    with h5py.File(new_path,'r') as datfile:
        Navg = int(Nshots)
        I = datfile['OPX_data/output_arrays/I/value'][:]
        Q = datfile['OPX_data/output_arrays/Q/value'][:]
        
    scale_factor = 1e2

    g0 = np.hstack([I.flatten().reshape(-1,1),Q.flatten().reshape(-1,1)])*scale_factor
    prob = clf.predict_proba(g0)

    prob_c = np.matmul(prob.reshape((-1, Navg, 3)), correction)
    
    p_avg = prob.reshape(-1, Navg, 3).mean(axis=1)
    p_avg_c = prob_c.reshape(-1, Navg, 3).mean(axis=1)
    
    if mesolve[0]:
        Htot = generate_Hamiltonian(3)
        
        starting, xpath, ypath, downsamp = mesolve[1]
        t = np.arange(0, len(xpath), 1)/downsamp
        
        concat_pulse = np.stack((xpath,ypath), axis=-1) *2*np.pi*1e-3
        
        prob_me_time, prob_me_gate, numgate = qt_mesolve(Htot, 3, starting, concat_pulse, t, numgate=maxM, cop=True)
    
    fig, ax = plt.subplots(2,2, figsize=(18, 10))
    fig.suptitle(new_path)
    
    for i in range(3):
        ax[0,0].plot(gates, p_avg[:,i], '.-', label=str(i))
    try:
        for i in range(4):
            ax[0,0].plot(prob_me_gate[i], '--', label=str(i))
    except:
        pass

    ax[0,0].grid()
    ax[0,0].set_xlabel('Gate repetition')
    ax[0,0].set_ylabel('Population')
    ax[0,0].set_title('As measured')
    ax[0,0].legend()
    
    if display_num is not None:
        ax[0,0].set_xlim(0, display_num)

    for i in range(3):
        ax[0,1].plot(gates, p_avg_c[:,i], '.-', label=str(i))
    try:
        for i in range(4):
            ax[0,1].plot(prob_me_gate[i], '--', label=str(i))
    except:
        pass
    
    ax[0,1].grid()
    ax[0,1].set_xlabel('Gate repetition')
    ax[0,1].set_ylabel('Population')
    ax[0,1].set_title('Confusion matrix')
    ax[0,1].legend()

    cost = []
    try:
        for i in range(3):
            ax[1,0].plot(gates, p_avg_c[:,i] - prob_me_gate[i], label='Corrected '+str(i))
            ax[1,0].plot(gates, p_avg[:,i] - prob_me_gate[i], '--', label='As measured '+str(i))
            cost.append(p_avg[:,i] - prob_me_gate[i])
    except:
        pass
    
    ax[1,0].grid()
    ax[1,0].set_xlabel('Gate repetition')
    ax[1,0].set_ylabel('Cost function')
    ax[1,0].set_title('Cost function')
    ax[1,0].legend()
    
    ax[1,1].hist2d(I, Q, bins=160);
    plt.show()

    return np.array(cost)
    

def plot_optimizer_opx(path):
    with h5py.File(path, 'r') as datfile:
            amps=datfile['OPX_data/output_arrays/x/value'][:].reshape((-1,3))
            damps=datfile['OPX_data/output_arrays/y/value'][:].reshape((-1,3))
            xfinal = datfile['OPX_data/output_arrays/xf/value'][:]
            
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(amps.flatten(),damps.flatten(),'o', label=str(xfinal))
    ax.legend()
    plt.grid()
    
    return xfinal



#####################################################
### Reduce the number of files after optimization ###
#####################################################
def data_reduction(lastfile, nit, clf, file_delete = True, Nshots=1000):
    '''
    Create a summary file from the files created during optimization.
    User can choose to delete or keep the files using 'file_delete' option.
    
    Parameters
    ----------
    lastfile: path
        Last file measured at the end of optimization process
    nit: int
        Number of iterations. 
        result.nit
    clf: classifier object
        classifier
    file_delete : bool
        keep or delete the files
    Nshots: inte
        Needed to reshape the I, Q array if the measurement was not parsable. 
        
    Returns
    -------
    sumfname : path
        summary file name
    lastnum : int
        The number in the last file
    
    '''
    
    tag = lastfile.split('_')[-1]
    
    # find the last number of the iteration
    lastnum = int(tag.split('.hdf5')[0])
    
    fbase = lastfile.split('.hdf5')[0]
    
    sumfname = f'{fbase}_summary.hdf5'
    
    filelist = []
    
    with h5py.File(sumfname, 'w') as f:
        pthstart = fbase[:-(len(str(lastnum))+1)]
        
        for i in np.arange(lastnum +1 - nit,lastnum+1):
            try:
                if i == 0:
                    pth = pthstart + '.hdf5'
                else:
                    pth = pthstart + '_' + str(i) + '.hdf5'
                data, _ = read_opx_hdf5(pth, clf=clf, rs_override=[-1,Nshots])
                
                filelist.append(pth)
                
                a = f.create_dataset(f'data{i}', data = data['classified'])
                with h5py.File(pth, 'r') as f2:
                    metadata = {}
                    for k in f2['OPX_data'].attrs.keys():
                        metadata[k] = f2['OPX_data'].attrs[k]
                    metadata['amp'] = f2['OPX_data'].attrs['QuA_program'].split('amp')[1].split('value')[1].split('"')[1]
                a.attrs.update(metadata)
            except Exception as e:
                print(e)
                pass
        
        if 'tevol' in sumfname:
            f.create_dataset('t (ns)', data = data['array_0'])
        
    if file_delete:
        for f in filelist:
            os.remove(f)
    return sumfname, lastnum


def plot_summary(filename, ax=None):
    '''
    Plot the traces measured during optimization.
    '''
    ax = ax or plt.gca()
    
    with h5py.File(filename, 'r') as datfile:
        datlist = [key for key in datfile.keys()]    
        
        for name in datlist:
            if 'data' in name:
                try:
                    time = datfile['t (ns)'][:]
                    ax.plot(time, datfile[name])
                    ax.set_title('Time evolution', fontsize=16)
                    ax.set_xlabel('Time (ns)', fontsize=16)
                    ax.set_ylabel('Population', fontsize=16)
                except:
                    ax.plot(datfile[name])
                    ax.set_title('Gate repetition', fontsize=16)
                    ax.set_xlabel('Gate repetition', fontsize=16)
                    ax.set_ylabel('Population', fontsize=16)
                    
                    pass
                
                
def plot_optimized(filename, starting, pulse_info, opt, ax=None, maxM = 20):
    '''
    Plot optimized trace and ME solve.
    Optimized trace is when the 'cost' is the lowest. 
    '''
    
    with h5py.File(filename, 'r') as datfile:

        try:
            time = datfile['t (ns)'][:]
            ax.plot(time, datfile[opt])
        except:
            ax.plot(datfile[opt])

    try:
        Htot = generate_Hamiltonian(3)

        xpath, ypath, downsamp = pulse_info
        t = np.arange(0, len(xpath), 1)/downsamp

        concat_pulse = np.stack((xpath,ypath), axis=-1) *2*np.pi*1e-3

        prob_me_time, prob_me_gate, numgate = qt_mesolve(Htot, 3, starting, concat_pulse, t, numgate = maxM, cop=True)
    except:
        pass

    for i in range(3):
        if 'tevol' in filename:
            ax.plot(t, prob_me_time[i].real, '--', label=str(i))
            ax.set_xlabel('Time (ns)', fontsize=16)
        elif 'gate_rep' in filename:
            ax.plot(np.arange(numgate), prob_me_gate[i].real, '--', label=str(i))
            ax.set_xlabel('Gate repetition', fontsize=16)
        ax.set_title('Optimized', fontsize=16)
        ax.set_ylabel('Population', fontsize=16)
        ax.legend()

##################################
### plot amplitude calibration ###
##################################

def plot_amp_calibration(new_path, ampi, ampf, ampstep, Nshots, clf=None, target=None):
    '''
    Plot the results of gate_amp_calibration. 
    
    Parameters
    ----------
    new_path : path to data
    ampi : initial amplitude of scan
    ampf : final amplitude of scan
    ampstep : step in scan
    Nshots : measurements per amplitude
    clf : classifier
    target (optional) : array [state, population]. Use this to pick a state
       and the target population. When provided, the function will print out
       the location where the specified state is closest to the target population.
    
    Returns
    -------
    amplitude dataframe
    
    '''

    amp_array = np.arange(ampi, ampf, ampstep)
    
    if clf == None:
        clf, _ = get_classifier()

    with h5py.File(new_path,'r') as datfile:
        Navg = int(Nshots)
        I = datfile['OPX_data/output_arrays/I/value'][:]
        Q = datfile['OPX_data/output_arrays/Q/value'][:]
    
    scale_factor = 1e2

    g0 = np.hstack([I.flatten().reshape(-1,1),Q.flatten().reshape(-1,1)])*scale_factor
    prob = clf.predict_proba(g0)

    p_avg = prob.reshape(-1, Navg, 3).mean(axis=1)
    
    # because the amplitude scan is from [ampi,ampf) (excluding ampf),
    # sometimes the amp_array we calculate has an extra point. If so, 
    # remove it.
    if len(amp_array) > len(p_avg[:,0]):
        amp_array = amp_array[:len(p_avg[:,0])]
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    fig.suptitle(new_path)
    
    for i in range(3):
        ax[0].plot(amp_array, p_avg[:,i], label=str(i))
    
    ax[0].grid()
    ax[0].set_xlabel('Amplitude ratio')
    ax[0].set_ylabel('Population')
    ax[0].set_title('Amplitude calibration')
    
    amp_list = []
    for i in range(3):
        amp_list.append([amp_array[p_avg[:,i].argmin()], amp_array[p_avg[:,i].argmax()]])
        print(f'min {i}: {amp_array[p_avg[:,i].argmin()]}, max {i}: {amp_array[p_avg[:,i].argmax()]}')
    
    
    amp_df = pd.DataFrame(np.array(amp_list).reshape(-1,2), columns=['min', 'max'])
    
    if target is not None:
        opt_amp = amp_array[np.abs((p_avg[:, target[0]] - target[1])).argmin()]
        amp_df.loc[len(amp_df.index)] = [opt_amp, np.nan]
        amp_df.rename(index={3: 'Optimal'})
        print(f'Optimized ampltude: {opt_amp}')
            
    ax[1].hist2d(I,Q, bins=200);
    plt.show()

    return amp_df


################################
### Plot Process Matrix data ###
################################

def plot_gate_ops_sub(new_path, ax, maxM, Nshots, clf=None, tags='', correction=[True, [[1,0,0],[0,1,0],[0,0,1]]], save=False):

    gates = np.linspace(0,maxM-1,maxM)
    plot_title = 'i'+new_path.split('init')[1].split(tags)[0]
    
    if clf == None:
        clf, _ = get_classifier()
    
    with h5py.File(new_path,'r') as datfile:
        Navg = int(Nshots)
        I = datfile['OPX_data/output_arrays/I/value'][:]
        Q = datfile['OPX_data/output_arrays/Q/value'][:]

    scale_factor = 1e2

    g0 = np.hstack([I.flatten().reshape(-1,1),Q.flatten().reshape(-1,1)])*scale_factor
    prob = clf.predict_proba(g0)
    
    p_avg = prob.reshape(-1, Navg, 3).mean(axis=1)
    
    if correction[0]:
        prob_c = np.matmul(prob.reshape((-1, Navg, 3)), correction[1])
        p_avg_c = prob_c.reshape(-1, Navg, 3).mean(axis=1)
        p_plot = p_avg_c
    else:
        p_plot = p_avg        
    
    if save:
        if tags[0] == '_':
            tags = tags[1:]
        else:
            tags = tags
            
        df = pd.DataFrame(p_plot)
        today = date.today().strftime("%y%m%d")
        fdir = os.getcwd() + '/saved_dat/process/' + today + '/' + tags+ '/'
        isExist = os.path.exists(fdir)
        
        if not isExist:
            os.makedirs(fdir)

        fname = new_path.split('\\')[-1].split('.hdf5')[0]
        filename = duplicate(fdir, fname, '.dat')
        df.to_csv(fdir+filename)
    
    
    for i in range(3):
        ax.plot(gates, p_avg[:,i], '.-', label=str(i))
    
    try:
        ax.set_title(plot_title, fontsize=10)
    except:
        pass
    ax.legend()
    ax.set_ylim(0,1)
    ax.set_xlim(0,20)
    

def duplicate(fdir, filename, extension):
    
    file_list = os.listdir(fdir)
    fexist = [fname for fname in file_list if filename in fname]
    
    if len(fexist) != 0:
        filename = filename + '_' + str(len(fexist)) + extension
    else:
        filename = filename + extension
    
    return filename
    
    
def save_unitary(tag, randU):
    today = date.today().strftime("%y%m%d")
    
    if tag[0] == '_':
        tag = tag[1:]
    else:
        tag = tag
        
    fdir = os.getcwd() + '/saved_dat/process/' + today + '/' + tag + '/'
    isExist = os.path.exists(fdir)
        
    if not isExist:
        os.makedirs(fdir)
            
    fname = duplicate(fdir, 'unitary'+tag, '.dat')
    
    with open(fdir+fname, 'w') as f:
        f.write('# random Unitary\n')
        for data_slice in randU.full():
            for d in data_slice:
                f.write(str(d))
                f.write('\t')
            f.write('\n')
            

def save_measurement_info(tag, init_states, post_ops, amp, Nshots, maxM):
    today = date.today().strftime("%y%m%d")
    
    if tag[0] == '_':
        tag = tag[1:]
    else:
        tag = tag
        
    fdir = os.getcwd() + '/saved_dat/process/' + today + '/' + tag+ '/'
    isExist = os.path.exists(fdir)
        
    if not isExist:
        os.makedirs(fdir)
            
    fname = duplicate(fdir, 'measurement_info'+tag, '.dat')
    with open(fdir+fname, 'w') as f:
        f.write('# init_states :\n')
        f.write(str(init_states))
        f.write('\n')
        
        f.write('# post operators :\n')
        f.write(str(post_ops))
        f.write('\n')
        
        f.write('# amplitude :\n')
        f.write(str(amp))
        f.write('\n')
        
        f.write('# number of shots :\n')
        f.write(str(Nshots))
        f.write('\n')
        
        f.write('# number of gates : \n')
        f.write(str(maxM))
        f.write('\n')
            
    
def plot_process_matrix_data(datfiles, froot, tags, init_array, post_op_array, maxM, Nshots, clf=None, correction=[True, [[1,0,0],[0,1,0],[0,0,1]]], save=False):
    
    if clf == None:
        clf, _ = get_classifier()
    
    if datfiles == []:
        for fdir in froot:
            for file in glob.glob(fdir+f'\*pop*{tags}.hdf5'):
                datfiles.append(file)
        
    if len(datfiles) == len(init_array) * len(post_op_array):
        rows = len(init_array)
        cols = len(post_op_array)
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4,rows*3))
        fig.subplots_adjust(hspace = .2, wspace=.2)
        axes = axes.flatten()

        for i, path in enumerate(datfiles):
            initial_state = int(path.split('init')[1].split('_')[0])
            plot_gate_ops_sub(path, ax=axes[initial_state*cols+i%cols], maxM=maxM, Nshots=Nshots, clf=clf, tags=tags, correction=correction, save=save)

    else:
        print('Check the data file list. len(data files) should be the same as len(initial states)*len(post operators)')
        
        
##############################
### Plot Phase measurement ###
##############################

def sine(x, x0):
    return np.sin(2*np.pi*(x-x0)/2)**2

def plot_phase(new_path, Nshots, clf=None, ai=0.5, af=1, astep=0.1):
    angle = np.arange(ai, af+astep, astep)
    
    if clf == None:
        clf, _ = get_classifier()

    with h5py.File(new_path,'r') as datfile:
        
        I = datfile['OPX_data/output_arrays/I/value'][:]
        Q = datfile['OPX_data/output_arrays/Q/value'][:]
        # i_avg = I.reshape(-1,Navg).mean(axis=1)
        # q_avg = Q.reshape(-1,Navg).mean(axis=1)

        IData = I.reshape((-1, len(angle), int(Nshots)))
        QData = Q.reshape((-1, len(angle), int(Nshots)))

        ii = IData*1e2
        qq = QData*1e2

        g0 = np.hstack([ii.flatten().reshape(-1,1),qq.flatten().reshape(-1,1)])
        prob = clf.predict_proba(g0)

        p_avg = prob.reshape(-1, int(Nshots), 3).mean(axis=1)

        fig, ax = plt.subplots()
        
        for i in range(3):
            ax.plot(angle, p_avg[:,i], '.-', label=str(i))
            
        ax.plot(angle, sine(angle, 0),'--',label='Initial')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:1.1g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/2))
        plt.legend()

        popt, pcov = sp.optimize.curve_fit(sine, angle, p_avg[:,1], p0=0)
        print('angle: ', popt)
        ax.plot(angle, sine(angle, *popt), '--', label='Fit')
        ax.set_title(new_path)
        plt.show()

