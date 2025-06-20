def my_plot_time_evolution(new_path, starting, Nshots, tsteps, clf=None, mesolve=[False, []], correction=[[1,0,0],[0,1,0],[0,0,1]]):
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
    diff, prob_avg, prob_corr_avg

    '''
    print("Inside my_plot_time_evolution")
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
       
    ax[0].grid()
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Population')
    ax[0].set_title('As measured')
    ax[0].legend()
        
    for i in range(3):
        ax[1].plot(tsteps, p_avg_c[:,i], label=str(i))
    
    ax[1].set_xlabel('Time (ns)')
    ax[1].set_ylabel('Population')
    ax[1].set_title('After correction')
    ax[1].grid()
    ax[1].legend()

    cost = []
    # Add simulated populations
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
            ax[1].plot(t, prob_me_time[i].real, '--', label=str(i))

        try:
            for i in range(3):
                cost.append(prob_me_time[i][(tsteps)*downsamp-1] - p_avg[:,i])
        
        except Exception as e:
            print(e)
            for i in range(3):
                cost.append(prob_me_time[i][(tsteps*downsamp-1)[:-1]] - p_avg[:,i])
    # end if mesolve[0]
    
    ax[2].hist2d(I, Q, bins=200);
        
    plt.show()
    
    return np.array(cost), p_avg, p_avg_c
