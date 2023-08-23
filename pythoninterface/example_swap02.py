from quandary import * 

## One qubit test case ##
Ne = [3]  # Number of essential energy levels
Ng = [0]  # Number of extra guard levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198]
# Frequency of rotations for computational frame [GHz] per oscillator
rotfreq = freq01

# Set the total time duration (ns)
T = 100.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = [10.0]  

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level
# print(unitary)

# # Potentially create custom Hamiltonian model, if different from standard model
# Ntotal = [sum(x) for x in zip(Ne, Ng)]
# Hsys, Hc_re, Hc_im = hamiltonians(Ntotal, freq01, selfkerr, crosskerr, Jkl, rotfreq=rotfreq, verbose=True)

# Potentially load initial control parameters from a file. Use absolute path!
# pcof0_filename = os.getcwd() + "/SWAP02_params.dat"
# pcof0=np.zeros(30)

# Set the location of the quandary executable (absolute path!)
quandary_exec="/Users/guenther5/Numerics/quandary/main"
# quandary_exec="/cygdrive/c/Users/scada-125/quandary/main.exe"

# Print out stuff
verbose = True

# Prepare Quandary
myconfig = QuandaryConfig(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, rotfreq=rotfreq, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, verbose=verbose)

# Execute quandary
time, pt, qt, ft, expectedEnergy, popt, infidelity, optim_hist = quandary_run(myconfig, quandary_exec=quandary_exec)


print(f"\nFidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if False:
    plot_pulse(myconfig.Ne, time, pt, qt)
    plot_expectedEnergy(myconfig.Ne, time, expectedEnergy)


# TODO:
#   * All function call arguments should be keyword only.  
#   * Switch between standard model and self-created Hamiltonian operators -> modify pulse_gen arguments
#   * Create high-level functions for pulse_gen vs simulation
#   * Change return of pulse_gen to match Tensorflow returns
#   * Ander: Get resonances: Iterating over non-zeros of U'HcU, should be all elements? Previously it was only lower triangular matrix, but it is not hermitian, so this makes a difference!
#   * Anders: Culled and sorted carrier waves, is that needed? 
    #   # CNOT case with Jkl coupling is not converging!
#   * Add custom decoherence operators to Quandary? 
#   * Gather all configuration in a dictionary (or other struct) that contains all defaults and allows for changes. Dump it with 
    # with open("config.cfg", 'w') as f:
    #     for key, value in config_dict.items():
    #         f.write('%s = %s' %(key, value))

# Note: 
#   * pcof0 uses Quandaries initialization
#   * leakage_weights = [0.0, 0.0] is disabled.
#   * "use_eigenbasis" disabled.