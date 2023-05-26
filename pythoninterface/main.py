import os
import numpy as np
from QuandaryPy import preprocess
from QuandaryPy import init_control
from QuandaryPy import quandary

## One qubit test case ##

Ne = [2]  # Number of essential energy levels
Ng = [2]  # Number of extra guard levels

# 01 transition frequencies
freq01 = [5.12] 
# Anharmonicities
selfkerr = [0.34]
# Coupling
Jkl = []   # no Jaynes-Cummings coupling
crosskerr = []  # no crossker coupling
# Setup frequency of rotations in computational frame
rotfreq = freq01
# Set the initial duration (ns)
T = 80.0
# Bspline spacing (ns) and number of splines
dtau = 3.33
nsplines = int(np.max([np.ceil(T/dtau + 2), 5])) # 10
# Points per shortest period
Pmin = 40  # 60 # 40 # 80
# If Lindblad solver: Specify decay (T1) and dephasing (T2) 
T1 = [10.0]
T2 = [20.0]

# Bounds on the ctrl vector (rot frame) for each qubit
maxctrl_MHz = [5.0]  # Will be divided by Nfreq internally

# Maximum amplitude of initial control vector
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(maxctrl_MHz))]

# Directory for running quandary 
datadir = "."
os.makedirs(datadir, exist_ok=True)


# Set up Hamiltonians in essential levels only
Hsys, Hc_re, Hc_im = preprocess.hamiltonians(Ne, freq01, selfkerr, crosskerr, Jkl, rotfreq=rotfreq)
print(f"{Hsys =}")
print(f"{Hc_re=}")
print(f"{Hc_im=}")

# Estimate number of time steps
nsteps = preprocess.estimate_timesteps(T, Hsys, Hc_re, Hc_im, maxctrl_MHz, Pmin=Pmin)
print("Final time: ",T,"ns, Number of timesteps: ", nsteps,", dt=", T/nsteps, "ns")
print("Maximum control amplitudes: ", maxctrl_MHz, "MHz")

# Estimate carrier wave frequencies
carrierfreq, growth_rate = preprocess.get_resonances(Ne, Hsys, Hc_re, Hc_im, verbose=True) 
# Keyword argument defaults:
# cw_amp_thres = 6e-2   
# cw_prox_thres = 1e-3  
print("Carrier frequencies: ", carrierfreq)


# Set up the initial control parameter
rand_seed = 1234
randomize_init_ctrl = True
pcof0 = init_control.init_control(initctrl_MHz=initctrl_MHz, nsplines=nsplines, carrierfreq=carrierfreq, rand_seed=rand_seed, randomize=randomize_init_ctrl, verbose=True)
# Keyword argument defaults:
# startfile=""

# Write pcof0 to file. This could be inside the init_control function. TODO.
initialpcof_filename =  datadir + "/params.dat"
with open(initialpcof_filename, "w") as f:
    for value in pcof0:
        f.write("{:20.13e}\n".format(value))
print("Initial control params written to ", initialpcof_filename)


# TODO: Set up a target gate (full dimensions, including guard levels??
dim = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])
targetgate = np.zeros((dim,dim))
targetgate[0,1] = 1.0
targetgate[1,0] = 1.0
print(targetgate)

# Write target gate to file
gatefilename = datadir + "targetgate.dat"
gate_vectorized = np.concatenate((np.real(targetgate).ravel(), np.imag(targetgate).ravel()))
with open(gatefilename, "w") as f:
    for value in gate_vectorized:
        f.write("{:20.13e}\n".format(value))
print("Target gate written to ", initialpcof_filename)


# Optimization options
costfunction = "Jtrace"
initialcondition = "basis"  # TODO? 
gamma_tik0 = 1e-4
# gamma_dpdm = 0.01 
gamma_energy = 0.01
tol_infidelity = 1e-3
tol_costfunc= 1e-3
maxiter = 100


# Write Quandary configuration file
runtype = "simulation"
config_filename = quandary.write_config(Ne=Ne, Ng=Ng, T=T, nsteps=nsteps, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, crosskerr=crosskerr, Jkl=Jkl, nsplines=nsplines, carrierfreq=carrierfreq, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, maxctrl_MHz=maxctrl_MHz, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, initialpcof_filename=initialpcof_filename, gatefilename=gatefilename)
# other keyword arguments defaults:
# print_frequency_iter = 1


# Call Quandary
quandary_exec="./main"
ncores = np.prod(Ne)
quandary.execute(runtype=runtype, ncores=ncores, quandary_exec=quandary_exec, config_filename=config_filename)


popt, infidelity, optim_hist = quandary.get_results(datadir)
print(f"{infidelity = }")

# TODO:
#   * Ubasis set up as matrix or specify via string 
#   * Get results from quandary
#   * Gather configuration in a dictionary (or other struct) that contains all defaults and allows for changes. Pass that dictionary around
#   * Test init_control loading from file
#   * CHECK quandary branch to add dpdm!

# Note: 
#   * leakage_weights = [0.0, 0.0] is disabled.
#   * "use_eigenbasis" disabled.
#   * Init controls for standard control parameterization. Always use initctrl_MHz to specify the amplitude (aka no growthrate!)

