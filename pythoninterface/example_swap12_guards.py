from quandary import * 

## Two qubit test case for the SWAP gate using two essential and two guard levels each ##

Ne = [2,2]  # Number of essential energy levels
Ng = [2,2]  # Number of extra guard levels
# -> Hilbert space dimension is (2+2)^2.

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.914, 5.114] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.33, 0.23]
# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [5.0e-3]       

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the time duration (ns)
T = 200.0

# Bspline spacing (ns) for control pulse parameterization. // The number of Bspline basis functions is then T/dtau + 2.
dtau = 10.0 # 3.33

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 30.0*np.ones(len(freq01))  

# Set the amplitude of initial (randomized) control vector for each oscillator 
amp_frac = 0.5
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(freq01))]

# Number of points to resolve the shortest period of the dynamics
Pmin = 40  # 60 # 40 # 80

# Set up a target gate (in essential level dimensions). Here: SWAP gate
unitary = np.identity(4)
unitary[1,1] = 0.0
unitary[1,2] = 1.0
unitary[2,1] = 1.0
unitary[2,2] = 0.0
# print("Target gate: ", unitary)


# Quandary run options
runtype = "simulation" # "optimization" #  # "simulation", or "gradient", or "optimization"
# quandary_exec="/Users/guenther5/Numerics/quandary/main"
quandary_exec="/Users/petersson1/src/quandary/main"
# quandary_exec="/cygdrive/c/Users/scada-125/quandary/main.exe"
ncores = 4  # Number of cores 
# ncores = 1
datadir = "./SWAP12_guard_run_dir"  # Compute and output directory 
verbose = True

# Prepare Quandary
myconfig = QuandaryConfig(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, Jkl=Jkl, rotfreq=rotfreq, T=T, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, targetgate=unitary, verbose=verbose, dtau=dtau, Pmin=Pmin)


# Potentially load previously optimized control parameters from a file. Use absolute path here.
# myconfig.pcof0_filename = os.getcwd() + "/" + datadir + "/params.dat" 

# Execute quandary
pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir, runtype=runtype)


print(f"Fidelity = {1.0 - infidelity}")
print("\n Quandary data directory: ", datadir)

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, myconfig.time, pt, qt)
	plot_expectedEnergy(myconfig.Ne, myconfig.time, expectedEnergy)