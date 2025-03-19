#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 

## QFT gates on qubit chain, two levels, no guard levels, dipole-dipole coupling 5KHz (chain topology) ##

# ALL PROBLEMS encoded here
freq01_all = [5.18, 5.12, 5.06, 5.0, 4.94] 
T_all = [50, 200, 500, 900, 1500]
Jkl_coupling = 5e-3  	# Dipole-Dipole for CHAIN topology

## Number of qubits ##
nqubits = 3

# Runtypes
verbose = False
do_optim = True
do_plot = True 

rand_seed=1234
maxcores=8
maxiter=500

# Penalty terms
gamma_energy = 1e-4  	# Control pulse energy integral
gamma_tik0 = 1e-3		# Tikhonov regularization
gamma_dpdm = 0.0		# State variation

# Control pulse settings 
maxctrl_MHz = 25.0 			 # Maximum control pulse amplitude
spline_knot_spacing =  3.0   # Bspline spacing [ns]

# Carrier wave thresholds
cw_amp_thres = 5e-2  # Min. theshold on growth rate for each carrier
cw_prox_thres = 1e-3 # Max. threshold on carrier proximity

# Step-size control: Min. number of time-steps per fastest oscillation in Hsys,Hcontrol
Pmin = 150*2	# Default: 150 (might be enough)

# Set up frequency vector
freq01 =  []	# 01 transition frequencies [GHz] 
Ne = []
Ng = []
for i in range(nqubits):
	freq01.append(freq01_all[len(freq01_all)-i-1])
	Ne.append(2)
	Ng.append(0)
print("Frequencies: ", freq01)
print("Ne: ", Ne)

# Set up qubit CHAIN coupling 
Jkl=[]
for i in range(len(Ne)):
	for j in range(i+1, len(Ne)):
		if j == i+1:
			val = Jkl_coupling
		else:
			val = 0.0
		Jkl.append(val)
print("Coupling: ", Jkl)

# Set the pulse duration (ns)
T = T_all[nqubits-1]
print("T=", T)

# Set up rotational frame frequency
favg = sum(freq01)/len(freq01) 
rotfreq = favg*np.ones(len(freq01))

# Define target gate: d-dimensional Discrete Fourier Transform
def get_QFT_gate(dim):
	gate_Hd =  np.zeros((dim, dim), dtype=complex)
	om_d = np.exp(1j*2*np.pi/dim)

	for j in range(dim):
		for k in range(dim):
			gate_Hd[j,k] = om_d**(j*k) / np.sqrt(dim)

	return gate_Hd
unitary = get_QFT_gate(np.prod(Ne))
# print("Target gate: ", unitary)

# Set up the Quandary configuration for this test case
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, Pmin=Pmin, spline_knot_spacing=spline_knot_spacing, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, maxiter=maxiter, maxctrl_MHz=maxctrl_MHz, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, gamma_energy=gamma_energy, gamma_tik0=gamma_tik0, gamma_dpdm=gamma_dpdm) 


# Potentially, load initial control parameters from a file. 
# quandary.pcof0_filename = os.getcwd() + "/"+datadir+"/params.dat"  # absolute path!

# Execute quandary
datadir = "./QFT_"+str(nqubits)+"_run_dir"  
if do_optim:
	t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(datadir=datadir, maxcores=maxcores)
	print(f"Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if do_plot:
	plot_pulse(quandary.Ne, t, pt, qt)
	plot_expectedEnergy(quandary.Ne, t, expectedEnergy) 
	# plot_population(quandary.Ne, t, population)


