#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 
np.random.seed(9001)

### Define some case's coefficients
def get_deterministic_params(N, h_amp, U_amp, J_amp):
    h = h_amp * np.ones(N) 
    U = U_amp * np.ones(N-1)
    J = J_amp * np.ones(N-1)
    return h, U, J

def get_xx_params(N, J_amp):
    h = np.zeros(N)
    U = np.zeros(N-1)
    J = J_amp * np.ones(N-1)
    return h, U, J

def get_disordered_xx_params(N, h_amp, J_amp):
    h = np.random.uniform(-h_amp, h_amp, N)
    U = np.zeros(N)
    J = J_amp * np.ones(N)
    return h, U, J

def get_xxz_params(N, U_amp, J_amp):
    h = np.zeros(N)
    U = U_amp * np.ones(N)
    J = J_amp * np.ones(N)
    return h, U, J

def get_disordered_xxz_params(N, h_amp, U_amp, J_amp):
    h = np.random.uniform(-h_amp, h_amp, N)
    U = U_amp * np.ones(N)
    J = J_amp * np.ones(N)
    return h, U, J

def mapCoeffs_SpinChainToQuandary(N:int, h:list, U:list, J:list):
	"""
	Map spin chain coefficents J, U and h onto Quandary coefficients

	Parameters:
	------------
 	N	: int			:  number of spin sites
 	J	: float			:  J-coefficient per linear chain coupling: J[i] couples i and i+1
 	U	: list(float)	:  U-coefficient per linear chain coupling: J[i] couples i and i+1
 	h	: list(float)	:  h-coefficient per spin site
	
	Output:
	--------
 	freq01	  :  01-transition frequency per qubit (omega_j)
 	crossker  :  crossker coupling (linear chain) (xi_ij)
 	Jkl       :  dipole-dipole coupling (linear chain) (J_ij)
	"""

	# 01 transition frequencies [GHz] per site (omega_q) 
	freq01 = np.zeros(N)
	for i in range(1,N-1):
		freq01[i] = (-2*h[i] -2*U[i] -2*U[i-1]) / (2*np.pi)
	freq01[0]   = (-2*h[0]   - 2*U[0] ) / (2*np.pi)
	freq01[N-1] = (-2*h[N-1] - 2*U[N-2]) / (2*np.pi)
	# Jkl and Xi term [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
	Jkl=[]
	crosskerr=[]
	couplingID = 0
	for i in range(N):
		for j in range(i+1, N):
			if j == i+1:  # linear chain coupling
				valJ = - 2*J[couplingID] / (2*np.pi)
				valC = - 4*U[couplingID] / (2*np.pi)
			else:
				valJ = 0.0
				valC = 0.0
			Jkl.append(valJ)
			crosskerr.append(valC)
		couplingID += 1
	return freq01, crosskerr, Jkl
#####

# Specify the number of spin sites
N = 8

# Get spin chain case coefficients
U_amp = 1.0 
J_amp = 1.0 
h_amp = 1.0
h, U, J = get_disordered_xx_params(N, U_amp, J_amp)

# Specify the initial state (list of 0 or 1 per site). Here: domain wall |111...000>
initstate= np.zeros(N, dtype=int)
for i in range(int(N/2)):
   initstate[i] = 1 

# Set the simulation duration and step size
T = 10.0
# T = 4.0
dT = 0.01 	# Double check if this is small enough (e.g. by re-running with smaller dT and compare results)

# Quandary run options
verbose = True
ncores=8   				# Numbers of cores for Quandary 
mpi_exec="mpirun -np" 	# MPI executable, e.g. "srun -n" for LC 

# Prepare Quandary
initcondstr = "pure, "
for e in initstate:
	initcondstr += str(e) +", "
freq01, crosskerr, Jkl = mapCoeffs_SpinChainToQuandary(N, h, U, J)
quandary = Quandary(Ne=[2 for _ in range(N)], Ng=[0 for _ in range(N)], freq01=freq01, rotfreq=np.zeros(N), crosskerr=crosskerr, Jkl=Jkl, initialcondition=initcondstr, T=T, dT=dT, initctrl_MHz=0.0, carrier_frequency=[[0.0] for _ in range(N)], verbose=verbose)

# Storage for averaged magnetization. Matrix rows = sites, cols = time
magnet = np.zeros((N,quandary.nsteps+1))
# Storage for averaged domain wall spread
nhalf = np.zeros(quandary.nsteps+1)

# Number of samples for h
# nsamples = 10
nsamples = 1

# Iterate over randomized h and run quandary
for isample in range(nsamples):

	# Sample new set of coefficients and map to Quandary's coeffs
	h, U, J = get_disordered_xx_params(N, U_amp, J_amp)
	quandary.freq01, quandary.crosskerr, quandary.Jkl = mapCoeffs_SpinChainToQuandary(N, h, U, J)
	quandary.update() 

	# Run forward simulation
	datadir = "./N"+str(N)+"_sample" + str(isample)+"_run_dir_parallel"  
	t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(datadir=datadir, maxcores=ncores, mpi_exec=mpi_exec)

	# Compute magnetization from expected Energy (-2*expEnergy + 1) and add to average
	for isite in range(N):
		for nt in range(len(t)):
			exp = expectedEnergy[isite][0][nt] ## expected energy for this site and time
			magnet_this = -2.0*exp + 1.0
			magnet[isite,nt] += magnet_this / nsamples

	# Compute domain wall spread (N_1/2) from expected Energy and add to average
	for nt in range(len(t)): # iterate over time
		nhalf_i = 0.0
		for isite in range(int(N/2)): # iterate over first half of the sites
			nhalf_i -= expectedEnergy[isite][0][nt] 
		nhalf_i += int(N/2)
		nhalf[nt] += nhalf_i / nsamples

# # Plot sigmaz dynamics of one spin site
# isite = 0
# myexp = [-2*e+1 for e in expectedEnergy[isite][0]]
# plt.plot(t, myexp)
# plt.show()

# Plot heatmap of magnetization
fig, ax = plt.subplots(figsize=(6,4))
mycmap = plt.get_cmap('coolwarm')
plt.imshow(magnet, interpolation='none', aspect='auto', cmap=mycmap)
plt.colorbar()
plt.title(r"Heat Map of Spin Chain $\langle \sigma^z_j \rangle$")
plt.xlabel("Time-step index")
plt.ylabel("Spin Site Index $j$")
plt.show()

# Plot domain wall spread:
plt.plot(t, nhalf)
plt.xlabel("Time-step index")
plt.ylabel("N_1/2")
plt.grid(True)
plt.legend()
plt.show()

