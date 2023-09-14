include("./quandary.jl")

# One qudit test case: Swap the 0 and 2 state of a three-level qudit
Ne = [3]  # Number of essential energy levels

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
unitary = Matrix{ComplexF64}([0.0 0.0 1.0; 0.0 1.0 0.0; 1.0 0.0 0.0])  # Swaps first and last level

# # Set the location of the quandary executable (absolute path!)
quandary_exec = "/Users/guenther5/Numerics/quandary/main"
# # quandary_exec = "/cygdrive/c/Users/scada-125/quandary/main.exe"

# Print out stuff
verbose = true

# Prepare Quandary configuration.
# The dataclass 'QuandaryConfig' gathers all default settings. You can change the defaults by passing them to the constructor.
myconfig = QuandaryConfig(Ne=Ne, freq01=freq01, selfkerr=selfkerr, rotfreq=rotfreq, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, verbose=verbose)


pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec, datadir="./SWAP02_run_dir_julia")

# println("\nFidelity = ", 1.0 - infidelity)