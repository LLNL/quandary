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
verbose = false

# Prepare Quandary configuration.
# The dataclass 'QuandaryConfig' gathers all default settings. You can change the defaults by passing them to the constructor.
myconfig = QuandaryConfig(Ne=Ne, freq01=freq01, selfkerr=selfkerr, rotfreq=rotfreq, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, verbose=verbose)


# Execute quandary. Default number of executing cores is the essential Hilbert space dimension.
pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec, datadir="./SWAP02_run_dir_julia")
println("\nFidelity = ", 1.0 - infidelity)

# Plot the control pulse and expected energy level evolution
if true
    pl1 = plot_pulse(myconfig.Ne, myconfig.time, pt, qt)
    pl2 = plot_expectedEnergy(myconfig.Ne, myconfig.time, expectedEnergy)
end


# Other optimization results can be accessed from the myconfig, in particular:
#   myconfig.popt         :  Optimized control parameters
#   myconfig.optim_hist   :  Optimization convergence history
#   myconfig.time         :  Time points where the expected energy is stored


# You can change configuration options directly, without creating a new QuandaryConfig instance. In most cases however, it is advised to call myconfig.update() afterwards to ensure that number of time steps and carrier wave frequencies are being re-computed.
# E.g. if you want to change the pulse length, this will work:
#    myconfig.T = 200.0
#    myconfig.update()
# pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec)

# If you want to run Quandary using previously optimized control parameters, this will do it:
#    myconfig.pcof0= myconfig.popt
#    pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec, runtype="simulation")     # Note the runtype.
# (myconfig.update() is not needed in this case)