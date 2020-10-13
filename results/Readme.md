3x20_AliceCavity_cooling:
  * optimized Alice-Cavity cooling problem (3x20 levels), configured for Nut resonator.
  * configuration file 'AxC.cfg' is set to simulate T=2.5us, using a stepsize of dt=1e-5 (i.e. N=250000 time steps). 
  * applies optimized control parameters (param_optimized.dat) for driving the expected value of Alice's and the cavities energy level to zero.

2x2_cnot:
  * optimized a 2x2 level system for the CNOT gate, configured for Geb resonator.
  * T = 200ns, time step size = 0.1ns
  * run optimization from scratch (using random initial controls) with config file 'cnot.cfg', or 
  * run forward simulation of the optimized controls (param_optimized.dat) using 'cnot_FWD_optimized.cfg'

