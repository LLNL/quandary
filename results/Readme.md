2x2_cnot:
  * Optimizes for a CNOT gate on two coupled qubits with 2 levels each. System configuration for the 'Geb' resonator at LLNL.
  * T = 200ns, time step size = 0.1ns
  * run optimization from scratch (using random initial controls) with config file 'cnot.cfg', or 
  * run forward simulation of the optimized controls (params_optimized.dat) using 'cnot_FWD_optimized.cfg'


3x20_AliceCavity_cooling:
  * Models a qudit (3 levels, "Alice") coupled to a readout cavity (20 levels). System configuration for the 'Nut' resonator at LLNL.
  * Optimization towards the pure |00> state of the coupled system (ground-state reset)
  * The initial state is the ensemble state over the qudit's space dimension (spanning all possible qudit states), coupled to the ground state in the cavity.
  * The configuration file 'AxC.cfg' is set to simulate T=2.5us, using a stepsize of dt=1e-5 (i.e. N=250000 time steps). 
  * Applies previously optimized control parameters (param_optimized.dat) that drive the qudit and the cavity (almost) to the ground state. (Parameters are not fully optimal.)
  * Note that since many tiny time-steps are performed, this testcase takes a while to simulate (and hence to optimize)...


3x20_pipulse_experiment:
  * Same system configuration as for the alice-cavity cooling test case above.
  * The configuration file 'AxC.cfg' is set up to apply a pi-pulse to the qudit ("Alice") at ground frequency starting at T=2us$ for a total of 0.104us with amplitude 15.10381. No drive for the cavity.


