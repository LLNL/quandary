Gate optimization CNOT:
  * Optimizes for a CNOT gate on two coupled qubits each modelled with 2 energy levels. System configuration for the 'Geb' resonator at LLNL.
  * T = 200ns, time step size = 0.1ns
  * 'cnot.cfg': Runs a closed-system (Schroedinger eq.) optimization using random initial control parameters. Can be run on up to 4 cores (one for each initial basis state)
  * 'cnot_FWD_optimized.cfg': Evaluates the fidelity of the control parameters stored in 'params_optimized.dat' by forward simulation (Schroedinger's equation)
  * 'cnot_FWD_optimized_withnoise.cfg': Same as above, but simulates with Lindblads master equation (with decoherence). 

Gate optimization SWAP02:
  * Considers a qudid modelled with 3 essential energy levels and one guard level
  * Optimizes for a SWAP02 gate that swaps the |0> with the |1> state. 
  * Schroedinger solver (-> closed-system optimization)
  * Can be run on up to 3 compute cores (one for each initial condition)

State-to-state:
  * Optimized for pulses that transfer the ground state of a 2-level qubit to the maximally mixed state [1/sqrt(2), 1/sqrt(2)].
  * Schroedinger's solver (closed-system optimization)
  * Can run on one core (one initial condition)

3x20_AliceCavity_cooling:
  * Models a qudit (3 levels, "Alice") coupled to a readout cavity (20 levels). System configuration for the 'Nut' resonator at LLNL.
  * Optimization towards the pure |00> state of the coupled system (ground-state reset)
  * The initial state is the ensemble state over the qudit's space dimension (spanning all possible qudit states), coupled to the ground state in the cavity.
  * The configuration file 'AxC.cfg' is set to simulate T=2.5us, using a stepsize of dt=1e-5 (i.e. N=250000 time steps). 
  * Applies previously optimized control parameters (param_optimized.dat) that drive the qudit and the cavity (almost) to the ground state. (Parameters are not fully optimal.)
  * Note that since many tiny time-steps are performed, this testcase takes a while to simulate (and hence to optimize)...

