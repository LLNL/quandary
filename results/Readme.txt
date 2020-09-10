 AxC_cooling:
 * optimized Alice-Cavity cooling problem (3x20 levels), configured for Spencer's resonator 'Nut": 
    - configuration file 'AxC.cfg' is set to simulate T=2.5us, using a stepsize of dt=1e-5 (i.e. N=250000 time steps). 
    - applies optimized control parameters (param_optimized.dat) for driving the expected value of Alice's and the cavities energy level to zero.

cnot2-pcof-opt-alpha-0.15_orig.dat:
 * Ander's optimized control spline amplitudes for 2-oscillator, 2-level test case. 150 coefficients per control functions, 4 control functions. 
 * Ordering is (c1re, c2re, c1im, c2im) where c1re corresponds to the matrix (a1+a1.dag)
 * Reordered for my design storage being (c1re, c1im, c2re, c2im) in file cnot2-pcof-opt-alpha-0.15.dat.

 param_optimized_CNOT.dat:
 * These are optimized parameters realizing a CNOT gate. Optimization has been performed on the Diagonal of the density only. The initial parameters were Ander's parameters. 

