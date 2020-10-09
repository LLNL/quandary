# Pi-Pulse experiment. Calibrating Nut. 

AxC.cfg is set for Nut resonator, simulating a bipartite system with 3x20 levels: Alice ("0") and  cavity ("1")

## Constant-then-PiPulse experiment:
1) Start with alice in |1>, cavity in |0> state.
2) Drive cavity at ground frequency for <t> us with amplitude <amp>, zero drive for alice.
3) Apply pi-pulse to alice for 0.104us with amplitude 15.68rad/us.
4) At <t>+0.104 us, measure expected energy level for alice (0) and cavity (1), and photon number (population) for alice. 

Current setting in AxC.cfg simulates for 2.104us, driving the cavity at ground frequency with amplitude 5rad/us for 2us, then applies a pi-pulse to alice for 0.104us.

## Submit a bunch of experiments and plot 
To repeat this experiment for various durations <t> and cavity drive strengths <amp>, use the python script `submit_pipulse_study.py' (tested for python3, likely works with python2). 
Adapt the setting in the beginning of the file! 

I've added data files for experiments sweeping over amplitudes 1,2,...,9 rad/us (columns), with cavity drive durations in [0.05, 2.0] (rows), spacing 0.05, see *.dat files. Use the gnuplot script for visualization:
gnuplot> load 'plot_expected.plt'

