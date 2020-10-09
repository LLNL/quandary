# Pi-Pulse experiment. Calibrating Nut. 

AxC.cfg is set for Nut resonator, simulating a bipartite system with 3x20 levels: Alice ("0") and  cavity ("1"). It runs the following experiment:

1) Start with alice in |1>, cavity in |0> state.
2) Drive cavity at ground frequency for <t> us with amplitude <amp>, zero drive for alice.
3) Apply pi-pulse to alice for 0.104us with amplitude 15.68rad/us.
4) At <t>+0.104 us, measure expected energy level for alice (0) and cavity (1), and photon number (population) for alice. 

Current setting in AxC.cfg simulates for a total of 2.104us, driving the cavity at ground frequency with amplitude <amp>=5rad/us for <t>=2us, then applies a pi-pulse to alice for 0.104us.

## Submit a bunch of experiments and plot 
To repeat this experiment and gather results for various durations <t> and cavity drive strengths <amp>, use the python script `submit_pipulse_study.py' (tested for python3, likely works with python2). Adapt the setting in the beginning of the file! Current setting sweeps over amp=5,9 and 20 durations in [0.1,2.0]us.

The script creates subfolders for each of the specified amplitudes and durations, flushes the config file there, and executes the job in that subfolder (if it doesn't exist yet). Then, the results are gathered from the subfolders and flushed to files expected_<name>.dat, and population_alice.dat, in matrix form: Rows are for different time-durations, first column lists time-duration, other columns list the result for each of the amplitudes: expected energy level (one value per amplitude) or population (three values per amplitude)

Results for experiments sweeping over amplitudes 1,2,...,9 rad/us, with cavity drive durations in [0.05, 2.0], spacing 0.05 have been added to the repo:
    * expected_<name>.dat : one row per time duration, first column is time, remaining 9 columns are expected energy level after the pi pulse for each of the 9 amplitudes. 
    * population_alice.dat : one row per time duration, first column is time, remaining 27 columns are alice's population (3 values) after the pi pulse for each of the 9 amplitude. 


Use 'plot_results.py' to plot the results from those matrix files using python's matplotlib. 
