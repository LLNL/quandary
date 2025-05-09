from quandary import * 
from bqskit import MachineModel
from bqskit.ir.gates import ConstantGate, QubitGate
from bqskit import Circuit
from bqskit import compile
from bqskit.ir import CircuitIterator, CycleInterval
from bqskit.ir.operation import Operation
from bqskit.ir.gates import CZGate, RZGate, SXGate, IdentityGate
## Note: BQSKIT install as developer in ~/Numerics/bqskit/

# get all operations at a current circuit cycle
def get_cycle_operations(circuit, cycle_idx):
	current_cycle_ops = [] 
	for qudit_idx in range(circuit.num_qudits):

		try:
			operation = QFT_circuit.get_operation((cycle_idx, qudit_idx))
			if operation not in current_cycle_ops:
				current_cycle_ops.append(operation)
		except:
			# continue  # Skip idle qubits
			# Add identity gate for idling qubits
			identity_gate = IdentityGate()
			identity_operation = Operation(identity_gate, [qudit_idx])
			current_cycle_ops.append(identity_operation)

	return current_cycle_ops



nqubits = 2
with_guard_level=False			# Switch using one guard level
constant_rotating_frame=True    # True: Use average qubit frequency as rotating frame. False: Rotating in each qubit frequency individually.
prefixfolder = "quandary_rundir/"			# Folder where to put all quandary run data
rand_seed = 34321

# 01 transition frequencies and anharmonicities for all qubits
freq01 = [5.18, 5.12, 5.06, 5.0, 4.94]    # GHz
selfkerr = [0.21, 0.21, 0.21, 0.21, 0.21] # GHz
rotfreq = freq01
if constant_rotating_frame:
	favg = sum(freq01[0:nqubits])/len(freq01[0:nqubits])
	rotfreq = favg*np.ones(len(freq01[0:nqubits]))
# Dipole-dipole coupling strength (same for all couplings for now), linear chain
coupling_strength = 0.005  # GHz
coupling_graph = [] 
for i in range(nqubits-1):
	coupling_graph.append((i,i+1)) # Qubit coupling graph: Linear chain of qubits

# Max control pulse amplitude
maxctrl_MHz = 15.0

# Gate durations for 1,2,3,4,5 - qubit gates
T_all = [66.6, 250.0, 500.0, 900.0, 1500.0]  # ns
# Hardcoding a fixed time-step size for Quandary that is small enough for both single- and two-qubit gates. 
# 2-levels single-qubit: ~0.5ns, two-qubit: ~0.15ns
# 3-levels single-qubit: ~0.05ns, two-qubit: ~0.015ns
dT = 0.015 if with_guard_level else 0.15  # ns

# Decoherence times for all qubits (for simulation of concatenated pulses only)
T1=[80000.0, 85000.0]  #ns
T2=[30000.0, 35000.0]  #ns

## Target algorithm: QFT 
def QFT_gate(nqubits):
	dim = 2**nqubits
	gate_Hd =  np.zeros((dim, dim), dtype=complex)
	om_d = np.exp(1j*2*np.pi/dim)
	for j in range(dim):
		for k in range(dim):
			gate_Hd[j,k] = om_d**(j*k) / np.sqrt(dim)
	return gate_Hd
QFT_unitary = QFT_gate(nqubits)

# Compile QFT gate with BQSKit
gate_set = {CZGate(), RZGate(), SXGate(), IdentityGate()} 
machine_model = MachineModel(nqubits,  coupling_graph=coupling_graph, gate_set=gate_set)

QFT_circuit = compile(QFT_unitary, model=machine_model, optimization_level=3, seed=1234, error_threshold=1e-4)
QFT_unitary = QFT_circuit.get_unitary() # Apparently, this is not exactly the QFT gate anymore! 
# print("Fidelity of exact gate vs compiled gate: ", fidelity_(QFT_unitary, QFT_circuit.get_unitary()))

## FOR TESTING: A simple circuit
# QFT_circuit = Circuit(nqubits)
# # QFT_circuit.append_gate(RZGate(), 0, [6.283188424662465])
# # # # # QFT_circuit.append_gate(RZGate(), 0, [4.7124])
# QFT_circuit.append_gate(SXGate(), 0 )
# QFT_circuit.append_gate(SXGate(), 0 )
# # # # QFT_circuit.append_gate(RZGate(), 0, [4.901146114674484e-06])
# QFT_circuit.append_gate(SXGate(), 1 )
# # QFT_circuit.append_gate(RZGate(), 1, [3.262])
# # QFT_circuit.append_gate(CZGate(), (0,1) )
# # # QFT_circuit.append_gate(CZGate(), (0,1) )
# QFT_circuit.append_gate(SXGate(), 1 )
# QFT_circuit.append_gate(CZGate(), (0,1) )
# QFT_circuit.append_gate(RZGate(), 0, [4.901146114674484e-06])
# # QFT_circuit.append_gate(CZGate(), (0,1) )
# QFT_unitary = QFT_circuit.get_unitary()

# Convert to qiskit and draw the circuit
# from bqskit.ext import bqskit_to_qiskit
# from qiskit import QuantumCircuit
# # qs = bqskit_to_qiskit(QFT_circuit)
# qs.draw(output="mpl", filename="QFT_mpl.png") # the matplotlib output is in color
# # qs.draw(output="latex", filename="QFT_ltx.png")
# # qs.draw()
# # stop

# THIS IS THE STORAGE FOR THE CONCATENATED PULSES
t_concat = [[] for _ in range(nqubits)]
p_concat = [[] for _ in range(nqubits)]
q_concat = [[] for _ in range(nqubits)]

# Storage for intermediate targets after each cycle: [ (U_int, time, nsteps) ]
intermediate_targets = []

# Iterate over circuit cycles, optimize for pulses for each operation per cycle:
for cycle_idx in range(QFT_circuit.num_cycles):

	# Get all operations at this cycle
	current_cycle_ops = get_cycle_operations(QFT_circuit, cycle_idx)
	print(f"Cycle {cycle_idx}", " # ops = ", len(current_cycle_ops), f" operations: {current_cycle_ops}")

	# For each operation, either optimize pulses, or grab from storage
	for op in current_cycle_ops:

		# Optimize for pulse, if it doesn't exist yet
		if not op.get_pulse():

			# Currently only 2-qubit gates!
			assert(op.num_qudits<=2)

			# Set up a Quandary instance to optimize for this gate
			spline_order = 2   # 2nd order Bspline parameterization
			spline_knot_spacing = 3.0  # ns. Maybe 8.25 if op.num_qudits==1 
			quandary_i = Quandary(
				Ne = [op.radixes[i] for i in range(op.num_qudits)],
				Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)], 
				freq01 = [freq01[op.location[i]] for i in range(op.num_qudits)], 
				selfkerr= [selfkerr[op.location[i]] for i in range(op.num_qudits)], 
				rotfreq=[rotfreq[op.location[i]] for i in range(op.num_qudits)], 
				Jkl = coupling_strength*np.ones(np.sum([i for i in range(op.num_qudits)])), # This only works if no larger than two-qubit gate! 
				T = T_all[op.num_qudits-1], 
				targetgate=op.get_unitary(), 
				dT = dT, 
				maxctrl_MHz=maxctrl_MHz,
				control_enforce_BC=True,
				spline_order=spline_order,
				spline_knot_spacing=spline_knot_spacing, 
				verbose=False, 
				rand_seed=rand_seed)

			print(" -> Optimizing pulses for operation ", op)
			datadir = prefixfolder+str(op.gate)+str(op.location)+ str(op.params) + "_rundir"
			t, pt, qt, infidelity, expectedEnergy, population = quandary_i.optimize(datadir=datadir)
			# If didn't converge, retry with new random seed
			if infidelity > 1e-3:
				quandary_i.rand_seed = quandary_i.rand_seed+34234
				t, pt, qt, infidelity, expectedEnergy, population = quandary_i.optimize(datadir=datadir)
				if infidelity > 1e-3:
					print("\nQuandary did not converge. CHECK LOGS in", datadir, "\n")
					stop

			# Store optimized pulse in this operation's gate class
			op.add_pulse(t, pt, qt)

		# Now concatenate the pulse to the global storage
		append_pulse(op.location, op.get_pulse(), dT, t_concat, p_concat, q_concat)

	# Compute and store this cycle's intermediate target for later use
	t_int = t_concat[0][-1]
	nsteps_int = len(p_concat[0])
	for i in range(len(current_cycle_ops)):
		# Kronecker all gates from this cycle. qubit order 0 x 1 x 2 ...
		if i == 0:
			U_int = current_cycle_ops[0].get_unitary() 
		else:
			U_int = np.kron(U_int, current_cycle_ops[i].get_unitary())
	# Multiply intermediate target with gates from previous cycles and store
	if len(intermediate_targets) > 0:
		Uprev = intermediate_targets[-1][0]
		U_int = U_int @ Uprev
	intermediate_targets.append((U_int, t_int, nsteps_int))


# Simulate the concatenated pulses (Schroedinger)
print("\nSimulate the concatenated pulses...")
quandary_concat= Quandary(
    Ne = [2 for i in range(nqubits)],
	Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)],
	freq01 = [freq01[i] for i in range(nqubits)],
	rotfreq =[rotfreq[i] for i in range(nqubits)],
	selfkerr =[selfkerr[i] for i in range(nqubits)],
	Jkl = coupling_strength*np.ones(np.sum([i for i in range(nqubits)])), # This only works if no larger than two-qubit gate!
	T = t_concat[0][-1],
	dT = dT,
	targetgate = QFT_unitary,
	spline_order = 0,
	spline_knot_spacing = dT,
	control_enforce_BC = True,
	verbose=False,
	rand_seed=rand_seed,
)
datadir = prefixfolder+"./ConcatenatedGates_rundir"
t, pt, qt, infidelity, expectedEnergy, population= quandary_concat.simulate(pt0=p_concat, qt0=q_concat, datadir=datadir, maxcores=8)
# Compute the averaged state fidelity: 
infidelity = 1.0 - avg_fidelity_(quandary_concat.uT, QFT_unitary)
print(f"Averaged state fidelity of concatenated pulses (Schroedinger): {1.0 - infidelity}")

# Plot concatenated pulses
# plot_pulse(quandary_concat.Ne, t_concat[0], p_concat, q_concat)

# ## TEST RECOMPUTE WITH LINDBLAD and T1=T2=0.0. This should give the SAME fidelity as above Schroedinger 
# print("\nSimulate with Lindblad T1=T2=0.0 for testing...")
# quandary_concat.T1 = [0.0, 0.0]
# quandary_concat.T2 = [0.0, 0.0]
# quandary_concat.initialcondition="diagonal"
# quandary_concat.update()
# datadir = prefixfolder+"./ConcatenatedGates_lindblad_rundir"
# t, pt, qt, infidelity_lind, expectedEnergy, population= quandary_concat.simulate(pt0=p_concat, qt0=q_concat, datadir=datadir, maxcores=4)
# print(f"Fidelity of concatenated pulses (Lindblad): {1.0 - infidelity_lind}")
# stop

# Compute intermediate fidelities along the way first  
## First using Schroedinger's eqation (grab the intermediate fidelities):
print("\nIntermediate averaged state fidelity (Schroedinger):")
fidelities_schroed = [1.0]
times = [0.0]
for i in range(len(intermediate_targets)):
	U_int, t_int, nsteps_int = intermediate_targets[i] 
	U_sim = quandary_concat.uInt[nsteps_int-1]  # Not available for lindblad
	# fid_int_operator = fidelity_(U_int,U_sim)
	fid_int = avg_fidelity_(U_int, U_sim)
	# print("Intermediate operator fidelity at t=", round(t_int,2), ": ", fid_int_operator)
	print("  t=", round(t_int,2), ": ", fid_int)
	times.append(t_int)
	fidelities_schroed.append(fid_int)

## Now using Lindblad's solver
print("\nIntermediate averaged state fidelity (Lindblad):")
fidelities_lind = [1.0]
times = [0.0]
for i in range(len(intermediate_targets)):
	U_int, t_int, nsteps_int = intermediate_targets[i] 
	quandary_concat.T1=T1
	quandary_concat.T2=T2
	quandary_concat.initialcondition="diagonal"
	quandary_concat.targetgate= U_int
	quandary_concat.T = t_int
	quandary_concat.nsplines = -1 # need to reset this so that update() recomputes it from scratch using spline_knot_spacing and T
	quandary_concat.update()
	p_curr = [ [p_concat[iq][nt] for nt in range(nsteps_int)] for iq in range(len(p_concat))]
	q_curr = [ [q_concat[iq][nt] for nt in range(nsteps_int)] for iq in range(len(q_concat))]
	t, pt, qt, infidelity, expectedEnergy, population= quandary_concat.simulate(pt0=p_curr, qt0=q_curr, datadir=datadir+"_int"+str(i), maxcores=4)	
	fid_int = 1.0 - infidelity
	print("  t=", round(t_int,2), ": ", fid_int)
	times.append(t_int)
	fidelities_lind.append(fid_int)

# Plot fidelities over time
plt.plot(times, fidelities_schroed, marker='o', linestyle='-', label="Schroedinger")
plt.plot(times, fidelities_lind, marker='o', linestyle='-', label="Lindblad")
plt.xlabel("Time (ns)")
plt.ylabel("Averaged State Fidelity")
plt.ylim(0.5,1.05)
plt.grid(True)
plt.legend()
plt.show()


###########
# Now now optimize for only 2-qubit gates per cycle:
###########
print("\n#####")
print("Optimize 2-qubit gates per cycle")
print("#####\n")

t_concat_2qubit = [[] for _ in range(nqubits)]
p_concat_2qubit = [[] for _ in range(nqubits)]
q_concat_2qubit = [[] for _ in range(nqubits)]
intermediate_targets_2qubit = []
for cycle_idx in range(QFT_circuit.num_cycles):

	# Get all operations happening at this cycle
	current_cycle_ops = get_cycle_operations(QFT_circuit, cycle_idx)
	print(f"Cycle {cycle_idx}", " # ops = ", len(current_cycle_ops), f" operations: {current_cycle_ops}")

	# Set up 2-qubit gate
	if current_cycle_ops[0].num_qudits == 2:
		Utar = current_cycle_ops[0].get_unitary()
		duration = T_all[1]
	else:
		assert(len(current_cycle_ops)==2)
		Utar = np.kron(current_cycle_ops[0].get_unitary(), current_cycle_ops[1].get_unitary())
		duration = T_all[0]

	# Optimize for pulse, if it doesn't exist yet
	spline_order = 2   # 2nd order Bspline parameterization
	spline_knot_spacing = 3.0  # ns. Maybe 8.25 if op.num_qudits==1 
	quandary_i = Quandary(
		Ne = [2 for i in range(op.num_qudits)],
		Ng = [1 if with_guard_level else 0 for _ in range(nqubits)], 
		freq01 = [freq01[i] for i in range(nqubits)], 
		selfkerr= [selfkerr[i] for i in range(nqubits)], 
		rotfreq=[rotfreq[i] for i in range(nqubits)], 
		Jkl = coupling_strength*np.ones(np.sum([i for i in range(nqubits)])), # This only works if no larger than two-qubit gate! 
		T = duration, 
		targetgate=Utar, 
		dT = dT, 
		maxctrl_MHz=maxctrl_MHz,
		control_enforce_BC=True,
		spline_order=spline_order,
		spline_knot_spacing=spline_knot_spacing, 
		verbose=False, 
		rand_seed=rand_seed)

	print(" -> Optimizing pulses for operation ", op)
	datadir = prefixfolder+"cycleID"+str(cycle_idx)+"_rundir"
	t, pt, qt, infidelity, expectedEnergy, population = quandary_i.optimize(datadir=datadir)
	# If didn't converge, retry with new random seed
	if infidelity > 1e-3:
		quandary_i.rand_seed = quandary_i.rand_seed+34234
		t, pt, qt, infidelity, expectedEnergy, population = quandary_i.optimize(datadir=datadir)
		if infidelity > 1e-3:
			print("\nQuandary did not converge. CHECK LOGS in", datadir, "\n")
			stop
		
	pulse_dict = {"location": (0,1),
                  "params"  : [0.0],
                  "times"   : t,
                  "p_pulse" : pt,
                  "q_pulse" : qt
                  }

	# Now concatenate the pulse to the global storage
	append_pulse((0,1), pulse_dict, dT, t_concat_2qubit, p_concat_2qubit, q_concat_2qubit)

	# Store this cycle's intermediate target for later use
	t_int = t_concat_2qubit[0][-1]
	nsteps_int = len(p_concat_2qubit[0])
	if len(intermediate_targets_2qubit) > 0:
		Uprev = intermediate_targets_2qubit[-1][0]
		Utar = Utar @ Uprev
	intermediate_targets_2qubit.append((Utar, t_int, nsteps_int))

# Simulate the concatenated pulses (Schroedinger)
quandary_concat_2qubit = Quandary(
    Ne = [2 for i in range(nqubits)],
	Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)],
	freq01 = [freq01[i] for i in range(nqubits)],
	rotfreq =[rotfreq[i] for i in range(nqubits)],
	selfkerr =[selfkerr[i] for i in range(nqubits)],
	Jkl = coupling_strength*np.ones(np.sum([i for i in range(nqubits)])), # This only works if no larger than two-qubit gate!
	T = t_concat[0][-1],
	dT = dT,
	targetgate = QFT_unitary,
	spline_order = 0,
	spline_knot_spacing = dT,
	control_enforce_BC = True,
	verbose=False,
	rand_seed=rand_seed,
)
datadir = prefixfolder+"./ConcatenatedGates_rundir"
t, pt, qt, infidelity, expectedEnergy, population= quandary_concat_2qubit.simulate(pt0=p_concat_2qubit, qt0=q_concat_2qubit, datadir=datadir, maxcores=8)
print(f"\nAveraged state fidelity of concatenated pulses (Schroedinger): {1.0 - infidelity}")

# Plot concatenated pulses
# plot_pulse(quandary_concat.Ne, t_concat[0], p_concat, q_concat)

# Compute intermediate fidelities along the way
## First using Schroedinger's eqation (grab the intermediate fidelities):
print("\nIntermediate averaged state fidelity (Schroedinger):")
fidelities_2qubit_schroed = [1.0]
times_2qubit = [0.0]
for i in range(len(intermediate_targets_2qubit)):
	U_int, t_int, nsteps_int = intermediate_targets_2qubit[i] 
	U_sim = quandary_concat_2qubit.uInt[nsteps_int-1]
	# fid_int = fidelity_(U_int,U_sim)
	fid_int = avg_fidelity_(U_int, U_sim)
	print("  t=", round(t_int,2), ": ", fid_int)
	times_2qubit.append(t_int)
	fidelities_2qubit_schroed.append(fid_int)

## Now using Lindblad's solver
print("\nIntermediate averaged state fidelity (Lindblad):")
fidelities_2qubit_lind = [1.0]
times_2qubit = [0.0]
for i in range(len(intermediate_targets_2qubit)):
	U_int, t_int, nsteps_int = intermediate_targets_2qubit[i] 
	quandary_concat_2qubit.T1=T1
	quandary_concat_2qubit.T2=T2
	quandary_concat_2qubit.initialcondition="diagonal"
	quandary_concat_2qubit.targetgate= U_int
	quandary_concat_2qubit.T = t_int
	quandary_concat_2qubit.nsplines = -1 # need to reset this so that update() recomputes it from scratch using spline_knot_spacing and T
	quandary_concat_2qubit.update()
	p_curr = [ [p_concat_2qubit[iq][nt] for nt in range(nsteps_int)] for iq in range(len(p_concat_2qubit))]
	q_curr = [ [q_concat_2qubit[iq][nt] for nt in range(nsteps_int)] for iq in range(len(q_concat_2qubit))]
	t, pt, qt, infidelity, expectedEnergy, population= quandary_concat_2qubit.simulate(pt0=p_curr, qt0=q_curr, datadir=datadir+"_int"+str(i), maxcores=4)	
	fid_int = 1.0 - infidelity
	print("  t=", round(t_int,2), ": ", fid_int)
	times_2qubit.append(t_int)
	fidelities_2qubit_lind.append(fid_int)

# Plot fidelities over time
plt.plot(times_2qubit, fidelities_2qubit_schroed, marker='o', linestyle='-', label="Schroedinger")
plt.plot(times_2qubit, fidelities_2qubit_lind, marker='o', linestyle='-', label="Lindblad")
plt.xlabel("Time (ns)")
plt.ylabel("Averaged State Fidelity")
plt.ylim(0.5,1.05)
plt.grid(True)
plt.legend()
plt.show()


######################

# IBM backends: ibm_brisbane, _kyiv, _sherbrook. All are 'Eagle' processors. Only two-qubit gate is the ECR cross-resonance gate. Somehow Bqskit doesn't find an entangling gate for those backends.
# from qiskit_ibm_runtime import QiskitRuntimeService 
# service = QiskitRuntimeService(channel="ibm_quantum", token="b07cf37c6c014fc5dee9bce811efce1f8954a1ceba63eecaa49254c0e4d6c4ce9f6046116d5c293c50404c3008c756fb7c46ed2561c39020716f0f90d6df448c")
# # print available backends and their instructions
# for backend in service.backends():
#     config = backend.configuration()
#     if "simulator" in config.backend_name:
#         continue
#     print(f"Backend: {config.backend_name}")
#     print(f"    Processor type: {config.processor_type}")
#     print(f"    Supported instructions:")
#     for instruction in config.supported_instructions:
#         print(f"        {instruction}")
#     print()
# bc = service.backend("ibm_heron")
# modelBrisbane = model_from_backend(bc)
# Now compile on IBM Brisbane
# QFT_circuit = compile(QFT_unitary, model=modelBrisbane, optimization_level=3)


## This is how you iterate over circuit operations rather than circuit cycles:
# for op in CircuitIterator(QFT_circuit):
	# print("Operation:", op, " params", op.params)

