from quandary import * 
from bqskit import MachineModel
from bqskit.ir.gates import ConstantGate, QubitGate
from bqskit import Circuit
from bqskit import compile
from bqskit.ir import CircuitIterator, CycleInterval
from bqskit.ir.operation import Operation
from bqskit.ir.gates import CZGate, RZGate, SXGate, IdentityGate
from bqskit.compiler.machine import QubitSpec 
from bqskit.qis.graph import CouplingGraph
## Note: BQSKIT installed as developer in ~/Numerics/bqskit/

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

with_guard_level=False			# Switch using one guard level
simulate_lindblad = False
prefixfolder = "quandary_rundir/"			# Folder where to put all quandary run data
rand_seed = 34321

# Define Qubit specs (01-frequency and anharmonicity per qubit [GHz], optional T1 and T2 times [ns])
specs = [
    QubitSpec(freq01=5.18, anharmonicity=0.210), 
    QubitSpec(freq01=5.12, anharmonicity=0.211),
    QubitSpec(freq01=5.06, anharmonicity=0.212),
    # QubitSpec(freq01=4.94, anharmonicity=0.21),
    # QubitSpec(freq01=5.18, anharmonicity=0.210, T1=80000.0, T2=30000.0), 
    # QubitSpec(freq01=5.12, anharmonicity=0.211, T1=85000.0, T2=35000.0),
    # QubitSpec(freq01=5.06, anharmonicity=0.212, T1=90000.0, T2=40000.0),
]
nqubits = len(specs)

# Define qubit connectivity and Dipole-dipole coupling strength
Jkl_coupling = {(0,1): 0.005, 
				# (0,2): 0.005,
				(1,2): 0.005
				}

# Set up the Machine Model
gate_set = {CZGate(), RZGate(), SXGate(), IdentityGate()} 
edges = [tuple(sorted(e)) for e in Jkl_coupling]
machine_model = MachineModel(nqubits, 
							 gate_set=gate_set, 
							 coupling_graph=CouplingGraph(edges, coupling_strengths=Jkl_coupling), 
							 qubit_specs=specs)

print("Qubit specs: ")
for i in range(nqubits):
	print("  01 freq: ", machine_model.freq01(i), ", anharmonicity: ", machine_model.anharmonicity(i), ", T1: ", machine_model.T1(i), ", T2: ", machine_model.T2(i))
print("Coupling: ")
for q1, q2 in machine_model.coupling_graph: # each tuple is an edge
    J = machine_model.J(q1, q2)  # or model.coupling_graph.coupling(q1,q2)
    print(f"  edge {q1}-{q2}:  J = ", J ," GHz")


# Specify the gate durations for 1,2,3,4,5 - qubit gates
def get_pulse_duration(nqubits):
	T_all = [66.6, 4*66.6, 8*66.6, 900.0, 1500.0]  # ns
	if nqubits>5:
		print("ERROR: Specify gate duration for ", nqubits, "-gates.")
		stop
	return T_all[nqubits-1]

# Set constant rotating frame frequency
rotfreq_const = sum([machine_model.freq01(i) for i in range(nqubits)]) / nqubits

# Max control pulse amplitude
maxctrl_MHz = 15.0

# Hardcoding a fixed time-step size for Quandary that is small enough for both single- and two-qubit gates. 
# 2-levels single-qubit: ~0.5ns, two-qubit: ~0.15ns
# 3-levels single-qubit: ~0.05ns, two-qubit: ~0.015ns
# dT = 0.015 if with_guard_level else 0.15  # ns
dT = 0.015   # ns

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
QFT_circuit = compile(QFT_unitary, model=machine_model, optimization_level=3, seed=1234, error_threshold=1e-4)
QFT_unitary = QFT_circuit.get_unitary() # Apparently, this is not exactly the QFT gate anymore! 
# print("Fidelity of exact gate vs compiled gate: ", fidelity_(QFT_unitary, QFT_circuit.get_unitary()))

# # FOR TESTING: A simple circuit
# QFT_circuit = Circuit(nqubits)
# QFT_circuit.append_gate(RZGate(), 0, [1.57])
# QFT_circuit.append_gate(RZGate(), 1, [3.95])
# QFT_circuit.append_gate(RZGate(), 2, [1.18])
# QFT_circuit.append_gate(SXGate(), 0)
# QFT_circuit.append_gate(SXGate(), 1)
# QFT_circuit.append_gate(SXGate(), 2)
# QFT_circuit.append_gate(RZGate(), 0, [4.21])
# QFT_circuit.append_gate(RZGate(), 1, [6.26])
# QFT_circuit.append_gate(SXGate(), 2)
# QFT_circuit.append_gate(SXGate(), 0)
# QFT_circuit.append_gate(SXGate(), 1)
# QFT_circuit.append_gate(RZGate(), 0, [1.48])
# QFT_circuit.append_gate(RZGate(), 1, [3.15])
# QFT_circuit.append_gate(CZGate(), (0,1) )
# QFT_circuit.append_gate(RZGate(), 0, [4.8])
# QFT_circuit.append_gate(SXGate(), 1)
# QFT_circuit.append_gate(SXGate(), 0)
# QFT_circuit.append_gate(RZGate(), 1, [5.38])
# QFT_circuit.append_gate(RZGate(), 0, [5.5])
# QFT_circuit.append_gate(RZGate(), 1, [3.02])
# QFT_circuit.append_gate(SXGate(), 0)
# QFT_circuit.append_gate(CZGate(), (1,2) )
# QFT_circuit.append_gate(RZGate(), 0, [11.3])
# QFT_circuit.append_gate(RZGate(), 1, [4.17])
# QFT_circuit.append_gate(SXGate(), 2)
# QFT_unitary = QFT_circuit.get_unitary()

# # Convert to qiskit and draw the circuit
# from bqskit.ext import bqskit_to_qiskit
# from qiskit import QuantumCircuit
# qs = bqskit_to_qiskit(QFT_circuit)
# qs.draw(output="mpl", filename="QFT_mpl.png") # the matplotlib output is in color
# # qs.draw(output="latex", filename="QFT_ltx.png")
# # qs.draw()
# stop

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

	# Get the cycle duration (duration of the largest gate in this cycle)
	maxduration = get_pulse_duration(max(op.num_qudits for op in current_cycle_ops))
		
	# For each operation, either optimize pulses, or grab from storage
	for op in current_cycle_ops:

		# Optimize for pulse, if it doesn't exist yet
		if not op.get_pulse():

			duration = get_pulse_duration(op.num_qudits)
			gate = op.get_unitary()
			gate_name = str(op.gate)+str(op.location)+str(op.params)
			t, pt, qt = QuandaryOptimize(op.location, op.radixes, gate, gate_name, machine_model, duration, dT, maxctrl_MHz, with_guard_level, rotfreq_const, prefixfolder, rand_seed)

			# Store optimized pulse in this operation's gate class
			op.add_pulse(t, pt, qt)

		# Now concatenate the pulse to the global storage
		append_pulse(op.location, op.get_pulse()["times"],op.get_pulse()["p_pulse"], op.get_pulse()["q_pulse"], dT, t_concat, p_concat, q_concat)

		# If current gate duration is smaller than the longest gate duration in this cycle, we need to insert an identity pulse (or stretch the gate optimization pulse to span this longer duration, TODO.)
		if  op.get_pulse()["times"][-1] < maxduration:

			duration = maxduration - op.get_pulse()["times"][-1]
			gate = IdentityGate().get_unitary()
			gate_name = "Identity" + str(op.location)
			t, pt, qt = QuandaryOptimize(op.location, op.radixes, gate, gate_name, machine_model, duration, dT, maxctrl_MHz, with_guard_level, rotfreq_const, prefixfolder, rand_seed)			

			append_pulse(op.location, t, pt, qt, dT, t_concat, p_concat, q_concat)

	# Compute and store this cycle's intermediate target for later use
	t_int = t_concat[0][-1]
	nsteps_int = len(p_concat[0])
	for i in range(len(current_cycle_ops)):
		# Kronecker all gates from this cycle. qubit order 0 x 1 x 2 ...
		if i == 0:
			U_int = current_cycle_ops[0].get_unitary() 
			# print("i=", i, " location ", current_cycle_ops[0].location, " U_int = ", current_cycle_ops[0].gate)
		else:
			U_int = np.kron(U_int, current_cycle_ops[i].get_unitary())
			# print("i=", i, " location ", current_cycle_ops[i].location, " U_int = kron(U_int,", current_cycle_ops[i].gate)
	# Multiply intermediate target with gates from previous cycles and store
	if len(intermediate_targets) > 0:
		Uprev = intermediate_targets[-1][0]
		U_int = U_int @ Uprev
	intermediate_targets.append((U_int, t_int, nsteps_int))


# Simulate the concatenated pulses (Schroedinger)
print("\nSimulate the concatenated pulses...")
gate = QFT_unitary
tstop = t_concat[0][-1]
infidelity, uInt = QuandarySimulate(tstop, p_concat, q_concat, gate, machine_model, dT, with_guard_level, rotfreq_const, prefixfolder)
print(f"Averaged state fidelity of concatenated pulses (Schroedinger): {1.0 - infidelity}")

# Plot concatenated pulses
# plot_pulse([2 for i in range(3)], t_concat[0], p_concat, q_concat)

# Compute intermediate fidelities along the way first  
## First using Schroedinger's eqation (grab the intermediate fidelities):
print("\nIntermediate averaged state fidelity (Schroedinger):")
fidelities_schroed = [1.0]
times = [0.0]
for i in range(len(intermediate_targets)):
	U_int, t_int, nsteps_int = intermediate_targets[i] 
	U_sim = uInt[nsteps_int-1]  # Not available for lindblad
	# fid_int_operator = fidelity_(U_int,U_sim)
	fid_int = avg_fidelity_(U_int, U_sim)
	# print("Intermediate operator fidelity at t=", round(t_int,2), ": ", fid_int_operator)
	print("  t=", round(t_int,2), ": ", fid_int)
	times.append(t_int)
	fidelities_schroed.append(fid_int)

## Now using Lindblad's solver
if simulate_lindblad:
	# machine_model.set_qubit_spec(0, T1=0.0, T2=0.0)
	# machine_model.set_qubit_spec(1, T1=0.0, T2=0.0)
	# machine_model.set_qubit_spec(2, T1=0.0, T2=0.0)
	machine_model.set_qubit_spec(0, T1=80000.0, T2=30000.0)
	machine_model.set_qubit_spec(1, T1=85000.0, T2=35000.0)
	machine_model.set_qubit_spec(2, T1=90000.0, T2=40000.0)
	print("Simulating with Decoherence model:")
	for i in range(nqubits):
		print("  qubit ", i, ":  T1= ", machine_model.T1(i), ", T2= ", machine_model.T2(i))

	print("\nIntermediate averaged state fidelity (Lindblad):")
	fidelities_lind = [1.0]
	times = [0.0]
	for i in range(len(intermediate_targets)):
		U_int, t_int, nsteps_int = intermediate_targets[i] 
		targetgate= U_int
		p_curr = [ [p_concat[iq][nt] for nt in range(nsteps_int)] for iq in range(len(p_concat))]
		q_curr = [ [q_concat[iq][nt] for nt in range(nsteps_int)] for iq in range(len(q_concat))]
		infidelity, _ = QuandarySimulate(t_int, p_curr, q_curr, targetgate, machine_model, dT, with_guard_level, rotfreq_const, prefixfolder+"/Lindblad/", maxcores=4)

		fid_int = 1.0 - infidelity
		print("  t=", round(t_int,2), ": ", fid_int)
		times.append(t_int)
		fidelities_lind.append(fid_int)

# Plot fidelities over time
plt.plot(times, fidelities_schroed, marker='o', linestyle='-', label="Schroedinger")
# if simulate_lindblad:
# plt.plot(times, fidelities_lind, marker='o', linestyle='-', label="Lindblad")
plt.xlabel("Time (ns)")
plt.ylabel("Averaged State Fidelity")
# plt.ylim(0.5,1.05)
plt.grid(True)
plt.legend()
plt.show()

stop

###########
# Now now optimize for 3-qubit gates per cycle:
###########
print("\n#####")
print("Optimize 3-qubit gates per cycle")
print("#####\n")

t_concat_3qubit = [[] for _ in range(nqubits)]
p_concat_3qubit = [[] for _ in range(nqubits)]
q_concat_3qubit = [[] for _ in range(nqubits)]
intermediate_targets_3qubit = []
for cycle_idx in range(QFT_circuit.num_cycles):

	# Get all operations happening at this cycle
	current_cycle_ops = get_cycle_operations(QFT_circuit, cycle_idx)
	print(f"Cycle {cycle_idx}", " # ops = ", len(current_cycle_ops), f" operations: {current_cycle_ops}")


	# Set up 3-qubit target gate
	gate = IdentityGate(num_qudits=1).get_unitary()
	for op in current_cycle_ops:
		gate = np.kron(gate, op.get_unitary())

	# Get the cycle duration (duration of the largest gate in this cycle)
	duration = get_pulse_duration(3)
			
	# Optimize for pulse, if it doesn't exist yet
	gate_name = "cycle"+str(cycle_idx)

	print(" -> Optimizing pulses for operation ", op)
	t, pt, qt = QuandaryOptimize((0,1,2), [2,2,2], gate, gate_name, machine_model, duration, dT, maxctrl_MHz, with_guard_level, rotfreq_const, prefixfolder+"3qubit_gates", rand_seed)

	# Now concatenate the pulse to the global storage
	append_pulse((0,1,2), t, pt, qt, dT, t_concat_3qubit, p_concat_3qubit, q_concat_3qubit)

	# Store this cycle's intermediate target for later use
	t_int = t_concat_3qubit[0][-1]
	nsteps_int = len(p_concat_3qubit[0])
	if len(intermediate_targets_3qubit) > 0:
		Uprev = intermediate_targets_3qubit[-1][0]
		gate = gate @ Uprev
	intermediate_targets_3qubit.append((gate, t_int, nsteps_int))

# Simulate the concatenated pulses (Schroedinger)
print("\nSimulate the concatenated 3-qubit pulses...")
gate = QFT_unitary
tstop = t_concat_3qubit[0][-1]
infidelity, uInt = QuandarySimulate(tstop, p_concat_3qubit, q_concat_3qubit, gate, machine_model, dT, with_guard_level, rotfreq_const, prefixfolder)
print(f"Averaged state fidelity of concatenated pulses (Schroedinger): {1.0 - infidelity}")


# Plot concatenated pulses
# plot_pulse([2 for _ in range(3)], t_concat_3qubit[0], p_concat_3qubit, q_concat_3qubit)

# Compute intermediate fidelities along the way
## First using Schroedinger's eqation (grab the intermediate fidelities):
print("\nIntermediate averaged state fidelity (Schroedinger):")
fidelities_3qubit_schroed = [1.0]
times_3qubit = [0.0]
for i in range(len(intermediate_targets_3qubit)):
	U_int, t_int, nsteps_int = intermediate_targets_3qubit[i] 
	U_sim = uInt[nsteps_int-1]
	# fid_int = fidelity_(U_int,U_sim)
	fid_int = avg_fidelity_(U_int, U_sim)
	print("  t=", round(t_int,2), ": ", fid_int)
	times_3qubit.append(t_int)
	fidelities_3qubit_schroed.append(fid_int)

## Now using Lindblad's solver
if simulate_lindblad:
	# machine_model.set_qubit_spec(0, T1=0.0, T2=0.0)
	# machine_model.set_qubit_spec(1, T1=0.0, T2=0.0)
	# machine_model.set_qubit_spec(2, T1=0.0, T2=0.0)
	machine_model.set_qubit_spec(0, T1=80000.0, T2=30000.0)
	machine_model.set_qubit_spec(1, T1=85000.0, T2=35000.0)
	machine_model.set_qubit_spec(2, T1=90000.0, T2=40000.0)
	print("Simulating with Decoherence model:")
	for i in range(nqubits):
		print("  qubit ", i, ":  T1= ", machine_model.T1(i), ", T2= ", machine_model.T2(i))

	print("\nIntermediate averaged state fidelity (Lindblad):")
	fidelities_3qubit_lind = [1.0]
	times = [0.0]
	for i in range(len(intermediate_targets_3qubit)):
		U_int, t_int, nsteps_int = intermediate_targets_3qubit[i] 
		targetgate= U_int
		p_curr = [ [p_concat_3qubit[iq][nt] for nt in range(nsteps_int)] for iq in range(len(p_concat_3qubit))]
		q_curr = [ [q_concat_3qubit[iq][nt] for nt in range(nsteps_int)] for iq in range(len(q_concat_3qubit))]
		infidelity, _ = QuandarySimulate(t_int, p_curr, q_curr, targetgate, machine_model, dT, with_guard_level, rotfreq_const, prefixfolder+"3qubit_gates"+"/Lindblad/", maxcores=4)

		fid_int = 1.0 - infidelity
		print("  t=", round(t_int,2), ": ", fid_int)
		times.append(t_int)
		fidelities_3qubit_lind.append(fid_int)

# Plot fidelities over time
plt.plot(times_3qubit, fidelities_3qubit_schroed, marker='o', linestyle='-', label="Schroedinger")
if simulate_lindblad:
	plt.plot(times_3qubit, fidelities_3qubit_lind, marker='o', linestyle='-', label="Lindblad")
plt.xlabel("Time (ns)")
plt.ylabel("Averaged State Fidelity")
# plt.ylim(0.5,1.05)
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

