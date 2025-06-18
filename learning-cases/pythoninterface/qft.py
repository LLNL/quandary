from quandary import * 
from bqskit import MachineModel
from bqskit.ir.gates import ConstantGate, QubitGate
from bqskit import Circuit
from bqskit import compile
from bqskit.ir import CircuitIterator
from bqskit.ir.gates import CZGate, RZGate, SXGate


###
# TODO:
# For single rotated frame: HOW TO DEAL WITH IDLE QUBITS?
# For doubly-rotated frame: Can't translate the pulses because Jkl is time-dependent

nqubits = 2
with_guard_level=False			# Switch using one guard level
constant_rotating_frame=True    # True: Use average qubit frequency as rotating frame. False: Rotating in each qubit frequency individually.

# 01 transition frequencies and anharmonicities for all qubits
freq01 = [5.18, 5.12, 5.06, 5.0, 4.94]    # GHz
selfkerr = [0.21, 0.21, 0.21, 0.21, 0.21] # GHz
rotfreq = freq01
if constant_rotating_frame:
	favg = sum(freq01[0:nqubits])/len(freq01[0:nqubits])
	rotfreq = favg*np.ones(len(freq01[0:nqubits]))

# Qubit coupling graph: Linear chain of qubits
coupling_graph = []
for i in range(nqubits-1):
	coupling_graph.append((i,i+1))

# Dipole-dipole coupling strength (same for all couplings for now).
coupling_strength = 0.005  # GHz
# coupling_strength = 0.0  # GHz

# Max control pulse amplitude
maxctrl_MHz = 15.0

# Gate durations for 1,2,3,4,5 - qubit gates
# T_all = [85.0, 250.0, 500.0, 900.0, 1500.0]  # ns
T_all = [66.6, 250.0, 500.0, 900.0, 1500.0]  # ns

# Hardcoding a fixed time-step size for Quandary that is small enough for both single- and two-qubit gates. 
# 2-levels single-qubit: ~0.5ns, two-qubit: ~0.15ns
# 3-levels single-qubit: ~0.05ns, two-qubit: ~0.015ns
dT = 0.015 if with_guard_level else 0.15  # ns

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
gate_set = {CZGate(), RZGate(), SXGate()} 
machine_model = MachineModel(nqubits,  coupling_graph=coupling_graph, gate_set=gate_set)

QFT_circuit = compile(QFT_unitary, model=machine_model, optimization_level=1, seed=1234)
#QFT_circuit = compile(QFT_unitary, model=machine_model, optimization_level=3)
QFT_unitary = QFT_circuit.get_unitary() # Apparently, this is not exactly the QFT gate anymore! 

# # ## FOR TESTING: A simple circuit
# QFT_circuit = Circuit(nqubits)
# # QFT_circuit.append_gate(RZGate(), 0, [6.283188424662465])
# # # # # QFT_circuit.append_gate(RZGate(), 0, [4.7124])
# QFT_circuit.append_gate(SXGate(), 0 )
# # # # QFT_circuit.append_gate(SXGate(), 0 )
# # # # QFT_circuit.append_gate(RZGate(), 0, [4.901146114674484e-06])
# QFT_circuit.append_gate(SXGate(), 1 )
# # QFT_circuit.append_gate(RZGate(), 1, [3.262])
# # QFT_circuit.append_gate(CZGate(), (0,1) )
# # # QFT_circuit.append_gate(CZGate(), (0,1) )
# # QFT_circuit.append_gate(SXGate(), 1 )
# QFT_circuit.append_gate(CZGate(), (0,1) )
# QFT_circuit.append_gate(RZGate(), 0, [4.901146114674484e-06])
# # QFT_circuit.append_gate(CZGate(), (0,1) )
# QFT_unitary = QFT_circuit.get_unitary()

# Convert to qiskit and draw the circuit
from bqskit.ext import bqskit_to_qiskit
from qiskit import QuantumCircuit
qs = bqskit_to_qiskit(QFT_circuit)
qs.draw()


# Optimize for pulses for each circuit operations
t_op = []
pt_op = []
qt_op = []
pop_op = []
exp_op = []
quandary_op = []
for op in CircuitIterator(QFT_circuit):
	print("Operation:", op, " params", op.params)

	# Check if a pulse for this operation exists, other wise optimize for it
	pulse = op.get_pulse()
	if not pulse:

		# Currently only one and two-qubit gate optimization
		if op.num_qudits > 2:
			print("Gates on more than 2 qubits not ready for setup with Quandary.")
			stop

		# Set up a Quandary instance to optimize for this gate
		Ne = [op.radixes[i] for i in range(op.num_qudits)]
		Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)]
		w01 = [freq01[op.location[i]] for i in range(op.num_qudits)]
		selfk = [selfkerr[op.location[i]] for i in range(op.num_qudits)]
		rotf = [rotfreq[op.location[i]] for i in range(op.num_qudits)]
		gkl = coupling_strength*np.ones(np.sum([i for i in range(op.num_qudits)])) # This only works if no larger than two-qubit gate!
		pulse_duration = T_all[op.num_qudits-1]
		targetgate=op.get_unitary()
		spline_order = 2   # 2nd order Bspline parameterization
		spline_knot_spacing = 8.25 if op.num_qudits==1 else 3.0 # ns 
		control_enforce_BC=True

		quandary_i = Quandary(
			Ne = Ne,
			Ng = Ng,
			freq01 = w01,
			selfkerr=selfk,
			rotfreq=rotf,
			Jkl = gkl,
			targetgate=targetgate,
			T = pulse_duration,
			dT = dT,
			maxctrl_MHz=maxctrl_MHz,
			control_enforce_BC=control_enforce_BC,
			spline_order=spline_order,
			spline_knot_spacing=spline_knot_spacing,
			verbose=False,
			rand_seed=1234,
		)

		# Optimize
		print("Optimizing pulses for gate ", op, ", params=", op.params, ", freq01 = ", w01, ", coupling = ", gkl)
		print(" -> target gate =", targetgate)
		datadir = str(op.gate)+str(op.location)+ str(op.params) + "_rundir"
		print(" -> datadir=", datadir)
		t, pt, qt, infidelity, expectedEnergy, population = quandary_i.optimize(datadir=datadir)
		if infidelity > 1e-3:
			# Try again with new random seed
			quandary_i.rand_seed = quandary_i.rand_seed+34234
			t, pt, qt, infidelity, expectedEnergy, population = quandary_i.optimize(datadir=datadir)
			if infidelity > 1e-3:
				print("\nQuandary did not converge. CHECK LOGS in", datadir, "\n")
				stop

		
		# quandary_i.spline_order=0
		# quandary_i.spline_knot_spacing=20*dT
		# t_pw, pt_pw, qt_pw, infidelity_pw, expectedEnergy, population = quandary_i.simulate(pt0=pt, qt0=qt, datadir=datadir+"_FWD")
		# print("Optimized infid:", 1.0 - infidelity)
		# print("TEST infid pw:", 1.0 - infidelity_pw)
		# stop

		# Store pulse in the operation's gate, for this location and this parameters.
		op.add_pulse(t, pt, qt)

		# Store individual pulses for debugging and plotting later
		quandary_op.append(quandary_i)
		t_op.append(t)
		pt_op.append(pt)
		qt_op.append(qt)
		pop_op.append(population)
		exp_op.append(expectedEnergy)
	else:
		print(" -> Found pulse for gate ", op.get_unitary())


print("Done optimizing for pulses\n")

# Now concatenate while grabbing pulses from the operations
t_global = [[] for _ in range(nqubits)]
p_global = [[] for _ in range(nqubits)]
q_global = [[] for _ in range(nqubits)]
intermediate_targets = []
for op in CircuitIterator(QFT_circuit):
	print("Concatenate pulse for operation:", op, " params ", op.params, " location=", op.location)

	pulse = op.get_pulse()
	assert(pulse)

	append_pulse(op.location, pulse, dT, t_global, p_global, q_global)

	# Store intermediate target
	Ui = op.get_unitary()
	if op.num_qudits == 1:
		Id = np.identity(3 if with_guard_level else 2)
		if op.location[0] == 0:
			Ui = np.kron(Ui, Id)
		else:
			Ui = np.kron(Id, Ui)
	tmax = np.max([t_global[iqubit][-1] if len(t_global[iqubit]) > 0 else 0.0 for iqubit in op.location])
	nstepsmax = np.max([len(t_global[iqubit]) for iqubit in op.location]) -1

	if len(intermediate_targets) == 0:
		intermediate_targets.append((tmax,Ui, nstepsmax))
	else:
		Uprev = intermediate_targets[-1][1]
		intermediate_targets.append((tmax,Ui@Uprev, nstepsmax))

# Synchronize all qubits to final time, padding idle qubits zeros if needed.
synch_zero_padding([i for i in range(nqubits)], dT, t_global, p_global, q_global)
# print("Done concatenating pulses.")

t_concat = t_global[0]  # should be the same on all qubits now!
p_concat = p_global
q_concat = q_global

# Simulate with the concatenated pulses
w01 = [freq01[i] for i in range(nqubits)]
selfk = [selfkerr[i] for i in range(nqubits)]
rotf = [rotfreq[i] for i in range(nqubits)]
gkl = coupling_strength*np.ones(np.sum([i for i in range(nqubits)])) # This only works if no larger than two-qubit gate!
pulse_duration = t_concat[-1]
#
spline_order = 0 # Simulate with step-wise control pulses
spline_knot_spacing = dT # ns. Should equal the discretization size of p(t) and q(t)
control_enforce_BC=True
quandary_concat = Quandary(
    Ne = [2 for i in range(nqubits)],
	Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)],
	freq01 = w01,
	rotfreq=rotf,
	selfkerr=selfk,
	Jkl = gkl,
	T = pulse_duration,
	dT=dT,
	targetgate=QFT_unitary,
	spline_order=spline_order,
	spline_knot_spacing=spline_knot_spacing,
	control_enforce_BC=control_enforce_BC,
	verbose=False,
	rand_seed=1234,
)
datadir = "./ConcatenatedGates_rundir"
t, pt, qt, infidelity, expectedEnergy, population= quandary_concat.simulate(pt0=p_concat, qt0=q_concat, datadir=datadir)
uT = quandary_concat.uT.copy()
print(f"\nFidelity of concatenated pulses (unitary): {1.0 - infidelity}")

# # Fidelity of individual initial states:
# fid = []
# fid.append(fidelity_(uT@[1,0,0,0], QFT_unitary[:,0]))
# fid.append(fidelity_(uT@[0,1,0,0], QFT_unitary[:,1]))
# fid.append(fidelity_(uT@[0,0,1,0], QFT_unitary[:,2]))
# fid.append(fidelity_(uT@[0,0,0,1], QFT_unitary[:,3]))
# print("Fidelity of concatenated pulses on individual initial states: ", fid)

# Plot concatenated pulses
plot_pulse(quandary_concat.Ne, t_concat, p_concat, q_concat)
# # Plot individual pulses
# for i in range(len(t_op)):
	# plot_pulse(quandary_op[i].Ne, t_op[i], pt_op[i], qt_op[i])
	# # plot_expectedEnergy(quandary_op[i].Ne, t_op[i], exp_op[i])

# Simulate up to intermediate targets
for i in range(len(intermediate_targets)):
	t, Ut, nsteps =intermediate_targets[i] 

	# March forward until the time stamp changes
	if len(intermediate_targets)>i+1:
		tnext, Unext, nstepsnext = intermediate_targets[i+1]
		if tnext<= t:
			continue

	# # Get the time and timestep index 
	# t = round(t, 2)
	# tc_rounded = [round(ts,2) for ts in t_concat]
	# index = tc_rounded.index(t)

	# # Cut pulses to this timestep
	# p_cut = [pi[0:index+1] for pi in p_concat]
	# q_cut = [qi[0:index+1] for qi in q_concat]
	p_cut = [pi[0:nsteps] for pi in p_concat]
	q_cut = [qi[0:nsteps] for qi in q_concat]
	t = nsteps*dT

	# Simulate forward until this time 
	quandary_sim = Quandary(
	    Ne = [2 for i in range(nqubits)],
		Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)],
		freq01 = w01,
		rotfreq=rotf,
		selfkerr=selfk,
		Jkl = gkl,
		T = t,
		dT=dT,
		targetgate=Ut,
		spline_order=spline_order,
		spline_knot_spacing=spline_knot_spacing,
		control_enforce_BC=control_enforce_BC,
		verbose=False,
		rand_seed=1234,
	)
	datadir = "./ConcatenatedGates_to"+str(t)+"_rundir"
	t_int, pt_int, qt_int, infidelity_int, expectedEnergy, population= quandary_sim.simulate(pt0=p_cut, qt0=q_cut, datadir=datadir)

	print("Intermediate fidelity at t=", t, ": ", 1.0 - infidelity_int)

stop

####
# Now optimize for the two single qubit gates as GxId and IdxG
###
for op in CircuitIterator(QFT_circuit):
	print("Operation:", op, " params ", op.params, " location=", op.location)

	# Get pulse for this operation
	pulse = op.get_pulse()
	assert(pulse)
	timesteps = pulse["times"]
	p_pulse = pulse["p_pulse"]
	q_pulse = pulse["q_pulse"]

	# Only single-qubit for now !!
	assert(len(op.location) == 1)
	iqubit = op.location[0]

	# Target gate is G x Id or Id x G
	targetgate=op.get_unitary()
	Id = np.identity(3 if with_guard_level else 2)
	if iqubit == 0:
		targetgate = np.kron(targetgate, Id)
	else:
		targetgate = np.kron(Id, targetgate)
	print(targetgate)


	quandary_concat.T = timesteps[-1]
	# quandary_concat.optimize



# # Now optimize on the QFT gate directly
# w01 = [freq01[i] for i in range(nqubits)]
# selfk = [selfkerr[i] for i in range(nqubits)]
# gkl = coupling_strength*np.ones(np.sum([i for i in range(nqubits)])) # This only works if no larger than two-qubit gate!
# pulse_duration = 70.0 # ns
# costfunction = "Jtrace"
# spline_order = 2 # 2
# spline_knot_spacing = 3.0 # ns
# nsplines = int(pulse_duration/spline_knot_spacing + 2)
# control_enforce_BC=True
# quandary_QFT = Quandary(
#   Ne = [2 for i in range(nqubits)],
#   Ng = [1 if with_guard_level else 0 for _ in range(op.num_qudits)]
# 	freq01 = w01,
#	selfkerr = self,
# 	Jkl = gkl,
# 	T = pulse_duration,
# 	targetgate=QFT_unitary,
# 	maxctrl_MHz=maxctrl_MHz,
# 	spline_order=spline_order,
# 	nsplines=nsplines,
# 	control_enforce_BC=control_enforce_BC,
# 	verbose=False,
# 	rand_seed=1234,
# )
# datadir = "./QFT_rundir"
# t_QFT, pt_QFT, qt_QFT, infidelity_QFT, expectedEnergy_QFT, population_QFT = quandary_QFT.optimize(datadir=datadir)
# print(f"\nFidelity of direct QFT pulse: {1.0 - infidelity_QFT}")

# Plot QFT pulse
# plot_pulse(quandary_QFT.Ne, t_QFT, pt_QFT, qt_QFT)
 
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

