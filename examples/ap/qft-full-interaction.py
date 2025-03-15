from quandary import * 
from bqskit import MachineModel
from bqskit.ir.gates import ConstantGate, QubitGate
from bqskit import Circuit
from bqskit import compile
from bqskit.ir import CircuitIterator, CycleInterval
from bqskit.ir.gates import CZGate, RZGate, SXGate, IdentityGate

def my_append_pulse(timesteps, p_pulse, q_pulse, t_global, p_global, q_global):
	num_qubits = 2
	pulse_location = [0, 1]
	
	# Append pulses for each qubit
	# Here 'i' is a loop counter, starting from 0 and 'qubitid' is the i'th element of the list 'pulse_location'
	for i, qubitid in enumerate(pulse_location):
		tlast = t_global[qubitid][-1] if len(t_global[qubitid]) > 0 else 0.0
		# NOTE: the first time point in timesteps is left out because it coincides with the end of the previous segment
		j0 = 1 if len(t_global[qubitid]) > 0 else 0
		for item in timesteps[j0:]:
			t_global[qubitid].append(item + tlast)
		for item in p_pulse[i][j0:]:
			p_global[qubitid].append(item)
		for item in q_pulse[i][j0:]:
			q_global[qubitid].append(item)
# end my_append_pulse

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
gate_set = {CZGate(), RZGate(), SXGate(), IdentityGate()} 
machine_model = MachineModel(nqubits,  coupling_graph=coupling_graph, gate_set=gate_set)

QFT_circuit = compile(QFT_unitary, model=machine_model, optimization_level=1, seed=1234, error_threshold=1e-4)

# Optimize for pulses for each circuit cycle
t_op = []
pt_op = []
qt_op = []
pop_op = []
exp_op = []
quandary_op = []

# traverse the circuit one cycle at a time
for cycle_idx in range(QFT_circuit.num_cycles):
	current_cycle_ops = []
	for qudit_idx in range(QFT_circuit.num_qudits):
		try:
			operation = QFT_circuit.get_operation((cycle_idx, qudit_idx))
			if operation not in current_cycle_ops:
				current_cycle_ops.append(operation)
		except:
			continue  # Skip idle points

	print(f"Cycle {cycle_idx}", " # ops = ", len(current_cycle_ops), f" operations: {current_cycle_ops}")
	if len(current_cycle_ops) == 1:
		print("Processing one two-qubit gate")
		op = current_cycle_ops[0] 		# this should be exactly one 2-qubit gate
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
	elif len(current_cycle_ops) == 2:
		print("Processing two single qubit gates in parallel")
		oparr = current_cycle_ops 		# there should be exactly two single-qubit gates 
		if oparr[0].num_qudits > 1 or oparr[1].num_qudits > 1:
			print("Each gate within this cycle must operate on exactly one qubit.")
			stop

		# Set up a Quandary instance to optimize for this gate
		Ne = [oparr[i].radixes[0] for i in range(len(oparr))]
		Ng = [1 if with_guard_level else 0 for _ in range(len(oparr))]
		w01 = [freq01[oparr[i].location[0]] for i in range(len(oparr))]
		selfk = [selfkerr[oparr[i].location[0]] for i in range(len(oparr))]
		rotf = [rotfreq[oparr[i].location[0]] for i in range(len(oparr))]
		gkl = coupling_strength*np.ones(1) # This only works if no larger than two-qubit gate!
		pulse_duration = T_all[0] # 2 single qubit gates

		print(f"Ne {Ne}, Ng {Ng}, w01 {w01}, selfk {selfk}, rotf {rotf}, gkl {gkl}, pulse_dur {pulse_duration}")
	
		targetgate= np.kron(oparr[0].get_unitary(), oparr[1].get_unitary())
		
	# unified optimization for both cases
	spline_knot_spacing = 3.0 # ns Must be the same for all gates for the concatenation to work!
	spline_order = 2   # 2nd order Bspline parameterization
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
	print("Optimizing pulses for cycle ", cycle_idx, ", freq01 = ", w01, ", coupling = ", gkl)
	print(" -> target gate =", targetgate)
	datadir = "QFT_cycle_" + str(cycle_idx) + "_rundir"
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
	# op.add_pulse(t, pt, qt) # this won't work for 2 single qubit gates in parallel

	# Store individual pulses for debugging and plotting later
	quandary_op.append(quandary_i)
	t_op.append(t)
	pt_op.append(pt)
	qt_op.append(qt)
	pop_op.append(population)
	exp_op.append(expectedEnergy)          

print("Done optimizing for pulses\n")

# Now concatenate while grabbing pulses from the operations
t_global = [[] for _ in range(nqubits)]
p_global = [[] for _ in range(nqubits)]
q_global = [[] for _ in range(nqubits)]
intermediate_targets = []

for i in range(len(t_op)):
	print("Concatenate pulse for circuit cycle #", i)
	my_append_pulse(t_op[i], pt_op[i], qt_op[i], t_global, p_global, q_global)

# print("Done concatenating pulses.")

t_concat = t_global[0]  # should be the same on all qubits now!
p_concat = p_global
q_concat = q_global

# Simulate with the concatenated pulses
w01 = [freq01[i] for i in range(nqubits)]
selfk = [selfkerr[i] for i in range(nqubits)]
rotf = [rotfreq[i] for i in range(nqubits)]
gkl = coupling_strength*np.ones(1) # This only works for a two-qubit gate!
pulse_duration = t_concat[-1]

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

if True:
	plot_pulse(quandary_concat.Ne, t, pt, qt)
	plot_expectedEnergy(quandary_concat.Ne, t, expectedEnergy)

