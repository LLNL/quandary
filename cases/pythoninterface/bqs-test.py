from quandary import * 
from bqskit import MachineModel
from bqskit.ir.gates import ConstantGate, QubitGate
from bqskit import Circuit
from bqskit import compile
from bqskit.ir import CircuitIterator, CycleInterval
from bqskit.ir.gates import CZGate, RZGate, SXGate, IdentityGate


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

####
# Now optimize for the two single qubit gates as GxId and IdxG
####
for op in CircuitIterator(QFT_circuit, and_cycles=True):
	print("Cycle idx= ", op[0], "Operation:", op[1], " params ", op[1].params, " location=", op[1].location, " num qubits=", op[1].num_qudits)
	# if op.num_qudits == 1:
	# 	# pull another operation from the circuit
	# 	op2 = CircuitIterator(QFT_circuit)
	# 	print("Parallel Operation:", op2, " params ", op2.params, " location=", op2.location, " num qubits=", op2.num_qudits)

#    
# 	# Get pulse for this operation
# 	pulse = op.get_pulse()
# 	assert(pulse)
# 	timesteps = pulse["times"]
# 	p_pulse = pulse["p_pulse"]
# 	q_pulse = pulse["q_pulse"]

# 	# Only single-qubit for now !!
# 	assert(len(op.location) == 1)
# 	iqubit = op.location[0]

# 	# Target gate is G x Id or Id x G
# 	targetgate=op.get_unitary()
# 	Id = np.identity(3 if with_guard_level else 2)
# 	if iqubit == 0:
# 		targetgate = np.kron(targetgate, Id)
# 	else:
# 		targetgate = np.kron(Id, targetgate)
# 	print(targetgate)


# 	quandary_concat.T = timesteps[-1]
# 	# quandary_concat.optimize

