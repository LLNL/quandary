from bqskit import Circuit
from bqskit.ir.gates import HGate, CNOTGate

from bqskit.compiler import Compiler

from bqskit.passes.partitioning import QuickPartitioner
from bqskit.passes.control import ForEachBlockPass

from pulsesynthesis import PulseSynthesisPass

from foreachsetup import ForEachBlockSetupPass


if __name__ == '__main__':

    # Create a circuit
    circ = Circuit(3)
    circ.append_gate(HGate(), [0])
    circ.append_gate(CNOTGate(), [0, 1])
    circ.append_gate(CNOTGate(), [1, 2])

    pulse_length = 240.0
    max_ctrl_MHz = 4.0
    exec_path = ...

    # Passes
    pulse = PulseSynthesisPass(
        pulse_length=pulse_length,
        maxctrl_MHz=max_ctrl_MHz,
        quandary_exec_path=exec_path,
    )
    partitioner = QuickPartitioner(2)
    setup = ForEachBlockSetupPass('quandary_datadir')
    foreach = ForEachBlockPass([pulse])

    # Run the circuit
    with Compiler() as compiler:
        result, data = compiler.compile(
            circ,
            workflow=[partitioner, setup, foreach],
            request_data=True,
        )

    foreach_data = data['ForEachBlockPass_data'][0]
    for i, data in enumerate(foreach_data):
        # Print the result
        infidelity = data['quandary_infidelity']
        exp_energy = data['quandary_exp_energy']
        population = data['quandary_population']
        print(f'block {i} infidelity: {infidelity}')
        print(f'block {i} exp_energy: {exp_energy}')
        print(f'block {i} population: {population}')
