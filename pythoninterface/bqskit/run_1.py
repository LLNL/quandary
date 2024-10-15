from bqskit import Circuit
from bqskit.ir.gates import HGate, CNOTGate

from bqskit.compiler import Compiler

from pulsesynthesis import PulseSynthesisPass


if __name__ == '__main__':

    # Create a circuit
    circ = Circuit(2)
    circ.append_gate(HGate(), [0])
    circ.append_gate(CNOTGate(), [0, 1])

    pulse_length = 400.0
    max_ctrl_MHz = 10.0
    exec_path = "/Users/guenther5/Numerics/quandary/"

    ## Need to specify qubit frequencies and topology for Quandary
    freq01 = [4.10, 4.15] # 01-transition frequencies
    Jkl = [0.005]       # dipole-dipole coupling strength
    qubit_data = {"01-Transition":freq01, "Coupling":Jkl}

    # TODO use Module from Bqskit to specify the hardware

    # Passes
    pulse = PulseSynthesisPass(
        qubit_data=qubit_data,
        pulse_length=pulse_length,
        maxctrl_MHz=max_ctrl_MHz,
        quandary_exec_path=exec_path,
    )

    # Run the circuit
    with Compiler() as compiler:
        result, data = compiler.compile(circ, pulse, request_data=True)

    # Print the result
    t = data['quandary_t']
    p = data['quandary_p']
    q = data['quandary_q']
    infidelity = data['quandary_infidelity']
    exp_energy = data['quandary_exp_energy']
    population = data['quandary_population']
    # print(f't: {t}')
    # print(f'p: {p}')
    # print(f'q: {q}')
    print(f'infidelity: {infidelity}')
    # print(f'exp_energy: {exp_energy}')
    # print(f'population: {population}')