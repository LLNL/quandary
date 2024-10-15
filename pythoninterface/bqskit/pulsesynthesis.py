"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

# from quandary import QuandaryConfig
# from quandary import quandary_run
from quandary import *


class PulseSynthesisPass(BasePass):

    pass_data_prefix = 'quandary_'

    def __init__(
            self,
            qubit_data: dict,
            pulse_length: float,
            maxctrl_MHz: float,
            quandary_exec_path: str,
        ) -> None:
        """
        Construct a new Quandary PulseSynthesisPass.

        Args:
            qubit_data (dict): "01-Transition": list, "Coupling": list

            pulse_length (float): The length of the pulses in nanoseconds.

            maxctrl_MHz (float): The maximum control frequency in MHz.

            quandary_exec_path (str): The path to the Quandary executable.
        """
        self.qubit_data = qubit_data
        self.pulse_length = pulse_length
        self.maxctrl_MHz = maxctrl_MHz
        self.exec_path = quandary_exec_path

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        single_key = self.pass_data_prefix + 'datadir'
        batch_key = 'ForEachBlockPass_specific_pass_down_' + single_key
        if single_key in data:
            datadir = data[single_key]
        elif batch_key in data:
            datadir = data[batch_key]
        else:
            datadir = './run_dir'
        
        unitary = circuit.get_unitary().numpy
        Ne = [circuit.radixes[i] for i in range(circuit.num_qudits)]

        # Switch using guard levels or not
        # Ng = [1 for _ in Ne]
        Ng = []

        # print("Optimizing with Ne", Ne)
        quandary = Quandary(
            Ne=Ne,
            Ng=Ng,
            freq01=self.qubit_data["01-Transition"],
            Jkl=self.qubit_data["Coupling"],
            targetgate=unitary,
            T=self.pulse_length,
            maxctrl_MHz=self.maxctrl_MHz,
            rand_seed=1234
            # verbose=True
        )
        print(self.exec_path)
        qexec = self.exec_path+"/quandary"
        t, p, q, infidelity, exp_energy, population = quandary.optimize(
            quandary_exec=qexec,
            datadir=datadir,
        )
        print("Done optimizing.")
        data[self.pass_data_prefix + 't'] = t
        data[self.pass_data_prefix + 'p'] = p
        data[self.pass_data_prefix + 'q'] = q
        data[self.pass_data_prefix + 'infidelity'] = infidelity
        data[self.pass_data_prefix + 'exp_energy'] = exp_energy
        data[self.pass_data_prefix + 'population'] = population