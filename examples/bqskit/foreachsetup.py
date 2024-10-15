"""This module implements the ForEachBlockSetupPass class."""
from __future__ import annotations

from typing import Callable

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
from bqskit.passes.control.foreach import ForEachBlockPass


class ForEachBlockSetupPass(BasePass):

    pass_down_block_specific_key_prefix = (
        ForEachBlockPass.pass_down_block_specific_key_prefix
    )

    def __init__(
        self,
        setup_suffix: str,
    ) -> None:
        """
        Prime the PassData with block-specific keys.

        Args:
            setup_suffix (str): The suffix to use for the passdown key.

            setup_function (Callable): The function to call to set up the
                data that will go into PassData for each block.
        """
        self.setup_suffix = setup_suffix

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        key = self.pass_down_block_specific_key_prefix + self.setup_suffix
        if key not in data:
            data[key] = {}

        for op_num, op in enumerate(circuit):
            if isinstance(op.gate, CircuitGate):
                _setup = 'rundir_block_' + str(op_num)
                data[key][op_num] = _setup