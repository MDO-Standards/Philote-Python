# Philote-Python
#
# Copyright 2026 IRT Saint Exupery
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# originally authored by Francois Gallard, IRT Saint Exupery.
"""A GEMSEO discipline wrapping a remote Philote explicit discipline server."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammars.factory import GrammarType
from numpy import prod

import philote_mdo.general as pm
import philote_mdo.generated.data_pb2 as data

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable


class PhiloteDiscipline(Discipline):
    """A GEMSEO discipline connected to a remote Philote explicit discipline.

    This discipline acts as a Philote-MDO client: at construction, it
    connects to a Philote explicit discipline server through a gRPC
    channel, triggers the remote ``Setup`` and requests the input/output
    and partials metadata. This metadata is used to build the GEMSEO
    input and output grammars, so that the resulting discipline can be
    used like any other GEMSEO discipline, e.g. in an
    :class:`~gemseo.mda.base_mda.BaseMDA`.

    Executing the discipline (:meth:`.execute`) and linearizing it
    (:meth:`.linearize`) transparently call the remote server to compute
    the outputs and the Jacobian, respectively.
    """

    default_grammar_type: ClassVar[GrammarType] = GrammarType.SIMPLE
    """The default type of grammar."""

    def __init__(self, channel, name="", **options):
        """Initialize the discipline and connect its Philote client.

        Args:
            channel: The gRPC channel to the Philote discipline server,
                e.g. created with ``grpc.insecure_channel("localhost:50051")``.
            name: The name of the discipline.
                If empty, use the name of the remote discipline class.
            **options: The discipline options to send to the server,
                if any.

        Raises:
            ValueError: When ``channel`` is empty or ``None``.
        """
        if not channel:
            msg = "No channel provided, the Philote client will not be able to connect."
            raise ValueError(msg)
        # The shape of each input and output variable, indexed by name;
        # filled in by _initialize_grammars() and used by _compute_jacobian()
        # to reshape the flattened partial derivatives sent by the server.
        self._shapes = {}
        # generic Philote client
        self._client = pm.ExplicitClient(channel=channel)

        # call the init function of the explicit component
        super().__init__(name=name)

        self._client.send_stream_options()
        if options:
            self._client.send_options(options)

        # run setup
        self._client.run_setup()
        self._client.get_variable_definitions()
        self._client.get_partials_definitions()
        self._initialize_grammars()

    def _run(self, input_data: dict) -> dict:
        """Compute the function evaluation.

        This sends the input values to the remote Philote discipline server
        through the ``ComputeFunction`` RPC and returns the resulting output
        values.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The output data computed by the remote discipline.
        """
        return self._client.run_compute(input_data)

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute one Jacobian matrix per input-output pair.

        This sends the current input values to the remote Philote discipline
        server through the ``ComputeGradient`` RPC, reshapes the resulting
        flattened partial derivatives according to the input and output
        shapes obtained by :meth:`._initialize_grammars`, and stores them
        in :attr:`.jac` as a dictionary
        ``{output_name: {input_name: jacobian_matrix}}``.

        Note:
            The remote discipline always returns the Jacobian of every
            declared output with respect to every declared input, so
            ``input_names`` and ``output_names`` are currently not used to
            restrict the RPC request: the full Jacobian is always computed
            and stored.

        Args:
            input_names: The names of the inputs against which to differentiate the
                outputs. If empty, use all the inputs.
            output_names: The names of the outputs to be differentiated.
                If empty, use all the outputs.
        """
        jac_flat = self._client.run_compute_partials(self.get_input_data())
        for jac_key, jac_val in jac_flat.items():
            out_name, in_name = jac_key
            out_size = int(prod(self._shapes[out_name]))
            in_size = int(prod(self._shapes[in_name]))
            jac_loc = jac_val.reshape(out_size, in_size)
            if out_name not in self.jac:
                self.jac[out_name] = {in_name: jac_loc}
            else:
                self.jac[out_name][in_name] = jac_loc

    def _initialize_grammars(self):
        """Set up the GEMSEO discipline input and output grammars.

        This builds the input and output grammars, and populates
        :attr:`._shapes`, from the variable metadata already retrieved
        from the remote discipline server by
        :meth:`~philote_mdo.general.discipline_client.DisciplineClient.get_variable_definitions`.
        It does not perform any RPC call itself.
        """
        input_names = []
        output_names = []
        for var in self._client._var_meta:
            if var.type == data.kInput:
                input_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

            if var.type == data.kOutput:
                output_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)
        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)
