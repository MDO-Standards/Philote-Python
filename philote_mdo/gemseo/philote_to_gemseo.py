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
from typing import Any
from typing import ClassVar

from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammars.factory import GrammarType
from numpy import prod

import philote_mdo.general as pm
import philote_mdo.generated.data_pb2 as data

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from collections.abc import Mapping


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

    The discrete variables of the remote discipline are part of the
    grammars, next to the continuous ones, and are read from and written to
    the local data like any other variable. Since a Philote discrete
    variable may carry any JSON-compatible value, its grammar element is
    bound to no type. Discrete variables are never differentiated: they are
    excluded from the Jacobian, including when it is requested in full with
    ``compute_all_jacobians=True``.
    """

    default_grammar_type: ClassVar[GrammarType] = GrammarType.SIMPLE
    """The default type of grammar."""

    def __init__(self, channel, name="", **options):
        """Initialize the discipline and connect its Philote client.

        Args:
            channel: The gRPC channel to the Philote discipline server,
                e.g. created with ``grpc.insecure_channel("localhost:50051")``.
            name: The name of the discipline.
                If empty, use the name reported by the server,
                or the class name if the server reports none.
            **options: The discipline options to send to the server,
                if any.

        Raises:
            ValueError: When ``channel`` is empty or ``None``.
            NotImplementedError: When the server declares dynamic-shape
                variables, which are not supported yet.
        """
        if not channel:
            msg = "No channel provided, the Philote client will not be able to connect."
            raise ValueError(msg)
        # The shape of each continuous input and output variable, indexed by
        # name; filled in by _initialize_grammars() and used by
        # _compute_jacobian() to reshape the flattened partial derivatives
        # sent by the server.
        self._shapes = {}
        # The names of the discrete input and output variables, filled in by
        # _initialize_grammars(); discrete inputs travel on their own side of
        # the wire protocol and so must be split off the GEMSEO input data.
        self._discrete_input_names = set()
        self._discrete_output_names = set()
        # generic Philote client
        self._client = pm.ExplicitClient(channel=channel)

        # call the init function of the explicit component; an empty name
        # makes GEMSEO fall back to the class name
        self._client.get_discipline_info()
        super().__init__(name=name or self._client._name)

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
        values. Continuous and discrete inputs are sent separately, and the
        discrete outputs returned by the server, if any, are merged back into
        the output data.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The output data computed by the remote discipline.
        """
        inputs, discrete_inputs = self._split_input_data(input_data)
        outputs = self._client.run_compute(inputs, discrete_inputs=discrete_inputs)
        # run_compute returns (outputs, discrete_outputs) when the server
        # sends back discrete output data, and a plain dictionary otherwise.
        if isinstance(outputs, tuple):
            outputs, discrete_outputs = outputs
            outputs.update(discrete_outputs)
        return outputs

    def _split_input_data(self, input_data: Mapping[str, Any]) -> tuple[dict, dict]:
        """Split the GEMSEO input data into continuous and discrete inputs.

        The GEMSEO input grammar holds the continuous and the discrete inputs
        side by side, while the Philote wire protocol carries them in two
        distinct kinds of message.

        Args:
            input_data: The input data, without namespace prefixes.

        Returns:
            The continuous input values and the discrete input values,
            both indexed by variable name.
        """
        inputs = {}
        discrete_inputs = {}
        for name, value in input_data.items():
            if name in self._discrete_input_names:
                discrete_inputs[name] = value
            else:
                inputs[name] = value
        return inputs, discrete_inputs

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
        inputs, discrete_inputs = self._split_input_data(self.get_input_data())
        jac_flat = self._client.run_compute_partials(
            inputs, discrete_inputs=discrete_inputs
        )
        for jac_key, jac_val in jac_flat.items():
            out_name, in_name = jac_key
            out_size = int(prod(self._shapes[out_name]))
            in_size = int(prod(self._shapes[in_name]))
            jac_loc = jac_val.reshape(out_size, in_size)
            if out_name not in self.jac:
                self.jac[out_name] = {in_name: jac_loc}
            else:
                self.jac[out_name][in_name] = jac_loc

    def _get_differentiated_io(
        self,
        compute_all_jacobians: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the inputs and outputs used in the differentiation.

        The discrete variables are filtered out, as they cannot be
        differentiated. GEMSEO already does this when the differentiated
        variables are selected with :meth:`.add_differentiated_inputs` and
        :meth:`.add_differentiated_outputs`, but not when the whole Jacobian
        is requested.

        Args:
            compute_all_jacobians: Whether to compute the Jacobians of all the
                outputs with respect to all the inputs.

        Returns:
            The names of the differentiated inputs
            and the names of the differentiated outputs.
        """
        input_names, output_names = super()._get_differentiated_io(
            compute_all_jacobians
        )
        if not compute_all_jacobians:
            return input_names, output_names

        return (
            tuple(
                name for name in input_names if name not in self._discrete_input_names
            ),
            tuple(
                name
                for name in output_names
                if name not in self._discrete_output_names
            ),
        )

    def _initialize_grammars(self):
        """Set up the GEMSEO discipline input and output grammars.

        This builds the input and output grammars, and populates
        :attr:`._shapes`, from the variable metadata already retrieved
        from the remote discipline server by
        :meth:`~philote_mdo.general.discipline_client.DisciplineClient.get_variable_definitions`.
        It does not perform any RPC call itself.

        Continuous variables are bound to :class:`~numpy.ndarray`, while
        discrete variables are bound to no type at all, since a Philote
        discrete variable may hold any JSON-compatible value.

        Raises:
            NotImplementedError: When the server declares dynamic-shape
                variables, whose shapes would have to be sent to the server.
        """
        unsupported = [
            f"{var.name} (dynamic shape)"
            for var in self._client._var_meta
            if var.dynamic_shape
        ]
        if unsupported:
            msg = (
                "PhiloteDiscipline does not support these server variables yet: "
                + ", ".join(unsupported)
            )
            raise NotImplementedError(msg)

        input_names = []
        output_names = []
        for var in self._client._var_meta:
            if var.type == data.kInput:
                input_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

            if var.type == data.kOutput:
                output_names.append(var.name)
                self._shapes[var.name] = tuple(var.shape)

        for var in self._client._discrete_var_meta:
            if var.type == data.kDiscreteInput:
                self._discrete_input_names.add(var.name)

            if var.type == data.kDiscreteOutput:
                self._discrete_output_names.add(var.name)

        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)
        # A None type means that the grammar accepts any value for that name,
        # which is what a Philote discrete variable may carry.
        self.input_grammar.update_from_types(
            dict.fromkeys(self._discrete_input_names)
        )
        self.output_grammar.update_from_types(
            dict.fromkeys(self._discrete_output_names)
        )
