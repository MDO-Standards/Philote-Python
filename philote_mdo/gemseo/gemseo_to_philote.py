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
"""A Philote explicit discipline wrapping a GEMSEO discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.serializable import GoogleDocstringInheritanceMeta
from numpy import asarray
from numpy import ndarray
from scipy.sparse import issparse

from philote_mdo.general.explicit_discipline import ExplicitDiscipline

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.core.grammars.base_grammar import BaseGrammar


class GEMSEOtoPhiloteDiscipline(
    ExplicitDiscipline, metaclass=GoogleDocstringInheritanceMeta
):
    """A Philote-MDO explicit discipline wrapping a GEMSEO discipline.

    This adapts a GEMSEO :class:`~gemseo.core.discipline.discipline.Discipline`
    to the :class:`~philote_mdo.general.explicit_discipline.ExplicitDiscipline`
    interface, so that it can be exposed to remote Philote clients (e.g. an
    OpenMDAO component or a :class:`~philote_mdo.gemseo.philote_to_gemseo.PhiloteDiscipline`)
    through a :class:`~philote_mdo.general.explicit_server.ExplicitServer`,
    for example::

        server = ExplicitServer(
            discipline=GEMSEOtoPhiloteDiscipline(my_discipline)
        )
        server.attach_to_server(grpc_server)

    The input and output variables of the wrapped discipline that hold
    continuous data, as reported by the ``is_continuous`` method of their
    grammar's data converter, are exposed as flat (1D) continuous Philote
    variables. Every other variable is exposed as a Philote discrete
    variable, which carries any JSON-compatible value.

    This is the split of the Philote-MDO protocol itself: a continuous
    variable travels as an array of doubles, while a discrete variable
    travels as a ``google.protobuf.Value``. An integer-valued variable
    therefore belongs to the discrete side, as it does in OpenMDAO.

    The Jacobian of every output with respect to every input is made
    available, computed on demand using GEMSEO's own differentiation
    capabilities (see
    :meth:`~gemseo.core.discipline.discipline.Discipline.linearize`).
    The discrete variables are left out of it.
    """

    gemseo_discipline: Discipline
    """The GEMSEO discipline to be wrapped."""

    default_data_size: int
    """The default size used to declare a variable that has no default value."""

    def __init__(self, gemseo_discipline: Discipline, default_data_size: int = 1):
        """Initialize the GEMSEO discipline.

        Args:
            gemseo_discipline: The GEMSEO discipline to be wrapped.
            default_data_size: The default data size to be declared for an
                input or output variable when the wrapped GEMSEO discipline
                does not define a default value (as a
                :class:`~numpy.ndarray`) for it in its grammars.
        """
        super().__init__()
        self.gemseo_discipline = gemseo_discipline
        self.default_data_size = default_data_size
        # The names of the variables of the wrapped discipline, split by
        # setup() into the ones Philote carries as continuous arrays and the
        # ones it carries as discrete values.
        self._input_names = ()
        self._output_names = ()
        self._discrete_input_names = ()
        self._discrete_output_names = ()

    def setup(self):
        """Declare the Philote inputs and outputs from the GEMSEO grammars.

        Every name of the wrapped GEMSEO discipline's input and output
        grammars that holds continuous data is declared as a flat (1D)
        continuous Philote variable. Its size is taken from the
        corresponding default value in
        :attr:`~gemseo.core.discipline.discipline.Discipline.default_input_data`
        or
        :attr:`~gemseo.core.discipline.discipline.Discipline.default_output_data`
        when it is a :class:`~numpy.ndarray`, and from
        :attr:`.default_data_size` otherwise.

        Every other name is declared as a discrete Philote variable. A
        discrete input takes the default value of the wrapped discipline, if
        any, which the server uses when a client does not send that input.
        """
        disc = self.gemseo_discipline
        self._input_names, self._discrete_input_names = self._split_names(
            disc.input_grammar
        )
        self._output_names, self._discrete_output_names = self._split_names(
            disc.output_grammar
        )

        defaults = disc.default_input_data
        for input_name in self._input_names:
            self.add_input(input_name, shape=(self._get_size(defaults, input_name),))
        for input_name in self._discrete_input_names:
            self.add_discrete_input(input_name, default=defaults.get(input_name))

        defaults = disc.default_output_data
        for output_name in self._output_names:
            self.add_output(output_name, shape=(self._get_size(defaults, output_name),))
        for output_name in self._discrete_output_names:
            self.add_discrete_output(output_name, default=defaults.get(output_name))

    @staticmethod
    def _split_names(grammar: BaseGrammar) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split the names of a grammar into continuous and discrete ones.

        Args:
            grammar: The input or output grammar of the wrapped discipline.

        Returns:
            The names bound to continuous data,
            which Philote carries as arrays of doubles,
            and the other names,
            which Philote carries as discrete variables.
        """
        is_continuous = grammar.data_converter.is_continuous
        continuous_names = []
        discrete_names = []
        for name in grammar.names:
            if is_continuous(name):
                continuous_names.append(name)
            else:
                discrete_names.append(name)
        return tuple(continuous_names), tuple(discrete_names)

    def _get_size(self, defaults: Mapping[str, Any], name: str) -> int:
        """Return the size to declare for a continuous Philote variable.

        Args:
            defaults: The default input or output data of the wrapped
                discipline.
            name: The name of the variable.

        Returns:
            The size of the default value when it is a NumPy array,
            and :attr:`.default_data_size` otherwise.
        """
        value = defaults.get(name)
        if isinstance(value, ndarray):
            return value.size
        return self.default_data_size

    def setup_partials(self):
        """Declare the Jacobian of every output with respect to every input.

        No sparsity pattern is assumed: all the continuous input-output
        pairs of the wrapped GEMSEO discipline are declared as partial
        derivatives to be computed by :meth:`.compute_partials`. The
        discrete variables are not differentiable, so they take no part in
        the Jacobian.
        """
        for output_name in self._output_names:
            for input_name in self._input_names:
                self.declare_partials(output_name, input_name)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Execute the wrapped GEMSEO discipline.

        Args:
            inputs: The continuous input values, indexed by input name.
            outputs: The dictionary to be filled with the continuous output
                values computed by the wrapped GEMSEO discipline, indexed by
                output name.
            discrete_inputs: The discrete input values, indexed by input
                name, if any.
            discrete_outputs: The dictionary to be filled with the discrete
                output values computed by the wrapped GEMSEO discipline,
                indexed by output name, if the discipline has any.
        """
        out = self.gemseo_discipline.execute(
            self._merge_inputs(inputs, discrete_inputs)
        )
        for output_name in self._output_names:
            outputs[output_name] = out[output_name]

        if discrete_outputs is not None:
            for output_name in self._discrete_output_names:
                discrete_outputs[output_name] = out[output_name]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        """Linearize the wrapped GEMSEO discipline.

        The discrete inputs are passed to the wrapped discipline along with
        the continuous ones, so that it can use them to compute the
        Jacobian, but the discipline is not differentiated with respect to
        them.

        Args:
            inputs: The continuous input values, indexed by input name.
            partials: The dictionary to be filled with the Jacobian matrices
                computed by the wrapped GEMSEO discipline, indexed by
                ``(output_name, input_name)`` pairs.
            discrete_inputs: The discrete input values, indexed by input
                name, if any.
        """
        if not self._partials_meta:
            # The discipline has no continuous input or no continuous output,
            # so setup_partials() declared nothing to differentiate.
            return

        disc = self.gemseo_discipline
        # Restrict the differentiation to the continuous variables, which is
        # exactly the subset that setup_partials() declared. GEMSEO keeps the
        # differentiated variables as discipline state, so restore them
        # afterwards: the wrapped discipline may also be used locally.
        saved_input_names = list(disc._differentiated_input_names)
        saved_output_names = list(disc._differentiated_output_names)
        try:
            disc.add_differentiated_inputs(self._input_names)
            disc.add_differentiated_outputs(self._output_names)
            jac = disc.linearize(self._merge_inputs(inputs, discrete_inputs))
        finally:
            disc._differentiated_input_names = saved_input_names
            disc._differentiated_output_names = saved_output_names

        for out_k, jac_in in jac.items():
            if out_k not in self._output_names:
                # Some GEMSEO disciplines (e.g. the Sellar ones) populate
                # their Jacobian with extra entries beyond their own
                # declared outputs; only forward what setup_partials()
                # actually declared to the Philote-MDO client.
                continue
            for in_k, jac_loc in jac_in.items():
                if in_k not in self._input_names:
                    continue
                # The Philote-MDO wire protocol only transfers dense arrays,
                # while a GEMSEO discipline may return a sparse Jacobian
                # matrix (e.g. scipy.sparse.dia_matrix).
                if issparse(jac_loc):
                    jac_loc = jac_loc.toarray()
                partials[out_k, in_k] = jac_loc

    def _merge_inputs(self, inputs, discrete_inputs):
        """Merge the continuous and the discrete inputs into GEMSEO input data.

        A discrete input set to ``None`` is left out, so that the wrapped
        discipline uses its own default value; this is what an OpenMDAO
        client sends for an unconnected discrete input. The other discrete
        values are converted back to the types the GEMSEO grammar expects,
        which the Philote wire protocol does not preserve (see
        :meth:`._to_gemseo_value`).

        Args:
            inputs: The continuous input values, indexed by input name.
            discrete_inputs: The discrete input values, indexed by input
                name, if any.

        Returns:
            The input data of the wrapped GEMSEO discipline.
        """
        if not discrete_inputs:
            return inputs
        merged = dict(inputs)
        for name, value in discrete_inputs.items():
            if value is not None:
                merged[name] = self._to_gemseo_value(name, value)
        return merged

    def _to_gemseo_value(self, name: str, value: Any) -> Any:
        """Convert a discrete value received from a client to its GEMSEO type.

        A discrete value travels as a ``google.protobuf.Value``, so a NumPy
        array, e.g. an integer array of a JSON grammar, arrives as a (nested)
        list. The array is restored using the default value of the wrapped
        discipline when there is one, and otherwise when the input grammar
        reports the variable as numeric.

        Args:
            name: The name of the discrete input.
            value: The value received from the client.

        Returns:
            The value, converted to the type the GEMSEO grammar expects
            when it can be inferred, and unchanged otherwise.
        """
        default = self._discrete_input_defaults.get(name)
        if isinstance(default, ndarray):
            return asarray(value, dtype=default.dtype)
        if isinstance(value, list) and self._has_numeric_type(name):
            return asarray(value)
        return value

    def _has_numeric_type(self, name: str) -> bool:
        """Return whether the input grammar declares a name as numeric.

        The question is put to the grammar's data converter, which answers
        it for every kind of GEMSEO grammar. Reading the JSON schema
        instead would only work for a
        :class:`~gemseo.core.grammars.json_grammar.JSONGrammar`, and a
        :class:`~gemseo.core.grammars.pydantic_grammar.PydanticGrammar`
        holding a NumPy array raises rather than returning no schema, since
        such a model has no JSON schema at all.

        Args:
            name: The name of the discrete input.

        Returns:
            Whether the name is declared by the wrapped discipline and
            bound to numeric data.
        """
        if name not in self._discrete_input_names:
            # A name the client sent that setup() never declared; the
            # grammar would raise rather than report a type for it.
            return False
        converter = self.gemseo_discipline.input_grammar.data_converter
        return converter.is_numeric(name)
