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

from gemseo.core.serializable import GoogleDocstringInheritanceMeta
from numpy import ndarray
from scipy.sparse import issparse

from philote_mdo.general.explicit_discipline import ExplicitDiscipline

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline


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

    All the input and output variables of the wrapped discipline are
    exposed as flat (1D) continuous Philote variables, and the Jacobian of
    every output with respect to every input is made available, computed
    on demand using GEMSEO's own differentiation capabilities
    (see :meth:`~gemseo.core.discipline.discipline.Discipline.linearize`).
    """

    gemseo_discipine: Discipline
    """The GEMSEO discipline to be wrapped."""

    default_data_size: int
    """The default size used to declare a variable that has no default value."""

    def __init__(self, gemseo_discipine: Discipline, default_data_size: int = 1):
        """Initialize the GEMSEO discipline.

        Args:
            gemseo_discipine: The GEMSEO discipline to be wrapped.
            default_data_size: The default data size to be declared for an
                input or output variable when the wrapped GEMSEO discipline
                does not define a default value (as a
                :class:`~numpy.ndarray`) for it in its grammars.
        """
        super().__init__()
        self.gemseo_discipine = gemseo_discipine
        self.default_data_size = default_data_size

    def setup(self):
        """Declare the Philote inputs and outputs from the GEMSEO grammars.

        Every name of the wrapped GEMSEO discipline's input and output
        grammars is declared as a flat (1D) continuous Philote variable.
        Its size is taken from the corresponding default value in
        :attr:`~gemseo.core.discipline.discipline.Discipline.default_input_data`
        or
        :attr:`~gemseo.core.discipline.discipline.Discipline.default_output_data`
        when it is a :class:`~numpy.ndarray`, and from
        :attr:`.default_data_size` otherwise.
        """
        disc = self.gemseo_discipine
        for input_name in disc.input_grammar.names:
            data = disc.default_input_data.get(input_name)
            size = self.default_data_size
            if isinstance(data, ndarray):
                size = data.size
            self.add_input(input_name, shape=(size,))
        for output_name in disc.output_grammar.names:
            data = disc.default_output_data.get(output_name)
            size = self.default_data_size
            if isinstance(data, ndarray):
                size = data.size
            self.add_output(output_name, shape=(size,))

    def setup_partials(self):
        """Declare the Jacobian of every output with respect to every input.

        No sparsity pattern is assumed: all the input-output pairs of the
        wrapped GEMSEO discipline are declared as partial derivatives to
        be computed by :meth:`.compute_partials`.
        """
        for output_name in self.gemseo_discipine.output_grammar.names:
            for input_name in self.gemseo_discipine.input_grammar.names:
                self.declare_partials(output_name, input_name)

    def compute(self, inputs, outputs):
        """Execute the wrapped GEMSEO discipline.

        Args:
            inputs: The input values, indexed by input name.
            outputs: The dictionary to be filled with the output values
                computed by the wrapped GEMSEO discipline, indexed by
                output name.
        """
        out = self.gemseo_discipine.execute(inputs)
        outputs.update({k: out[k] for k in self.gemseo_discipine.output_grammar.names})

    def compute_partials(self, inputs, partials):
        """Linearize the wrapped GEMSEO discipline.

        Args:
            inputs: The input values, indexed by input name.
            partials: The dictionary to be filled with the Jacobian matrices
                computed by the wrapped GEMSEO discipline, indexed by
                ``(output_name, input_name)`` pairs.
        """
        jac = self.gemseo_discipine.linearize(inputs, compute_all_jacobians=True)
        output_names = self.gemseo_discipine.output_grammar.names
        input_names = self.gemseo_discipine.input_grammar.names

        for out_k, jac_in in jac.items():
            if out_k not in output_names:
                # Some GEMSEO disciplines (e.g. the Sellar ones) populate
                # their Jacobian with extra entries beyond their own
                # declared outputs; only forward what setup_partials()
                # actually declared to the Philote-MDO client.
                continue
            for in_k, jac_loc in jac_in.items():
                if in_k not in input_names:
                    continue
                # The Philote-MDO wire protocol only transfers dense arrays,
                # while a GEMSEO discipline may return a sparse Jacobian
                # matrix (e.g. scipy.sparse.dia_matrix).
                if issparse(jac_loc):
                    jac_loc = jac_loc.toarray()
                partials[out_k, in_k] = jac_loc
