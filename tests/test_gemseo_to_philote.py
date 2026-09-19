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
from concurrent import futures
import unittest
from unittest.mock import patch
import grpc
from numpy import array
from numpy import eye
from numpy import ndarray
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammars.factory import GrammarType
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
import philote_mdo.general as pmdo
from philote_mdo.gemseo import GEMSEOtoPhiloteDiscipline
from philote_mdo.gemseo import PhiloteDiscipline

PORT = "[::]:50051"
CHANNEL = "localhost:50051"


class ScalingDiscipline(Discipline):
    """A GEMSEO discipline with a string input and a string output.

    The string input ``mode`` scales the array output ``y``, so that both
    the outputs and the Jacobian depend on it, and the string output
    ``used_mode`` echoes it back.
    """

    default_grammar_type = GrammarType.SIMPLE

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.input_grammar.update_from_types({"mode": str})
        self.output_grammar.update_from_names(["y"])
        self.output_grammar.update_from_types({"used_mode": str})
        self.default_input_data = {"x": array([1.0, 1.0]), "mode": "double"}
        self.default_output_data = {"y": array([0.0, 0.0]), "used_mode": "double"}

    @staticmethod
    def _get_factor(mode):
        """Return the factor scaling the output.

        Args:
            mode: The value of the string input.

        Returns:
            The factor by which the input array is scaled.
        """
        return 2.0 if mode == "double" else 3.0

    def _run(self, input_data):
        factor = self._get_factor(input_data["mode"])
        return {"y": factor * input_data["x"], "used_mode": input_data["mode"]}

    def _compute_jacobian(self, input_names=(), output_names=()):
        self._init_jacobian(input_names, output_names)
        self.jac["y"]["x"] = self._get_factor(self.io.data["mode"]) * eye(2)


class IntegerDiscipline(Discipline):
    """A GEMSEO discipline with an integer input and an integer output.

    A Philote continuous variable travels as an array of doubles, so an
    integer variable belongs to the discrete side of the protocol.
    """

    default_grammar_type = GrammarType.SIMPLE

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.input_grammar.update_from_types({"n": int})
        self.output_grammar.update_from_names(["y"])
        self.output_grammar.update_from_types({"count": int})
        self.default_input_data = {"x": array([1.0]), "n": 2}

    def _run(self, input_data):
        return {"y": input_data["n"] * input_data["x"], "count": input_data["n"]}


class DiscreteOnlyDiscipline(Discipline):
    """A GEMSEO discipline whose variables are all discrete."""

    default_grammar_type = GrammarType.SIMPLE

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_types({"mode": str})
        self.output_grammar.update_from_types({"used_mode": str})
        self.default_input_data = {"mode": "double"}

    def _run(self, input_data):
        return {"used_mode": input_data["mode"]}


class GEMSEOToPhiloteTests(unittest.TestCase):
    """
    Integration tests for GEMSEOtoPhiloteDiscipline, which wraps a GEMSEO
    discipline as a Philote explicit discipline server.

    A GEMSEO Sellar1 discipline is served, and a PhiloteDiscipline client is
    used to check that the remote outputs and Jacobian match those of a
    local (non-remote) Sellar1 instance.
    """

    def test_sellar1_compute(self):
        """
        The remote GEMSEO discipline produces the same output as a local one.
        """
        sellar1 = Sellar1()

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(
            discipline=GEMSEOtoPhiloteDiscipline(sellar1)
        )
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        remote_sellar1 = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        out = remote_sellar1.execute(sellar1.default_input_data)
        out_local = sellar1.execute()

        server.stop(0)

        assert_allclose(out["y_1"], out_local["y_1"])

    def test_sellar1_linearize(self):
        """
        The Jacobian of the remote GEMSEO discipline matches the local one.
        """
        sellar1 = Sellar1()

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(
            discipline=GEMSEOtoPhiloteDiscipline(sellar1)
        )
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        remote_sellar1 = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        remote_sellar1.execute(sellar1.default_input_data)
        remote_sellar1.linearize(
            sellar1.default_input_data, compute_all_jacobians=True
        )
        jac_local = sellar1.linearize(
            sellar1.default_input_data, compute_all_jacobians=True
        )

        server.stop(0)

        for input_name in ("x_1", "x_shared", "y_2", "gamma"):
            jac_loc = jac_local["y_1"][input_name]
            if issparse(jac_loc):
                jac_loc = jac_loc.toarray()
            assert_allclose(
                remote_sellar1.jac["y_1"][input_name].ravel(),
                jac_loc.ravel(),
                atol=1e-8,
            )

    def test_setup_uses_default_output_data_size(self):
        """
        The size of a Philote output declared by setup() is taken from the
        wrapped discipline's default_output_data when it holds an ndarray,
        instead of falling back to default_data_size.
        """
        sellar1 = Sellar1()
        sellar1.default_output_data.update({"y_1": array([0.0, 0.0])})

        wrapper = GEMSEOtoPhiloteDiscipline(sellar1)
        wrapper.setup()

        y_1_meta = next(m for m in wrapper._var_meta if m.name == "y_1")
        self.assertEqual(y_1_meta.shape, [2])

    def test_compute_partials_skips_output_not_in_grammar(self):
        """
        Jacobian entries returned by the wrapped discipline for an output
        that is not part of its own output grammar are ignored, rather than
        being forwarded to the Philote-MDO client.
        """
        sellar1 = Sellar1()
        wrapper = GEMSEOtoPhiloteDiscipline(sellar1)
        wrapper.setup()
        wrapper.setup_partials()

        inputs = sellar1.default_input_data
        real_jac = sellar1.linearize(inputs, compute_all_jacobians=True)
        jac_with_extra_output = dict(real_jac)
        jac_with_extra_output["not_a_declared_output"] = {
            "x_1": array([[1.0]])
        }

        partials = {}
        with patch.object(sellar1, "linearize", return_value=jac_with_extra_output):
            wrapper.compute_partials(inputs, partials)

        self.assertIn(("y_1", "x_1"), partials)
        self.assertNotIn(("not_a_declared_output", "x_1"), partials)


class DiscreteVariableTests(unittest.TestCase):
    """
    Tests for the handling of the variables of a wrapped GEMSEO discipline
    that do not hold numeric data, which are exposed as Philote discrete
    variables.
    """

    def test_setup_splits_numeric_and_discrete_variables(self):
        """
        The variables whose grammar data converter reports them as numeric
        are declared as continuous Philote variables, and the others as
        discrete ones.
        """
        wrapper = GEMSEOtoPhiloteDiscipline(ScalingDiscipline())
        wrapper.setup()

        self.assertEqual([m.name for m in wrapper._var_meta], ["x", "y"])
        self.assertEqual(
            [m.name for m in wrapper._discrete_var_meta], ["mode", "used_mode"]
        )

    def test_setup_partials_skips_discrete_variables(self):
        """
        Only the variables holding continuous data take part in the
        Jacobian, so no partial derivative is declared for a discrete one.
        """
        wrapper = GEMSEOtoPhiloteDiscipline(ScalingDiscipline())
        wrapper.setup()
        wrapper.setup_partials()

        self.assertEqual(
            [(m.name, m.subname) for m in wrapper._partials_meta], [("y", "x")]
        )

    def test_discrete_variables_round_trip(self):
        """
        A discrete input travels to the wrapped GEMSEO discipline and a
        discrete output travels back, through a remote PhiloteDiscipline.
        """
        remote = self._serve_scaling_discipline()

        self.assertEqual(
            {n: remote.input_grammar[n] for n in remote.input_grammar},
            {"x": ndarray, "mode": None},
        )
        self.assertEqual(
            {n: remote.output_grammar[n] for n in remote.output_grammar},
            {"y": ndarray, "used_mode": None},
        )

        out = remote.execute({"x": array([1.0, 2.0]), "mode": "triple"})

        assert_allclose(out["y"], array([3.0, 6.0]))
        self.assertEqual(out["used_mode"], "triple")

    def test_discrete_input_used_by_compute_partials(self):
        """
        The discrete inputs are passed to the wrapped discipline when it is
        linearized, and are themselves left out of the Jacobian.
        """
        remote = self._serve_scaling_discipline()

        jac = remote.linearize(
            {"x": array([1.0, 2.0]), "mode": "triple"}, compute_all_jacobians=True
        )

        self.assertEqual(list(jac), ["y"])
        self.assertEqual(list(jac["y"]), ["x"])
        assert_allclose(jac["y"]["x"], 3.0 * eye(2))

    def _serve_scaling_discipline(self):
        """
        Serves a ScalingDiscipline for the duration of the test.

        Returns:
            A PhiloteDiscipline client connected to the served discipline.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pmdo.ExplicitServer(
            discipline=GEMSEOtoPhiloteDiscipline(ScalingDiscipline())
        ).attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()
        self.addCleanup(server.stop, 0)
        return PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

    def test_integer_variables_are_discrete(self):
        """
        A Philote continuous variable is an array of doubles, so an
        integer-valued variable is served as a discrete one and takes no
        part in the Jacobian.
        """
        wrapper = GEMSEOtoPhiloteDiscipline(IntegerDiscipline())
        wrapper.setup()
        wrapper.setup_partials()

        self.assertEqual([m.name for m in wrapper._var_meta], ["x", "y"])
        self.assertEqual(
            [m.name for m in wrapper._discrete_var_meta], ["n", "count"]
        )
        self.assertEqual(
            [(m.name, m.subname) for m in wrapper._partials_meta], [("y", "x")]
        )

    def test_compute_partials_without_declared_partials(self):
        """
        When no variable holds continuous data, setup_partials() declares
        nothing and compute_partials() does not linearize the wrapped
        discipline at all.
        """
        discipline = DiscreteOnlyDiscipline()
        wrapper = GEMSEOtoPhiloteDiscipline(discipline)
        wrapper.setup()
        wrapper.setup_partials()

        partials = {}
        with patch.object(discipline, "linearize") as linearize:
            wrapper.compute_partials({}, partials, {"mode": "double"})

        linearize.assert_not_called()
        self.assertEqual(partials, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
