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
from contextlib import contextmanager
import unittest
from unittest.mock import patch
import grpc
from numpy import array
from numpy import asarray
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


# Every kind of GEMSEO grammar, so that a mechanism which happens to work
# for one of them, e.g. by reading a JSON schema, is caught by the tests.
GRAMMAR_TYPES = (
    GrammarType.JSON,
    GrammarType.SIMPLE,
    GrammarType.SIMPLER,
    GrammarType.PYDANTIC,
)


class IntegerArrayDiscipline(Discipline):
    """A GEMSEO discipline with integer arrays, using a JSON grammar.

    A JSON grammar does not report an integer array as continuous, so the
    integer arrays are served as discrete variables: ``n`` has no default
    value, while ``offsets`` has one.
    """

    default_grammar_type = GrammarType.JSON

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_data(
            {"x": array([1.0]), "n": array([1, 2]), "offsets": array([0, 0])}
        )
        self.output_grammar.update_from_data({"counts": array([0, 0])})
        self.default_input_data = {"x": array([1.0]), "offsets": array([0, 0])}
        self.received_types = {}

    def _run(self, input_data):
        self.received_types = {
            name: type(input_data[name]) for name in ("n", "offsets")
        }
        return {"counts": input_data["n"] + input_data["offsets"]}


class TagDiscipline(Discipline):
    """A GEMSEO discipline with a list input, using a Pydantic grammar.

    The list input ``tags`` has no default value, and the grammar also
    holds a NumPy array, so it has no JSON schema at all: asking such a
    Pydantic model for its schema raises.
    """

    default_grammar_type = GrammarType.PYDANTIC

    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_data({"x": array([1.0]), "tags": ["a", "b"]})
        self.output_grammar.update_from_data({"y": array([1.0]), "joined": "ab"})
        self.default_input_data = {"x": array([1.0])}
        self.default_output_data = {"y": array([0.0]), "joined": ""}

    def _run(self, input_data):
        return {"y": 2.0 * input_data["x"], "joined": "".join(input_data["tags"])}


def build_integer_array_discipline(grammar_type):
    """Create an integer-array discipline using a given kind of grammar.

    Every variable is given a default value, so that the size of the ones
    served as continuous Philote variables can be taken from it; the
    wrapper has no other way to size a variable.

    Args:
        grammar_type: The kind of grammar the discipline uses.

    Returns:
        A discipline adding two integer arrays, which records the types
        its inputs were received as.
    """

    class ParametrizedIntegerArrayDiscipline(Discipline):
        """A discipline adding two integer arrays."""

        default_grammar_type = grammar_type

        def __init__(self):
            super().__init__()
            data = {"x": array([1.0]), "n": array([1, 2]), "offsets": array([0, 0])}
            self.input_grammar.update_from_data(data)
            self.output_grammar.update_from_data({"counts": array([0, 0])})
            self.default_input_data = dict(data)
            self.default_output_data = {"counts": array([0, 0])}
            self.received_types = {}

        def _run(self, input_data):
            self.received_types = {
                name: type(input_data[name]) for name in ("n", "offsets")
            }
            return {"counts": input_data["n"] + input_data["offsets"]}

    return ParametrizedIntegerArrayDiscipline()


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

    def test_unsent_discrete_input_uses_gemseo_default(self):
        """
        A discrete input the client leaves out takes the default value of
        the wrapped GEMSEO discipline, which the server applies.
        """
        remote = self._serve_scaling_discipline()

        out = remote.execute({"x": array([1.0, 2.0])})

        assert_allclose(out["y"], array([2.0, 4.0]))
        self.assertEqual(out["used_mode"], "double")

    def test_none_discrete_input_uses_gemseo_default(self):
        """
        A discrete input set to None, as an OpenMDAO client sends for an
        unconnected discrete input, is left to the GEMSEO default value.
        """
        wrapper = GEMSEOtoPhiloteDiscipline(ScalingDiscipline())
        wrapper.setup()

        outputs = {}
        discrete_outputs = {}
        wrapper.compute(
            {"x": array([1.0, 2.0])}, outputs, {"mode": None}, discrete_outputs
        )

        assert_allclose(outputs["y"], array([2.0, 4.0]))
        self.assertEqual(discrete_outputs["used_mode"], "double")

    def test_integer_arrays_round_trip(self):
        """
        Integer arrays served as discrete variables reach the wrapped
        discipline as NumPy arrays, whether or not they have a default
        value, and come back to the client as lists of ints rather than
        as strings.
        """
        discipline = IntegerArrayDiscipline()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pmdo.ExplicitServer(
            discipline=GEMSEOtoPhiloteDiscipline(discipline)
        ).attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()
        self.addCleanup(server.stop, 0)
        remote = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        out = remote.execute(
            {"x": array([1.0]), "n": array([1, 2]), "offsets": array([10, 20])}
        )

        self.assertEqual(
            discipline.received_types, {"n": ndarray, "offsets": ndarray}
        )
        self.assertEqual(out["counts"], [11, 22])

    def test_discrete_list_input_of_a_grammar_without_a_json_schema(self):
        """
        A discrete list input with no default value reaches the wrapped
        discipline unchanged, for a grammar that has no JSON schema.

        A Pydantic grammar holding a NumPy array cannot produce one, and
        asking it for its schema raises rather than reporting that it has
        none, so the type of a discrete value has to be settled through
        the grammar's data converter.
        """
        discipline = TagDiscipline()

        with self._serve(GEMSEOtoPhiloteDiscipline(discipline)) as remote:
            out = remote.execute({"x": array([1.0]), "tags": ["a", "b"]})

        assert_allclose(out["y"], array([2.0]))
        self.assertEqual(out["joined"], "ab")

    def test_undeclared_discrete_input_is_left_unchanged(self):
        """
        A discrete input the client sent but setup() never declared is
        passed on unchanged, rather than being looked up in a grammar that
        does not know the name and would raise.
        """
        wrapper = GEMSEOtoPhiloteDiscipline(ScalingDiscipline())
        wrapper.setup()

        merged = wrapper._merge_inputs({}, {"not_declared": [1, 2]})

        self.assertEqual(merged, {"not_declared": [1, 2]})

    def test_integer_arrays_classified_for_every_grammar_type(self):
        """
        Whether an integer array is a discrete or a continuous Philote
        variable follows the grammar's data converter, so it depends on the
        kind of grammar: only a JSON grammar reports an integer array as
        non-continuous.
        """
        discrete_names = {
            GrammarType.JSON: (("n", "offsets"), ("counts",)),
            GrammarType.SIMPLE: ((), ()),
            GrammarType.SIMPLER: ((), ()),
            GrammarType.PYDANTIC: ((), ()),
        }
        for grammar_type in GRAMMAR_TYPES:
            with self.subTest(grammar_type=grammar_type.name):
                expected_inputs, expected_outputs = discrete_names[grammar_type]
                wrapper = GEMSEOtoPhiloteDiscipline(
                    build_integer_array_discipline(grammar_type)
                )

                wrapper.setup()

                self.assertEqual(wrapper._discrete_input_names, expected_inputs)
                self.assertEqual(wrapper._discrete_output_names, expected_outputs)

    def test_integer_arrays_round_trip_for_every_grammar_type(self):
        """
        An integer array reaches the wrapped discipline as a NumPy array,
        and its values survive the round trip, for every kind of GEMSEO
        grammar.

        The arrays travel as discrete variables with a JSON grammar and as
        continuous ones otherwise, so this covers both paths. A mechanism
        that only works for a JSON grammar, such as reading its schema,
        fails here.
        """
        for grammar_type in GRAMMAR_TYPES:
            with self.subTest(grammar_type=grammar_type.name):
                discipline = build_integer_array_discipline(grammar_type)

                with self._serve(GEMSEOtoPhiloteDiscipline(discipline)) as remote:
                    out = remote.execute({
                        "x": array([1.0]),
                        "n": array([1, 2]),
                        "offsets": array([10, 20]),
                    })

                self.assertEqual(
                    discipline.received_types, {"n": ndarray, "offsets": ndarray}
                )
                assert_allclose(asarray(out["counts"]), array([11, 22]))

    @contextmanager
    def _serve(self, discipline):
        """
        Serves a Philote discipline for the duration of the block.

        Unlike addCleanup, this releases the port before the next
        iteration of a subTest loop binds it again.

        Args:
            discipline: The Philote discipline to serve.

        Yields:
            A PhiloteDiscipline client connected to the served discipline.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pmdo.ExplicitServer(discipline=discipline).attach_to_server(server)
        self.assertNotEqual(
            server.add_insecure_port(PORT), 0, "the test port is already in use"
        )
        server.start()
        try:
            yield PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))
        finally:
            server.stop(0)

    def test_compute_partials_restores_differentiated_variables(self):
        """
        Linearizing on behalf of a client leaves the differentiated inputs
        and outputs of the wrapped discipline as they were, since that
        discipline may also be used locally.
        """
        discipline = ScalingDiscipline()
        wrapper = GEMSEOtoPhiloteDiscipline(discipline)
        wrapper.setup()
        wrapper.setup_partials()

        partials = {}
        wrapper.compute_partials({"x": array([1.0, 2.0])}, partials, {})

        assert_allclose(partials["y", "x"], 2.0 * eye(2))
        self.assertEqual(discipline._differentiated_input_names, [])
        self.assertEqual(discipline._differentiated_output_names, [])

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
