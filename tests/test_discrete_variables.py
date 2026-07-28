# Philote-Python
#
# Copyright 2022-2025 Christopher A. Lupp
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
#
# This work has been cleared for public release, distribution unlimited, case
# number: AFRL-2023-5713.
#
# The views expressed are those of the authors and do not reflect the
# official guidance or position of the United States Government, the
# Department of Defense or of the United States Air Force.
#
# Statement from DoD: The Appearance of external hyperlinks does not
# constitute endorsement by the United States Department of Defense (DoD) of
# the linked websites, of the information, products, or services contained
# therein. The DoD does not exercise any editorial, security, or other
# control over the information you may find at these locations.
"""
Unit tests for discrete variable support across the Philote stack.
"""
import unittest
from unittest.mock import Mock, MagicMock, patch

import numpy as np
from google.protobuf import struct_pb2

import philote_mdo.generated.data_pb2 as data
from philote_mdo.general import (
    Discipline,
    DisciplineClient,
    DisciplineServer,
    ExplicitDiscipline,
    ExplicitServer,
    ExplicitClient,
    ImplicitDiscipline,
    ImplicitServer,
    ImplicitClient,
)
from philote_mdo.general.discipline_server import _value_to_python, _python_to_value
import philote_mdo.openmdao.utils as om_utils


# ---------------------------------------------------------------------------
#  Value conversion helpers
# ---------------------------------------------------------------------------
class TestValueConversion(unittest.TestCase):
    """Tests for _value_to_python and _python_to_value round-trip conversion."""

    def test_none(self):
        val = _python_to_value(None)
        self.assertIsNone(_value_to_python(val))

    def test_bool_true(self):
        val = _python_to_value(True)
        self.assertIs(_value_to_python(val), True)

    def test_bool_false(self):
        val = _python_to_value(False)
        self.assertIs(_value_to_python(val), False)

    def test_int(self):
        val = _python_to_value(42)
        result = _value_to_python(val)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_float(self):
        val = _python_to_value(3.14)
        result = _value_to_python(val)
        self.assertAlmostEqual(result, 3.14)
        self.assertIsInstance(result, float)

    def test_string(self):
        val = _python_to_value("hello")
        self.assertEqual(_value_to_python(val), "hello")

    def test_list(self):
        original = [1, "two", 3.0, True]
        val = _python_to_value(original)
        result = _value_to_python(val)
        self.assertEqual(result, [1, "two", 3, True])

    def test_tuple_converts_to_list(self):
        val = _python_to_value((1, 2))
        result = _value_to_python(val)
        self.assertEqual(result, [1, 2])

    def test_dict(self):
        original = {"key": "value", "n": 5}
        val = _python_to_value(original)
        result = _value_to_python(val)
        self.assertEqual(result, original)

    def test_nested_structure(self):
        original = {"mesh": "coarse", "params": [1, 2, 3], "opts": {"tol": 1e-6}}
        val = _python_to_value(original)
        result = _value_to_python(val)
        self.assertEqual(result["mesh"], "coarse")
        self.assertEqual(result["params"], [1, 2, 3])

    def test_unsupported_type_becomes_string(self):
        """Unsupported types are serialized via str()."""
        val = _python_to_value(object)
        result = _value_to_python(val)
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
#  Discipline base class
# ---------------------------------------------------------------------------
class TestDisciplineDiscreteVars(unittest.TestCase):
    """Tests for add_discrete_input / add_discrete_output on Discipline."""

    def test_add_discrete_input(self):
        d = Discipline()
        d.add_discrete_input("mode")
        self.assertEqual(len(d._discrete_var_meta), 1)
        meta = d._discrete_var_meta[0]
        self.assertEqual(meta.name, "mode")
        self.assertEqual(meta.type, data.VariableType.kDiscreteInput)

    def test_add_discrete_output(self):
        d = Discipline()
        d.add_discrete_output("status")
        self.assertEqual(len(d._discrete_var_meta), 1)
        meta = d._discrete_var_meta[0]
        self.assertEqual(meta.name, "status")
        self.assertEqual(meta.type, data.VariableType.kDiscreteOutput)

    def test_clear_data_resets_discrete_meta(self):
        d = Discipline()
        d.add_discrete_input("x")
        d.add_discrete_output("y")
        d._clear_data()
        self.assertEqual(len(d._discrete_var_meta), 0)


# ---------------------------------------------------------------------------
#  DisciplineServer – discrete metadata streaming and process_inputs
# ---------------------------------------------------------------------------
class TestDisciplineServerDiscrete(unittest.TestCase):
    """Tests for discrete variable handling in DisciplineServer."""

    def test_get_variable_definitions_includes_discrete(self):
        """GetVariableDefinitions should stream both continuous and discrete metadata."""
        server = DisciplineServer()
        server._discipline = Discipline()
        server._discipline.add_input("x", shape=(1,))
        server._discipline.add_discrete_input("mode")

        responses = list(server.GetVariableDefinitions(None, None))
        self.assertEqual(len(responses), 2)
        types = [r.type for r in responses]
        self.assertIn(data.VariableType.kInput, types)
        self.assertIn(data.VariableType.kDiscreteInput, types)

    def test_process_inputs_with_discrete(self):
        """process_inputs should demux continuous and discrete messages."""
        server = DisciplineServer()

        flat_inputs = {"x": np.zeros(2)}
        discrete_inputs = {}
        discrete_outputs = {}

        request_iterator = [
            data.VariableMessage(
                continuous=data.Array(
                    name="x", start=0, end=1,
                    type=data.VariableType.kInput, data=[1.0, 2.0],
                )
            ),
            data.VariableMessage(
                discrete=data.DiscreteVariable(
                    name="mode",
                    type=data.VariableType.kDiscreteInput,
                    value=_python_to_value("forward"),
                )
            ),
            data.VariableMessage(
                discrete=data.DiscreteVariable(
                    name="status",
                    type=data.VariableType.kDiscreteOutput,
                    value=_python_to_value(0),
                )
            ),
        ]

        di, do = server.process_inputs(
            request_iterator, flat_inputs,
            discrete_inputs=discrete_inputs,
            discrete_outputs=discrete_outputs,
        )

        np.testing.assert_array_equal(flat_inputs["x"], [1.0, 2.0])
        self.assertEqual(di["mode"], "forward")
        self.assertEqual(do["status"], 0)


# ---------------------------------------------------------------------------
#  DisciplineClient – discrete variable definitions, assembly, recovery
# ---------------------------------------------------------------------------
class TestDisciplineClientDiscrete(unittest.TestCase):
    """Tests for discrete variable handling in DisciplineClient."""

    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_get_variable_definitions_separates_discrete(self, mock_stub_cls):
        mock_channel = Mock()
        mock_stub = mock_stub_cls.return_value
        mock_stub.GetVariableDefinitions.return_value = [
            data.VariableMetaData(name="x", type=data.kInput, shape=[1]),
            data.VariableMetaData(
                name="mode", type=data.VariableType.kDiscreteInput
            ),
            data.VariableMetaData(name="f", type=data.kOutput, shape=[1]),
            data.VariableMetaData(
                name="status", type=data.VariableType.kDiscreteOutput
            ),
        ]

        client = DisciplineClient(mock_channel)
        client.get_variable_definitions()

        self.assertEqual(len(client._var_meta), 2)
        self.assertEqual(len(client._discrete_var_meta), 2)

    def test_assemble_input_messages_with_discrete(self):
        """_assemble_input_messages should include discrete messages."""
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)
        client._stream_options.num_double = 10

        inputs = {"x": np.array([1.0])}
        discrete_inputs = {"mode": "forward", "order": 3}

        messages = client._assemble_input_messages(
            inputs, discrete_inputs=discrete_inputs
        )

        # 1 continuous + 2 discrete = 3 messages
        self.assertEqual(len(messages), 3)

        # Check that discrete messages are present
        discrete_msgs = [
            m for m in messages if m.WhichOneof("payload") == "discrete"
        ]
        self.assertEqual(len(discrete_msgs), 2)

        names = {m.discrete.name for m in discrete_msgs}
        self.assertEqual(names, {"mode", "order"})

    def test_assemble_input_messages_with_discrete_outputs(self):
        """_assemble_input_messages should include discrete output messages."""
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)
        client._stream_options.num_double = 10

        inputs = {"x": np.array([1.0])}
        discrete_outputs = {"status": 0}

        messages = client._assemble_input_messages(
            inputs, discrete_outputs=discrete_outputs
        )

        # 1 continuous + 1 discrete output
        self.assertEqual(len(messages), 2)
        discrete_msgs = [
            m for m in messages if m.WhichOneof("payload") == "discrete"
        ]
        self.assertEqual(len(discrete_msgs), 1)
        self.assertEqual(discrete_msgs[0].discrete.name, "status")
        self.assertEqual(
            discrete_msgs[0].discrete.type, data.VariableType.kDiscreteOutput
        )

    def test_recover_outputs_with_discrete(self):
        """_recover_outputs should return (outputs, discrete_outputs) tuple."""
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)
        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(1,)),
        ]

        responses = [
            data.VariableMessage(
                continuous=data.Array(
                    name="f", type=data.kOutput, start=0, end=0, data=[42.0]
                )
            ),
            data.VariableMessage(
                discrete=data.DiscreteVariable(
                    name="status",
                    type=data.VariableType.kDiscreteOutput,
                    value=_python_to_value("converged"),
                )
            ),
        ]

        result = client._recover_outputs(responses)

        # Should be a tuple (continuous_outputs, discrete_outputs)
        self.assertIsInstance(result, tuple)
        outputs, d_outputs = result
        np.testing.assert_array_equal(outputs["f"], [42.0])
        self.assertEqual(d_outputs["status"], "converged")


# ---------------------------------------------------------------------------
#  ExplicitServer – discrete data flow
# ---------------------------------------------------------------------------
class TestExplicitServerDiscrete(unittest.TestCase):
    """Tests for discrete variable handling in ExplicitServer."""

    def test_compute_function_with_discrete(self):
        """ComputeFunction should pass discrete data to discipline.compute."""
        server = ExplicitServer()
        discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(1,))
        discipline.add_output("f", shape=(1,))
        discipline.add_discrete_input("mode")
        discipline.add_discrete_output("status")
        server._discipline = discipline
        server._stream_opts.num_double = 10

        captured = {}

        def compute(inputs, outputs, discrete_inputs, discrete_outputs):
            captured["discrete_inputs"] = dict(discrete_inputs)
            outputs["f"] = inputs["x"] * 2
            discrete_outputs["status"] = "ok"

        discipline.compute = compute

        request_iterator = [
            data.VariableMessage(
                continuous=data.Array(
                    name="x", start=0, end=0,
                    type=data.VariableType.kInput, data=[3.0],
                )
            ),
            data.VariableMessage(
                discrete=data.DiscreteVariable(
                    name="mode",
                    type=data.VariableType.kDiscreteInput,
                    value=_python_to_value("fast"),
                )
            ),
        ]

        responses = list(server.ComputeFunction(request_iterator, None))

        # Should have received the discrete input
        self.assertEqual(captured["discrete_inputs"]["mode"], "fast")

        # Should have continuous and discrete output responses
        continuous_responses = [
            r for r in responses if r.WhichOneof("payload") == "continuous"
        ]
        discrete_responses = [
            r for r in responses if r.WhichOneof("payload") == "discrete"
        ]

        self.assertEqual(len(continuous_responses), 1)
        self.assertEqual(continuous_responses[0].continuous.data[0], 6.0)

        self.assertEqual(len(discrete_responses), 1)
        self.assertEqual(discrete_responses[0].discrete.name, "status")

    def test_compute_gradient_with_discrete(self):
        """ComputeGradient should pass discrete data to compute_partials."""
        server = ExplicitServer()
        discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(1,))
        discipline.add_output("f", shape=(1,))
        discipline.add_discrete_input("mode")
        discipline.declare_partials("f", "x")
        server._discipline = discipline
        server._stream_opts.num_double = 10

        captured = {}

        def compute_partials(inputs, jac, discrete_inputs):
            captured["discrete_inputs"] = dict(discrete_inputs)
            jac["f", "x"] = np.array([2.0])

        discipline.compute_partials = compute_partials

        request_iterator = [
            data.VariableMessage(
                continuous=data.Array(
                    name="x", start=0, end=0,
                    type=data.VariableType.kInput, data=[3.0],
                )
            ),
            data.VariableMessage(
                discrete=data.DiscreteVariable(
                    name="mode",
                    type=data.VariableType.kDiscreteInput,
                    value=_python_to_value("fast"),
                )
            ),
        ]

        responses = list(server.ComputeGradient(request_iterator, None))

        self.assertEqual(captured["discrete_inputs"]["mode"], "fast")
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].continuous.data[0], 2.0)


# ---------------------------------------------------------------------------
#  ImplicitServer – discrete data flow
# ---------------------------------------------------------------------------
class TestImplicitServerDiscrete(unittest.TestCase):
    """Tests for discrete variable handling in ImplicitServer."""

    def _make_discipline(self):
        discipline = ImplicitDiscipline()
        discipline.add_input("x", shape=(1,))
        discipline.add_output("f", shape=(1,))
        discipline.add_discrete_input("mode")
        discipline.declare_partials("f", "x")
        return discipline

    def _make_request(self, x_val=1.0, f_val=0.0, mode_val="fast"):
        return [
            data.VariableMessage(
                continuous=data.Array(
                    name="x", start=0, end=0,
                    type=data.VariableType.kInput, data=[x_val],
                )
            ),
            data.VariableMessage(
                continuous=data.Array(
                    name="f", start=0, end=0,
                    type=data.VariableType.kOutput, data=[f_val],
                )
            ),
            data.VariableMessage(
                discrete=data.DiscreteVariable(
                    name="mode",
                    type=data.VariableType.kDiscreteInput,
                    value=_python_to_value(mode_val),
                )
            ),
        ]

    def test_compute_residuals_with_discrete(self):
        server = ImplicitServer()
        discipline = self._make_discipline()
        server._discipline = discipline
        server._stream_opts.num_double = 10

        captured = {}

        def compute_residuals(inputs, outputs, residuals, di, do):
            captured["mode"] = di["mode"]
            residuals["f"] = outputs["f"] - inputs["x"]

        discipline.compute_residuals = compute_residuals

        responses = list(
            server.ComputeResiduals(self._make_request(), None)
        )

        self.assertEqual(captured["mode"], "fast")
        self.assertGreater(len(responses), 0)

    def test_solve_residuals_with_discrete(self):
        server = ImplicitServer()
        discipline = self._make_discipline()
        server._discipline = discipline
        server._stream_opts.num_double = 10

        captured = {}

        def solve_residuals(inputs, outputs, di):
            captured["mode"] = di["mode"]
            outputs["f"] = inputs["x"]

        discipline.solve_residuals = solve_residuals

        responses = list(
            server.SolveResiduals(self._make_request(), None)
        )

        self.assertEqual(captured["mode"], "fast")
        self.assertGreater(len(responses), 0)

    def test_compute_residual_gradients_with_discrete(self):
        server = ImplicitServer()
        discipline = self._make_discipline()
        server._discipline = discipline
        server._stream_opts.num_double = 10

        captured = {}

        def residual_partials(inputs, outputs, jac, di, do):
            captured["mode"] = di["mode"]
            jac["f", "x"] = np.array([1.0])

        discipline.residual_partials = residual_partials

        responses = list(
            server.ComputeResidualGradients(self._make_request(), None)
        )

        self.assertEqual(captured["mode"], "fast")
        self.assertGreater(len(responses), 0)


# ---------------------------------------------------------------------------
#  ExplicitClient – discrete round-trip
# ---------------------------------------------------------------------------
class TestExplicitClientDiscrete(unittest.TestCase):
    """Tests for discrete inputs through ExplicitClient."""

    @patch("philote_mdo.generated.disciplines_pb2_grpc.ExplicitServiceStub")
    def test_run_compute_with_discrete(self, mock_stub_cls):
        mock_channel = Mock()
        mock_stub = mock_stub_cls.return_value
        client = ExplicitClient(mock_channel)
        # pin the streaming transport; the unary path is covered in
        # tests/test_unary_transport.py
        client.transport = "stream"
        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(1,)),
        ]

        mock_stub.ComputeFunction.return_value = [
            data.VariableMessage(
                continuous=data.Array(
                    name="f", type=data.kOutput, start=0, end=0, data=[6.0]
                )
            ),
        ]

        result = client.run_compute(
            {"x": np.array([3.0])},
            discrete_inputs={"mode": "fast"},
        )

        # Should have been called with VariableMessage including discrete
        call_args = mock_stub.ComputeFunction.call_args
        messages = list(call_args[0][0])
        discrete_msgs = [
            m for m in messages if m.WhichOneof("payload") == "discrete"
        ]
        self.assertEqual(len(discrete_msgs), 1)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.ExplicitServiceStub")
    def test_run_compute_partials_with_discrete(self, mock_stub_cls):
        mock_channel = Mock()
        mock_stub = mock_stub_cls.return_value
        client = ExplicitClient(mock_channel)
        # pin the streaming transport; the unary path is covered in
        # tests/test_unary_transport.py
        client.transport = "stream"
        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(1,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(1,)),
        ]
        client._partials_meta = [
            data.PartialsMetaData(name="f", subname="x"),
        ]

        mock_stub.ComputeGradient.return_value = [
            data.VariableMessage(
                continuous=data.Array(
                    name="f", subname="x", type=data.kPartial,
                    start=0, end=0, data=[2.0],
                )
            ),
        ]

        partials = client.run_compute_partials(
            {"x": np.array([3.0])},
            discrete_inputs={"mode": "fast"},
        )

        np.testing.assert_array_equal(partials[("f", "x")], [2.0])


# ---------------------------------------------------------------------------
#  OpenMDAO utils – discrete variable setup and extraction
# ---------------------------------------------------------------------------
class TestOpenMdaoUtilsDiscrete(unittest.TestCase):
    """Tests for discrete variable support in OpenMDAO utility functions."""

    def test_client_setup_declares_discrete_vars(self):
        comp = MagicMock()
        comp._client._var_meta = [
            data.VariableMetaData(name="x", type=data.kInput, shape=[1], units="m"),
        ]
        comp._client._discrete_var_meta = [
            data.VariableMetaData(
                name="mode", type=data.VariableType.kDiscreteInput
            ),
            data.VariableMetaData(
                name="status", type=data.VariableType.kDiscreteOutput
            ),
        ]

        om_utils.client_setup(comp)

        comp.add_discrete_input.assert_called_once_with("mode", val=None)
        comp.add_discrete_output.assert_called_once_with("status", val=None)

    def test_create_local_discrete_inputs(self):
        discrete_inputs = {"mode": "forward", "extra": 99}
        meta = [
            data.VariableMetaData(
                name="mode", type=data.VariableType.kDiscreteInput
            ),
        ]

        result = om_utils.create_local_discrete_inputs(discrete_inputs, meta)
        self.assertEqual(result, {"mode": "forward"})

    def test_create_local_discrete_inputs_none(self):
        result = om_utils.create_local_discrete_inputs(None, [])
        self.assertIsNone(result)

    def test_create_local_discrete_inputs_empty_returns_none(self):
        """When no matching vars, should return None."""
        discrete_inputs = {"other": 1}
        meta = [
            data.VariableMetaData(
                name="mode", type=data.VariableType.kDiscreteOutput
            ),
        ]
        result = om_utils.create_local_discrete_inputs(discrete_inputs, meta)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
#  OpenMDAO Explicit – discrete tuple return handling
# ---------------------------------------------------------------------------
@patch("openmdao.api.ExplicitComponent.__init__")
class TestOpenMdaoExplicitDiscrete(unittest.TestCase):
    """Tests for discrete data flow through RemoteExplicitComponent."""

    def test_compute_with_discrete_tuple_result(self, om_patch):
        from philote_mdo.openmdao import RemoteExplicitComponent

        mock_channel = Mock()
        comp = RemoteExplicitComponent(channel=mock_channel)

        client_mock = MagicMock()
        client_mock._var_meta = [
            data.VariableMetaData(name="x", type=data.kInput, shape=[1]),
            data.VariableMetaData(name="f", type=data.kOutput, shape=[1]),
        ]
        client_mock._discrete_var_meta = [
            data.VariableMetaData(
                name="mode", type=data.VariableType.kDiscreteInput
            ),
            data.VariableMetaData(
                name="status", type=data.VariableType.kDiscreteOutput
            ),
        ]
        # Simulate server returning tuple (outputs, discrete_outputs)
        client_mock.run_compute.return_value = (
            {"f": np.array([6.0])},
            {"status": "ok"},
        )
        comp._client = client_mock
        comp.name = "test"

        inputs = {"x": np.array([3.0])}
        outputs = {"f": np.zeros(1)}
        discrete_inputs = {"mode": "fast"}
        discrete_outputs = {"status": None}

        comp.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        np.testing.assert_array_equal(outputs["f"], [6.0])
        self.assertEqual(discrete_outputs["status"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
