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
Tests for the unary compute transport.

Three layers are covered:

- TestTransportSelection exercises the client's transport ladder as pure
  logic, with no gRPC involved.
- TestUnaryServicers calls the unary servicer methods directly.
- TestUnaryIntegration runs a real gRPC server on a dedicated port.
"""
from concurrent import futures
import unittest
from unittest.mock import Mock, patch

import grpc
import numpy as np

import philote_mdo.general as pmdo
import philote_mdo.generated.data_pb2 as data
from philote_mdo.examples import Paraboloid, QuadradicImplicit
from philote_mdo.general import (
    ExplicitClient,
    ExplicitDiscipline,
    ExplicitServer,
    ImplicitServer,
)
from philote_mdo.utils.validation import PhiloteServerError, PhiloteValidationError


# ports dedicated to this module, so that it never collides with the servers
# that test_integration.py and test_dynamic_shapes.py bind on 50051
UNARY_PORT = 50071
FALLBACK_PORT = 50072
IMPLICIT_PORT = 50073


def make_rpc_error(code, details="boom"):
    """
    Builds a fake RpcError, matching the pattern used in test_explicit_client.
    """
    error = grpc.RpcError()
    error.code = lambda: code
    error.details = lambda: details
    return error


class TestTransportSelection(unittest.TestCase):
    """
    Tests the client-side choice between the unary and streaming transports.
    """

    def setUp(self):
        patcher = patch(
            "philote_mdo.generated.disciplines_pb2_grpc.ExplicitServiceStub"
        )
        self.stub_cls = patcher.start()
        self.addCleanup(patcher.stop)

        self.stub = self.stub_cls.return_value
        self.client = ExplicitClient(Mock())
        self.client._var_meta = [
            data.VariableMetaData(name="x", type=data.kInput, shape=(2,)),
            data.VariableMetaData(name="f", type=data.kOutput, shape=(2,)),
        ]

        # a well-formed unary response, so the happy path can be asserted
        self.stub.ComputeFunctionUnary.return_value = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="f", type=data.kOutput, start=0, end=1, data=[1.0, 2.0]
                    )
                )
            ]
        )
        self.stub.ComputeFunction.return_value = [
            data.VariableMessage(
                continuous=data.Array(
                    name="f", type=data.kOutput, start=0, end=1, data=[3.0, 4.0]
                )
            )
        ]

        self.inputs = {"x": np.array([1.0, 2.0])}

    def test_auto_attempts_unary_when_untried(self):
        outputs = self.client.run_compute(self.inputs)

        self.assertTrue(self.stub.ComputeFunctionUnary.called)
        self.assertFalse(self.stub.ComputeFunction.called)
        self.assertTrue(np.array_equal(outputs["f"], np.array([1.0, 2.0])))

    def test_transport_stream_forces_streaming(self):
        self.client.transport = "stream"

        outputs = self.client.run_compute(self.inputs)

        self.assertFalse(self.stub.ComputeFunctionUnary.called)
        self.assertTrue(self.stub.ComputeFunction.called)
        self.assertTrue(np.array_equal(outputs["f"], np.array([3.0, 4.0])))

    def test_transport_unary_bypasses_size_gate(self):
        """
        An explicit pin overrides the size gate, even for a payload the gate
        would otherwise reject.
        """
        self.client.transport = "unary"
        self.client._var_meta.append(
            data.VariableMetaData(name="big", type=data.kOutput, shape=(1_000_000,))
        )

        self.client.run_compute(self.inputs)

        self.assertTrue(self.stub.ComputeFunctionUnary.called)
        self.assertNotIn("ComputeFunction", self.client._stream_only_rpcs)

    def test_setup_estimate_demotes_large_variables(self):
        """
        A payload known at setup time to be too large never reaches the unary
        stub, and the verdict is cached rather than recomputed.
        """
        self.client._var_meta.append(
            data.VariableMetaData(name="big", type=data.kOutput, shape=(1_000_000,))
        )

        self.client.run_compute(self.inputs)

        self.assertFalse(self.stub.ComputeFunctionUnary.called)
        self.assertTrue(self.stub.ComputeFunction.called)
        self.assertIn("ComputeFunction", self.client._stream_only_rpcs)

    def test_large_discrete_falls_back_for_this_call_only(self):
        """
        An oversized discrete value falls back without demoting the RPC, since
        the next call may carry a small one.
        """
        self.client._discrete_var_meta = [
            data.VariableMetaData(name="payload", type=data.kDiscreteInput)
        ]
        big = {"payload": ["x" * 1000] * 1000}

        self.client.run_compute(self.inputs, discrete_inputs=big)

        self.assertFalse(self.stub.ComputeFunctionUnary.called)
        self.assertTrue(self.stub.ComputeFunction.called)
        self.assertNotIn("ComputeFunction", self.client._stream_only_rpcs)

    def test_unimplemented_demotes_client_wide(self):
        """
        An old server answers UNIMPLEMENTED; the client retries on the
        streaming transport and stops attempting unary altogether.
        """
        self.stub.ComputeFunctionUnary.side_effect = make_rpc_error(
            grpc.StatusCode.UNIMPLEMENTED
        )

        outputs = self.client.run_compute(self.inputs)

        self.assertTrue(self.stub.ComputeFunction.called)
        self.assertIs(self.client._unary_supported, False)
        self.assertTrue(np.array_equal(outputs["f"], np.array([3.0, 4.0])))

    def test_resource_exhausted_permanent_without_discrete_vars(self):
        """
        With no discrete variables the payload size cannot vary, so a
        RESOURCE_EXHAUSTED will recur and the RPC is demoted permanently.
        """
        self.stub.ComputeFunctionUnary.side_effect = make_rpc_error(
            grpc.StatusCode.RESOURCE_EXHAUSTED
        )

        self.client.run_compute(self.inputs)

        self.assertTrue(self.stub.ComputeFunction.called)
        self.assertIn("ComputeFunction", self.client._stream_only_rpcs)

    def test_resource_exhausted_transient_with_discrete_vars(self):
        """
        With discrete variables present the size can vary from call to call,
        so the RPC is retried but not demoted.
        """
        self.client._discrete_var_meta = [
            data.VariableMetaData(name="payload", type=data.kDiscreteInput)
        ]
        self.stub.ComputeFunctionUnary.side_effect = make_rpc_error(
            grpc.StatusCode.RESOURCE_EXHAUSTED
        )

        self.client.run_compute(self.inputs)

        self.assertTrue(self.stub.ComputeFunction.called)
        self.assertNotIn("ComputeFunction", self.client._stream_only_rpcs)

    def test_internal_error_propagates_without_fallback(self):
        """
        A real server bug must surface rather than being silently retried on
        the other transport.
        """
        self.stub.ComputeFunctionUnary.side_effect = make_rpc_error(
            grpc.StatusCode.INTERNAL, "discipline exploded"
        )

        with self.assertRaises(PhiloteServerError) as ctx:
            self.client.run_compute(self.inputs)

        self.assertIn("discipline exploded", str(ctx.exception))
        self.assertFalse(self.stub.ComputeFunction.called)

    def test_server_limit_narrows_client_limit(self):
        """
        The effective limit is the smaller of the client's performance
        threshold and the server's advertised capacity.
        """
        self.client.unary_max_bytes = 4096
        self.client._server_unary_max_bytes = 512
        self.assertEqual(self.client._unary_limit(), 512)

        self.client._server_unary_max_bytes = 8192
        self.assertEqual(self.client._unary_limit(), 4096)

        self.client._server_unary_max_bytes = 0
        self.assertEqual(self.client._unary_limit(), 4096)

    def test_get_discipline_info_stores_transport_fields(self):
        with patch(
            "philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub"
        ) as disc_cls:
            client = ExplicitClient(Mock())
            disc_cls.return_value.GetInfo.return_value = data.DisciplineProperties(
                name="d", version="1", supports_unary=True, max_unary_bytes=4096
            )
            client._disc_stub = disc_cls.return_value
            client.get_discipline_info()

        self.assertIs(client._unary_supported, True)
        self.assertEqual(client._server_unary_max_bytes, 4096)

    def test_unassembled_messages_are_unchunked(self):
        """
        The unary transport must not fragment at the stream chunk size.
        """
        self.client._stream_options.num_double = 1

        chunked = self.client._assemble_input_messages(self.inputs)
        unchunked = self.client._assemble_input_messages(self.inputs, chunked=False)

        self.assertEqual(len(chunked), 2)
        self.assertEqual(len(unchunked), 1)
        self.assertEqual(list(unchunked[0].continuous.data), [1.0, 2.0])


class TestUnaryServicers(unittest.TestCase):
    """
    Tests the unary servicer methods directly, with a Mock context.
    """

    def test_compute_function_unary(self):
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(2,), units="")
        discipline.add_output("f", shape=(2,), units="")

        def compute(inputs, outputs):
            outputs["f"] = inputs["x"] * 3.0

        discipline.compute = compute

        request = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="x", type=data.kInput, start=0, end=1, data=[1.0, 2.0]
                    )
                )
            ]
        )

        response = server.ComputeFunctionUnary(request, Mock())

        self.assertEqual(len(response.variables), 1)
        arr = response.variables[0].continuous
        self.assertEqual(arr.name, "f")
        self.assertEqual(arr.start, 0)
        self.assertEqual(arr.end, 1)
        self.assertEqual(list(arr.data), [3.0, 6.0])

    def test_compute_gradient_unary(self):
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(1,), units="")
        discipline.add_output("f", shape=(1,), units="")
        discipline.declare_partials("f", "x")

        def compute_partials(inputs, partials):
            partials["f", "x"] = np.array([7.0])

        discipline.compute_partials = compute_partials

        request = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="x", type=data.kInput, start=0, end=0, data=[1.0]
                    )
                )
            ]
        )

        response = server.ComputeGradientUnary(request, Mock())

        self.assertEqual(len(response.variables), 1)
        arr = response.variables[0].continuous
        self.assertEqual(arr.name, "f")
        self.assertEqual(arr.subname, "x")
        self.assertEqual(list(arr.data), [7.0])

    def test_unary_ignores_stream_chunking(self):
        """
        The unary response is one message per variable regardless of the
        negotiated stream chunk size.
        """
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        server._stream_opts.num_double = 1
        discipline.add_input("x", shape=(1,), units="")
        discipline.add_output("f", shape=(5,), units="")

        def compute(inputs, outputs):
            outputs["f"] = np.arange(5.0)

        discipline.compute = compute

        request = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="x", type=data.kInput, start=0, end=0, data=[1.0]
                    )
                )
            ]
        )

        response = server.ComputeFunctionUnary(request, Mock())

        self.assertEqual(len(response.variables), 1)
        self.assertEqual(list(response.variables[0].continuous.data), [0, 1, 2, 3, 4])

    def test_unary_discrete_outputs_round_trip(self):
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(1,), units="")
        discipline.add_output("f", shape=(1,), units="")
        discipline.add_discrete_output("flag", default=0)

        def compute(inputs, outputs, discrete_inputs, discrete_outputs):
            outputs["f"] = inputs["x"]
            discrete_outputs["flag"] = 42

        discipline.compute = compute

        request = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="x", type=data.kInput, start=0, end=0, data=[1.0]
                    )
                )
            ]
        )

        response = server.ComputeFunctionUnary(request, Mock())

        discrete = [
            v.discrete for v in response.variables if v.WhichOneof("payload") == "discrete"
        ]
        self.assertEqual(len(discrete), 1)
        self.assertEqual(discrete[0].name, "flag")

    def test_unary_aborts_on_validation_error(self):
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(1,), units="")
        discipline.add_output("f", shape=(1,), units="")

        def bad_compute(inputs, outputs):
            raise PhiloteValidationError("bad input data")

        discipline.compute = bad_compute

        context = Mock()
        request = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="x", type=data.kInput, start=0, end=0, data=[1.0]
                    )
                )
            ]
        )

        server.ComputeFunctionUnary(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("bad input data", args[0][1])

    def test_unary_aborts_on_discipline_error(self):
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        discipline.add_input("x", shape=(1,), units="")
        discipline.add_output("f", shape=(1,), units="")

        def bad_compute(inputs, outputs):
            raise RuntimeError("kaboom")

        discipline.compute = bad_compute

        context = Mock()
        request = data.VariableSet(
            variables=[
                data.VariableMessage(
                    continuous=data.Array(
                        name="x", type=data.kInput, start=0, end=0, data=[1.0]
                    )
                )
            ]
        )

        server.ComputeFunctionUnary(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INTERNAL)
        self.assertIn("kaboom", args[0][1])

    def test_get_info_advertises_unary_support(self):
        server = ExplicitServer(discipline=Paraboloid())

        response = server.GetInfo(None, Mock())

        self.assertTrue(response.supports_unary)
        self.assertEqual(response.max_unary_bytes, 0)

    def test_base_server_does_not_advertise_unary(self):
        server = pmdo.DisciplineServer(discipline=Paraboloid())

        response = server.GetInfo(None, Mock())

        self.assertFalse(response.supports_unary)

    def test_implicit_unary_rpcs(self):
        server = ImplicitServer(discipline=QuadradicImplicit())
        server._discipline.setup()
        server._discipline.setup_partials()

        def variable_set(names, values, var_type):
            return data.VariableSet(
                variables=[
                    data.VariableMessage(
                        continuous=data.Array(
                            name=n, type=t, start=0, end=0, data=[v]
                        )
                    )
                    for n, v, t in zip(names, values, var_type)
                ]
            )

        request = variable_set(
            ["a", "b", "c", "x"],
            [1.0, 2.0, 1.0, 1.0],
            [data.kInput, data.kInput, data.kInput, data.kOutput],
        )

        residuals = server.ComputeResidualsUnary(request, Mock())
        self.assertEqual(len(residuals.variables), 1)
        self.assertEqual(residuals.variables[0].continuous.type, data.kResidual)

        solve_request = variable_set(
            ["a", "b", "c"], [1.0, 2.0, 1.0], [data.kInput] * 3
        )
        outputs = server.SolveResidualsUnary(solve_request, Mock())
        self.assertEqual(len(outputs.variables), 1)
        self.assertEqual(outputs.variables[0].continuous.name, "x")

        gradients = server.ComputeResidualGradientsUnary(request, Mock())
        self.assertEqual(len(gradients.variables), 4)
        for message in gradients.variables:
            self.assertEqual(message.continuous.type, data.kPartial)


class TestUnaryIntegration(unittest.TestCase):
    """
    End-to-end tests over a real gRPC connection.
    """

    def _serve(self, server_impl, port):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        server_impl.attach_to_server(server)
        server.add_insecure_port("[::]:%d" % port)
        server.start()
        self.addCleanup(server.stop, 0)
        return server

    def _client(self, client_cls, port):
        client = client_cls(channel=grpc.insecure_channel("localhost:%d" % port))
        client.get_discipline_info()
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()
        return client

    def test_explicit_matches_across_transports(self):
        self._serve(ExplicitServer(discipline=Paraboloid()), UNARY_PORT)
        client = self._client(ExplicitClient, UNARY_PORT)

        self.assertTrue(client._unary_supported)

        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}

        for transport in ("unary", "stream"):
            with self.subTest(transport=transport):
                client.transport = transport

                outputs = client.run_compute(inputs)
                self.assertEqual(outputs["f_xy"][0], 39.0)

                partials = client.run_compute_partials(inputs)
                self.assertAlmostEqual(partials["f_xy", "x"][0], -2.0)
                self.assertAlmostEqual(partials["f_xy", "y"][0], 13.0)

    def test_implicit_matches_across_transports(self):
        self._serve(ImplicitServer(discipline=QuadradicImplicit()), IMPLICIT_PORT)
        client = self._client(pmdo.ImplicitClient, IMPLICIT_PORT)

        inputs = {"a": np.array([1.0]), "b": np.array([-4.0]), "c": np.array([3.0])}

        for transport in ("unary", "stream"):
            with self.subTest(transport=transport):
                client.transport = transport

                outputs = client.run_solve_residuals(inputs)
                self.assertAlmostEqual(outputs["x"][0], 3.0)

                residuals = client.run_compute_residuals(inputs, outputs)
                self.assertAlmostEqual(residuals["x"][0], 0.0)

                partials = client.run_residual_gradients(inputs, outputs)
                self.assertEqual(len(partials), 4)

    def test_falls_back_against_stream_only_server(self):
        """
        A server that does not implement the unary RPCs still works: the
        client demotes on UNIMPLEMENTED and retries over the stream.
        """

        class StreamOnlyServer(ExplicitServer):
            _supports_unary = False

            def ComputeFunctionUnary(self, request, context):
                context.abort(grpc.StatusCode.UNIMPLEMENTED, "no unary here")

        self._serve(StreamOnlyServer(discipline=Paraboloid()), FALLBACK_PORT)
        client = self._client(ExplicitClient, FALLBACK_PORT)

        # the server advertises no unary support, so the client should not
        # even attempt it
        self.assertFalse(client._unary_supported)

        outputs = client.run_compute({"x": np.array([1.0]), "y": np.array([2.0])})
        self.assertEqual(outputs["f_xy"][0], 39.0)

    def test_probes_unary_without_get_info(self):
        """
        A client that never queried the discipline properties still discovers
        the unary transport by attempting it.
        """
        self._serve(ExplicitServer(discipline=Paraboloid()), UNARY_PORT + 10)

        client = ExplicitClient(
            channel=grpc.insecure_channel("localhost:%d" % (UNARY_PORT + 10))
        )
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        self.assertIsNone(client._unary_supported)

        outputs = client.run_compute({"x": np.array([1.0]), "y": np.array([2.0])})
        self.assertEqual(outputs["f_xy"][0], 39.0)


if __name__ == "__main__":
    unittest.main()
