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
from concurrent import futures
import unittest

from philote_mdo.general import Discipline
from conftest import job_context, make_server, make_server_from_instance
from unittest.mock import Mock

import grpc
import numpy as np
import numpy.testing as npt

from philote_mdo.general import (
    Discipline,
    DisciplineServer,
    ExplicitDiscipline,
    ExplicitServer,
    ExplicitClient,
    ImplicitDiscipline,
    ImplicitServer,
    ImplicitClient,
)
from philote_mdo.utils.validation import PhiloteValidationError
from philote_mdo.examples import FlexibleDiscipline
import philote_mdo.generated.data_pb2 as data


# ---------------------------------------------------------------
# Unit tests: Discipline base class
# ---------------------------------------------------------------
class TestDynamicShapeDiscipline(unittest.TestCase):
    """Unit tests for dynamic_shape flag on the Discipline base class."""

    def test_add_input_dynamic_shape(self):
        """add_input with dynamic_shape=True stores the flag and omits shape."""
        disc = Discipline()
        disc.add_input("x", dynamic_shape=True)

        self.assertEqual(len(disc._var_meta), 1)
        meta = disc._var_meta[0]
        self.assertEqual(meta.name, "x")
        self.assertTrue(meta.dynamic_shape)
        self.assertEqual(list(meta.shape), [])

    def test_add_output_dynamic_shape(self):
        """add_output with dynamic_shape=True stores the flag and omits shape."""
        disc = Discipline()
        disc.add_output("y", dynamic_shape=True)

        self.assertEqual(len(disc._var_meta), 1)
        meta = disc._var_meta[0]
        self.assertEqual(meta.name, "y")
        self.assertTrue(meta.dynamic_shape)
        self.assertEqual(list(meta.shape), [])

    def test_add_input_static_shape_unchanged(self):
        """add_input without dynamic_shape behaves as before."""
        disc = Discipline()
        disc.add_input("x", shape=(3, 2), units="m")

        meta = disc._var_meta[0]
        self.assertFalse(meta.dynamic_shape)
        self.assertEqual(list(meta.shape), [3, 2])

    def test_add_output_static_shape_unchanged(self):
        """add_output without dynamic_shape behaves as before."""
        disc = Discipline()
        disc.add_output("y", shape=(4,))

        meta = disc._var_meta[0]
        self.assertFalse(meta.dynamic_shape)
        self.assertEqual(list(meta.shape), [4])

    def test_dynamic_shape_skips_shape_validation(self):
        """dynamic_shape=True should not validate the default shape arg."""
        disc = Discipline()
        # Should not raise even though we pass no valid shape
        disc.add_input("x", dynamic_shape=True)
        disc.add_output("y", dynamic_shape=True)
        self.assertEqual(len(disc._var_meta), 2)

    def test_implicit_output_dynamic_shape_creates_residual(self):
        """For implicit disciplines, dynamic output also creates a residual entry."""
        disc = Discipline()
        disc._is_implicit = True
        disc.add_output("y", dynamic_shape=True, units="m")

        # Should have output and residual
        self.assertEqual(len(disc._var_meta), 2)
        out = disc._var_meta[0]
        res = disc._var_meta[1]
        self.assertEqual(out.type, data.VariableType.kOutput)
        self.assertTrue(out.dynamic_shape)
        self.assertEqual(res.type, data.VariableType.kResidual)
        self.assertTrue(res.dynamic_shape)


# ---------------------------------------------------------------
# Unit tests: DisciplineServer.SetVariableShapes
# ---------------------------------------------------------------
class TestSetVariableShapesRPC(unittest.TestCase):
    """Unit tests for the SetVariableShapes RPC handler."""

    def _make_server_with_dynamic_disc(self):
        disc = Discipline()
        disc.add_input("x", dynamic_shape=True)
        disc.add_output("y", dynamic_shape=True)
        disc.add_input("z", shape=(2,))  # static
        return make_server_from_instance(DisciplineServer, disc)

    def test_set_shapes_for_dynamic_variables(self):
        """SetVariableShapes updates shapes on dynamic variables."""
        server, job, context = self._make_server_with_dynamic_disc()

        x_meta = data.VariableMetaData(
            name="x", type=data.VariableType.kInput, shape=[5]
        )
        y_meta = data.VariableMetaData(
            name="y", type=data.VariableType.kOutput, shape=[5]
        )
        server.SetVariableShapes(iter([x_meta, y_meta]), context)

        # verify shapes were updated
        for var in job.discipline._var_meta:
            if var.name == "x":
                self.assertEqual(list(var.shape), [5])
            if var.name == "y" and var.type == data.VariableType.kOutput:
                self.assertEqual(list(var.shape), [5])

    def test_reject_shape_for_static_variable(self):
        """SetVariableShapes aborts when targeting a non-dynamic variable."""
        server, job, context = self._make_server_with_dynamic_disc()

        z_meta = data.VariableMetaData(
            name="z", type=data.VariableType.kInput, shape=[10]
        )
        server.SetVariableShapes(iter([z_meta]), context)
        context.abort.assert_called_once()

    def test_reject_unknown_variable(self):
        """SetVariableShapes aborts when the variable name is not found."""
        server, job, context = self._make_server_with_dynamic_disc()

        meta = data.VariableMetaData(
            name="nope", type=data.VariableType.kInput, shape=[3]
        )
        server.SetVariableShapes(iter([meta]), context)
        context.abort.assert_called_once()

    def test_reject_invalid_shape(self):
        """SetVariableShapes aborts on invalid (non-positive) shape."""
        server, job, context = self._make_server_with_dynamic_disc()

        meta = data.VariableMetaData(
            name="x", type=data.VariableType.kInput, shape=[-1]
        )
        server.SetVariableShapes(iter([meta]), context)
        context.abort.assert_called_once()

    def test_preallocate_raises_when_shape_unset(self):
        """preallocate_inputs raises if a dynamic variable has no shape."""
        server, job, context = self._make_server_with_dynamic_disc()

        with self.assertRaises(PhiloteValidationError):
            server.preallocate_inputs(job, {}, {})


# ---------------------------------------------------------------
# Unit tests: DisciplineClient helpers
# ---------------------------------------------------------------
class TestDynamicShapeClient(unittest.TestCase):
    """Unit tests for client-side dynamic shape helpers."""

    def test_set_variable_shape(self):
        """set_variable_shape creates correct VariableMetaData."""
        client = ExplicitClient.__new__(ExplicitClient)
        client._var_meta = []

        meta = client.set_variable_shape("x", (3, 2), data.VariableType.kInput)

        self.assertEqual(meta.name, "x")
        self.assertEqual(list(meta.shape), [3, 2])
        self.assertEqual(meta.type, data.VariableType.kInput)

    def test_get_dynamic_variables(self):
        """get_dynamic_variables filters correctly."""
        client = ExplicitClient.__new__(ExplicitClient)

        static_var = data.VariableMetaData(
            name="a", type=data.kInput, shape=[1], dynamic_shape=False
        )
        dynamic_var = data.VariableMetaData(
            name="b", type=data.kInput, dynamic_shape=True
        )
        client._var_meta = [static_var, dynamic_var]

        result = client.get_dynamic_variables()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "b")


# ---------------------------------------------------------------
# Integration test: FlexibleDiscipline round-trip
# ---------------------------------------------------------------
class TestFlexibleDisciplineIntegration(unittest.TestCase):
    """End-to-end test: dynamic shapes over gRPC."""

    def test_flexible_compute(self):
        """Client sets shapes, then computes with the FlexibleDiscipline."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = ExplicitServer(discipline_factory=FlexibleDiscipline)
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        try:
            client = ExplicitClient(
                channel=grpc.insecure_channel("localhost:50051")
            )
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()

            # verify dynamic_shape flags came through
            dynamic = client.get_dynamic_variables()
            self.assertEqual(len(dynamic), 2)

            # set shapes
            shapes = [
                client.set_variable_shape("x", (4,), data.VariableType.kInput),
                client.set_variable_shape(
                    "y", (4,), data.VariableType.kOutput
                ),
            ]
            client.send_variable_shapes(shapes)
            client.get_partials_definitions()

            # compute
            inputs = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
            outputs = client.run_compute(inputs)

            npt.assert_array_almost_equal(
                outputs["y"], [2.0, 4.0, 6.0, 8.0]
            )
        finally:
            server.stop(0)

    def test_flexible_compute_partials(self):
        """Client sets shapes, then computes partials."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = ExplicitServer(discipline_factory=FlexibleDiscipline)
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        try:
            client = ExplicitClient(
                channel=grpc.insecure_channel("localhost:50051")
            )
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()

            shapes = [
                client.set_variable_shape("x", (3,), data.VariableType.kInput),
                client.set_variable_shape(
                    "y", (3,), data.VariableType.kOutput
                ),
            ]
            client.send_variable_shapes(shapes)
            client.get_partials_definitions()

            inputs = {"x": np.array([1.0, 2.0, 3.0])}
            jac = client.run_compute_partials(inputs)

            expected = 2.0 * np.eye(3)
            npt.assert_array_almost_equal(jac["y", "x"], expected)
        finally:
            server.stop(0)

    def test_backward_compat_static_shapes(self):
        """Existing static-shape disciplines work without calling SetVariableShapes."""
        from philote_mdo.examples import Paraboloid

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = ExplicitServer(discipline_factory=Paraboloid)
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        try:
            client = ExplicitClient(
                channel=grpc.insecure_channel("localhost:50051")
            )
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()
            client.get_partials_definitions()

            # no dynamic variables
            self.assertEqual(len(client.get_dynamic_variables()), 0)

            inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
            outputs = client.run_compute(inputs)
            self.assertAlmostEqual(outputs["f_xy"][0], 39.0)
        finally:
            server.stop(0)


# ---------------------------------------------------------------
# Integration test: implicit discipline with dynamic shapes
# ---------------------------------------------------------------
class DynamicImplicit(ImplicitDiscipline):
    """Implicit discipline with dynamic shapes for residual testing."""

    def setup(self):
        self.add_input("a", shape=(1,))
        self.add_output("x", dynamic_shape=True)

    def setup_partials(self):
        self.declare_partials("x", "a")
        self.declare_partials("x", "x")

    def compute_residuals(self, inputs, outputs, residuals):
        residuals["x"] = outputs["x"] ** 2 - inputs["a"]

    def solve_residuals(self, inputs, outputs):
        outputs["x"] = np.sqrt(np.abs(inputs["a"]))

    def compute_residual_partials(self, inputs, outputs, partials):
        partials["x", "a"] = -np.ones(inputs["a"].shape)
        partials["x", "x"] = 2.0 * np.diag(outputs["x"].ravel())


class TestDynamicImplicitIntegration(unittest.TestCase):
    """Integration test for implicit discipline with dynamic shapes."""

    def test_implicit_dynamic_shape_residual(self):
        """SetVariableShapes updates residual entries for implicit disciplines."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = ImplicitServer(discipline_factory=DynamicImplicit)
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        try:
            client = ImplicitClient(
                channel=grpc.insecure_channel("localhost:50051")
            )
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()

            # set shape for the dynamic output (and its residual)
            shapes = [
                client.set_variable_shape(
                    "x", (2,), data.VariableType.kOutput
                ),
            ]
            client.send_variable_shapes(shapes)
            client.get_partials_definitions()

            inputs = {"a": np.array([4.0])}
            outputs = {"x": np.array([3.0, 1.0])}
            residuals = client.run_compute_residuals(inputs, outputs)

            # x**2 - a = [9-4, 1-4] = [5, -3]
            npt.assert_array_almost_equal(residuals["x"], [5.0, -3.0])
        finally:
            server.stop(0)


# ---------------------------------------------------------------
# Unit test: SetVariableShapes generic exception path
# ---------------------------------------------------------------
class TestSetVariableShapesGenericException(unittest.TestCase):
    """Tests the generic exception handler in SetVariableShapes."""

    def test_generic_exception_aborts(self):
        server, job, context = make_server(DisciplineServer, Discipline)
        disc = Discipline()
        disc.add_input("x", dynamic_shape=True)
        job.discipline = disc

        context = job_context(job=job)

        # Craft an iterator that raises a non-validation exception
        def bad_iterator():
            raise RuntimeError("unexpected error")
            yield  # pragma: no cover

        server.SetVariableShapes(bad_iterator(), context)
        context.abort.assert_called_once()
        self.assertEqual(
            context.abort.call_args[0][0], grpc.StatusCode.INTERNAL
        )


# ---------------------------------------------------------------
# Unit test: OpenMDAO utils dynamic shape paths
# ---------------------------------------------------------------
class TestOpenMdaoUtilsDynamicShapes(unittest.TestCase):
    """Tests the OpenMDAO utils functions with dynamic-shape variables."""

    def test_client_setup_with_dynamic_input(self):
        """client_setup passes shape_by_conn=True for dynamic inputs."""
        comp = Mock()
        var = Mock()
        var.name = "x"
        var.units = "m"
        var.type = data.kInput
        var.shape = []
        var.dynamic_shape = True

        comp._client._var_meta = [var]
        comp._client._discrete_var_meta = []
        comp._client.options_list = {}

        from philote_mdo.openmdao.utils import client_setup

        client_setup(comp)

        comp.add_input.assert_called_once_with(
            "x", shape_by_conn=True, units="m"
        )

    def test_client_setup_with_dynamic_output(self):
        """client_setup passes shape_by_conn=True for dynamic outputs."""
        comp = Mock()
        var = Mock()
        var.name = "y"
        var.units = ""
        var.type = data.kOutput
        var.shape = []
        var.dynamic_shape = True

        comp._client._var_meta = [var]
        comp._client._discrete_var_meta = []
        comp._client.options_list = {}

        from philote_mdo.openmdao.utils import client_setup

        client_setup(comp)

        comp.add_output.assert_called_once_with(
            "y", shape_by_conn=True, units=None
        )

    def test_send_resolved_shapes_with_dynamic_vars(self):
        """send_resolved_shapes reads OpenMDAO metadata and sends shapes."""
        comp = Mock()
        var = data.VariableMetaData(
            name="x", type=data.kInput, dynamic_shape=True
        )
        comp._client._var_meta = [var]
        comp._var_rel2meta = {"x": {"shape": (5,)}}
        comp._client.set_variable_shape.return_value = data.VariableMetaData(
            name="x", type=data.kInput, shape=[5]
        )

        from philote_mdo.openmdao.utils import send_resolved_shapes

        send_resolved_shapes(comp)

        comp._client.set_variable_shape.assert_called_once_with(
            "x", (5,), data.kInput
        )
        comp._client.send_variable_shapes.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
