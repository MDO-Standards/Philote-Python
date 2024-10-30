# Philote-Python
#
# Copyright 2022-2024 Christopher A. Lupp
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
import grpc
import numpy as np
from numpy.testing import assert_almost_equal
import openmdao.api as om
import philote_mdo.general as pmdo
import philote_mdo.openmdao as pmdo_om
from philote_mdo.examples import Paraboloid, QuadradicImplicit, Rosenbrock


class OpenMDAOIntegrationTests(unittest.TestCase):
    """
    Integration tests for the paraboloid discipline.
    """

    def test_openmdao_paraboloid_compute(self):
        """
        Integration test for the Paraboloid compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        prob = om.Problem()
        model = prob.model

        paraboloid_comp = pmdo_om.RemoteExplicitComponent(channel=grpc.insecure_channel("localhost:50051"))
        model.add_subsystem("Paraboloid", paraboloid_comp)

        # run setup
        prob.setup()

        # define some inputs
        prob.set_val("Paraboloid.x", 1.0)
        prob.set_val("Paraboloid.y", 2.0)

        # run a function evaluation
        prob.run_model()

        self.assertEqual(prob.get_val("Paraboloid.f_xy")[0], 39.0)

        # stop the server
        server.stop(0)

    def test_paraboloid_compute_partials(self):
        """
        Integration test for the Paraboloid compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        prob = om.Problem()
        model = prob.model

        paraboloid_comp = pmdo_om.RemoteExplicitComponent(channel=grpc.insecure_channel("localhost:50051"))
        model.add_subsystem("Paraboloid", paraboloid_comp)

        # setup the problem
        prob.setup()

        # define some inputs
        prob.set_val("Paraboloid.x", 1.0)
        prob.set_val("Paraboloid.y", 2.0)

        # run a function evaluation
        jac = prob.compute_totals("Paraboloid.f_xy", ["Paraboloid.x", "Paraboloid.y"])

        self.assertEqual(jac["Paraboloid.f_xy", "Paraboloid.x"][0], -2.0)
        self.assertEqual(jac["Paraboloid.f_xy", "Paraboloid.y"][0], 13.0)

        # stop the server
        server.stop(0)

    def test_rosenbrock_compute(self):
        """
        Integration test for the Paraboloid compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ExplicitServer(discipline=Rosenbrock())
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        prob = om.Problem()
        model = prob.model

        comp = pmdo_om.RemoteExplicitComponent(channel=grpc.insecure_channel("localhost:50051"), dimension=2)
        model.add_subsystem("Rosenbrock", comp)

        # run setup
        prob.setup()

        # define some inputs
        prob.set_val("Rosenbrock.x", np.zeros(2))

        # run a function evaluation
        prob.run_model()

        self.assertEqual(prob.get_val("Rosenbrock.f")[0], 1.0)

        # stop the server
        server.stop(0)

    def test_rosenbrock_compute_partials(self):
        """
        Integration test for the Paraboloid compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ExplicitServer(discipline=Rosenbrock())
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        prob = om.Problem()
        model = prob.model

        comp = pmdo_om.RemoteExplicitComponent(channel=grpc.insecure_channel("localhost:50051"), dimension=2)
        model.add_subsystem("Rosenbrock", comp)

        # setup the problem
        prob.setup()

        # define some inputs
        prob.set_val("Rosenbrock.x", np.zeros(2))

        # run a gradient evaluation
        jac = prob.compute_totals("Rosenbrock.f", ["Rosenbrock.x"])

        assert_almost_equal(jac["Rosenbrock.f", "Rosenbrock.x"], np.array([[-2.0, 0.0]]))

        # stop the server
        server.stop(0)

    def test_quadratic_compute_function(self):
        """
        Integration test for the OpenMDAO implicit client using the quadratic
        example.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ImplicitServer(discipline=QuadradicImplicit())
        discipline.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        prob = om.Problem()
        model = prob.model
        client = model.add_subsystem("Quadratic",
                                     pmdo_om.RemoteImplicitComponent(channel=grpc.insecure_channel("localhost:50051"))
                                     )

        # define some inputs
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}

        # run setup
        prob.setup()

        prob.set_val("Quadratic.a", 1.0)
        prob.set_val("Quadratic.b", 2.0)
        prob.set_val("Quadratic.c", -2.0)

        # run a function evaluation
        prob.run_model()

        self.assertAlmostEqual(prob.get_val("Quadratic.x")[0], 0.73205081, places=8)

        # stop the server
        server.stop(0)

    # def test_quadratic_compute_function(self):
    #     """
    #     Integration test for the OpenMDAO implicit client using the quadratic
    #     example.
    #     """
    #     # server code
    #     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    #
    #     discipline = pmdo.ImplicitServer(discipline=QuadradicImplicit())
    #     discipline.attach_to_server(server)
    #
    #     server.add_insecure_port("[::]:50051")
    #     server.start()
    #
    #     # client code
    #     prob = om.Problem()
    #     model = prob.model
    #     client = model.add_subsystem("Quadratic",
    #                                  pmdo_om.RemoteImplicitComponent(channel=grpc.insecure_channel("localhost:50051"))
    #                                  )
    #     client.linear_solver = om.DirectSolver()
    #
    #     # define some inputs
    #     inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
    #     outputs = {"x": np.array([4.0])}
    #
    #     # run setup
    #     prob.setup()
    #
    #     prob.set_val("Quadratic.a", 1.0)
    #     prob.set_val("Quadratic.b", 2.0)
    #     prob.set_val("Quadratic.c", -2.0)
    #
    #     # run a gradient evaluation
    #     jac = prob.compute_totals("Quadratic.x", ["Quadratic.a", "Quadratic.b", "Quadratic.c"])
    #
    #     # check the gradient values
    #     assert_almost_equal(jac["Quadratic.x", "Quadratic.a"], np.array([[-0.25]]))
    #     assert_almost_equal(jac["Quadratic.x", "Quadratic.b"], np.array([[-0.25]]))
    #     assert_almost_equal(jac["Quadratic.x", "Quadratic.c"], np.array([[-0.25]]))
    #
    #     # stop the server
    #     server.stop(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
