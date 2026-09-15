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
from numpy.testing import assert_allclose
from scipy.sparse import issparse
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
import philote_mdo.general as pmdo
from philote_mdo.gemseo import GEMSEOtoPhiloteDiscipline
from philote_mdo.gemseo import PhiloteDiscipline

PORT = "[::]:50051"
CHANNEL = "localhost:50051"


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
