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
try:
    import openmdao.api as om
    omdao_installed = True
except ImportError:
    omdao_installed = False
    om = None
import philote_mdo.general as pm
import philote_mdo.generated.data_pb2 as data
import philote_mdo.openmdao.utils as utils


if omdao_installed:
    class RemoteImplicitComponent(om.ImplicitComponent):
        """
        An OpenMDAO component that acts as a client to an implicit analysis server.
        """

        def __init__(self, channel=None, num_par_fd=1, **kwargs):
            """
            Initialize the component and client.
            """
            if not channel:
                raise ValueError('No channel provided, the Philote client will not'
                                 'be able to connect.')

            # generic Philote client
            # The setting of OpenMDAO options requires the list of available
            # Philote discipline options to be known during initialize. That
            # means that the server must be reachable to query the
            # available options on this discipline.
            self._client = pm.ImplicitClient(channel=channel)

            # call the init function of the explicit component
            super().__init__(num_par_fd=1, **kwargs)

            # assign and send the option values to the server
            # this must be done here and not in initialize, as the values of the
            # OpenMDAO options are only set after intialize has been called in the
            # init function. That is why the parent init function must be called
            # before sending the options values to the philote server.
            # options = {}
            # for key, val in self.options.items():
            #     options[key] = val
            self._client.send_options(kwargs)

        def initialize(self):
            """
            Define the OpenMDAO component options.
            """
            # get the available options from the philote discipline
            self._client.get_available_options()

            # add to the OpenMDAO component options
            for name, type_str in self._client.options_list.items():
                type = None
                if type_str == 'bool':
                    type = bool
                elif type_str == 'int':
                    type = int
                elif type_str == 'float':
                    type = float
                elif type_str == 'str':
                    type = str

                self.options.declare(name, types=type)

        def setup(self):
            """
            Set up the OpenMDAO component.
            """
            utils.client_setup(self)

        def setup_partials(self):
            """
            Set up the partials of the OpenMDAO component.
            """
            utils.client_setup_partials(self)

        def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
            """
            Compute the residual evaluation.
            """
            local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
            local_outputs = utils.create_local_inputs(outputs, self._client._var_meta, data.kOutput)

            res = self._client.run_compute_residuals(local_inputs, local_outputs)
            utils.assign_global_outputs(res, residuals)

        def solve_nonlinear(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
            """
            Solves the residual for the implicit discipline.
            """
            local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
            out = self._client.run_solve_residuals(local_inputs)
            utils.assign_global_outputs(out, outputs)

        def linearize(self, inputs, outputs, partials,  discrete_inputs=None, discrete_outputs=None):
            """
            Computes the residual gradients for the implicit discipline.
            """
            local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
            local_outputs = utils.create_local_inputs(outputs, self._client._var_meta, data.kOutput)
            jac = self._client.run_residual_gradients(local_inputs, local_outputs)
            utils.assign_global_outputs(jac, partials)
