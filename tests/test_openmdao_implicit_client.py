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
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import philote_mdo.generated.data_pb2 as data
from philote_mdo.openmdao import RemoteImplicitComponent


@patch('openmdao.api.ImplicitComponent.__init__')
class TestOpenMdaoImplicitClient(unittest.TestCase):
    """
    Unit tests for the OpenMDAO implicit component/client.
    """

    @patch('philote_mdo.general.ImplicitClient')
    def test_constructor(self, mock_explicit_client, mock_implicit_component):
        """
        Tests the constructor of the OpenMDAO implicit Client.
        """
        mock_channel = Mock()
        num_par_fd = 1
        options = {'option1': True, 'option2': 20, 'option3': 3.14, 'option4': 'test'}

        # mock the client and its behavior
        mock_client_instance = MagicMock()
        mock_client_instance.options_list = {'option1': 'bool', 'option2': 'int', 'option3': 'float', 'option4': 'str'}
        mock_client_instance.get_available_options.return_value = None

        # set the mock client instance as the return value of pm.ExplicitClient
        mock_explicit_client.return_value = mock_client_instance

        # Create an instance of the class
        comp = RemoteImplicitComponent(channel=mock_channel, num_par_fd=num_par_fd, **options)

        # Verify that pm.ExplicitClient is initialized with the correct channel
        mock_explicit_client.assert_called_once_with(channel=mock_channel)

        # Verify that super().__init__ is called with the correct arguments
        mock_implicit_component.assert_called_once_with(num_par_fd=num_par_fd, **options)

        # Verify that send_options is called with the correct arguments
        expected_send_options_args = options.copy()
        comp._client.send_options.assert_called_once_with(expected_send_options_args)

    def test_initialize(self, om_implicit_component_patch):
        """
        Tests the initialize function of the OpenMDAO implicit client.
        """
        mock_channel = Mock()

        # mock the client and its behavior
        client_mock = MagicMock()
        client_mock.options_list = {'option1': 'bool', 'option2': 'int', 'option3': 'float'}

        # mock the options object
        options_mock = MagicMock()

        # create an instance of the class
        comp = RemoteImplicitComponent(channel=mock_channel)
        comp._client = client_mock
        comp.options = options_mock

        # call the method to be tested
        comp.initialize()

        # assert that get_available_options is called
        client_mock.get_available_options.assert_called_once()

        # assert that options.declare is called for each option
        options_mock.declare.assert_any_call('option1', types=bool)
        options_mock.declare.assert_any_call('option2', types=int)
        options_mock.declare.assert_any_call('option3', types=float)

    @patch('philote_mdo.openmdao.utils.client_setup')
    def test_setup(self, mock_openmdao_client_setup, om_explicit_component_patch):
        """
        Tests the setup function of the OpenMDAO implicit client.
        """
        var1 = Mock()
        var1.name = "input1"
        var1.units = "m"
        var1.type = data.kInput
        var1.shape = [1]

        var2 = Mock()
        var2.name = "input2"
        var2.units = None
        var2.type = data.kInput
        var2.shape = [1]

        var3 = Mock()
        var3.name = "output1"
        var3.units = None
        var3.type = data.kOutput
        var3.shape = [1]

        var4 = Mock()
        var4.name = "output2"
        var4.units = None
        var4.type = data.kOutput
        var4.shape = [1]

        mock_channel = Mock()
        component = RemoteImplicitComponent(channel=mock_channel)
        component._client = Mock()
        component._client._var_meta = [var1, var2, var3, var4]

        # call the function
        component.setup()

        # check that the setup utility function was called
        mock_openmdao_client_setup.assert_called_once_with(component)

    @patch('philote_mdo.openmdao.utils.client_setup_partials')
    def test_setup_partials(self, mock_openmdao_client_setup_partials, om_explicit_component_patch):
        """
        Tests the setup partials function of the OpenMDAO implicit client.
        """
        par1 = Mock()
        par1.name = "partial1"
        par1.subname = "subpartial1"

        par2 = Mock()
        par2.name = "partial2"
        par2.subname = "subpartial2"

        mock_channel = Mock()
        component = RemoteImplicitComponent(channel=mock_channel)
        component._client = Mock()
        component._client._partials_meta = [par1, par2]

        # call the function
        component.setup_partials()

        # check that the setup utility function was called
        mock_openmdao_client_setup_partials.assert_called_once_with(component)

    def test_apply_nonlinear(self, mock):
        """
        Tests the apply_nonlinear function of the OpenMDAO implicit client.
        """
        # mocking the client and its methods
        client_mock = MagicMock()
        client_mock._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="f", type=data.kResidual, shape=(3,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]

        expected_residuals = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([8.0, 9.0, 10.0]),
        }

        client_mock.run_compute_residuals.return_value = expected_residuals

        # creating instance of the class to be tested
        mock_channel = Mock()
        comp = RemoteImplicitComponent(channel=mock_channel)
        comp._client = client_mock
        # mock the component name
        comp.name = 'test'

        # inputs and outputs
        inputs = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2),
        }
        outputs = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([7.0, 6.0, 5.0]),
        }
        discrete_inputs = None
        discrete_outputs = None

        residuals = {
            "f": np.zeros(3),
            "g": np.zeros(3),
        }

        # calling the function to be tested
        comp.apply_nonlinear(inputs, outputs, residuals, discrete_inputs, discrete_outputs)

        # asserting that the method calls are made correctly
        client_mock.run_compute_residuals.assert_called_once_with(inputs, outputs)
        for res_name, expected_data in expected_residuals.items():
            self.assertTrue(res_name in residuals)
            np.testing.assert_array_equal(residuals[res_name], expected_data)

    def test_solve_nonlinear(self, mock):
        """
        Tests the solve_nonlinear function of the OpenMDAO implicit client.
        """
        # mocking the client and its methods
        client_mock = MagicMock()
        client_mock._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="f", type=data.kResidual, shape=(3,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]

        expected_outputs = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([8.0, 9.0, 10.0]),
        }

        client_mock.run_solve_residuals.return_value = expected_outputs

        # creating instance of the class to be tested
        mock_channel = Mock()
        comp = RemoteImplicitComponent(channel=mock_channel)
        comp._client = client_mock
        # mock the component name
        comp.name = 'test'

        # inputs and outputs
        inputs = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2),
        }
        discrete_inputs = None
        discrete_outputs = None

        outputs = {
            "f": np.zeros(3),
            "g": np.zeros(3),
        }

        # calling the function to be tested
        comp.solve_nonlinear(inputs, outputs, discrete_inputs, discrete_outputs)

        # asserting that the method calls are made correctly
        client_mock.run_solve_residuals.assert_called_once_with(inputs)
        for output_name, expected_data in expected_outputs.items():
            self.assertTrue(output_name in outputs)
            np.testing.assert_array_equal(outputs[output_name], expected_data)

    def test_linearize(self, om_explicit_component_patch):
        """
        Tests the linearize function of the OpenMDAO implicit client.
        """
        # mocking the client and its methods
        client_mock = MagicMock()
        client_mock._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="f", type=data.kResidual, shape=(3,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(1,)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]
        client_mock._partials_meta = [
            data.PartialsMetaData(name="f", subname="x"),
            data.PartialsMetaData(name="f", subname="f"),
            data.PartialsMetaData(name="g", subname="x"),
            data.PartialsMetaData(name="g", subname="g")
        ]

        expected_jac = {
            ("f", "x"): np.array([5.0, 6.0, 7.0]),
            ("f", "f"): np.array([8.0, 9.0, 10.0, 8.0, 9.0, 10.0, 8.0, 9.0, 10.0]).reshape((3,3)),
            ("g", "x"): np.array([1.0, 6.0, 7.0]),
            ("g", "g"): np.array([1.0, 9.0, 10.0, 8.0, 9.0, 10.0, 8.0, 9.0, 1.0]).reshape((3, 3)),
        }
        client_mock.run_residual_gradients.return_value = expected_jac

        # creating instance of the class to be tested
        mock_channel = Mock()
        comp = RemoteImplicitComponent(channel=mock_channel)
        comp._client = client_mock
        # mock the component name
        comp.name = 'test'

        # inputs and outputs
        inputs = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2),
        }
        outputs = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([7.0, 6.0, 5.0]),
        }
        discrete_inputs = None
        discrete_outputs = None

        partials = {
            ("f", "x"): np.zeros(3),
            ("f", "f"): np.zeros((3, 3)),
            ("g", "x"): np.zeros(3),
            ("g", "g"): np.zeros((3, 3))
        }

        # calling the function to be tested
        comp.linearize(inputs, outputs, partials, discrete_inputs, discrete_outputs)

        # asserting that the method calls are made correctly
        client_mock.run_residual_gradients.assert_called_once_with(inputs, outputs)
        for jac_key, expected_data in expected_jac.items():
            self.assertTrue(jac_key in partials)
            np.testing.assert_array_equal(partials[jac_key], expected_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)