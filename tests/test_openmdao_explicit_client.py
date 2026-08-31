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
import unittest

from conftest import patch_discipline_stub
from unittest.mock import Mock, MagicMock, patch
import philote_mdo.generated.data_pb2 as data
from philote_mdo.openmdao import RemoteExplicitComponent


@patch("openmdao.api.ExplicitComponent.__init__")
class TestOpenMdaoExplicitClient(unittest.TestCase):
    """
    Unit tests for the OpenMDAO explicit component/client.
    """

    def setUp(self):
        # the component claims a job from inside __init__, so the stub has
        # to be mocked before construction rather than after
        self._job_patch = patch_discipline_stub()
        self._job_patch.start()

    def tearDown(self):
        self._job_patch.stop()


    @patch("philote_mdo.general.ExplicitClient")
    def test_constructor(self, mock_explicit_client, mock_explicit_component):
        """
        Tests the initialize function of the OpenMDAO Explicit Client.
        """
        mock_channel = Mock()
        num_par_fd = 1
        options = {"option1": True, "option2": 20, "option3": 3.14, "option4": "test"}

        # mock the client and its behavior
        mock_client_instance = MagicMock()
        mock_client_instance.options_list = {
            "option1": "bool",
            "option2": "int",
            "option3": "float",
            "option4": "str",
        }
        mock_client_instance.get_available_options.return_value = None

        # set the mock client instance as the return value of pm.ExplicitClient
        mock_explicit_client.return_value = mock_client_instance

        # Create an instance of the class
        comp = RemoteExplicitComponent(
            channel=mock_channel, num_par_fd=num_par_fd, **options
        )

        # Verify that pm.ExplicitClient is initialized with the correct channel
        mock_explicit_client.assert_called_once_with(channel=mock_channel)

        # Verify that super().__init__ is called with the correct arguments
        mock_explicit_component.assert_called_once_with(
            num_par_fd=num_par_fd, **options
        )

        # The options are sent at setup time from the resolved OpenMDAO
        # options, not from the constructor kwargs, so nothing is
        # transmitted during construction.
        comp._client.send_options.assert_not_called()

    def test_initialize(self, om_explicit_component_patch):
        mock_channel = Mock()

        # mock the client and its behavior
        client_mock = MagicMock()
        client_mock.options_list = {
            "option1": "bool",
            "option2": "int",
            "option3": "float",
        }

        # mock the options object
        options_mock = MagicMock()

        # create an instance of the class
        comp = RemoteExplicitComponent(channel=mock_channel)
        comp._client = client_mock
        comp.options = options_mock

        # call the method to be tested
        comp.initialize()

        # assert that get_available_options is called
        client_mock.get_available_options.assert_called_once()

        # assert that options.declare is called for each option
        options_mock.declare.assert_any_call("option1", types=bool)
        options_mock.declare.assert_any_call("option2", types=int)
        options_mock.declare.assert_any_call("option3", types=float)

    @patch("philote_mdo.openmdao.utils.client_setup")
    def test_setup(self, mock_openmdao_client_setup, om_explicit_component_patch):
        """
        Tests the setup function of the OpenMDAO Explicit Client.
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
        component = RemoteExplicitComponent(channel=mock_channel)
        component._client = Mock()
        component._client._var_meta = [var1, var2, var3, var4]

        # call the function
        component.setup()

        # check that the setup utility function was called
        mock_openmdao_client_setup.assert_called_once_with(component)

    @patch("philote_mdo.openmdao.utils.client_setup_partials")
    def test_setup_partials(
        self, mock_openmdao_client_setup_partials, om_explicit_component_patch
    ):
        """
        Tests the setup partials function of the OpenMDAO Explicit Client.
        """
        par1 = Mock()
        par1.name = "partial1"
        par1.subname = "subpartial1"

        par2 = Mock()
        par2.name = "partial2"
        par2.subname = "subpartial2"

        mock_channel = Mock()
        component = RemoteExplicitComponent(channel=mock_channel)
        component._client = Mock()
        component._client._partials_meta = [par1, par2]
        component._client._var_meta = []

        # call the function
        component.setup_partials()

        # check that the setup utility function was called
        mock_openmdao_client_setup_partials.assert_called_once_with(component)

    def test_compute(self, om_explicit_component_patch):
        """
        Tests the compute function of the OpenMDAO explicit client.
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

        # Mocking necessary objects
        inputs = {"input1": 10, "input2": 20}
        outputs = {"output1": None, "output2": None}
        discrete_inputs = None
        discrete_outputs = None

        # Mocking the client and its methods
        client_mock = MagicMock()
        client_mock._var_meta = [var1, var2, var3, var4]
        client_mock.run_compute.return_value = {"output1": 30, "output2": 40}

        # Creating instance of the class to be tested
        mock_channel = Mock()
        instance = RemoteExplicitComponent(channel=mock_channel)
        instance._client = client_mock
        # mock the component name
        instance.name = "test"

        # Calling the function to be tested
        instance.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        # Asserting that the method calls are made correctly
        client_mock.run_compute.assert_called_once_with(
            {"input1": 10, "input2": 20}, discrete_inputs=None
        )
        self.assertEqual(outputs["output1"], 30)
        self.assertEqual(outputs["output2"], 40)

    def test_compute_partials(self, om_explicit_component_patch):
        # Mocking necessary objects
        inputs = {"input1": 10, "input2": 20}
        partials = {
            "output1": {"input1": None, "input2": None},
            "output2": {"input1": None, "input2": None},
        }
        discrete_inputs = None
        discrete_outputs = None

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

        # Mocking the client and its methods
        client_mock = MagicMock()
        client_mock._var_meta = [var1, var2, var3, var4]
        client_mock.run_compute_partials.return_value = {
            "output1": {"input1": 1, "input2": 2},
            "output2": {"input1": 3, "input2": 4},
        }

        # Creating instance of the class to be tested
        instance = RemoteExplicitComponent(channel=Mock())
        instance._client = client_mock
        # mock the component name
        instance.name = "test"

        # Calling the function to be tested
        instance.compute_partials(inputs, partials, discrete_inputs, discrete_outputs)

        # Asserting that the method calls are made correctly
        client_mock.run_compute_partials.assert_called_once_with(
            {"input1": 10, "input2": 20}, discrete_inputs=None
        )
        self.assertEqual(partials["output1"]["input1"], 1)
        self.assertEqual(partials["output1"]["input2"], 2)
        self.assertEqual(partials["output2"]["input1"], 3)
        self.assertEqual(partials["output2"]["input2"], 4)

    def test_constructor_no_channel_raises_error(self, om_explicit_component_patch):
        """
        Tests that the constructor raises ValueError when no channel is provided.
        """
        with self.assertRaises(ValueError) as context:
            RemoteExplicitComponent(channel=None)
        
        self.assertIn("No channel provided", str(context.exception))
        
        # Test with channel not provided at all
        with self.assertRaises(ValueError) as context:
            RemoteExplicitComponent()
        
        self.assertIn("No channel provided", str(context.exception))

    def test_invalid_num_par_fd_raises(self, mock_explicit_component):
        """
        Tests that an invalid num_par_fd raises a ValueError.
        """
        mock_channel = Mock()
        with self.assertRaises(ValueError) as context:
            RemoteExplicitComponent(channel=mock_channel, num_par_fd=0)
        self.assertIn("num_par_fd must be a positive integer", str(context.exception))


class TestOpenMdaoExplicitComponentOptions(unittest.TestCase):
    """
    Tests that the resolved options reach the server (issue #77).

    These tests use the real OpenMDAO component machinery so the option
    values are actually declared, validated and defaulted.
    """

    def _make_component(self, **kwargs):
        client = MagicMock()
        client.options_list = {"dimension": "int", "scale": "float"}
        client._var_meta = []
        client._discrete_var_meta = []

        with patch("philote_mdo.general.ExplicitClient", return_value=client):
            comp = RemoteExplicitComponent(channel=Mock(), **kwargs)

        return comp, client

    def test_constructor_does_not_send_options(self):
        """
        The constructor no longer sends the raw keyword arguments.
        """
        _, client = self._make_component(dimension=2)

        client.send_options.assert_not_called()

    def test_setup_sends_constructor_options(self):
        """
        Options passed to the constructor are sent when setup runs.
        """
        comp, client = self._make_component(dimension=2, scale=1.5)

        comp.setup()

        client.send_options.assert_called_once_with(
            {"dimension": 2, "scale": 1.5}
        )

    def test_setup_sends_post_construction_assignment(self):
        """
        An option assigned after construction reaches the server.
        """
        comp, client = self._make_component(dimension=2, scale=1.5)
        comp.options["dimension"] = 10

        comp.setup()

        client.send_options.assert_called_once_with(
            {"dimension": 10, "scale": 1.5}
        )

    def test_setup_omits_unset_options(self):
        """
        An option the caller never set is left to the server's default.
        """
        comp, client = self._make_component(dimension=2)

        comp.setup()

        client.send_options.assert_called_once_with({"dimension": 2})

    def test_setup_omits_unrelated_openmdao_options(self):
        """
        OpenMDAO options the server never declared are not forwarded.
        """
        comp, client = self._make_component(dimension=2)

        comp.setup()

        sent = client.send_options.call_args[0][0]
        self.assertEqual(set(sent), {"dimension"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
