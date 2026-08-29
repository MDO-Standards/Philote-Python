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
import grpc
import philote_mdo.generated.disciplines_pb2_grpc as disc
import philote_mdo.generated.data_pb2 as data
from philote_mdo.general.discipline_server import (
    DisciplineServer,
    _python_to_value,
)
from philote_mdo.utils import get_chunk_indices, set_array_data
from philote_mdo.utils.validation import PhiloteValidationError


class ExplicitServer(DisciplineServer, disc.ExplicitServiceServicer):
    """
    Base class for remote explicit components.
    """

    def __init__(self, discipline_factory=None, **kwargs):
        super().__init__(discipline_factory=discipline_factory, **kwargs)

    def attach_to_server(self, server):
        """
        Attaches this discipline server class to a gRPC server.
        """
        super().attach_to_server(server)
        disc.add_ExplicitServiceServicer_to_server(self, server)

    def ComputeFunction(self, request_iterator, context):
        """
        Computes the function evaluation and sends the result to the client.
        """
        job = self._resolve_job(context)

        # serialise calls within this job. Separate jobs never contend here,
        # which is what lets two clients evaluate at the same time.
        job.lock.acquire()

        try:
            inputs = {}
            flat_inputs = {}
            outputs = {}
            discrete_inputs = {}
            discrete_outputs = {}

            self.preallocate_inputs(job, inputs, flat_inputs)
            discrete_inputs, _ = self.process_inputs(
                request_iterator, flat_inputs, discrete_inputs=discrete_inputs
            )

            # Call compute with discrete data when discrete variables are present
            if discrete_inputs or job.discipline._discrete_var_meta:
                job.discipline.compute(
                    inputs, outputs, discrete_inputs, discrete_outputs
                )
            else:
                job.discipline.compute(inputs, outputs)

            # Stream continuous outputs
            for output_name, value in outputs.items():
                for b, e in get_chunk_indices(value.size, job.stream_opts.num_double):
                    message = data.VariableMessage(
                        continuous=data.Array(
                            name=output_name,
                            type=data.kOutput,
                            start=b,
                            end=e - 1,
                        )
                    )
                    set_array_data(message.continuous, value.ravel()[b:e])

                    yield message

            # Stream discrete outputs
            for name, value in discrete_outputs.items():
                yield data.VariableMessage(
                    discrete=data.DiscreteVariable(
                        name=name,
                        type=data.VariableType.kDiscreteOutput,
                        value=_python_to_value(value),
                    )
                )
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"ComputeFunction failed: {e}"
            )
        finally:
            job.lock.release()

    def ComputeGradient(self, request_iterator, context):
        """
        Computes the gradient evaluation and sends the result to the client.
        """
        job = self._resolve_job(context)

        # serialise calls within this job. Separate jobs never contend here,
        # which is what lets two clients evaluate at the same time.
        job.lock.acquire()

        try:
            inputs = {}
            flat_inputs = {}
            discrete_inputs = {}

            self.preallocate_inputs(job, inputs, flat_inputs)
            jac = self.preallocate_partials(job)
            discrete_inputs, _ = self.process_inputs(
                request_iterator, flat_inputs, discrete_inputs=discrete_inputs
            )

            if discrete_inputs or job.discipline._discrete_var_meta:
                job.discipline.compute_partials(inputs, jac, discrete_inputs)
            else:
                job.discipline.compute_partials(inputs, jac)

            for jac, value in jac.items():
                for b, e in get_chunk_indices(value.size, job.stream_opts.num_double):
                    message = data.VariableMessage(
                        continuous=data.Array(
                            name=jac[0],
                            subname=jac[1],
                            type=data.kPartial,
                            start=b,
                            end=e - 1,
                        )
                    )
                    set_array_data(message.continuous, value.ravel()[b:e])

                    yield message
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"ComputeGradient failed: {e}"
            )
        finally:
            job.lock.release()
