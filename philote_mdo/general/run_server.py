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
from concurrent import futures
import philote_mdo.generated.explicit_pb2_grpc as explicit_pb2_grpc
import philote_mdo.generated.implicit_pb2_grpc as implicit_pb2_grpc


def run_server(service, port="50051", max_workers=10):
    """
    Helper function for running an analysis server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # add an explicit or implicit server
    if isinstance(service, explicit_pb2_grpc.ExplicitDisciplineServicer):
        explicit_pb2_grpc.add_ExplicitDisciplineServicer_to_server(service, server)
    elif isinstance(service, implicit_pb2_grpc.ImplicitDisciplineServicer):
        implicit_pb2_grpc.add_ImplicitDisciplineServicer_to_server(service, server)
    else:
        raise ValueError("Unexpected object type provided for variable " '"service".')

    server.add_insecure_port("[::]:" + port)
    server.start()

    server.wait_for_termination()
