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
import os
import importlib.resources as resources
import grpc_tools.protoc
import protoletariat.__main__ as protol


def main():

    print("Compiling proto files.")

    proto_include = os.path.join(resources.files("grpc_tools"), "_proto")
    generated_dir = "./philote_mdo/generated/"

    # proto files
    proto_files = ["data.proto", "disciplines.proto"]

    # A FileDescriptorSet, generated alongside the Python code below, lets
    # protoletariat rewrite the generated imports without shelling out to a
    # standalone "protoc" executable, which most environments do not have on
    # PATH (grpc_tools bundles its own protoc, used in-process below).
    descriptor_set_path = os.path.join(generated_dir, "_descriptor.bin")

    # compile the proto files for use in python
    return_code = grpc_tools.protoc.main(
        [
            "grpc_tools.protoc",
            "-I{}".format(proto_include),
            "-I{}".format("./proto"),
            "--python_out={}".format(generated_dir),
            "--pyi_out={}".format(generated_dir),
            "--grpc_python_out={}".format(generated_dir),
            "--descriptor_set_out={}".format(descriptor_set_path),
            "--include_imports",
        ]
        + proto_files
    )
    if return_code != 0:
        msg = "Compiling the proto files failed (see the protoc output above)."
        raise RuntimeError(msg)

    try:
        # call protoletariat to convert absolute imports to relative ones,
        # using the FileDescriptorSet generated above.
        protol.main(
            [
                "--python-out={}".format(generated_dir),
                "--in-place",
                "--dont-create-package",
                "raw",
                descriptor_set_path,
            ]
        )
    finally:
        os.remove(descriptor_set_path)


if __name__ == "__main__":
    main()