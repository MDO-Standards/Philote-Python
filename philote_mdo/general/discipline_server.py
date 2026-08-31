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
import warnings

import grpc
import numpy as np

import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc
from google.protobuf.empty_pb2 import Empty
from google.protobuf import struct_pb2
from philote_mdo.utils import (
    PairDict,
    build_shape_index,
    get_flattened_view,
    get_function_shape,
    get_partials_shape,
    get_variable_shape,
    read_array_into,
)
from philote_mdo.utils.validation import (
    JobCapacityError,
    JobNotFoundError,
    JobStateError,
    PhiloteJobError,
    PhiloteValidationError,
    validate_shape,
)
from philote_mdo.general.job import (
    DEFAULT_MAX_JOBS,
    DEFAULT_SWEEP_INTERVAL,
    DEFAULT_TTL,
    JOB_METADATA_KEY,
    JobState,
    JobStore,
)


# gRPC status codes for the job errors raised by JobStore
_JOB_STATUS = (
    (JobNotFoundError, grpc.StatusCode.NOT_FOUND),
    (JobCapacityError, grpc.StatusCode.RESOURCE_EXHAUSTED),
    (JobStateError, grpc.StatusCode.FAILED_PRECONDITION),
)


def _job_status(exc):
    """
    Returns the gRPC status code for a job error.
    """
    for exc_type, code in _JOB_STATUS:
        if isinstance(exc, exc_type):
            return code

    return grpc.StatusCode.INTERNAL


class DisciplineServer(disc.DisciplineServiceServicer):
    """
    Base class for all server classes.

    The server owns no discipline of its own. It holds a factory and builds one
    discipline instance per job, so that the state each client accumulates --
    its options, its variable metadata, and anything the discipline stores on
    itself -- stays private to that client.
    """

    def __init__(
        self,
        discipline=None,
        max_jobs=DEFAULT_MAX_JOBS,
        ttl=DEFAULT_TTL,
        sweep_interval=DEFAULT_SWEEP_INTERVAL,
    ):
        self.verbose = False

        self._max_jobs = max_jobs
        self._ttl = ttl
        self._sweep_interval = sweep_interval

        # live jobs, created once a factory is attached
        self._jobs = None

        if discipline is not None:
            self.attach_discipline(discipline)

    def attach_to_server(self, server):
        """
        Attaches this discipline server class to a gRPC server.
        """
        self._warn_on_thread_pool(server)
        disc.add_DisciplineServiceServicer_to_server(self, server)

    def attach_discipline(self, factory):
        """
        Adds a discipline factory to the server.

        Parameters
        ----------
        factory : callable
            Zero-argument callable returning a fresh discipline. A class works
            directly when its ``initialize()`` performs its own configuration;
            a discipline configured from the outside needs a closure or a
            ``functools.partial``.
        """
        self._jobs = JobStore(
            factory,
            max_jobs=self._max_jobs,
            ttl=self._ttl,
            sweep_interval=self._sweep_interval,
        )

    def _warn_on_thread_pool(self, server):
        """
        Warns when the gRPC thread pool is smaller than the job limit.

        Every in-flight RPC holds a worker for its whole duration, including
        the entire bidirectional stream of a compute call. A server that
        allows more jobs than the pool has workers queues calls silently
        instead of running them, so the limit the operator set is not the one
        that binds.
        """
        try:
            workers = server._state.thread_pool._max_workers
        except AttributeError:  # pragma: no cover - private gRPC internals
            return

        if workers < self._max_jobs:
            warnings.warn(
                f"gRPC thread pool has {workers} workers but this server "
                f"allows {self._max_jobs} concurrent jobs. Every in-flight "
                f"RPC holds a worker for its full duration, so jobs beyond "
                f"the {workers}th will queue rather than run. Raise "
                f"max_workers on the ThreadPoolExecutor, or lower max_jobs.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _resolve_job(self, context):
        """
        Returns the job named by the request metadata.

        Aborts the call rather than returning when the header is missing or
        names a job the server does not hold. Call this outside the RPC's
        ``try`` block: ``context.abort`` raises, and a surrounding
        ``except Exception`` would otherwise swallow it and re-abort with the
        wrong status code.

        Parameters
        ----------
        context : grpc.ServicerContext
            Context of the call in progress.

        Returns
        -------
        Job
        """
        if self._jobs is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "no discipline factory is attached to this server.",
            )

        metadata = dict(context.invocation_metadata() or ())
        job_id = metadata.get(JOB_METADATA_KEY)

        if not job_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"missing '{JOB_METADATA_KEY}' metadata. Call StartJob and "
                f"send the returned id on every subsequent call.",
            )

        try:
            return self._jobs.get(job_id)
        except JobNotFoundError as e:
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))

    def StartJob(self, request, context):
        """
        RPC that starts a job and returns its handle.
        """
        if self._jobs is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "no discipline factory is attached to this server.",
            )

        try:
            job = self._jobs.create()
        except PhiloteJobError as e:
            context.abort(_job_status(e), str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"StartJob failed: {e}"
            )

        return data.JobHandle(job_id=job.job_id)

    def EndJob(self, request, context):
        """
        RPC that ends a job and releases what its discipline holds.
        """
        job = self._resolve_job(context)

        try:
            self._jobs.close(job.job_id)
            return Empty()
        except PhiloteJobError as e:
            context.abort(_job_status(e), str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"EndJob failed: {e}"
            )

    def KeepAlive(self, request, context):
        """
        RPC that defers eviction of an idle job.
        """
        # _resolve_job already refreshed the job's timestamp
        self._resolve_job(context)
        return Empty()

    def _describe(self, context):
        """
        Returns a discipline instance for the job-independent RPCs.

        ``GetInfo`` and ``GetAvailableOptions`` report properties of the
        discipline class rather than of any run, so they must answer before a
        client has a job. Build a throwaway instance for them.
        """
        if self._jobs is None:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "no discipline factory is attached to this server.",
            )

        return self._jobs.describe()

    def GetInfo(self, request, context):
        """
        RPC that sends the discipline information/properties to the client.

        Job-independent: these are properties of the discipline itself.
        """
        discipline = self._describe(context)

        return data.DisciplineProperties(
            continuous=discipline._is_continuous,
            differentiable=discipline._is_differentiable,
            provides_gradients=discipline._provides_gradients,
            name=discipline._name,
            version=discipline._version,
        )

    def SetStreamOptions(self, request, context):
        """
        Receives options from the client on how data will be transmitted to and
        received from the client. The options are stored on the job for use in
        the compute routines, since they are a per-client setting.
        """
        job = self._resolve_job(context)
        job.stream_opts = request
        return Empty()

    def GetAvailableOptions(self, request, context):
        """
        RPC that gets the names and types of all available discipline options.

        Job-independent: the option schema comes from ``initialize()`` and is a
        property of the discipline class, so a client may call this before it
        starts a job.
        """
        discipline = self._describe(context)

        try:
            opts_dict = discipline.options_list
            opts = data.OptionsList()

            for name, val in opts_dict.items():
                opts.options.append(name)

                # assign the correct data type
                if val == "bool":
                    type = data.kBool
                elif val == "int":
                    type = data.kInt
                elif val == "float":
                    type = data.kDouble
                elif val == "str":
                    type = data.kString
                elif val == "dict":
                    type = data.kStruct
                else:
                    raise PhiloteValidationError(
                        "Invalid value for discipline option '{}'".format(name)
                    )

                opts.type.append(type)

            return opts
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"GetAvailableOptions failed: {e}"
            )

    def SetOptions(self, request, context):
        """
        RPC that sets the discipline options.

        Rejected once the job has run ``Setup``: the variable metadata was
        built from the previous option values, so accepting new ones would
        leave the job describing itself inconsistently.
        """
        job = self._resolve_job(context)

        try:
            with job.lock:
                job.require_before_setup("SetOptions")
                job.discipline.set_options(request.options)

            return Empty()
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except PhiloteJobError as e:
            context.abort(_job_status(e), str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"SetOptions failed: {e}"
            )

    def Setup(self, request, context):
        """
        RPC that runs the setup function
        """
        job = self._resolve_job(context)

        try:
            with job.lock:
                job.state = JobState.SETUP
                job.discipline._clear_data()
                job.discipline.setup()
                job.discipline.setup_partials()
                job.state = JobState.READY

            return Empty()
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"Setup failed: {e}"
            )

    def GetVariableDefinitions(self, request, context):
        """
        Transmits variable metadata about the analysis discipline to the client.

        Both continuous and discrete variable metadata are streamed.
        """
        job = self._resolve_job(context)

        for var in job.discipline._var_meta:
            yield var

        for var in job.discipline._discrete_var_meta:
            yield var

    def GetPartialDefinitions(self, request, context):
        """
        Transmits partials metadata about the analysis discipline to the client.
        """
        job = self._resolve_job(context)

        for jac in job.discipline._partials_meta:
            yield jac

    def SetVariableShapes(self, request_iterator, context):
        """
        Receives client-defined shapes for variables flagged as
        dynamic_shape.

        The client must call this RPC after GetVariableDefinitions and
        before any compute RPCs for disciplines that contain variables
        with dynamic shapes.
        """
        job = self._resolve_job(context)

        try:
            # index by type and name once; searching the list per incoming
            # message is quadratic in the number of variables
            index = {}
            for var in job.discipline._var_meta:
                index.setdefault((var.type, var.name), var)

            for meta in request_iterator:
                validate_shape(tuple(meta.shape), "SetVariableShapes")

                var = index.get((meta.type, meta.name))

                if var is None:
                    raise PhiloteValidationError(
                        f"SetVariableShapes: variable '{meta.name}' "
                        f"not found."
                    )

                if not var.dynamic_shape:
                    raise PhiloteValidationError(
                        f"Variable '{meta.name}' does not allow "
                        f"dynamic shapes."
                    )

                var.shape[:] = []
                var.shape.extend(meta.shape)

                # if the variable is an output on an implicit discipline,
                # also update the matching residual entry
                if meta.type == data.VariableType.kOutput:
                    res = index.get((data.VariableType.kResidual, meta.name))

                    if res is not None and res.dynamic_shape:
                        res.shape[:] = []
                        res.shape.extend(meta.shape)

            return Empty()
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"SetVariableShapes failed: {e}"
            )

    def preallocate_inputs(
        self, job, inputs, flat_inputs, outputs=None, flat_outputs=None
    ):
        """
        Preallocates the inputs before receiving data from the client.

        Note, for implicit disciplines, the function values are considered
        inputs to evaluate the residuals and the partials of the residuals.

        Parameters
        ----------
        job : Job
            The job whose discipline supplies the variable metadata.
        """
        for var in job.discipline._var_meta:
            # validate that dynamic-shape variables have been resolved
            if var.dynamic_shape and len(var.shape) == 0:
                raise PhiloteValidationError(
                    f"Variable '{var.name}' has dynamic_shape=True but "
                    f"no shape has been set. Call SetVariableShapes "
                    f"before computing."
                )

            if var.type == data.kInput:
                inputs[var.name] = np.zeros(var.shape)
                flat_inputs[var.name] = get_flattened_view(inputs[var.name])

            if (
                var.type == data.kOutput
                and outputs is not None
                and flat_outputs is not None
            ):
                outputs[var.name] = np.zeros(var.shape)
                flat_outputs[var.name] = get_flattened_view(outputs[var.name])

    def preallocate_partials(self, job):
        """
        Preallocates the partials.

        Note: there are edge cases for this function, where either f or x, or
        both are scalar. In those cases the shapes of the partials must be
        treated differently.

        Parameters
        ----------
        job : Job
            The job whose discipline supplies the partials metadata.
        """
        jac = PairDict()

        # index the metadata by (type, name) once; scanning it per partial is
        # quadratic in the number of variables
        shapes = build_shape_index(job.discipline._var_meta)

        for pair in job.discipline._partials_meta:
            shape = get_partials_shape(
                get_function_shape(shapes, pair.name, "preallocate_partials"),
                get_variable_shape(shapes, pair.subname, "preallocate_partials"),
            )

            jac[(pair.name, pair.subname)] = np.zeros(shape)

        return jac

    def process_inputs(
        self,
        request_iterator,
        flat_inputs,
        flat_outputs=None,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Processes the message inputs from a gRPC stream.

        The stream consists of ``VariableMessage`` wrappers, each of which
        contains either a continuous ``Array`` or a ``DiscreteVariable``.

        Note, for implicit disciplines, the function values are considered
        inputs to evaluate the residuals and the partials of the residuals.
        """
        if discrete_inputs is None:
            discrete_inputs = {}
        if discrete_outputs is None:
            discrete_outputs = {}

        for message in request_iterator:
            variant = message.WhichOneof("payload")

            if variant == "continuous":
                arr = message.continuous

                if len(arr.data) > 0:
                    if arr.type == data.VariableType.kInput:
                        read_array_into(arr, flat_inputs[arr.name])
                    elif arr.type == data.VariableType.kOutput:
                        read_array_into(arr, flat_outputs[arr.name])
                else:
                    raise PhiloteValidationError(
                        "Expected continuous variables but arrays were"
                        " empty for variable %s." % (arr.name)
                    )

            elif variant == "discrete":
                dv = message.discrete
                # Convert protobuf Value to native Python type
                native_value = _value_to_python(dv.value)

                if dv.type == data.VariableType.kDiscreteInput:
                    discrete_inputs[dv.name] = native_value
                elif dv.type == data.VariableType.kDiscreteOutput:
                    discrete_outputs[dv.name] = native_value

        return discrete_inputs, discrete_outputs


def _value_to_python(value):
    """
    Converts a ``google.protobuf.Value`` to a native Python object.

    Parameters
    ----------
    value : google.protobuf.Value
        protobuf Value message

    Returns
    -------
    object
        Native Python equivalent (None, bool, int/float, str, list, or dict)
    """
    kind = value.WhichOneof("kind")

    if kind == "null_value":
        return None
    elif kind == "bool_value":
        return value.bool_value
    elif kind == "number_value":
        # protobuf stores all numbers as doubles; return int if lossless
        num = value.number_value
        if num == int(num):
            return int(num)
        return num
    elif kind == "string_value":
        return value.string_value
    elif kind == "list_value":
        return [_value_to_python(v) for v in value.list_value.values]
    elif kind == "struct_value":
        return {k: _value_to_python(v) for k, v in value.struct_value.fields.items()}
    else:  # pragma: no cover – all protobuf Value kinds are handled above
        return None


def _python_to_value(obj):
    """
    Converts a native Python object to a ``google.protobuf.Value``.

    Parameters
    ----------
    obj : object
        A Python scalar, list, or dict

    Returns
    -------
    google.protobuf.Value
        protobuf Value message
    """
    val = struct_pb2.Value()

    if obj is None:
        val.null_value = 0
    elif isinstance(obj, bool):
        val.bool_value = obj
    elif isinstance(obj, (int, float)):
        val.number_value = float(obj)
    elif isinstance(obj, str):
        val.string_value = obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            val.list_value.values.append(_python_to_value(item))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            val.struct_value.fields[str(k)].CopyFrom(_python_to_value(v))
    else:
        val.string_value = str(obj)

    return val
