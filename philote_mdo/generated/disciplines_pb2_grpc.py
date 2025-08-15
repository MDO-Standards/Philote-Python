"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings
from . import data_pb2 as data__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
GRPC_GENERATED_VERSION = '1.66.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False
try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True
if _version_not_supported:
    raise RuntimeError(f'The grpc package installed is at version {GRPC_VERSION},' + f' but the generated code in disciplines_pb2_grpc.py depends on' + f' grpcio>={GRPC_GENERATED_VERSION}.' + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}' + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.')

class DisciplineServiceStub(object):
    """Generic Discipline Definition
    This service is never used on its own. It will always be used in a base class
    or implementation. Base classes are necessary to avoid code duplication in
    bindings for implicit and explicit disciplines.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetInfo = channel.unary_unary('/philote.DisciplineService/GetInfo', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=data__pb2.DisciplineProperties.FromString, _registered_method=True)
        self.SetStreamOptions = channel.unary_unary('/philote.DisciplineService/SetStreamOptions', request_serializer=data__pb2.StreamOptions.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, _registered_method=True)
        self.GetAvailableOptions = channel.unary_unary('/philote.DisciplineService/GetAvailableOptions', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=data__pb2.OptionsList.FromString, _registered_method=True)
        self.SetOptions = channel.unary_unary('/philote.DisciplineService/SetOptions', request_serializer=data__pb2.DisciplineOptions.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, _registered_method=True)
        self.Setup = channel.unary_unary('/philote.DisciplineService/Setup', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, _registered_method=True)
        self.GetVariableDefinitions = channel.unary_stream('/philote.DisciplineService/GetVariableDefinitions', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=data__pb2.VariableMetaData.FromString, _registered_method=True)
        self.GetPartialDefinitions = channel.unary_stream('/philote.DisciplineService/GetPartialDefinitions', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=data__pb2.PartialsMetaData.FromString, _registered_method=True)

class DisciplineServiceServicer(object):
    """Generic Discipline Definition
    This service is never used on its own. It will always be used in a base class
    or implementation. Base classes are necessary to avoid code duplication in
    bindings for implicit and explicit disciplines.
    """

    def GetInfo(self, request, context):
        """Gets the fundamental properties of the discipline
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetStreamOptions(self, request, context):
        """RPC to set remote streaming options
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetAvailableOptions(self, request, context):
        """RPC that allows a client to query available discipline options
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetOptions(self, request, context):
        """Sets the discipline options
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Setup(self, request, context):
        """Sets up the discipline
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetVariableDefinitions(self, request, context):
        """Gets the variable definitions for the discipline
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetPartialDefinitions(self, request, context):
        """Gets the discipline partials definitions
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_DisciplineServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'GetInfo': grpc.unary_unary_rpc_method_handler(servicer.GetInfo, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=data__pb2.DisciplineProperties.SerializeToString), 'SetStreamOptions': grpc.unary_unary_rpc_method_handler(servicer.SetStreamOptions, request_deserializer=data__pb2.StreamOptions.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'GetAvailableOptions': grpc.unary_unary_rpc_method_handler(servicer.GetAvailableOptions, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=data__pb2.OptionsList.SerializeToString), 'SetOptions': grpc.unary_unary_rpc_method_handler(servicer.SetOptions, request_deserializer=data__pb2.DisciplineOptions.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'Setup': grpc.unary_unary_rpc_method_handler(servicer.Setup, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'GetVariableDefinitions': grpc.unary_stream_rpc_method_handler(servicer.GetVariableDefinitions, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=data__pb2.VariableMetaData.SerializeToString), 'GetPartialDefinitions': grpc.unary_stream_rpc_method_handler(servicer.GetPartialDefinitions, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=data__pb2.PartialsMetaData.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('philote.DisciplineService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('philote.DisciplineService', rpc_method_handlers)

class DisciplineService(object):
    """Generic Discipline Definition
    This service is never used on its own. It will always be used in a base class
    or implementation. Base classes are necessary to avoid code duplication in
    bindings for implicit and explicit disciplines.
    """

    @staticmethod
    def GetInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/philote.DisciplineService/GetInfo', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, data__pb2.DisciplineProperties.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def SetStreamOptions(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/philote.DisciplineService/SetStreamOptions', data__pb2.StreamOptions.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def GetAvailableOptions(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/philote.DisciplineService/GetAvailableOptions', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, data__pb2.OptionsList.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def SetOptions(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/philote.DisciplineService/SetOptions', data__pb2.DisciplineOptions.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def Setup(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_unary(request, target, '/philote.DisciplineService/Setup', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def GetVariableDefinitions(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/philote.DisciplineService/GetVariableDefinitions', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, data__pb2.VariableMetaData.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def GetPartialDefinitions(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.unary_stream(request, target, '/philote.DisciplineService/GetPartialDefinitions', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, data__pb2.PartialsMetaData.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

class ExplicitServiceStub(object):
    """Definition of the generic Explicit Component RPC
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ComputeFunction = channel.stream_stream('/philote.ExplicitService/ComputeFunction', request_serializer=data__pb2.Array.SerializeToString, response_deserializer=data__pb2.Array.FromString, _registered_method=True)
        self.ComputeGradient = channel.stream_stream('/philote.ExplicitService/ComputeGradient', request_serializer=data__pb2.Array.SerializeToString, response_deserializer=data__pb2.Array.FromString, _registered_method=True)

class ExplicitServiceServicer(object):
    """Definition of the generic Explicit Component RPC
    """

    def ComputeFunction(self, request_iterator, context):
        """Calls the discipline Compute function
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ComputeGradient(self, request_iterator, context):
        """Calls the discipline ComputePartials function
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ExplicitServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'ComputeFunction': grpc.stream_stream_rpc_method_handler(servicer.ComputeFunction, request_deserializer=data__pb2.Array.FromString, response_serializer=data__pb2.Array.SerializeToString), 'ComputeGradient': grpc.stream_stream_rpc_method_handler(servicer.ComputeGradient, request_deserializer=data__pb2.Array.FromString, response_serializer=data__pb2.Array.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('philote.ExplicitService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('philote.ExplicitService', rpc_method_handlers)

class ExplicitService(object):
    """Definition of the generic Explicit Component RPC
    """

    @staticmethod
    def ComputeFunction(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/philote.ExplicitService/ComputeFunction', data__pb2.Array.SerializeToString, data__pb2.Array.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def ComputeGradient(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/philote.ExplicitService/ComputeGradient', data__pb2.Array.SerializeToString, data__pb2.Array.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

class ImplicitServiceStub(object):
    """Definition of the generic Explicit Discipline RPC
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ComputeResiduals = channel.stream_stream('/philote.ImplicitService/ComputeResiduals', request_serializer=data__pb2.Array.SerializeToString, response_deserializer=data__pb2.Array.FromString, _registered_method=True)
        self.SolveResiduals = channel.stream_stream('/philote.ImplicitService/SolveResiduals', request_serializer=data__pb2.Array.SerializeToString, response_deserializer=data__pb2.Array.FromString, _registered_method=True)
        self.ComputeResidualGradients = channel.stream_stream('/philote.ImplicitService/ComputeResidualGradients', request_serializer=data__pb2.Array.SerializeToString, response_deserializer=data__pb2.Array.FromString, _registered_method=True)

class ImplicitServiceServicer(object):
    """Definition of the generic Explicit Discipline RPC
    """

    def ComputeResiduals(self, request_iterator, context):
        """Calls the discipline Compute function
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SolveResiduals(self, request_iterator, context):
        """Calls the discipline RPC that solves the nonlinear equations
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ComputeResidualGradients(self, request_iterator, context):
        """Calls the discipline ComputePartials function
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ImplicitServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {'ComputeResiduals': grpc.stream_stream_rpc_method_handler(servicer.ComputeResiduals, request_deserializer=data__pb2.Array.FromString, response_serializer=data__pb2.Array.SerializeToString), 'SolveResiduals': grpc.stream_stream_rpc_method_handler(servicer.SolveResiduals, request_deserializer=data__pb2.Array.FromString, response_serializer=data__pb2.Array.SerializeToString), 'ComputeResidualGradients': grpc.stream_stream_rpc_method_handler(servicer.ComputeResidualGradients, request_deserializer=data__pb2.Array.FromString, response_serializer=data__pb2.Array.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('philote.ImplicitService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('philote.ImplicitService', rpc_method_handlers)

class ImplicitService(object):
    """Definition of the generic Explicit Discipline RPC
    """

    @staticmethod
    def ComputeResiduals(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/philote.ImplicitService/ComputeResiduals', data__pb2.Array.SerializeToString, data__pb2.Array.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def SolveResiduals(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/philote.ImplicitService/SolveResiduals', data__pb2.Array.SerializeToString, data__pb2.Array.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)

    @staticmethod
    def ComputeResidualGradients(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/philote.ImplicitService/ComputeResidualGradients', data__pb2.Array.SerializeToString, data__pb2.Array.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata, _registered_method=True)