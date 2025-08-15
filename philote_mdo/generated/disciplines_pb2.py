"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 5, 27, 2, '', 'disciplines.proto')
_sym_db = _symbol_database.Default()
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from . import data_pb2 as data__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11disciplines.proto\x12\x07philote\x1a\x1bgoogle/protobuf/empty.proto\x1a\ndata.proto2\x84\x04\n\x11DisciplineService\x12B\n\x07GetInfo\x12\x16.google.protobuf.Empty\x1a\x1d.philote.DisciplineProperties"\x00\x12D\n\x10SetStreamOptions\x12\x16.philote.StreamOptions\x1a\x16.google.protobuf.Empty"\x00\x12E\n\x13GetAvailableOptions\x12\x16.google.protobuf.Empty\x1a\x14.philote.OptionsList"\x00\x12B\n\nSetOptions\x12\x1a.philote.DisciplineOptions\x1a\x16.google.protobuf.Empty"\x00\x129\n\x05Setup\x12\x16.google.protobuf.Empty\x1a\x16.google.protobuf.Empty"\x00\x12O\n\x16GetVariableDefinitions\x12\x16.google.protobuf.Empty\x1a\x19.philote.VariableMetaData"\x000\x01\x12N\n\x15GetPartialDefinitions\x12\x16.google.protobuf.Empty\x1a\x19.philote.PartialsMetaData"\x000\x012\x83\x01\n\x0fExplicitService\x127\n\x0fComputeFunction\x12\x0e.philote.Array\x1a\x0e.philote.Array"\x00(\x010\x01\x127\n\x0fComputeGradient\x12\x0e.philote.Array\x1a\x0e.philote.Array"\x00(\x010\x012\xc5\x01\n\x0fImplicitService\x128\n\x10ComputeResiduals\x12\x0e.philote.Array\x1a\x0e.philote.Array"\x00(\x010\x01\x126\n\x0eSolveResiduals\x12\x0e.philote.Array\x1a\x0e.philote.Array"\x00(\x010\x01\x12@\n\x18ComputeResidualGradients\x12\x0e.philote.Array\x1a\x0e.philote.Array"\x00(\x010\x01B\x11\n\x0forg.philote.mdob\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'disciplines_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'\n\x0forg.philote.mdo'
    _globals['_DISCIPLINESERVICE']._serialized_start = 72
    _globals['_DISCIPLINESERVICE']._serialized_end = 588
    _globals['_EXPLICITSERVICE']._serialized_start = 591
    _globals['_EXPLICITSERVICE']._serialized_end = 722
    _globals['_IMPLICITSERVICE']._serialized_start = 725
    _globals['_IMPLICITSERVICE']._serialized_end = 922