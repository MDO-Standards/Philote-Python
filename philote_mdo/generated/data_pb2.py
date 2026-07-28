"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(_runtime_version.Domain.PUBLIC, 5, 27, 2, '', 'data.proto')
_sym_db = _symbol_database.Default()
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ndata.proto\x12\x07philote\x1a\x1cgoogle/protobuf/struct.proto"\xae\x01\n\x14DisciplineProperties\x12\x12\n\ncontinuous\x18\x01 \x01(\x08\x12\x16\n\x0edifferentiable\x18\x02 \x01(\x08\x12\x1a\n\x12provides_gradients\x18\x03 \x01(\x08\x12\x0c\n\x04name\x18\x04 \x01(\t\x12\x0f\n\x07version\x18\x05 \x01(\t\x12\x16\n\x0esupports_unary\x18\x06 \x01(\x08\x12\x17\n\x0fmax_unary_bytes\x18\x07 \x01(\x03"#\n\rStreamOptions\x12\x12\n\nnum_double\x18\x01 \x01(\x03"?\n\x0bOptionsList\x12\x0f\n\x07options\x18\x01 \x03(\t\x12\x1f\n\x04type\x18\x02 \x03(\x0e2\x11.philote.DataType"=\n\x11DisciplineOptions\x12(\n\x07options\x18\x01 \x01(\x0b2\x17.google.protobuf.Struct"z\n\x10VariableMetaData\x12#\n\x04type\x18\x01 \x01(\x0e2\x15.philote.VariableType\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\r\n\x05shape\x18\x04 \x03(\x03\x12\r\n\x05units\x18\x05 \x01(\t\x12\x15\n\rdynamic_shape\x18\x06 \x01(\x08"@\n\x10PartialsMetaData\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07subname\x18\x02 \x01(\t\x12\r\n\x05shape\x18\x03 \x03(\x03"u\n\x05Array\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07subname\x18\x02 \x01(\t\x12\r\n\x05start\x18\x03 \x01(\x03\x12\x0b\n\x03end\x18\x04 \x01(\x03\x12#\n\x04type\x18\x05 \x01(\x0e2\x15.philote.VariableType\x12\x0c\n\x04data\x18\x06 \x03(\x01"l\n\x10DiscreteVariable\x12\x0c\n\x04name\x18\x01 \x01(\t\x12#\n\x04type\x18\x02 \x01(\x0e2\x15.philote.VariableType\x12%\n\x05value\x18\x03 \x01(\x0b2\x16.google.protobuf.Value"q\n\x0fVariableMessage\x12$\n\ncontinuous\x18\x01 \x01(\x0b2\x0e.philote.ArrayH\x00\x12-\n\x08discrete\x18\x02 \x01(\x0b2\x19.philote.DiscreteVariableH\x00B\t\n\x07payload":\n\x0bVariableSet\x12+\n\tvariables\x18\x01 \x03(\x0b2\x18.philote.VariableMessage*F\n\x08DataType\x12\t\n\x05kBool\x10\x00\x12\x08\n\x04kInt\x10\x01\x12\x0b\n\x07kDouble\x10\x02\x12\x0b\n\x07kString\x10\x03\x12\x0b\n\x07kStruct\x10\x04*m\n\x0cVariableType\x12\n\n\x06kInput\x10\x00\x12\x12\n\x0ekDiscreteInput\x10\x01\x12\r\n\tkResidual\x10\x02\x12\x0b\n\x07kOutput\x10\x03\x12\x13\n\x0fkDiscreteOutput\x10\x04\x12\x0c\n\x08kPartial\x10\x05B\x11\n\x0forg.philote.mdob\x06proto3')
_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'data_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
    _globals['DESCRIPTOR']._loaded_options = None
    _globals['DESCRIPTOR']._serialized_options = b'\n\x0forg.philote.mdo'
    _globals['_DATATYPE']._serialized_start = 989
    _globals['_DATATYPE']._serialized_end = 1059
    _globals['_VARIABLETYPE']._serialized_start = 1061
    _globals['_VARIABLETYPE']._serialized_end = 1170
    _globals['_DISCIPLINEPROPERTIES']._serialized_start = 54
    _globals['_DISCIPLINEPROPERTIES']._serialized_end = 228
    _globals['_STREAMOPTIONS']._serialized_start = 230
    _globals['_STREAMOPTIONS']._serialized_end = 265
    _globals['_OPTIONSLIST']._serialized_start = 267
    _globals['_OPTIONSLIST']._serialized_end = 330
    _globals['_DISCIPLINEOPTIONS']._serialized_start = 332
    _globals['_DISCIPLINEOPTIONS']._serialized_end = 393
    _globals['_VARIABLEMETADATA']._serialized_start = 395
    _globals['_VARIABLEMETADATA']._serialized_end = 517
    _globals['_PARTIALSMETADATA']._serialized_start = 519
    _globals['_PARTIALSMETADATA']._serialized_end = 583
    _globals['_ARRAY']._serialized_start = 585
    _globals['_ARRAY']._serialized_end = 702
    _globals['_DISCRETEVARIABLE']._serialized_start = 704
    _globals['_DISCRETEVARIABLE']._serialized_end = 812
    _globals['_VARIABLEMESSAGE']._serialized_start = 814
    _globals['_VARIABLEMESSAGE']._serialized_end = 927
    _globals['_VARIABLESET']._serialized_start = 929
    _globals['_VARIABLESET']._serialized_end = 987