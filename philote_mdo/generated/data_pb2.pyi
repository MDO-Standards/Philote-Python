from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union
DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    kBool: _ClassVar[DataType]
    kInt: _ClassVar[DataType]
    kDouble: _ClassVar[DataType]
    kString: _ClassVar[DataType]

class VariableType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    kInput: _ClassVar[VariableType]
    kDiscreteInput: _ClassVar[VariableType]
    kResidual: _ClassVar[VariableType]
    kOutput: _ClassVar[VariableType]
    kDiscreteOutput: _ClassVar[VariableType]
    kPartial: _ClassVar[VariableType]
kBool: DataType
kInt: DataType
kDouble: DataType
kString: DataType
kInput: VariableType
kDiscreteInput: VariableType
kResidual: VariableType
kOutput: VariableType
kDiscreteOutput: VariableType
kPartial: VariableType

class DisciplineProperties(_message.Message):
    __slots__ = ('continuous', 'differentiable', 'provides_gradients', 'name', 'version')
    CONTINUOUS_FIELD_NUMBER: _ClassVar[int]
    DIFFERENTIABLE_FIELD_NUMBER: _ClassVar[int]
    PROVIDES_GRADIENTS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    continuous: bool
    differentiable: bool
    provides_gradients: bool
    name: str
    version: str

    def __init__(self, continuous: bool=..., differentiable: bool=..., provides_gradients: bool=..., name: _Optional[str]=..., version: _Optional[str]=...) -> None:
        ...

class StreamOptions(_message.Message):
    __slots__ = ('num_double',)
    NUM_DOUBLE_FIELD_NUMBER: _ClassVar[int]
    num_double: int

    def __init__(self, num_double: _Optional[int]=...) -> None:
        ...

class OptionsList(_message.Message):
    __slots__ = ('options', 'type')
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    options: _containers.RepeatedScalarFieldContainer[str]
    type: _containers.RepeatedScalarFieldContainer[DataType]

    def __init__(self, options: _Optional[_Iterable[str]]=..., type: _Optional[_Iterable[_Union[DataType, str]]]=...) -> None:
        ...

class DisciplineOptions(_message.Message):
    __slots__ = ('options',)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _struct_pb2.Struct

    def __init__(self, options: _Optional[_Union[_struct_pb2.Struct, _Mapping]]=...) -> None:
        ...

class VariableMetaData(_message.Message):
    __slots__ = ('type', 'name', 'shape', 'units')
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    UNITS_FIELD_NUMBER: _ClassVar[int]
    type: VariableType
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    units: str

    def __init__(self, type: _Optional[_Union[VariableType, str]]=..., name: _Optional[str]=..., shape: _Optional[_Iterable[int]]=..., units: _Optional[str]=...) -> None:
        ...

class PartialsMetaData(_message.Message):
    __slots__ = ('name', 'subname', 'shape')
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUBNAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    subname: str
    shape: _containers.RepeatedScalarFieldContainer[int]

    def __init__(self, name: _Optional[str]=..., subname: _Optional[str]=..., shape: _Optional[_Iterable[int]]=...) -> None:
        ...

class Array(_message.Message):
    __slots__ = ('name', 'subname', 'start', 'end', 'type', 'data')
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUBNAME_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    name: str
    subname: str
    start: int
    end: int
    type: VariableType
    data: _containers.RepeatedScalarFieldContainer[float]

    def __init__(self, name: _Optional[str]=..., subname: _Optional[str]=..., start: _Optional[int]=..., end: _Optional[int]=..., type: _Optional[_Union[VariableType, str]]=..., data: _Optional[_Iterable[float]]=...) -> None:
        ...