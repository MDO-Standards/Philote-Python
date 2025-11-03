"""
Julia discipline wrapper for Philote.

This module provides Python wrappers for Julia Philote disciplines via juliacall,
allowing pure Julia code to be served via the Philote gRPC protocol.
"""
from .wrapper import JuliaWrapperDiscipline, JuliaImplicitWrapperDiscipline
from .config import PhiloteConfig, DisciplineConfig, ServerConfig
from .explicit import serve_explicit_discipline
from .implicit import serve_implicit_discipline

__all__ = [
    'JuliaWrapperDiscipline',
    'JuliaImplicitWrapperDiscipline',
    'PhiloteConfig',
    'DisciplineConfig',
    'ServerConfig',
    'serve_explicit_discipline',
    'serve_implicit_discipline',
]
