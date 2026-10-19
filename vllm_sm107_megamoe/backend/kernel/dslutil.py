# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / kernel helpers.
"""Small DSL helpers shared by the kernel modules (DESIGN.md section 4.1).

CuteDSL treats a Python object on which a METHOD is called inside a staged ``if``/``while``/``for`` region as potentially mutated and tries to
carry it through the region as a dynamic value; a plain Python class then fails with ``TYPE_DYNAMIC_EXPR_UNSUPPORTED``.  Objects used that
way (the context object, views, role-local helpers such as a scheduler) must therefore be declared with :func:`static_dataclass`
(frozen dataclass: "read-only") AND inherit :class:`DslStatic` (empty MLIR value list: carries nothing).  Both mechanisms are needed
because the two preprocessor paths of the DSL (native / python reference, selected e.g. by ``CUTE_DSL_KEEP``) treat them differently.

Usage::

    @static_dataclass
    class Foo(DslStatic):
        def __init__(self, a, b):
            init_fields(self, a=a, b=b)
        @cute.jit
        def bar(self): ...
"""

from __future__ import annotations

from dataclasses import dataclass


class DslStatic:
    """Mixin: the object carries NO dynamic MLIR values through staged regions (its SSA-valued fields are captured, never reassigned)."""

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self


def static_dataclass(cls):
    """``@dataclass(frozen=True, eq=False, init=False)``: write ``__init__`` yourself and set fields with :func:`init_fields`."""
    return dataclass(frozen=True, eq=False, init=False)(cls)


def init_fields(self, **kw) -> None:
    """Set attributes of a frozen dataclass instance (``object.__setattr__``)."""
    for k, v in kw.items():
        object.__setattr__(self, k, v)
