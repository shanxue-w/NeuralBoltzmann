"""Strict Onsager neural collision operator for Maxwell molecules."""

from .model import OnsagerCollision
from .physics import ConstantMaxwellKernel

__all__ = [
    "OnsagerCollision",
    "ConstantMaxwellKernel",
]
