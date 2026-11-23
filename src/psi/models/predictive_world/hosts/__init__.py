"""Registered layered VLA hosts for Zeta-0."""

from __future__ import annotations

from psi.models.predictive_world.hosts.base import (
    BackboneLoadResult,
    LayeredVLAHostSpec,
    describe_hosts,
    get_host,
    list_hosts,
    register_host,
)
from psi.models.predictive_world.hosts.gr00t_n1d7 import Gr00tN1d7Host
from psi.models.predictive_world.hosts.pi05 import Pi05PlannedHost
from psi.models.predictive_world.hosts.psi0 import Psi0NativeHost

register_host(Psi0NativeHost())
register_host(Gr00tN1d7Host())
register_host(Pi05PlannedHost())

__all__ = [
    "BackboneLoadResult",
    "Gr00tN1d7Host",
    "LayeredVLAHostSpec",
    "Pi05PlannedHost",
    "Psi0NativeHost",
    "describe_hosts",
    "get_host",
    "list_hosts",
    "register_host",
]
