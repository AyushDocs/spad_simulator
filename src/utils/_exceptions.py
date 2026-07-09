class SpadError(Exception):
    """Base exception for all SPAD simulator errors."""


class ConfigError(SpadError):
    """Invalid configuration (layers, doping, material, etc.)."""


class PhysicsError(SpadError):
    """Unphysical simulation state (divergence, NaN, etc.)."""


class ConvergenceError(SpadError):
    """A solver failed to converge."""
