"""Consent policy is lightweight; execution limits load on demand."""
from importlib import import_module
from .policy import ContributionPolicy

__all__ = ["ContributionPolicy", "ResourceGovernor", "ResourceLease", "ResourceRequirements"]

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".resource_governor", __name__), name)
    globals()[name] = value
    return value
