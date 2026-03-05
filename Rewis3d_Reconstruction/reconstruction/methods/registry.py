# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Reconstruction method registry.

Provides a factory pattern for loading reconstruction methods by name.
New methods can be registered using the @register_method decorator.
"""

from typing import Dict, List, Optional, Type

from .base import BaseReconstructionMethod

# Global registry of reconstruction methods
_METHOD_REGISTRY: Dict[str, Type[BaseReconstructionMethod]] = {}

# Cache for instantiated methods per device
_METHOD_CACHE: Dict[str, BaseReconstructionMethod] = {}


def register_method(name: Optional[str] = None):
    """
    Decorator to register a reconstruction method.

    Args:
        name: Name to register the method under. If None, uses the class's `name` attribute.

    Example:
        @register_method("my_method")
        class MyMethod(BaseReconstructionMethod):
            ...

        # Or use the class's name attribute:
        @register_method()
        class MyMethod(BaseReconstructionMethod):
            name = "my_method"
            ...
    """

    def decorator(cls: Type[BaseReconstructionMethod]):
        method_name = (
            name if name is not None else getattr(cls, "name", cls.__name__.lower())
        )
        _METHOD_REGISTRY[method_name] = cls
        return cls

    return decorator


def get_reconstruction_method(
    method_name: str, device_id: Optional[int] = None, use_cache: bool = True, **kwargs
) -> BaseReconstructionMethod:
    """
    Get or create a reconstruction method instance.

    Args:
        method_name: Name of the reconstruction method (e.g., "map_anything").
        device_id: GPU device ID. If None, uses default CUDA device or CPU.
        use_cache: If True, returns cached instance for same method+device combination.
        **kwargs: Additional arguments passed to the method constructor.

    Returns:
        Instance of the requested reconstruction method.

    Raises:
        ValueError: If the method name is not registered.

    Example:
        method = get_reconstruction_method("map_anything", device_id=0)
        output = method.reconstruct(image_paths, config)
    """
    # Ensure built-in methods are registered
    _ensure_builtin_methods_registered()

    if method_name not in _METHOD_REGISTRY:
        available = list(_METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown reconstruction method: '{method_name}'. "
            f"Available methods: {available}"
        )

    # Create cache key
    device_str = str(device_id) if device_id is not None else "default"
    cache_key = f"{method_name}_{device_str}"

    # Return cached instance if available and caching is enabled
    if use_cache and cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    # Create new instance
    method_class = _METHOD_REGISTRY[method_name]
    instance = method_class(device_id=device_id, **kwargs)

    # Cache the instance
    if use_cache:
        _METHOD_CACHE[cache_key] = instance

    return instance


def list_available_methods() -> List[str]:
    """
    List all registered reconstruction methods.

    Returns:
        List of method names.
    """
    _ensure_builtin_methods_registered()
    return list(_METHOD_REGISTRY.keys())


def clear_cache() -> None:
    """Clear the method cache and free resources."""
    for method in _METHOD_CACHE.values():
        method.cleanup()
    _METHOD_CACHE.clear()


def _ensure_builtin_methods_registered():
    """
    Ensure all built-in methods are registered.

    This is called lazily to avoid import issues.
    """
    if "map_anything" not in _METHOD_REGISTRY:
        from .map_anything import MapAnythingMethod

        register_method("map_anything")(MapAnythingMethod)

    if "depth_anything_v3" not in _METHOD_REGISTRY:
        from .depth_anything_v3 import DepthAnythingV3Method

        register_method("depth_anything_v3")(DepthAnythingV3Method)

    # Add more built-in methods here as they are implemented:
    # if "dust3r" not in _METHOD_REGISTRY:
    #     from .dust3r import DUSt3RMethod
    #     register_method("dust3r")(DUSt3RMethod)
