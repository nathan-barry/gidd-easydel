try:
    from .gidd_configuration import GiddConfig
    from .modeling_gidd import (
        GiddModel,
        GiddForDiffusionLM,
    )
except ImportError:
    from .configuration_gidd import GiddConfig
    from .modeling_gidd_hf import (
        GiddForDiffusionLM,
    )

__all__ = (
    "GiddConfig",
    "GiddForDiffusionLM",
)
