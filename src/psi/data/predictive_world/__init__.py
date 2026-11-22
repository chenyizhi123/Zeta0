"""Data contracts for the temporal latent predictive-world route."""

__all__ = [
    "FutureSupervision",
    "NativeActionBatch",
    "ObservedHistory",
    "PredictiveWorldBatch",
    "RaggedPatchBatch",
    "SampleMetadata",
]


def __getattr__(name):
    # Asset relocation and CLI diagnostics do not require Torch. Preserve the
    # public contracts while importing their tensor dependencies only on use.
    if name in __all__:
        from psi.data.predictive_world import contracts
        value = getattr(contracts, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
