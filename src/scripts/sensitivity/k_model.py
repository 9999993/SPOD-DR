"""Set K at original initialization, before any parameter is allocated.

Call install(K, parent_models) before Parent construction, including the B0
factory. The same constructor path must be used in continuation and inference.
Default K=6 consumes exactly the historical initialization RNG stream.
"""
import functools
import inspect

_QUERY_COUNT = 6


def install(query_count, parent_models):
    global _QUERY_COUNT
    query_count = int(query_count)
    if query_count not in (2, 4, 6, 8, 10):
        raise ValueError(f"unregistered local query count {query_count}")
    _QUERY_COUNT = query_count
    classes = {
        parent_models.p4._bootstrap_old_sources()[0],
        parent_models.s2_backend().ClinicalRiskRepresentationLearningModel,
    }
    for cls in classes:
        if cls.__dict__.get("_spod_k_installed", False):
            continue
        original = cls.__init__
        signature = inspect.signature(original)

        def make_init(original, signature):
            @functools.wraps(original)
            def initialize(self, *args, **kwargs):
                bound = signature.bind(self, *args, **kwargs)
                bound.arguments["local_query_count"] = _QUERY_COUNT
                original(*bound.args, **bound.kwargs)
            return initialize

        cls.__init__ = make_init(original, signature)
        cls._spod_k_installed = True


def assert_shape(model, query_count):
    branch = model.inner.base.local_branch
    shape = tuple(branch.queries.shape)
    if shape != (int(query_count), 256):
        raise RuntimeError(f"actual queries.shape {shape}, expected {(int(query_count), 256)}")
    return {
        "K": int(query_count),
        "queries_shape": list(shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "construction": "original initialization with local_query_count; strict state load",
    }
