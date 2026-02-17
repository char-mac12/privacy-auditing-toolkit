MODEL_REGISTRY = {}
DATASET_REGISTRY = {}
ATTACK_REGISTRY = {}
REPORTER_REGISTRY = {}
METRICS_REGISTRY = {}

def register(registry, name: str):
    def decorator(cls):
        registry[name] = cls
        return cls
    return decorator
