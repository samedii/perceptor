from typing import TypeVar
from functools import wraps
import weakref


T = TypeVar("T")


def cache(model: T) -> T:
    cached = weakref.WeakValueDictionary()

    @wraps(model)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)

        if key in cached:
            return cached[key]
        else:
            model_instance = model(*args, **kwargs)
            cached[key] = model_instance
            return model_instance

    return wrapper
