from functools import wraps
import weakref


def cache(model):
    cached = weakref.WeakValueDictionary()

    @wraps(model)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)

        if key in cached:
            print("cached")
            return cached[key]
        else:
            model_instance = model(*args, **kwargs)
            cached[key] = model_instance
            return model_instance

    return wrapper
