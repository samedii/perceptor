from functools import wraps


def cache(model):
    cached = dict()

    @wraps(model)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cached:
            cached[key] = model(*args, **kwargs)
        return cached[key]

    return wrapper
