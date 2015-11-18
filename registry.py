class Registry(object):
    """Container class that allows functions to be registered."""
    
    @classmethod
    def register(cls, func):
        if not hasattr(cls, func.__name__):
            setattr(cls, func.__name__, staticmethod(func))
        return func
