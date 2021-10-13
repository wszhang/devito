from devito.symbolics import MIN

__all__ = ['evalmin']


def evalmin(a, b):
    """
    Simplify min(a, b) if possible
    """
    try:
        bool(min(a, b))  # Can it be evaluated?
        return min(a, b)
    except TypeError:
        return MIN(a, b)
