from itertools import chain, combinations
import numpy as np


def powerset(iterable):
#    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
#
    len_powerset = 0
    powerset_vals = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return powerset_vals

def scale_free_sampler(lower_bound = 1/10, upper_bound=10,sample_size = 1):
    """
    This allows you to approximate the improper probability
    distribution P(λ) = 1/λ for the positive reals

    This is accomplished by identifying a range of values you are 
    interested in sampling from 

    For example, the default arguments for the sampler generate a
    single value λ from the range 10^{-1} to 10 with the pdf 1/λ.
    """
    if lower_bound >=upper_bound:
        error_statement = "Lower bound must be below upper bound."
        raise ValueError(error_statement)

    # if lower_bound <= 0:
    #     error_statement = "Lower bound must be greater than upper bound."
    #     raise ValueError(error_statement)


    bottom_val = np.log(lower_bound)
    top_val = np.log(upper_bound)
    
    return list(np.exp(np.random.uniform(bottom_val,top_val, sample_size)))

