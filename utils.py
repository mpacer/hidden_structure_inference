from itertools import chain, combinations

def powerset(iterable):
#    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
#
    len_powerset = 0
    powerset_vals = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return powerset_vals
