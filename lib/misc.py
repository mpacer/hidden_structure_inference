import numpy as np

def cond_to_data(cond):
    a,b,c,d = cond
    data_sequences = [
        [a,b,c,d], 
        [a,b,c,-np.inf], 
        [a,b,-np.inf,d],
        [a,b,-np.inf,-np.inf],
        [a,-np.inf,-np.inf,-np.inf]
    ]
    return data_sequences

