import numpy as np


generator_dictionary ={ "nodes" : ["A_int","A_obs","A_★","B_obs","B_★","C_obs","C_★","D_obs","D_★"],
    "query_edge_set" : [
        ("A_★",'B_★'),
        ("A_★",'C_★'),
        ("A_★",'D_★'),
        ("B_★",'C_★'),
        ("B_★",'D_★'),
        ("C_★",'B_★'),
        ("C_★",'D_★'),
        ("D_★",'B_★'),
        ("D_★",'C_★')
    ],
    "filters": {
        "explicit_child_parentage"  : [[
            ("A_int",[]),
            ("A_★",["A_int"]),
            ('A_obs',['A_int','A_★']),
            ('B_obs',['B_★']),
            ('C_obs',["C_★"]),
            ('D_obs',["D_★"])
        ]],
        "explicit_parent_offspring" : [[
            ('A_int',['A_obs','A_★']),
            ("A_obs",[]),
            ("B_obs",[]),
            ("C_obs",[]),
            ("D_obs",[])
        ]],
        "extract_remove_self_loops": []
    },
    "conditions": {
        "create_path_complete_condition" : [[("A_int","B_★"),("A_int","C_★"),("A_int","D_★")]],
    }
}



node_semantics={
        # ".*_int" → intervener
        "intervener": {
            "node_type":"intervener",
            "where":"suffix",
            "infix":"_",
            "code":"int"},
        # ".*_obs" → observed
        "observed": {
            "node_type":"observed",
            "where":"suffix",
            "infix":"_",
            "code":"obs"},
        # ".*_★" → hidden
        "hidden": {
            "node_type":"hidden",
            "where":"suffix",
            "infix":"_",
            "code":"★"}
}

edge_semantics={
    "hidden_sample":{
        "source_types":["hidden"],
        "target_types":["hidden"],
        "edge_type": "hidden_sample"
    },
    "observed":{
        "source_types":["hidden"],
        "target_types":["observed"],
        "edge_type": "observed"
    },
    "intervention":{
        "source_types":["intervener"],
        "target_types":None,
        "edge_type": "intervention"
    }
}

param_sample_size = 3000
stigma_sample_size = 10000

# scale_free_bounds = (10**(-.1),10**(.1))

# scale_free_bounds = (10**(-.25),10**(.25))
# scale_free_bounds = (10**-1,10**1)
scale_free_bounds = (10**-3,10**3)
small_val = .05

cond1 = [0,small_val,small_val,small_val]
cond2 = [0,1,3,2]
cond3 = [0,3,2,1]
cond4 = [0,1,2,2]

conds = np.array([cond1,cond2,cond3,cond4])

# from .misc import cond_to_data
# data_sets = cond_to_data(cond2)

data_sets = None
sparsity = .5

options = {
    'param_sample_size': param_sample_size,
    'stigma_sample_size': stigma_sample_size,
    'scale_free_bounds': scale_free_bounds,
    'sparsity': sparsity,
    'data_sets': data_sets,
    'num_data_samps': 100,
    'max_obs_time': 4,
    'data_probs':[0.512,0.128,0.128,0.032,.2]
}
