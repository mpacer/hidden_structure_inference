lag_slo_results = {("A_★","B_★"):[1.00, .96, .58,.92], ("A_★","C_★"):[.33, .13, .54, .33], ("A_★","D_★"):[.38, .17, .88, .29],
           ("B_★","C_★"):[.75, .79, .21, .79], ("B_★","D_★"):[.75, .96, .38, .88],
           ("C_★","B_★"):[.63, .38, .79, .50], ("C_★","D_★"):[.29, .21, .33, .29],
           ("D_★","B_★"):[.50, .46, .50, .46], ("D_★","C_★"):[.25, .83 ,.71 ,.21]}


edges_of_interest = {edge:0 for edge in lag_slo_results}

def extract_edge_weights_from_post(edge_query,graphs,posterior):
#     import ipdb; ipdb.set_trace()
    for i,g in enumerate(graphs):
        for edge in edge_query:
            if edge in g.edges():
                edge_query[edge]+=posterior[i]
    assert all([edge-1<=1e-14 for edge in edge_query.values()])
    pass
