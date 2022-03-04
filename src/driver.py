import numpy as np 

def train(path_to_sents, path_to_pcfg):
    
    train_sents = load_text(path_to_sents)
    unary_rules, binary_rules, nts = load_rules_from_pcfg(path_to_pcfg)

    print(unary_rules[0])
    print(binary_rules[0])
    print(nts)
    print("len(binary_rules)", len(binary_rules))
    print("len(unary_rules)", len(unary_rules))
    print("len rules", len(unary_rules) +  len(binary_rules))

    nts2idx = {}
    for i, nt in enumerate(nts):
        nts2idx[nt] = i
    
    G = {}
    binary_tuples = []
    for rule in binary_rules:
        A, B, C, g = rule
        key = (A,B,C)
        G[key] = g.astype(np.float)
        binary_tuples.append(key)
    unary_tuples = []
    for rule in unary_rules:
        A, w, g = rule
        key = (A,w)
        G[key] = g.astype(float)
        unary_tuples.append(key)
    rules = [unary_tuples, binary_tuples]
    # print(G)


    
    g, t, final_ll = EM(train_sents, rules, nts, nts2idx, G)
    return g,t, final_ll
  

# def parse_tree():

def test(path_to_sents, path_to_pcfg, path_to_dict):
    test_sents = load_text(path_to_sents)
    unary_rules, binary_rules, nts = load_rules_from_pcfg(path_to_pcfg)
   
    print(unary_rules[0])
    print(binary_rules[0])
    print(nts)
    print("len(binary_rules)", len(binary_rules))
    print("len(unary_rules)", len(unary_rules))

    nts2idx = {}
    for i, nt in enumerate(nts):
        nts2idx[nt] = i

    G_str = np.load(path_to_dict, allow_pickle=True)
    G_str = G_str[()]
    print(type(G_str))
    # print(G.items())
    print("G",len(G_str.items()),"R", len(unary_rules) + len(binary_rules))
    assert len(G_str.items()) == len(unary_rules) + len(binary_rules)
    G = {}
    binary_tuples = []
    for rule in binary_rules:
        A, B, C, g = rule
        key = (A,B,C)
        g = G_str[key]
        G[key] = g.astype(np.float)
        binary_tuples.append(key)
    unary_tuples = []
    for rule in unary_rules:
        A, w, g = rule
        key = (A,w)
        g = G_str[key]
        G[key] = g.astype(float)
        unary_tuples.append(key)
    rules = [unary_tuples, binary_tuples]


    expected_counts_vec, avg_likelihood = e_step2(rules, test_sents, nts, nts2idx, G)


    return expected_counts_vec, avg_likelihood


### run it