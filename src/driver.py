import numpy as np 
from expectation_maximization import load_text, load_rules_from_pcfg, EM, get_likelihood, e_step



# todo - input file discrepancy.


class Args():
    '''
    Simple class to keep track of static variables. should upgrade to command line arg...
    '''
    def __init__(self):
        # self.path_to_sents
        # self.path_to_pcfg
        self.unary_rules 
        self.binary_rules 
        self.nts 
        self.nts2idx 
        self.train_sents
        self.test_sents
        self.unary_rules
        self.binary_rules
        return



def train(path_to_sents, path_to_pcfg):
    args = Args()
    # args.path_to_sents = path_to_sents
    # args.path_to_pcfg = path_to_pcfg
    
    args.train_sents = load_text(path_to_sents)
    args.unary_rules, args.binary_rules, args.nts, args.nts2idx, G = load_rules_from_pcfg(path_to_pcfg)


    print(args.unary_rules[0])
    print(args.binary_rules[0])
    print(args.nts)
    print("len(binary_rules)", len(args.binary_rules))
    print("len(unary_rules)", len(args.unary_rules))
    print("len rules", len(args.unary_rules) +  len(args.binary_rules))
    print(type(G))

    
    g, t, final_ll = EM(args, G)

    # TODO save G in the text file. what are benefits of having it as a dict? idk

    return g, t, final_ll

  

def test(path_to_sents, path_to_pcfg):
    """
    This is too specific to my task... I think this is a fit parse type of deal"""
    args = Args()

    test_sents = load_text(path_to_sents)
    args.unary_rules, args.binary_rules, args.nts, args.nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=True)


    expected_counts_vec, avg_likelihood = e_step(args, G)
    log_likelihood = get_likelihood(expected_counts_vec, args.unary_rules, args.binary_rules, G)

    return expected_counts_vec, avg_likelihood


    # def parse():
