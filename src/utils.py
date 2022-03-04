import numpy as np


def load_text(path_to_text):
    
    words = np.loadtxt(path_to_text, delimiter="/n", dtype=str,)
    
    sents = [[sent.split()] for sent in words]
    sents = [[sent] for sent in sents if len(sent) > 2] # just a check
   
    return words


def load_rules_from_pcfg(path_to_pcfg):
    rules = np.loadtxt(path_to_pcfg, delimiter="/n", dtype=str)
    binary_rule_length = len(rules[0].split())
    rules = np.unique(rules)

    binary_rules = []
    unary_rules = []
    nts = []
    
    for rule in rules:
        rule = rule.split()
        
        if rule[0] not in nts:
            nts.append(rule[0])
        if len(rule) == binary_rule_length:
            # print(rule)
            prob = rand.random()
            rule = np.array([rule[0], rule[2], rule[3], prob])
            binary_rules.append(rule)

        elif len(rule) < binary_rule_length:
            prob = rand.random()
            rule = np.array([rule[0], rule[2], prob])
            unary_rules.append(rule)
        else:
            print("error")
            print(rule)
    

    
    binary_rules = np.array(binary_rules) 
    unary_rules = np.array(unary_rules) 
    return unary_rules, binary_rules, nts
    