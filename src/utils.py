import numpy as np
import random as rand
from nptyping import NDArray, Bool
import os


def load_text(path_to_text: str, delimiter : str = "/n") -> NDArray:
    '''
    Convert a text document of sentences into a NDArray of sentences.

    Parameters:
        path_to_text: path to .txt file containing 1 or more sentences.
        delimiter: token that denotes a new sentence - default is '/n'

    Returns:
        sents: ndarray of sentences
    '''
    path_to_text = os.path.normpath(path_to_text)
    sents = np.loadtxt(path_to_text, delimiter=delimiter, dtype=str,)
    
    # sents = [[sent.split()] for sent in words]
    # sents = [[sent] for sent in sents if len(sent) > 2] # just a check

    return sents

# def load_grammar(path_to_pcfg : str):
#     path_to_pcfg = os.path.normpath(path_to_pcfg)
#     sents = np.loadtxt(path_to_text, delimiter="/n", dtype=str,)
#     grammar = PCFG.from_string(sents)  
#     return grammar

class Args():
    '''
    Simple class to keep track of static variables. should upgrade to command line arg...
    '''
    def __init__(self):
        # self.path_to_sents
        # self.path_to_pcfg
        self.unary_rules = None
        self.binary_rules = None
        self.nts = None
        self.nts2idx = None
        self.train_sents = None
        # self.test_sents = None
        self.unary_rules = None
        self.binary_rules = None
        return

def convert_updated_pcfg_to_string(args, weights, path_to_pcfg, given_weights: bool =False) -> str:
    path_to_pcfg = os.path.normpath(path_to_pcfg)
    rules = np.loadtxt(path_to_pcfg, delimiter="/n", dtype=str,)
    grammar = ""
    for rule in args.unary_rules:
        rule = str(rule[0]) + " -> " + str(rule[1]) + ' [' + str(weights[rule]) + ']'
        grammar = grammar + '\n' + rule

    for rule in args.binary_rules:
        rule = str(rule[0]) + " -> " + str(rule[1]) + ' ' + str(rule[1]) + ' [' + str(weights[rule]) + ']'
        grammar = grammar + '\n' + rule

    print("Grammar:")
    print(grammar)

    return grammar




def load_rules_from_pcfg(path_to_pcfg : str, given_weights: bool =False, delimiter: str = "/n", path_to_prob_dict=[]):
    '''
    Convert a text document of PCFG rules into a list of binary and unary rules, and get set of all nonterminal tokens to be used.
    Also initialize probabilities of PCFG rules

    Parameters:
        path_to_pcfg (str): path to .txt file containing 1 or more pcfg rules
        given_weights (bool): True if text file contains wieght for each rule
        delimiter (str): token that denotes a new rule- default is "/n"

    Returns:
        binary_rules: list of tuple pairs for binary rules, ie (Parent, child, child)
        unary_rules: list of tuple pairs for unary rules, ie (Parent, child) where the child is a terminal word
        nts: set of all nonterminal tokens, ie unary rule labels and binary rule labels 
        nts2idx: dict util to convert nts to idx value.
        g: dict where the keys are the unary/binary rule, and the value is g, the probablity* of that rule. For now, g is initialized randomly

    '''
    # TODO - initialize g based of user input

    rules = np.loadtxt(path_to_pcfg, delimiter="/n", dtype=str)

    # check that every given pcfg list contains at least 1 unary rule and 1 binary rule
    # will also throw assertion error if given weights is true when should be false, or vice versa
    if given_weights:
        assert any(len(rule.split()) == 5 for rule in rules)
        assert any(len(rule.split()) == 4 for rule in rules)
        BINARY_RULE_LENGTH = 5
    else:
        assert any(len(rule.split()) == 4 for rule in rules)
        assert any(len(rule.split()) == 3 for rule in rules)
        BINARY_RULE_LENGTH = 4

    if len(rules) > len(np.unique(rules)):
        rules = np.unique(rules)
        print(f"There are repeated rules in the input file. Removing .{len(rules) - len(np.unique(rules))} redundant rules and continuing.")

    binary_rules = []
    unary_rules = []
    nts = set()

    G = {}

    for rule in rules:
        rule = rule.split()
        assert len(rule) >= 3 and len(rule) <= 5
        if given_weights:
                prob = rule[-1]
        else:
            prob = rand.random()

        nts.add(rule[0])

        if len(rule) == BINARY_RULE_LENGTH:
            key = (rule[0], rule[2], rule[3])
            G[key] = prob
            binary_rules.append(key)

        elif len(rule) < BINARY_RULE_LENGTH:
            key = (rule[0], rule[2])
            G[key] = prob
            unary_rules.append(key)
        else:
            print("error")
            print(rule)

    nts = list(nts)
    nts2idx = {}
    for i, nt in enumerate(nts):
        nts2idx[nt] = i
    

    return unary_rules, binary_rules, nts, nts2idx, G
    