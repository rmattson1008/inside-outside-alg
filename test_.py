# from inside_outside import outside, get_list_of_rules_starting_with
from pcfg.expectation_maximization import EM
from pcfg.utils import load_text, load_rules_from_pcfg
import os
import pytest

from pcfg.utils import load_text, load_rules_from_pcfg, convert_updated_pcfg_to_string, Args
# from expectation_maximization import  EM, e_step, get_total_log_likelihood, get_likelihood_from_parse_tree
from nltk.parse.viterbi import ViterbiParser
from nltk.grammar import PCFG

# how to check up on likelihood


def test_load_text():
    """assert correct number of sentences are loaded"""
    path_to_text = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/training.txt")
    sents = load_text(path_to_text)
    assert len(sents) == 64
    return

def test_load_pcfg_with_weights():
    """assert correct rules are loaded"""
    path_to_pcfg = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/pcfg_weighted.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=True)
    assert len(unary_rules) + len(binary_rules) == 12
    assert len(unary_rules) == 7
    assert len(binary_rules) == 5
    return

def test_load_pcfg():
    """assert correct rules are loaded"""
    path_to_pcfg = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/pcfg.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
    assert len(unary_rules) + len(binary_rules) == 12
    assert len(unary_rules) == 7
    assert len(binary_rules) == 5

    return

def test_pcfg_wrong_input():
    with pytest.raises(AssertionError) as e:
        path_to_pcfg = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/pcfg.txt")
        unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=True)
    with pytest.raises(AssertionError) as e:
        path_to_pcfg = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/pcfg_weighted.txt")
        unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=False)
    return

def test_load_nts():
    """nts are correct"""
    path_to_pcfg = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/pcfg.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
  
    assert len(nts) > 0
    assert len(nts) == len(nts2idx)
    nt = nts[0]
    assert nts2idx[nt] == 0
    return

def test_load_g():
    path_to_pcfg = os.path.normpath("/Users/rachelmattson/dev/inside-outside-alg/test/pcfg.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
    assert len(unary_rules) + len(binary_rules) == len(G)

    return


def test_form_of_trained_pcfg():

    path_to_sents = "/Users/rachelmattson/dev/inside-outside-alg/test/training.txt"
    path_to_pcfg = "/Users/rachelmattson/dev/inside-outside-alg/test/pcfg.txt"
    args = Args()
    args.train_sents = load_text(path_to_sents)
    args.unary_rules, args.binary_rules, args.nts, args.nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
    g, t, final_ll = EM(args, G)
    grammar = convert_updated_pcfg_to_string(args, t, path_to_pcfg)
    grammar = PCFG.fromstring(grammar)
    return


# def test_create_pcfg():
#     """assert correct rules are created when starting from nts."""
# # 
    # return

# def test_e_step_output():
#     args = Args()
#     args.train_sents = load_text(path_to_sents)
#     args.unary_rules, args.binary_rules, args.nts, args.nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
#     # e_step
#     # len(expected counts vec  = len sents)
#     # why tho can't I just sum as I go...

