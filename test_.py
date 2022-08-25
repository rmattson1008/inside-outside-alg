from src.expectation_maximization import e_step
from src.utils import load_text, load_rules_from_pcfg
import os
import pytest
# how to check up on likelihood


def test_load_text():
    """assert correct number of sentences are loaded"""
    path_to_text = os.path.normpath("test/training.txt")
    sents = load_text(path_to_text)
    assert len(sents) == 64
    return

def test_load_pcfg_with_weights():
    """assert correct rules are loaded"""
    path_to_pcfg = os.path.normpath("test/pcfg_weighted.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=True)
    assert len(unary_rules) + len(binary_rules) == 12
    assert len(unary_rules) == 7
    assert len(binary_rules) == 5
    return

def test_load_pcfg():
    """assert correct rules are loaded"""
    path_to_pcfg = os.path.normpath("test/pcfg.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
    assert len(unary_rules) + len(binary_rules) == 12
    assert len(unary_rules) == 7
    assert len(binary_rules) == 5

    return

def test_pcfg_wrong_input():
    with pytest.raises(AssertionError) as e:
        path_to_pcfg = os.path.normpath("test/pcfg.txt")
        unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=True)
    with pytest.raises(AssertionError) as e:
        path_to_pcfg = os.path.normpath("test/pcfg_weighted.txt")
        unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=False)
    return

def test_load_nts():
    """nts are correct"""
    path_to_pcfg = os.path.normpath("test/pcfg.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
  
    assert len(nts) > 0
    assert len(nts) == len(nts2idx)
    nt = nts[0]
    assert nts2idx[nt] == 0
    return

def test_load_g():
    path_to_pcfg = os.path.normpath("test/pcfg.txt")
    unary_rules, binary_rules, nts, nts2idx, G = load_rules_from_pcfg(path_to_pcfg)
    assert len(unary_rules) + len(binary_rules) == len(G)

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

