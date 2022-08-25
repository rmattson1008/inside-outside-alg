from utils import load_text, load_rules_from_pcfg, convert_updated_pcfg_to_string
from expectation_maximization import  EM, e_step, get_total_log_likelihood, get_likelihood_from_parse_tree
from nltk.parse.viterbi import ViterbiParser
from nltk.grammar import PCFG
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')





# todo - input file discrepancy.


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







    # def parse():



path_to_sents = "./test/training.txt"
path_to_pcfg = "./test/pcfg.txt"

args = Args()
args.train_sents = load_text(path_to_sents)
args.unary_rules, args.binary_rules, args.nts, args.nts2idx, G = load_rules_from_pcfg(path_to_pcfg)

# for i in range(100):
#     g, t, final_ll = EM(args, G)
g, t, final_ll = EM(args, G)



# path_to_test_sents = "./test/training.txt"
# def parse(path_to_test_sents):
#     args.train_sents = load_text(path_to_test_sents)
#     for sent in args.train_sents:
#         print()
#         print(sent)
#         expected_counts_vec, Zs = e_step(args, [sent], g)
#         print(len(expected_counts_vec[-1]))
#         # scalar = get_total_log_likelihood(args, expected_counts_vec, g)
#         scalar = get_likelihood_from_parse_tree(Zs)
#         print("Single likelihood output",scalar)
#         print("Z",Zs)

grammar = convert_updated_pcfg_to_string(args, t, path_to_pcfg)
grammar = PCFG.fromstring(grammar)
print(grammar)


viterbi = ViterbiParser(grammar)
s = "astronomers saw stars with ears"
# tokens = word_tokenize(s)
tokens = s.split()
tree = viterbi.parse_all(tokens)
print(tree)

# ok yeah something is broken, probably in the chart. you should not be getting these high values







# get likelihoods of all trees idk. on test sentences
# more control over avg and such. 
#  am I taking averages likelihood over sentences? ? 
# test_sents = load_text(path_to_sents)
# args.unary_rules, args.binary_rules, args.nts, args.nts2idx, G = load_rules_from_pcfg(path_to_pcfg, given_weights=True)


# expected_counts_vec, avg_likelihood = e_step(args, G)
# log_likelihood = get_likelihood(expected_counts_vec, args.unary_rules, args.binary_rules, G)

# return expected_counts_vec, avg_likelihood