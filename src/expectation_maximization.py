import numpy as np
# from inside_outside import outside, get_list_of_rules_starting_with #todo name_change? 
# from inside_outside import outside, get_list_of_rules_starting_with #todo name_change? 
import inside_outside 
from tqdm import tqdm

def e_step(args, sents, G):
    """
        Calculates expected counts i.e. the expected frequency of a rule occuring in the training sentences. 
    """
    # keep track of frequency of possible rules for each sentence
    expected_counts_vec = []
    Zs = []
    progress_bar = tqdm(sents)
    for sent in sents:
        if len(sent.split()) <= 2:
            print(f'Skipping sentence "{sent}" as it is too short.')
            continue
        count, Z = outside(sent, args, G)
        expected_counts_vec.append(count) #????? big problem here. 
        # TODO - Please sum up counts on each step? Is it better to sum that every time or store a vector of dictionaries n long? 
        # print(Z)
        Zs.append(Z)
        progress_bar.update()
    progress_bar.close()
    
    return expected_counts_vec, Zs



def m_step(args, expected_counts_vec):
    """Updates thetas, the model parameters, to the value that maximizes the cost function
    must follow eq 20 of tutorial
    ***
    """
    thetas = {}
    sum_occurrences_of_A = {}

    # get a dictionary of the expected frequncy of rules beginning with each nt
    for nt in args.nts:
        un_counts = 0
        bin_counts = 0
        A_rules = get_list_of_rules_starting_with(nt, args.unary_rules)
        if A_rules:
            for rule in A_rules:
                un_counts += np.sum([count[rule] for count in expected_counts_vec]) 
        A_rules = get_list_of_rules_starting_with(nt, args.binary_rules)
        if A_rules:
            for rule in A_rules:
                bin_counts += np.sum([count[rule] for count in expected_counts_vec]) 
        sum_occurrences_of_A[nt] = bin_counts + un_counts
        if sum_occurrences_of_A[nt] == 0.0:
            print("ZERO", nt)

    # Calculate unary rule weights
    for rule in args.unary_rules:
        A,w = rule
        summed_count = np.sum([count[rule] for count in expected_counts_vec]) 

        assert sum_occurrences_of_A[A] > 0.0
        assert summed_count <= sum_occurrences_of_A[A] + 0.0001
        # Note: this assertion will sometimes be thrown when summed count appears to == sum_occures, but is lightly greater in value. adding the .01 threshold stops this being thrown erroneosly. 
        if sum_occurrences_of_A[A] > 0.0:
            # thetas[rule] =  np.exp((summed_count / sum_occurrences_of_A[A]))
            # thetas[rule] =  np.log((summed_count / sum_occurrences_of_A[A]))
            thetas[rule] =  (summed_count / sum_occurrences_of_A[A])
        else:
            thetas[rule] =  0.0

    # Calculate binary rule weights
    for rule in args.binary_rules:

        A,B,C = rule
        summed_count = sum([count[rule] for count in expected_counts_vec]) 
       
        assert sum_occurrences_of_A[A] > 0.0
        assert summed_count <= sum_occurrences_of_A[A] + 0.0001

        if sum_occurrences_of_A[A] > 0.0:
            # thetas[rule] =  np.exp((summed_count / sum_occurrences_of_A[rule[0]]))
            thetas[rule] =  np.log(summed_count / sum_occurrences_of_A[rule[0]])
            thetas[rule] =  summed_count / sum_occurrences_of_A[rule[0]]
        else:
            thetas[rule] =  0.0

    return thetas


def EM(args, G, max_iter=20):
    """
    Carries out the Expectation-Maximization algorithm using the inside-outside theorem to 
    estimate the frequency of a pcfg rule in the E-step and a pcfg specific process to update model parameters
    during the M-step. The algorithm will iterate until updating the model parameters does not significantly 
    increase the likelihood it describes the given sentences, or until the maximum number of iterations is reached.

    parameters:
        args: argument manager
        G: dict containing initial rule weights
        max-iter: the maximum number of iterations of EM

    returns:
        g: the final rule weights
        thetas: the final rule weights in log form
        : The final likelihood of learned weights

    """
    THRESHOLD = .01
    iters = 0
    g = G.copy()
    thetas = {}
    avg_likelihoods = []

    while True:
        iters += 1 
        print(f"Iteration {iters}")
        expected_counts_vec, Zs = e_step(args, args.train_sents, g)
        thetas = m_step(args, expected_counts_vec)

        total_likelihood = get_total_log_likelihood(args, expected_counts_vec, g) 
        # other_likelihood = get_likelihood_from_parse_tree(Zs) 
        # print("other likelihood", other_likelihood) #this makes no sense
        avg_likelihood_per_rule = total_likelihood / len(expected_counts_vec)
        print("Avg log likelihood", avg_likelihood_per_rule) #this makes no sense

        avg_likelihoods.append(avg_likelihood_per_rule)
        if len(avg_likelihoods) > 2:
            print(f"Debugging: Difference in likelihoods is {abs(avg_likelihoods[-1] - avg_likelihoods[-2])} from iteration {iters-1} to {iters}", )

        if len(avg_likelihoods) > 2 and abs(avg_likelihoods[-1] - avg_likelihoods[-2]) < THRESHOLD:
            print(f"No significant improvement on iteration {iters}. Exiting.")
            break
        elif iters > max_iter:
            print("Reached maximum iterations specified. Exiting.")
            break

        # rewrite g based on updated thetas
        for rule in args.unary_rules:
                g[rule] = np.exp(thetas[rule])
        for rule in args.binary_rules:
                g[rule] = np.exp(thetas[rule])
        # g = thetas.copy()


        # TODO - test that min theta is no longer needed for vanishing/exploding weights

    return g, thetas, avg_likelihoods[-1]


# def get_total_log_likelihood( args, expected_counts_vec, g):
#     """
#     Total log likelihood of observed trees.
#     Equation 6 of https://www.borealisai.com/research-blogs/tutorial-19-parsing-iii-pcfgs-and-inside-outside-algorithm/
#     """
#     likelihood = 0 # shape 
#     for count in expected_counts_vec: 
#         for rule in args.unary_rules:
#             likelihood += count[rule] * np.log(g[rule])
#         for rule in args.binary_rules:
#             likelihood += count[rule] * np.log(g[rule])
#     return likelihood 

def get_total_log_likelihood( args, expected_counts_vec, g):
    """
    Total log likelihood of observed trees. (from all rules in system)
    Equation 6 of https://www.borealisai.com/research-blogs/tutorial-19-parsing-iii-pcfgs-and-inside-outside-algorithm/
    """
    likelihood = 0 # shape 
    for count in expected_counts_vec: 
        for A in args.nts:
            for rule in get_list_of_rules_starting_with(A, args.unary_rules):
                likelihood += count[rule] * np.log(g[rule])
            for rule in get_list_of_rules_starting_with(A, args.binary_rules):
                likelihood += count[rule] * np.log(g[rule])




        # for rule in args.binary_rules:
        #     likelihood += count[rule] * np.log(g[rule])
    return likelihood 


def get_likelihood_from_parse_tree(Zs):
    """
    "For each test document, compute its likelihood for each grammar Gi by multiplying the
        probability of the top PCFG parse for each
        sentence." - https://www.cs.utexas.edu/users/ml/papers/raghavan.acl10.pdf

        Ok yeah is Z a probability lol no i don't think Z is what u think it is. 

        parameters: 
        Zs: list of probability of top PCFG parse for each sentence
    """
    Zs = [Z for Z in Zs if Z > 0.0]
    log_likelihoods = np.log(Zs)
    avg_likelihood = np.mean(log_likelihoods)
    return avg_likelihood
