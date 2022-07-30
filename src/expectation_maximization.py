import numpy as np
from inside_outside import outside, get_list_of_rules_starting_with #todo name_change? 
import tqdm

def e_step(args, G):

    print("calculating expected counts...")
    expected_counts_vec = []
    Zs = []
    progress_bar = tqdm(args.sents)
    for sent in args.sents:
        # print(sent.split())
        if len(sent.split()) <= 2:
            continue
        count, Z = outside(sent, args, G)
        expected_counts_vec.append(count)
        Zs.append(Z)
        progress_bar.update()
    progress_bar.close()
    
    avg_likelihood = np.mean(Zs)
    Zs = [Z for Z in Zs if Z >0.0]
    avg_log_likelihood = np.log(Zs)
    var = np.std(avg_log_likelihood)
    avg_log_likelihood = np.mean(avg_log_likelihood)
    # print("avg_likelihood", avg_likelihood)
    print("avg_log_likelihood?", avg_log_likelihood)
    print("var", var)

     # likelihood = get_likelihood(expected_counts_vec, args.unary_rules, args.binary_rules, G)
    # print("new likelihood", likelihood) # I do think its log likelihood...
    return expected_counts_vec, avg_log_likelihood

# def e_step2(args, G):
# # expected_counts = outside()


#     print("calculating expected counts...")
#     expected_counts_vec = []
#     Zs = []
#     progress_bar = tqdm(args.sents)
#     for sent in args.sents:
#         # print(sent.split())
#         if len(sent.split()) <= 2:
#             continue
#         count, Z = outside(sent, args, G)
#         expected_counts_vec.append(count)
#         Zs.append(Z)
#         progress_bar.update()
#     progress_bar.close()
    
#     avg_likelihood = np.mean(Zs)
#     Zs = [Z for Z in Zs if Z >0.0]
#     avg_log_likelihood = np.log(Zs)
#     var = np.std(avg_log_likelihood)
#     avg_log_likelihood = np.mean(avg_log_likelihood)
#     # print("avg_likelihood", avg_likelihood)
#     # print("avg_log_likelihood?", avg_log_likelihood)
#     # print("var", var)

    # log_likelihood = get_likelihood(expected_counts_vec, args.unary_rules, args.binary_rules, G)
    # print("likelihood", log_likelihood) 
    # return expected_counts_vec, log_likelihood


def m_step(expected_counts_vec, args):
    # print("Example exp.counts", expected_counts_vec)
    print("M Step")

    
    # args.unary_rules = rules[0]
    # args.binary_rules = rules[1]
    thetas = {}
    sum_occurrences_of_A = {}

    # print("building dict summing all occurrences of A")
    # print(nts)
    for nt in args.nts:
        un_counts = 0
        bin_counts = 0
   
        A_rules = get_list_of_rules_starting_with(nt, args.unary_rules)
        # print("U", len(A_rules))
        if A_rules:
            for rule in A_rules:
                un_counts += np.sum([count[rule] for count in expected_counts_vec]) 
            # print(summed_counts)
        
        A_rules = get_list_of_rules_starting_with(nt, args.binary_rules)
        # print("B", len(A_rules))
        if A_rules:
            for rule in A_rules:
                bin_counts += np.sum([count[rule] for count in expected_counts_vec]) 
            
        sum_occurrences_of_A[nt] = bin_counts + un_counts
        if sum_occurrences_of_A[nt] == 0.0:
            print("ZERO", nt)



    # print("Calculating unary theta rules...")
    for rule in args.unary_rules:
        A,w = rule
        summed_count = np.sum([count[rule] for count in expected_counts_vec]) 

        # assert sum_occurrences_of_A[A] > 0.0
        # assert summed_count <= sum_occurrences_of_A[A]
        if sum_occurrences_of_A[A] > 0.0:
            thetas[rule] =  np.exp((summed_count / sum_occurrences_of_A[A]))
            # print("sum occurrences",sum_occurrences_of_A[A] )
            # print("nonzero rule", rule, thetas[rule])
        else:
            thetas[rule] =  0.0
    # except:
        # thetas[rule] =  0.0
            
    # print("Calculating binary theta rules...")
    for rule in args.binary_rules:

        A,B,C = rule
        summed_count = sum([count[rule] for count in expected_counts_vec]) 
       
        # assert summed_count <= sum_occurrences_of_A[A]
        if sum_occurrences_of_A[A] > 0.0:
            thetas[rule] =  np.exp((summed_count / sum_occurrences_of_A[rule[0]]))
            # print("sum occurrences",sum_occurrences_of_A[A] )
            # print("nonzero rule",rule, thetas[rule])
        else:
            thetas[rule] =  0.0

    return thetas


def EM(args, G, max_iter=10):
    THRESHOLD = .01
    THRESHOLD = .1
    MIN_THETA = .0001
    change = 1
    avg_loglikelihoods = []
    iters = 1
    g = G #TODO why can't I keep updating this. 
    thetas = []
    avg_likelihoods = []

    
    # args.unary_rules, args.binary_rules = rules
   
    # rules = [unary_tuples, binary_tuples]

    while True:
        print("ITER", iters)
        # _, loglikelihoods = e_step(rules, sents)
        expected_counts_vec, avg_likelihood = e_step(args, g)

        avg_likelihoods.append(avg_likelihood)
        if len(avg_likelihoods) > 2:
            print("DIFF", abs(avg_likelihoods[-1] - avg_likelihoods[-2]))

        if len(avg_likelihoods) > 2 and abs(avg_likelihoods[-1] - avg_likelihoods[-2]) < THRESHOLD:
            print("NO IMPROVEMENT")
            break
        elif iters >max_iter:
            break
        # E step:
        # u_thetas, b_thetas = m_step(expected_counts_vec, rules)
        thetas = m_step(args, expected_counts_vec)
        # g = thetas

        # TODO - flex g
        for rule in args.args.unary_rules:
            # these if statements are now unnecesary
            if thetas[rule] < MIN_THETA:
                g[rule] = np.exp(thetas[rule])
            else:
                g[rule] = np.exp(thetas[rule])
        for rule in args.args.binary_rules:
            if thetas[rule] < MIN_THETA:
                g[rule] = np.exp(thetas[rule])
            else:
                g[rule] = np.exp(thetas[rule])

        # break
        iters += 1 
        
    print("EXITING after", iters, "iters of e_step")
    return g, thetas, avg_likelihoods[-1]


def get_likelihood(expected_counts_vec, args, g):
    likelihood = 0 # shape 
    for count in expected_counts_vec: 
        for rule in args.args.unary_rules:
            likelihood += count[rule] * np.log(g[rule])
        for rule in args.args.binary_rules:
            likelihood += count[rule] * np.log(g[rule])
    return likelihood 
