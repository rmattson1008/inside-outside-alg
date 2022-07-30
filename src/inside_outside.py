from gettext import npgettext
from nptyping import Float
import numpy as np
import random as rand


class Chart():
    """dynamically keep track of inside and outside weight of all subtrees"""

    def __init__(self, nts, nts2idx, unary_rules, binary_rules, words):
        self.n = len(words)
        self.v = len(nts)
        self.nts = nts
        self.words = words
        self.nts2idx = nts2idx
        # self.nts2idx = nts2idx
        self.arr = np.zeros([self.n, self.n + 1, self.v])

    def set(self, i, j, A, value):
        rule_idx = self.nts2idx[A]
        self.arr[i, j, rule_idx] = value
        return

    def get(self, i, j, A) -> float:
        rule_idx = self.nts2idx[A]
        return self.arr[i, j, rule_idx]

    def add(self, i, j, A, value):
        rule_idx = self.nts2idx[A]
        self.arr[i, j, rule_idx] += value
        return

    def print(self, num_slices=1):
        print("shape:", self.arr.shape)
        if num_slices <= 0:
            num_slices = self.v
        # for r in range(num_slices):
            # print(self.nts[r])
        for i in range(self.n):
            for j in range(self.n+1):
                if i == j:
                    # print(self.words[i], end="| ")
                    print('{:.6s}'.format(self.words[i]), end="| ")
                    #  print(self.words[i].center(4), end= "| ")
                else:
                    # print('{:.6f}'.format(self.arr[i, j, r]), end="| ")
                    assert len(self.arr[i, j]) == self.v
                    max_value = np.max(self.arr[i, j])
                    if max_value > 0.0:
                        print('{:.6f}'.format(max_value), end="| ")
                    else:
                        print("000000", end="| ")
            print()


def inside(words, args, g) -> Chart:
    # print("ENTERING INSIDE")
    # print(type(args.unary_rules), type(args.binary_rules))

    chart = []
    # words = sent.split(" ")
    # print(words)
    n = len(words)
    num_unary_rules_used = 0
    chart = Chart(args.nts, args.nts2idx, args.unary_rules, args.binary_rules, words)

    # TODO- where is good?
    # print("pretty" in words)
    unary = [x for x in args.unary_rules if x[1] in words]

    v = max(len(unary), len(args.binary_rules))
    substitute_rule = rand.choice(args.unary_rules)

    for k in range(n):
        # print("word", words[k])
        relevant_rules = [x for x in unary if words[k] in x]
        # print(rules)
        # print("Chart span", k, k+1)
        if not relevant_rules:
            print("no unary rule found for ", words[k])
            prob = g[substitute_rule]
            chart.set(k, k+1, substitute_rule[0], value=prob)
        for rule in relevant_rules:
            prob = g[rule]
            chart.set(k, k+1, rule[0], value=prob)

    examined = []
    for sub_string_length in range(2, n+1):
        # print("substring length", sub_string_length)
        # print("L", l)
        for start_idx in range(n-sub_string_length+1):
            # start_idx +=1
            # end_idx = start_idx + sub_string_length+1
            end_idx = start_idx + sub_string_length
            # print("Chart span", start_idx, end_idx)
            for mid_idx in range(start_idx + 1, end_idx):  # IDX???????
                # print("subtrees:", start_idx, mid_idx, end_idx)
                examined.append((start_idx, end_idx))
                for rule in args.binary_rules:
                    A, B, C = rule
                    # print(rule)
                    # g = rule[-1].astype(np.float)
                    new_prob = g[rule] * chart.get(start_idx,
                                                   mid_idx, B) * chart.get(mid_idx, end_idx, C)
                    # new_prob = chart.get(start_idx, mid_idx, B) * chart.get(mid_idx, end_idx, C)
                    chart.add(start_idx, end_idx, A, value=new_prob)

    return chart


def outside(sent, args, g) -> "tuple[dict, float]":
    # print("ENTERING OUTSIDE")
    words = sent.split(" ")
    # # words = sent
    # if len(words) > 5:
    #     words = words[:5]
    if len(words) <= 1:
        print("sent TOO SHORT:", sent, ":")
    # print(words)
        # return g
    # words = sent
    n = len(words)
    v = len(args.nts)

    out_rules = {}
    for rule in args.binary_rules:
        # print("Rule_name", rule[:-1])
        out_rules[rule] = 0.0
    for rule in args.unary_rules:
        out_rules[rule] = 0.0
    counts = out_rules.copy()

    # TODO
    inside_weights = inside(words, args, g)
    # print("LEAVING INSIDE")
    Z = inside_weights.get(0, n, "S")

    # return counts, 1

    out_weights = Chart(args.nts, args.nts2idx, args.unary_rules, args.binary_rules, words)

    out_weights.add(0, n, "S", value=1)
    examined = []
    for sub_string_length in reversed(range(2, n+1)):  # ???
        for start_idx in range(n-sub_string_length + 1):
           # for start_idx in range(n-sub_string_length):
            # start_idx +=1
            end_idx = start_idx + sub_string_length
            # print("Chart span", start_idx, end_idx)

            for mid_idx in range(start_idx + 1, end_idx):  # IDX???????
                # print("subtrees:", start_idx, mid_idx, end_idx)
                examined.append((start_idx, mid_idx, end_idx))
                for rule in args.binary_rules:
                    # I think the first must be S
                    A, B, C = rule
                    # print("current prob", g)
                    # g = g.astype(np.float)
                    out_rules[rule] += out_weights.get(start_idx, end_idx, A) * inside_weights.get(
                        start_idx, mid_idx, B) * inside_weights.get(mid_idx, end_idx, C)
                    probB = out_weights.get(
                        start_idx, end_idx, A) * g[rule] * inside_weights.get(mid_idx, end_idx, C)
                    probC = out_weights.get(
                        start_idx, end_idx, A) * g[rule] * inside_weights.get(start_idx, mid_idx, B)
                    out_weights.add(start_idx, mid_idx, B, probB)
                    out_weights.add(mid_idx, end_idx, C, probC)

    for k in range(n-1):
        relevant_rules = [x for x in args.unary_rules if words[k] in x]
        # print("Chart span", k, k+1)
        for rule in relevant_rules:
            A, w = rule
            # print(rule)
            out_rules[rule] += out_weights.get(k, k+1, A)

    try:
        assert Z > 0.0
    except AssertionError:
        # what the hell is this?? 
        # I think if z is 0 I just take g instead of the calculated count
        # TODO - percolate this better
        for rule in args.unary_rules:
            counts[rule] = g[rule]
            # counts[rule] = 0
        for rule in args.binary_rules:
            counts[rule] = g[rule]
            # counts[rule] = 0
        # print("_____")
        print("NO TREE FOUND - Z:0")
        print("INSIDE")
        print(inside_weights.print())
        print("OUTSIDE")
        print(out_weights.print())
        # COUNT_Z_IS_ZERO +=1
        print(words)
        return counts, Z

    for rule in args.unary_rules:
        counts[rule] = (1/Z) * out_rules[rule] * g[rule]
    for rule in args.binary_rules:
        counts[rule] = (1/Z) * out_rules[rule] * g[rule]

    return counts, Z


def get_list_of_rules_starting_with(A: str, rules: list) -> list:
    '''
    Given a dictionary of keys and a token A, find all the rules in the dictionary that begin with token A. 
    Ex. given token A, get A -> BC,  A -> CB, A -> D, etc

    Parameters:
        A (str): a given token
        rules (list of tuples): list of rules

    Returns:
        filtered_rules (list): List of rule tuples starting with given token 

    '''
    filtered_rules = [rule for rule in rules if rule[0] == A]
    return filtered_rules
