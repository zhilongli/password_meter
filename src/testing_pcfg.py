import pickle as pkl
import numpy as np
import statistics
import math

def get_pcfg_prob(pw, pattern_prob, emission_prob):
    """ returns the log probability """
    pw_prob = 0
    prev_char = None # keep track of prev character, D, S, L
    overall_pattern = ''
    curr_char_count = 0
    curr_emission = ''
    for i, ch in enumerate(pw):
        if ch.isalpha():
            curr_char = 'L'
            if curr_char==prev_char:
                curr_char_count += 1
                curr_emission += ch
            else:
                if prev_char:
                    curr_pattern = prev_char + str(curr_char_count)
                    overall_pattern = overall_pattern + curr_pattern
                    try:
                        pattern_dict = emission_prob[curr_pattern]
                        try:
                            e_prob = pattern_dict[curr_emission]
                            pw_prob += e_prob
                        except KeyError:
                            pw_prob += min(pattern_dict.values())
                    except KeyError: # a very rare pattern, give a very low prob 
                        return math.inf
                curr_char_count = 1
                # reset the current emission
                curr_emission = ch
        elif ch.isnumeric():
            curr_char = 'D'
            if curr_char==prev_char:
                curr_char_count += 1
                curr_emission += ch
            else:
                if prev_char:
                    curr_pattern = prev_char + str(curr_char_count)
                    overall_pattern = overall_pattern + curr_pattern
                    try:
                        pattern_dict = emission_prob[curr_pattern]
                        try:
                            e_prob = pattern_dict[curr_emission]
                            pw_prob += e_prob
                        except KeyError:
                            pw_prob += min(pattern_dict.values())
                    except KeyError: # a very rare pattern, give a very low prob 
                        return -math.inf
                curr_char_count = 1
                curr_emission = ch
        else:
            curr_char = 'S'
            if curr_char==prev_char:
                curr_char_count += 1
                curr_emission += ch
            else:
                if prev_char:
                    curr_pattern = prev_char + str(curr_char_count)
                    overall_pattern = overall_pattern + curr_pattern
                    try:
                        pattern_dict = emission_prob[curr_pattern]
                        try:
                            e_prob = pattern_dict[curr_emission]
                            pw_prob += e_prob
                        except KeyError:
                            pw_prob += min(pattern_dict.values())
                    except KeyError: # a very rare pattern, give a very low prob 
                        return math.inf
                curr_char_count = 1
                curr_emission = ch
        prev_char = curr_char
        if prev_char and i==len(pw)-1: #reached the last character
            curr_pattern = prev_char + str(curr_char_count)
            overall_pattern = overall_pattern + curr_pattern
            try:
                pattern_dict = emission_prob[curr_pattern]
                try:
                    e_prob = pattern_dict[curr_emission]
                    pw_prob += e_prob
                except KeyError:
                    pw_prob += min(pattern_dict.values())
            except KeyError: # a very rare pattern, give a very low prob 
                return math.inf
    try:
        pw_prob += pattern_prob[overall_pattern]
    except KeyError:
        pw_prob += min(pattern_prob.values())
    return pw_prob  

def get_pw_rank(pw, A, C, pattern_prob, emission_prob):
    pw_prob = np.exp(get_pcfg_prob(pw, pattern_prob, emission_prob))
    # print('PW: ', pw)
    # print("PW prob: ", pw_prob)
    # search for rightmost i in A thats > p(pw)
    if pw_prob==math.inf:
        return None
    left= 0
    right = len(A)
    best_i = None
    while abs(right-left)>1:
        mid = left+(right-left)//2
        curr_p = A[mid]
        if curr_p<pw_prob: #need move left
            right -=1
        else:
            best_i = mid
            left +=1
    if not best_i:
        if pw_prob<A[-1]:
            best_i = len(A)-1
        else:
            best_i = 0
    rank = C[best_i]
    return rank

def rank_test_set(A, C):
    with open('models/pcfg_patterns.pkl', 'rb') as f:
        pattern_prob = pkl.load(f)
    with open('models/pcfg_emissions.pkl', 'rb') as f:
        emission_prob = pkl.load(f)
    with open('data/10000_test_pws.pkl', 'rb') as f:
        test_set = pkl.load(f)
    all_ranks = []
    for test in test_set:
        test = test.strip('\n')
        rank = get_pw_rank(test, A, C, pattern_prob, emission_prob)
        if rank:
            all_ranks.append(rank)
    # all_ranks.sort()
    # print(all_ranks)
    print("median rank of test set: ", statistics.median(all_ranks))    

def get_strong_weak(A, C):
    with open('models/pcfg_patterns.pkl', 'rb') as f:
        pattern_prob = pkl.load(f)
    with open('models/pcfg_emissions.pkl', 'rb') as f:
        emission_prob = pkl.load(f)
    with open('data/10000_test_pws.pkl', 'rb') as f:
        test_set = pkl.load(f)
    all_ranks = {}
    for test in test_set:
        test = test.strip('\n')
        rank = get_pw_rank(test, A, C, pattern_prob, emission_prob)
        if rank:
            all_ranks[test] = rank
    sorted_test = sorted(all_ranks.items(), key=lambda kv: kv[1]) # weakest to strongest
    # print(sorted_test[:100])
    # print(sorted_test)
    weak = []
    strong = []
    for i in range(1, 21):
        weak.append(sorted_test[10*i][0])
        strong.append(sorted_test[-10*i][0])
    print("strong pass: ", strong)
    print("weak pass: ", weak)


if __name__ == "__main__":
    with open('models/pcfg_A.pkl', 'rb') as f:
        A = pkl.load(f)
        # print(A)
    with open('models/pcfg_C.pkl', 'rb') as f:
        C = pkl.load(f)
        # print(C)
    pw = 'koobeanno.2'
    # print(get_pw_rank(pw, A, C))
    rank_test_set(A, C)
    get_strong_weak(A, C)
