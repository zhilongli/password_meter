import numpy as np
import pickle as pkl
import random

def train_pcfg():
    """trains an pcfg model on the training set"""
    with open('data/all_pws.pkl', 'rb') as f:
        training_data = pkl.load(f)
    pattern_count = {} #dictionary of {pattern: counts}
    cond_count = {} #dictionay of {pattern:{emission: counts}}
    cond_prob = {} #output model, same as ngram, {pattern: {emission: prob}}
    pattern_prob = {} #also output model {pattern: prob}
    for pw in training_data:
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
                        if curr_pattern not in cond_count:
                            cond_count[curr_pattern] = {}
                            cond_count[curr_pattern][curr_emission] =1
                        else:
                            if curr_emission not in cond_count[curr_pattern]:
                                cond_count[curr_pattern][curr_emission] = 1
                            else:
                                cond_count[curr_pattern][curr_emission] += 1
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
                        if curr_pattern not in cond_count:
                            cond_count[curr_pattern] = {}
                            cond_count[curr_pattern][curr_emission] =1
                        else:
                            if curr_emission not in cond_count[curr_pattern]:
                                cond_count[curr_pattern][curr_emission] = 1
                            else:
                                cond_count[curr_pattern][curr_emission] += 1
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
                        if curr_pattern not in cond_count:
                            cond_count[curr_pattern] = {}
                            cond_count[curr_pattern][curr_emission] =1
                        else:
                            if curr_emission not in cond_count[curr_pattern]:
                                cond_count[curr_pattern][curr_emission] = 1
                            else:
                                cond_count[curr_pattern][curr_emission] += 1
                    curr_char_count = 1
                    curr_emission = ch
            prev_char = curr_char
            if prev_char and i==len(pw)-1: #reached the last character
                curr_pattern = prev_char + str(curr_char_count)
                overall_pattern = overall_pattern + curr_pattern
                if curr_pattern not in cond_count:
                    cond_count[curr_pattern] = {}
                    cond_count[curr_pattern][curr_emission] =1
                else:
                    if curr_emission not in cond_count[curr_pattern]:
                        cond_count[curr_pattern][curr_emission] = 1
                    else:
                        cond_count[curr_pattern][curr_emission] += 1     
        if overall_pattern not in pattern_count:
            pattern_count[overall_pattern] = 1
        else:
            pattern_count[overall_pattern] +=1

    # after getting counts, get probs
    sum_patterns = sum(pattern_count.values())
    for p, c in pattern_count.items():
        pattern_prob[p] = np.log(c/sum_patterns)
    
    for p, emission_counts in cond_count.items():
        sum_emissions = sum(emission_counts.values())
        cond_prob[p] = {}
        for e, c in emission_counts.items():
            cond_prob[p][e] = np.log(c/sum_emissions)
    with open('models/pcfg_patterns.pkl', 'wb+') as f:
        pkl.dump(pattern_prob, f)
    with open('models/pcfg_emissions.pkl', 'wb+') as f:
        pkl.dump(cond_prob, f)
    print("Done training pcfg")
    
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
                    pw_prob += emission_prob[curr_pattern][curr_emission]
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
                    pw_prob += emission_prob[curr_pattern][curr_emission]
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
                    pw_prob += emission_prob[curr_pattern][curr_emission]
                curr_char_count = 1
                curr_emission = ch
        prev_char = curr_char
        if prev_char and i==len(pw)-1: #reached the last character
            curr_pattern = prev_char + str(curr_char_count)
            overall_pattern = overall_pattern + curr_pattern
            pw_prob += emission_prob[curr_pattern][curr_emission]
    pw_prob += pattern_prob[overall_pattern]
    return pw_prob  

def create_pcfg_sample(sample_size):
    """Returns a sample set of sample_size passwords sampled from training set,
        also return the A and C arrays as in ngram model"""
    # first do a random subsample of 100,000 from the training set
    with open('data/all_pws.pkl', 'rb') as f:
        all_pws = pkl.load(f)
    subsampled_probs = []
    sampled_prob = []
    with open('models/pcfg_patterns.pkl', 'rb') as f:
        pattern_prob = pkl.load(f)
    with open('models/pcfg_emissions.pkl', 'rb') as f:
        emission_prob = pkl.load(f)
    for pw in all_pws:
        this_prob = get_pcfg_prob(pw, pattern_prob, emission_prob)
        subsampled_probs.append(np.exp(this_prob))
    probs_array = subsampled_probs
    # p = probs_array/sum(probs_array)
    sampled_prob = np.random.choice(subsampled_probs, sample_size)
    # for pw in sampled_pw:
    #     ind = subsampled.index(pw)
    #     sampled_prob.append(subsampled_probs[ind])
    # with open('models/pcfg_samples.pkl', 'wb+') as f:
    #     pkl.dump(sampled_pw, f)
    # with open('models/pcfg_sample_probs.pkl', 'wb+') as f:
    #     pkl.dump(sampled_prob, f)
    # print(type(sampled_prob))
    sampled_prob[::-1].sort()
    n = len(sampled_prob)
    C = [1/n*1/sampled_prob[0]]
    for i in range(1, n):
        C.append(C[-1]+1/(n*sampled_prob[i]))
    with open('models/pcfg_A.pkl', 'wb+') as f:
        pkl.dump(sampled_prob, f)
    print(sampled_prob)
    with open('models/pcfg_C.pkl', 'wb+') as f:
        pkl.dump(C, f)


if __name__ == "__main__":
    # train_pcfg()
    create_pcfg_sample(10000)