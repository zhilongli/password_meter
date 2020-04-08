import pickle as pkl
import numpy as np
import statistics

def get_pw_prob(pw, model):
    """Runs the 4-gram model to get the probability associated with pw"""
    # with open('models/trained_ngram_10000000.pkl', 'rb') as f:
    #     model = pkl.load(f)
    pw = 3*' '+pw
    start = 0
    end = 3
    prob = None
    while end<=len(pw):
        state = pw[start:end]
        trans = " " if end==len(pw) else pw[end]
        start+=1
        end+=1
        try:
            curr_prob = model[(state,trans)]
        except KeyError: # just give this pair the lowest prob
            curr_prob = min(model.values())
        if not prob:
            prob = curr_prob
        else:
            prob+=curr_prob
    return prob

def get_pw_rank(pw, A, C, model):
    pw_prob = np.exp(get_pw_prob(pw, model))
    # print("PW prob: ", pw_prob)
    # search for rightmost i in A thats > p(pw)
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
    with open('models/trained_ngram_10000000.pkl', 'rb') as f:
        model = pkl.load(f)
    with open('data/10000_test_pws.pkl', 'rb') as f:
        test_set = pkl.load(f)
    all_ranks = []
    for test in test_set:
        test = test.strip('\n')
        all_ranks.append(get_pw_rank(test, A, C, model))
    print("median rank of test set: ", statistics.median(all_ranks))    

def get_strong_weak(A, C):
    with open('models/trained_ngram_10000000.pkl', 'rb') as f:
        model = pkl.load(f)
    with open('data/1000_test_pws.pkl', 'rb') as f:
        test_set = pkl.load(f)
    all_ranks = {}
    for test in test_set:
        test = test.strip('\n')
        all_ranks[test] = get_pw_rank(test, A, C, model)
    sorted_test = sorted(all_ranks.items(), key=lambda kv: kv[1]) # weakest to strongest
    weak = []
    strong = []
    for i in range(1, 21):
        weak.append(sorted_test[10*i][0])
        strong.append(sorted_test[-10*i][0])
    print("strong pass: ", strong)
    print("weak pass: ", weak)


if __name__ == "__main__":
    with open('models/4gram_A_1.pkl', 'rb') as f:
        A = pkl.load(f)
        # print(A)
    with open('models/4gram_C_1.pkl', 'rb') as f:
        C = pkl.load(f)
        # print(C)
    pw = 'koobeanno.2'
    # print(get_pw_rank(pw, A, C))
    rank_test_set(A, C)
    get_strong_weak(A, C)
