import numpy as np
import pickle as pkl
import random

def train_n_gram(n):
    """trains an n-gram model on the training set, n is an input argument.
        Returns a dictionary of conditional probabilities of n-grams, 
        in log space"""
    with open('data/all_pws.pkl', 'rb') as f:
        training_data = pkl.load(f)
    all_states = {}
    all_ngrams = {}
    state_ngram_map = {} # {state: set[ngram_1, ngram_2, ...]}
    cond_probs = {} # {(state, transition): log_prob}
    # note an ngram is state+transition
    for token in training_data:
        start = 0
        end = n-1
        token = " "*(n-1)+token  # i'm assuming that no passwords start with space, which also makes sense given my preprocessing
        while end<len(token):
            state = token[start:end]
            # accumulate the counts of ngrams, grouped by ngrams
            if state not in all_states:
                all_states[state] = 1
            else:
                all_states[state] += 1
            if end+1<len(token):
                ngram = token[start:end+1]
                if state not in state_ngram_map:
                    state_ngram_map[state] = set()
                    state_ngram_map[state].add(ngram)
                else:
                    state_ngram_map[state].add(ngram)
            else: # the next character is the end
                ngram = state+' '
                if state not in state_ngram_map:
                    state_ngram_map[state] = set()
                    state_ngram_map[state].add(ngram)
                else:
                    state_ngram_map[state].add(ngram)
            start += 1
            end +=1
        start = 0
        end = n
        while end<len(token)+1:
            if end<len(token):
                ngram = token[start:end]
            else:
                ngram = token[start:end-1]+' '
            start += 1
            end +=1
            if ngram not in all_ngrams:
                all_ngrams[ngram] = 1
            else:
                all_ngrams[ngram] += 1
    
    # now we have all the states, find transitions for each state
    for state, ng in state_ngram_map.items():
        for ngram in ng:
            transition = ngram[-1]
            cond_probs[(state, transition)] = np.log(all_ngrams[ngram]/all_states[state])
    with open('models/trained_ngram_10000000.pkl', 'wb+') as g:
        pkl.dump(cond_probs, g)
    print("done training n gram")

def get_ngram_C_t():
    with open('models/trained_ngram_10000000.pkl', 'rb') as f:
        cond_probs = pkl.load(f)
    with open('data/all_pws.pkl', 'rb') as f:
        training_data = pkl.load(f)
    tsi = {} # in descending order of occurrences, {state: {trans: count}}
    n=4 # consistent with prev training
    for token in training_data:
        token = (n-1)*" "+token
        start = 0
        end = n-1
        while end<len(token):
            state = token[start:end]
            if end+1<len(token):
                trans = token[end]
                if state not in tsi:
                    tsi[state] = {trans:1}
                else:
                    if trans in tsi[state]:
                        tsi[state][trans] +=1
                    else:
                        tsi[state][trans] = 1
            else: # the next character is the end
                if state not in tsi:
                    tsi[state] = {' ':1}
                else:
                    if ' ' not in tsi[state]:
                        tsi[state][' '] = 1
                    else:
                        tsi[state][' '] += 1
            start += 1
            end +=1
    sorted_tsi = {}
    for s, tc in tsi.items():
        sorted_tc = sorted(tc.items(), key=lambda kv: kv[1], reverse=True)
        sorted_tsi[s] = sorted_tc
    C = {}
    for s, tc in sorted_tsi.items():
        C[s] = []
        # print(tc)
        for trans in tc:
            if len(C[s])==0: # first element 
                C[s].append(cond_probs[(s, trans[0])])
            else:
                C[s].append(sum_log_probs(C[s][-1], cond_probs[(s, trans[0])])) # because all in log space
    return sorted_tsi, C

def generate_n_gram_samples(sample_size):
    """Generates a sample set of passwords using the ngram model"""
    with open('models/trained_ngram_10000000.pkl', 'rb') as f:
        model = pkl.load(f)
    pw_samples = []
    A = []
    # again, assuming 4-gram model
    # follow algo 1 in paper
    count = 0
    tsi, C = get_ngram_C_t()
    while count<sample_size:
        s = 3*" "
        g = ''
        while True:
            r = random.random()
            # binary search thru C
            left = 0
            right = len(C[s])
            best_i = None
            while abs(right-left)>1:
                mid = left+(right-left)//2
                curr_p = np.exp(C[s][mid])
                if curr_p<r: #need move right
                    left +=1
                else:
                    best_i = mid
                    right -=1
            if not best_i:
                trans = tsi[s][0][0]
                break
            else:
                trans = tsi[s][best_i][0]
            if trans==" ":
                prob = np.exp(get_pw_prob(g, model))
                pw_samples.append(g)
                print(g)
                A.append(prob)
                count +=1 
                break
            else:
                # print(trans)
                g+=trans
                s+=trans
                s = s[1:]
    return pw_samples, A

def get_pw_prob(pw, model):
    """Runs the 4-gram model to get the probability associated with pw"""
    pw = 3*' '+pw
    start = 0
    end = 3
    prob = None
    while end<=len(pw):
        state = pw[start:end]
        trans = " " if end==len(pw) else pw[end]
        start+=1
        end+=1
        curr_prob = model[(state, trans)]
        if not prob:
            prob = curr_prob
        else:
            prob+=curr_prob
    return prob
    
def sum_log_probs(a, b):
    """ Function for numerically stable sum of logs computation"""
    if a>b:
        return a+np.log1p(np.exp(b-a))
    else:
        return b+np.log1p(np.exp(a-b))

if __name__ == "__main__":
    train_n_gram(4)
    samples, A = generate_n_gram_samples(1000)
    # get the sample probs, A, C array
    A.sort(reverse=True)
    n = len(samples)
    C = [1/n*1/A[0]]
    for i in range(1, n):
        C.append(C[-1]+1/(n*A[i]))
    with open('models/4gram_A_1.pkl', 'wb+') as f:
        pkl.dump(A, f)
    with open('models/4gram_C_1.pkl', 'wb+') as f:
        pkl.dump(C, f)
    with open('models/ngram_samples_1.pkl', 'wb+') as f:
        pkl.dump(samples, f)
