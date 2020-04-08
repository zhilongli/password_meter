import sys
import pickle as pkl
from testing_ngram import get_pw_rank as get_ngram_rank
from testing_pcfg import get_pw_rank as get_pcfg_rank

if __name__ == "__main__":
    in_pass = sys.argv[1]
    out_pass = sys.argv[2]
    ngram_median = 11624799252.120903
    pcfg_median = 4745709.558933191
    with open('models/4gram_A_1.pkl', 'rb') as f:
        ngram_A = pkl.load(f)
    with open('models/4gram_C_1.pkl', 'rb') as f:
        ngram_C = pkl.load(f)
    with open('models/pcfg_A.pkl', 'rb') as f:
        pcfg_A = pkl.load(f)
    with open('models/pcfg_C.pkl', 'rb') as f:
        pcfg_C = pkl.load(f)
    with open('models/trained_ngram_10000000.pkl', 'rb') as f:
        ngram_model = pkl.load(f)
    with open('models/pcfg_emissions.pkl', 'rb') as f:
        pcfg_emissions = pkl.load(f)
    with open('models/pcfg_patterns.pkl', 'rb') as f:
        pcfg_patterns = pkl.load(f)
    with open(in_pass, 'r') as f:
        all_pws = f.readlines()
    strengths = []
    for pw in all_pws:
        pw = pw.strip('\n')
        print(pw)
        pcfg_rank = get_pcfg_rank(pw, pcfg_A, pcfg_C, pcfg_patterns, pcfg_emissions)
        ngram_rank = get_ngram_rank(pw, ngram_A, ngram_C, ngram_model)
        print('ngram: ', ngram_rank)
        print('pcfg: ', pcfg_rank)
        ngram_strength = 'strong' if ngram_rank>ngram_median else 'weak'
        pcfg_strength = 'strong' if (not pcfg_rank or pcfg_rank>pcfg_median) else 'weak'
        overall = 'strong' if (ngram_strength=='strong' and pcfg_strength=='strong') else 'weak'
        strengths.append(overall)
    with open(out_pass, 'a') as f:
        for s in strengths:
            f.write(s + '\n')
        f.write('\n')
    print("done all classification")