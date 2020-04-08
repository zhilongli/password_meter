import numpy as np
import pickle as pkl
import random

def get_all_pw():
    with open('data/10-million-combos.txt', 'r') as f:
        data = f.readlines()
    pws = []
    for line in data:
        tokens = ''.join(line.split()[1:])
        pws.append(tokens)
    with open('data/all_pws.pkl', 'wb+') as g:
        pkl.dump(pws, g)
    
def subsample_pws(sample_size):
    """Random sampling of sample_size passwords and store as a list in pickle"""
    with open('data/all_pws.pkl', 'rb') as f:
        all_pws = pkl.load(f)
    sample = random.sample(all_pws, sample_size)
    with open('data/{}_subsample_pw.pkl'.format(str(sample_size)), 'wb+') as g:
        pkl.dump(sample, g)

def sample_testing(sample_size):
    all_pw = []
    with open('data/rockyou.txt', 'r') as f:
        while True:
            try:
                pw = f.readline()
                if not pw:
                    break
                all_pw.append(pw)
            except:
                continue   

    sample = random.sample(all_pw, sample_size)
    with open('data/{}_test_pws.pkl'.format(str(sample_size)), 'wb+') as g:
        pkl.dump(sample, g)

if __name__=='__main__':
    # get_all_pw()
    # subsample_pws(10000)
    # subsample_pws(100000)
    # subsample_pws(1000000)
    sample_testing(1000)