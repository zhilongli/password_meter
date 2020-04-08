#!/bin/bash
pip3 install numpy
wget https://github.com/zhilongli/password_meter/raw/master/models/trained_ngram_10000000_small.pklz
wget https://github.com/zhilongli/password_meter/raw/master/models/pcfg_small.pklz
python3 src/classify.py $1 $2