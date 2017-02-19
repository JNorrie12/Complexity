import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import json
########DATA COLLECTION AND STORAGE###########
#n= power of 2 we start at
#m= number of system sizes we wish to collect/use
def Collect_data(p, n, m, t):
    L = [2 ** x for x in range(n, n + m)]
    for i in range(len(L)):

        # make a pile
        pile = oslo(L[i], p)
        # drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
        ssave, hsave, tc  = pile.transrec(t)

        file_path_s = 'Database/Avalanche_size' + str(L[i]) + 'Prob' + str(p) + 'drops' + str(t) + '.json'
        file_path_h = 'Database/Total_height' + str(L[i]) + 'Prob' + str(p) + 'drops' + str(t) + '.json'
        file_path_g = 'Database/Crossover_Time' +str(L[i]) + 'Prob' + str(p) + 'drops' + str(t) + '.json'

        with open(file_path_s, 'w') as fp:
            json.dump(ssave, fp)
        with open(file_path_h, 'w') as fp:
            json.dump(hsave, fp)
        with open(file_path_g, 'w') as fp:
            json.dump(tc, fp)

def Import_data(p, n , m, t) :
   #Import specific data, if x=TRUE
    L = [2 ** x for x in range(n, n + m)]
    sv = []
    hv= []
    tcv = []
    for i in range(len(L)) :
            file_path_s = 'Database/Avalanche_size' + str(L[i]) + 'Prob' + str(p) + 'drops' + str(t) + '.json'
            with open(file_path_s) as fp:
                s = [json.load(fp)]
            sv.extend(s)

            file_path_h = 'Database/Total_height' + str(L[i]) + 'Prob' + str(p) + 'drops' + str(t) + '.json'
            with open(file_path_h) as fp:
                h = [json.load(fp)]
            hv.extend(h)

            file_path_g = 'Database/Crossover_Time' + str(L[i]) + 'Prob' + str(p) + 'drops' + str(t) + '.json'
            with open(file_path_g) as fp:
                tc = [json.load(fp)]
            tcv.extend(tc)

    return sv, hv, tcv
