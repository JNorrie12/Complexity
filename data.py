import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import json

n = 3
m = 5
p = 0.5
L = [2 ** x for x in range(n, n + m)]
def Collect_data(p, n, m):

    for i in range(len(L)):

        # make a pile
        pile = oslo(L[i], p)
        # drop_grains(self,trans=500,recur=1000,from_zero=True,draw=False):
        ssave, hsave, tc  = pile.transrec(10000,True)

        file_path_s = 'Database/Avalanche_size' + str(L[i]) + 'Prob' + str(p) + '.json'
        file_path_h = 'Database/Total_height' + str(L[i]) + 'Prob' + str(p) +'.json'
        file_path_g = 'Database/Crossover_Time' +str(L[i]) + 'Prob' + str(p) + '.json'

        with open(file_path_s, 'w') as fp:
            json.dump(ssave, fp)
        with open(file_path_h, 'w') as fp:
            json.dump(hsave, fp)
        with open(file_path_g, 'w') as fp:
            json.dump(tc, fp)

def Import_data(Avalanche, Total, Sum, p) :
   #Import specific data, if x=TRUE
    sv = []
    hv= []
    tcv = []
    for i in range(len(L)) :
        if Avalanche== True:
            file_path_s = 'Database/Avalanche_size' + str(L[i]) + 'Prob' + str(p) + '.json'
            with open(file_path_s) as fp:
                s = [json.load(fp)]
            sv.extend(s)


        if Total == True:
            file_path_h = 'Database/Total_height' + str(L[i]) + 'Prob' + str(p) +'.json'
            with open(file_path_h) as fp:
                h = [json.load(fp)]
            hv.extend(h)


        if Sum == True :
            file_path_g = 'Database/Crossover_Time' + str(L[i]) + 'Prob' + str(p) + '.json'
            with open(file_path_g) as fp:
                tc = [json.load(fp)]
            tcv.extend(tc)

    return sv, hv, tcv

# # ##############PLOTTTER################
# sv, hv, tcv =Import_data(True, True, True)
# for i in range(len(hv)):
#     plt.plot(hv[i])
# plt.show()
# print tcv
# plt.plot(tcv)
# plt.show()
