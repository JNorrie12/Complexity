import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import json

input = sys.argv[1]

if input == 'a':
    aval, height, tc = Import_data(0.5, 3, 5)
    for i in range(len(height)):
        L = 2 ** (3 + i)
    ############## Change sample size here #######################################
        freq = np.bincount(np.array([int(aval[i][j]) for j in range(tc[i], len(aval[i]))]))
        prob = freq/float(len(aval[i])-tc[i])

        width = 1.5
        plt.bar(range(len(aval[i])-tc[i]), prob, width)
        plt.show()
# width = 1.5
# plt.plot(prob)
# plt.show()


        b , c = log_bin(aval[i]) ######set a in log_bin########
        plt.loglog(b,c )
        plt.show()

if input == 'b':
    aval, height, tc = Import_data(0.5, 3, 5)
    for i in range(len(height)):
        L = 2 ** (3 + i)
        b, c = log_bin(aval[i])  ######set a in log_bin########
        plt.loglog(b, c)
    plt.show()


