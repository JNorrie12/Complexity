import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
from data import *
import sys

#
# print(ssave)
# b, c = log_bin(ssave)
# plt.loglog(b,c )
# plt.show()
# pile = oslo(8, 0)
# while True:
#     raw_input()
#     pile.drive()
#     pile.relax()
#     pile.draw()

###################################
#***THINGS TO TEST***
#1)Plot heights for p=0, p=1, compare
#2)Plot avalanche size frequency
#3)Log_bin?
#
#
#
#
#
#
input = sys.argv[1]

if input is 'a':
##############NEED TO IMPORT P=0 P=1 ########################
    for p in range(0,2):
        n= 3
        m= 6
        aval, height, tc = Import_data(p, n, m)
        L = [2 ** x for x in range(n, n + m)]
        for i in range(len(height)):
            x = range(1, len(height[i])+1 )

        # plt.scatter(72, 16, color='red', zorder= 3)
            plt.scatter(tc[i],(p+1)*L[i] , color='yellow', zorder=2)
            plt.plot(x, height[i], drawstyle='steps-post', zorder=1)
            plt.title(' Time vs. Height of P(threshold changing to 2)=0')
            plt.xlabel('Time')
            plt.ylabel('Height')
    plt.show()


elif input is 'b':
    p=0.5
    L = [2 ** x for x in range(n, n + m)]
    aval, height, tc =Import_data(True, True, True, p)
    for i in aval:

        width = 1.5
        plt.bar( range(len(i)), i, width)
        plt.show()

