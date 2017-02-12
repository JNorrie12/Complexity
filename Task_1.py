import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
from data import *
import sys
import matplotlib.lines as mlines

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
    J=[0, 1]
    for p in J:
        n= 8
        m= 1
        aval, height, tc = Import_data(p, n, m, 100000)
        L = [2 ** x for x in range(n, n + m)]
        for i in range(len(height)):
            x = range(1, len(height[i])+1 )
            a=max(height[i])

            blue_line = mlines.Line2D([], [], color='blue', label='Threshold size=1')
            orange_line = mlines.Line2D([], [], color='orange', label='Threshold size=2')
            b= plt.scatter(tc[i], a, color='black' ,zorder=2, label= 'Respective Crossover Times')
            plt.annotate('(' + str(tc[i]) + ',' + str(int(a)) +')', xy=(tc[i], a), xytext=(tc[i]+0.01*np.array(L), a+0.03*np.array(L)))
            plt.legend(handles=[blue_line, orange_line, b])
            plt.plot(x, height[i], drawstyle='steps-post', zorder=1)
            plt.title('Constant Threshold Plots for System Size' + str(L))
            plt.xlabel('Time')
            plt.ylabel('Height')
    plt.show()


elif input is 'b':
    p=0.5
    aval, height, tc =Import_data(p, 8, 1)
    for i in aval:
        width = 1.5
        plt.bar( range(len(i)), i, width)
        plt.show()

