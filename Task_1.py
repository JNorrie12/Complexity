import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
from data import *
import sys
import matplotlib.lines as mlines

input = sys.argv[1]

if input is 'a':
    J=[0, .5,1]
    for p in J:

        n= 8
        m= 1
########8=100, 16=500, 32=2000, 64=5000 128=20000 256= 75000
        aval, height, tc = Import_data(p, n, m, 75000)
        L = [2 ** x for x in range(n, n + m)]
        for i in range(len(height)):
            x = range(1, len(height[i])+1 )
            blue_line = mlines.Line2D([], [], color='blue', label='Threshold size=1')
            green_line = mlines.Line2D([], [], color='green', label='Threshold size=2')
            orange_line = mlines.Line2D([], [], color='orange', label='Changing threshold with $P=0.5$')
            b= plt.scatter(tc[i], height[i][tc[i]], color='black' ,zorder=2, label= 'Respective Crossover Times')
            plt.annotate('(' + str(tc[i]) + ',' + str(int(height[i][tc[i]])) +')', xy=(tc[i], int(height[i][tc[i]])), xytext=(tc[i]+0.01*np.array(L),  int(height[i][tc[i]])+0.03*np.array(L)))
            plt.legend(handles=[blue_line, orange_line, green_line, b])
            plt.plot(x, height[i], drawstyle='steps-post', zorder=1)
            plt.title('Constant Threshold Plots for System Size' + str(L))
            plt.xlabel('Time')
            plt.ylabel('Height')
    plt.show()

