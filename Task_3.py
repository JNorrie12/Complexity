import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import sys
from data import *
import json

input = sys.argv[1]

if input == 'a':
    n=8
    m=1

    aval, height, tc = Import_data(0.5, n, m, 1000000) #CHANGE SYSTEM SIZE HERE####
    for i in range(len(height)):
        L = 2 ** (m + n)
        avalN= aval[i][tc[i]:1000000]
        vals , counts = lin_bin(avalN, range(len(avalN)))
        plt.loglog(vals, counts, '.')
        b , c = log_bin(avalN) ######set a in log_bin########
        plt.loglog(b,c )
        plt.show()


elif input == 'b':
    n=3
    m=6
    aval, height, tc = Import_data(0.5, n, m)
    for i in range(len(height)):
        L = 2 ** (n + i)
        b, c = log_bin(aval[i][tc[i]:])  ######set a in log_bin########
        plt.loglog(b, c)
    plt.show()

elif input == 'c': ###########################Not sure how to do s^-tP#################
    n= 3
    m=6
    aval, height, tc = Import_data(0.5, n, m,1000000)
    fit=[]
    if m+n == 9 :
        L = 2 ** (n + m)
        b, c = log_bin(aval[-1][tc[-1]:])
        logb = np.log(b)
        logc = np.log(c)
        fit = np.polyfit(logb, logc, 1)
    for i in range(len(aval)):
        L= 2**(n+i)
        vals,counts =log_bin(aval[i][tc[i]:])
        scale = (vals**-fit[0])*counts
        plt.loglog(scale)
    plt.show()
    for i in range(len(aval)):
        L = 2 ** (n + i)
        vals, counts = log_bin(aval[i][tc[i]:])
        scale = (vals ** -fit[0]) * counts
        vals2= np.array(vals)/(L**2.174)
        plt.loglog(vals2 , scale)
    plt.show()


elif input =='d' :
    n= 3
    m= 6
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    gradients=[]
    for k in range(1,6):
        Vmoment=[]
        Vlog=[]
        L=[]
        for i in range(len(height)):
            L.append(2**(n+i))
            moment=0
            for j in range(len(aval[i])):
                if j> tc[i]:
                    moment += aval[i][j]**k
            Vmoment.append(float(moment))
            # Vlog.append(np.log(moment))
        fit = np.polyfit(np.log(L), np.log(Vmoment), 1)
        gradients.append(fit[0])
        plt.loglog(L,Vmoment)
        plt.loglog(L,Vmoment, '.', color= 'black')
    plt.xlabel('System size)')
    plt.ylabel('Moment size')
    plt.show()

    fit2 = np.polyfit(range(1,6), gradients, 1)
    fitt2 = np.poly1d(fit2)
    # print(fitt2(range))
    plt.plot(fitt2(range(0,6)))
    plt.plot(range(1,6), gradients, '.')
    print(fit2[0])
    t=(fit2[1]/fit2[0])-1
    print(t)
    plt.xlabel('k')
    plt.ylabel('D(1+k-t)')
    plt.show()
