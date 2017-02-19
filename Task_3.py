import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import matplotlib.lines as mlines
import sys
from data import *
import json

input = sys.argv[1]

if input == 'a':
    n=8
    m=1
    a=1.5
    s=100000
    aval, height, tc = Import_data(0.5, n, m, 1000000) #CHANGE SYSTEM SIZE HERE####
    for i in range(len(height)):
        L = 2 ** (n + i)
        avalN= aval[i][tc[i]:s+tc[i]] #Always loaded data 1 order above so a could eliminate transient time steps.
        vals , counts = lin_bin(avalN, range(len(avalN)))
        plt.loglog(vals, counts, 'b.')
        b , c = log_bin(avalN, a=a, drop_zeros=False) ######set a in log_bin, a=2.2 for 1e4, a=1.7 for 1e5, a=1.25 for 1e6########
        plt.loglog(b,c, color='orange' )
        orange_line = mlines.Line2D([], [], color='orange', label='Log_Binned data')
        blue_line=mlines.Line2D([],[], color='blue', linestyle=' ', marker='o' ,label='Raw data')
        notes = mlines.Line2D([], [], linestyle=' ', label='Data size='+str(s))
        notes2=mlines.Line2D([],[], linestyle=' ',label='a='+str(a))

        plt.legend(handles=[orange_line, blue_line, notes, notes2])
        plt.title('Loglog Plot of Avalanche Size and Frequency')
        plt.xlabel('Avalanche Size')
        plt.ylabel('Avalanche Frequency)')
        plt.show()


elif input == 'b':
    n=3
    m=6
    a=1.5
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    for i in range(len(height)):
        L = 2 ** (n + i)
        b, c = log_bin(aval[i][tc[i]:], a=a, drop_zeros=False)  ######set a in log_bin########
        plt.loglog(b, c)
    blue_line = mlines.Line2D([], [], color='blue', label='L=8')
    orange_line = mlines.Line2D([], [], color='orange', label='L=16')
    green_line = mlines.Line2D([], [], color='green', label='L=32')
    red_line = mlines.Line2D([], [], color='red', label='L=64')
    purple_line = mlines.Line2D([], [], color='purple', label='L=128')
    brown_line = mlines.Line2D([], [], color='brown', label='L=256')
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, brown_line])
    plt.title('Log_Binned Data for L=16,32,64,128,256')
    plt.xlabel('Avalanche size(s)')
    plt.ylabel('Avalanche Frequency')

    plt.show()

elif input == 'c': ###########################Not sure how to do s^-tP#################
    n= 3
    m=6
    a=1.5
    aval, height, tc = Import_data(0.5, n, m,1000000)
    fit=[]
    for i in range(len(height)):
        L = 2 ** (n + i)
        b, c = log_bin(aval[i][tc[i]:], a=a, drop_zeros=False)  ######set a in log_bin########
        logb= np.log(b)
        logc= np.log(c)
        plt.plot(logb, logc)
        if i==5 :
            fit = np.polyfit(logb, logc, 1)
            x=np.linspace(-1,13,12)
            plt.plot(x, fit[0]*x+fit[1], color='gray', linestyle=':')
    blue_line = mlines.Line2D([], [], color='blue', label='L=8')
    orange_line = mlines.Line2D([], [], color='orange', label='L=16')
    green_line = mlines.Line2D([], [], color='green', label='L=32')
    red_line = mlines.Line2D([], [], color='red', label='L=64')
    purple_line = mlines.Line2D([], [], color='purple', label='L=128')
    brown_line = mlines.Line2D([], [], color='brown', label='L=256')
    grey_line = mlines.Line2D([], [], color='gray', linestyle=':', label='Loglog linear fit')
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, brown_line, grey_line])
    plt.title('Log_Binned Data for L=16,32,64,128,256')
    plt.xlabel('Log(Avalanche size)')
    plt.ylabel('Log(Avalanche Frequency)')
    plt.show()
    print fit[0]
    for i in range(len(aval)):
        L = 2 ** (n + i)
        a = 1.5
        vals ,counts =log_bin(aval[i][tc[i]:], a=a, drop_zeros=False)
        scale = (vals**-fit[0])*counts
        plt.loglog(scale)
    blue_line = mlines.Line2D([], [], color='blue', label='L=16')
    orange_line = mlines.Line2D([], [], color='orange', label='L=32')
    green_line = mlines.Line2D([], [],color='green', label='L=64')
    red_line = mlines.Line2D([], [], color='red', label='L=128')
    purple_line = mlines.Line2D([], [], color='purple', label='L=256')
    grey_line = mlines.Line2D([], [], color='gray', linestyle=':', label='Loglog linear fit')
    tau= mlines.Line2D([], [], linestyle=' ', label='$\\tau_s$ ='+str(round(fit[0],2)))
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, tau])
    plt.title('')
    plt.xlabel('$s$')
    plt.ylabel('$s^{\\tau_s}P(s;L)$')
    plt.show()
    for i in range(len(aval)):
        L = 2 ** (n + i)
        a=1.5
        vals, counts = log_bin(aval[i][tc[i]:], a=a, drop_zeros=False)
        scale = (vals ** -fit[0]) * counts

        vals2= np.array(vals)/(L**(1/(2+fit[0])))
        plt.loglog(vals2 , scale)
    print fit[0]
    #Prediction of Tau_s
    print 1/(2+fit[0])
    #Prediction of D
    blue_line = mlines.Line2D([], [], color='blue', label='L=16')
    orange_line = mlines.Line2D([], [], color='orange', label='L=32')
    green_line = mlines.Line2D([], [], color='green', label='L=64')
    red_line = mlines.Line2D([], [], color='red', label='L=128')
    purple_line = mlines.Line2D([], [], color='purple', label='L=256')
    grey_line = mlines.Line2D([], [], color='gray', linestyle=':', label='Loglog linear fit')
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line])
    plt.title('')
    plt.xlabel('Log(Avalanche size)')
    plt.ylabel('Log(Avalanche Frequency)')
    plt.show()



elif input =='d' :
    n= 1
    m= 8
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    gradients=[]
    s=[]
    for k in range(1,6):
        Vmoment=[]
        L=[]
        for i in range(len(height)):
            L.append(2**(n+i))
            moment=0
            for j in range(len(aval[i])):
                if j> tc[i]:
                    moment += aval[i][j]**k
            Vmoment.append(float(moment))
        fit = np.polyfit(np.log(L[-2:]), np.log(Vmoment[-2:]), 1)
        gradients.append(fit[0])
        x=np.linspace(2,300,300)
        plt.loglog(x, np.exp(fit[1])*x**fit[0])
        plt.loglog(L,Vmoment, '.', color= 'black')
        s.append(Vmoment)
    blue_line = mlines.Line2D([], [], color='blue', label='k=1')
    orange_line = mlines.Line2D([], [], color='orange', label='k=2')
    green_line = mlines.Line2D([], [], color='green', label='k=3')
    red_line = mlines.Line2D([], [], color='red', label='k=4')
    purple_line = mlines.Line2D([], [], color='purple', label='k=5')
    black= mlines.Line2D([], [], color='black', linestyle=' ', marker='o', label='$<s^k>$ for increasing L')
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, black])
    plt.title('Plots of the Moment Sizes for Each System')
    plt.xlabel('System size')
    plt.ylabel('Moment size')
    plt.show()


    fit2 = np.polyfit(range(4,6), gradients[-2:], 1)
    x= np.linspace(0,5,5)
    plt.plot(x, fit2[0]*x+fit2[1], 'r--')
    plt.plot(range(1,6), gradients, color='black', linestyle=' ', marker='o')
    print(fit2[0]) #Prediction of D
    t=(fit2[1]/fit2[0])-1
    print(t) #Prediction of Tau_s
    red = mlines.Line2D([], [], color='red', linestyle='--', label='Fit, $D(1+k+\\tau_s)$')
    black= mlines.Line2D([], [], color='black', linestyle=' ', marker='o', label='Sample $\\alpha$')

    plt.legend(handles=[red, black])
    plt.title('Plot of the power ($\\alpha$) vs. moment k')
    plt.xlabel('$k$')
    plt.ylabel('$\\alpha$')
    plt.show()

    scaling = np.array(gradients)/np.array(range(1,6))
    plt.plot(range(1, 6), scaling, color='black', linestyle=' ', marker='o', zorder=2)
    x = np.linspace(0, 5, 50)
    plt.plot(x, fit2[0]+fit2[1]/x , 'r--', zorder=1)
    black = mlines.Line2D([], [], color='black', linestyle=' ', marker='o', label='Data points')
    plt.legend(handles=[black])
    plt.title('Correction to Scaling')
    plt.xlabel('k')
    plt.ylabel('$\\frac{\\alpha}{k}$')
    plt.show()