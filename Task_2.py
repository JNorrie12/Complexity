import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import sys
from data import *
input = sys.argv[1]
################################################################################
#REMEMBER TO CHANGE M N IN DATA
def standard_dev():
    L = []
    Vsd = []
    Vmean = []
    for i in range(len(height)):
        L.append(2 ** (3 + i))
        summ = 0
        summsq = 0
        for j in range(len(height[i])):  ####CHANGE HERE FOR Ls##########
            if j > tc[i]:
                summ += height[i][j]
                summsq += (height[i][j]**2)
        Mean = summ / (len(height[i]) - tc[i])
        Square = summsq/ (len(height[i])-tc[i])
        sd= (Square - (Mean)**2)**.5
        Vsd.append(sd)
        Vmean.append(Mean)
    return Vmean, Vsd

if input is 'a':
    aval, height, tc = Import_data(0.5, 3, 6, 1000000)
    for i in height:
        plt.plot(i[:5000])
    plt.title('Heights')
    plt.xlabel('Time')
    plt.ylabel('Height')

    plt.show()
elif input is 'y':
    aval, height, tc = Import_data(0.5, 3, 6, 1000000)
    L=[]
    for i in range(len(tc)):
        L.append(2**(3+i))
    plt.plot(L, tc)
    coeff =np.polynomial.polynomial.polyfit(L, tc,[2] )
    coeffrev= [coeff[2], 0, 0] ####reverses the list #####
    Quad = np.poly1d(coeffrev)
    Fit = Quad(range(256))
    print Quad
    plt.plot(Fit)
    plt.title('Crossover Times')
    plt.xlabel('System size L')
    plt.ylabel('Time')
    plt.show()

#################################################################
#2b)
elif input is 'b':
    n=3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    for i in range(len(height)):
        W= 2*(2**(n+i)) ##Smoothing the data by W=2*L is nice##########
        Mov= [sum(height[i][j-W:j+W])/(2*W+1) for j in range(W,len(height[i][:75000])-W)] #CHANGE TIME HERE
        plt.plot(Mov)
        plt.title('Moving Average')
        plt.xlabel('System size L')
        plt.ylabel('Moving Average(W=2*L)')
    plt.show()

elif input is 'x':
    n=3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    ax1 = plt.subplot(111)
    ax1.set_xlim([-0.1,2.3])
    ax1.set_ylim([-0.1, 2.3])
    for i in range(m):
        L = (2 ** (n + i))
        W = 2*L
        Tshrink = [float(k)/(L**2) for k in range(W,len(height[i])- W)]#NEED TO CHANGE TIME DEPENDENT ON L
        Mov = [sum(height[i][j - W:j + W]) /(L*(2 * W + 1)) for j in range(W, len(height[i]) - W)]  # CHANGE TIME HERE
        ax1.plot(Tshrink, Mov)
        if i == m-1:
            Tsqrt=1.85*np.sqrt(Tshrink)
            Tconst= 1.73*np.ones(len(Tshrink))
            ax1.plot(Tshrink, Tsqrt, 'r--')
            ax1.plot(Tshrink, Tconst, 'r--')
    plt.title('Average Height Data Collapse')
    plt.xlabel('Scaled Time(t/L**2)')
    plt.ylabel('Scaled Height(h/L)')
    plt.show()

########################################################################################################
# #2c) ~~~~~WORKING~~~~~Making more streamline w/ sd function
if input is 'c':
    n= 3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000) #######Include 2 and 4 to see signs of scaling################
    L= []
    VMean=[]
    for i in range(len(height)):
        L.append(2 ** (n + i))
        summ = 0
        for j in range(len(height[i])): ####CHANGE HERE FOR Ls##########
            if j > tc[i] :
                summ += height[i][j]
            else:
                pass

        Mean= summ/(len(height[i])-(tc[i]))
        VMean.append(Mean)

    plt.title('Mean Height vs. System Size')
    plt.xlabel('System size L')
    plt.ylabel('Mean Height')
    plt.plot(L, VMean)
    plt.show()
    a0=[]
    finder =[]
    findera=[]
    finderb=[]
    for i in range(len(VMean)):
        a= VMean[i]/L[i]
        a0.append(a)

    a00 = a0[-1]+0.017
    a01 = a0[-1]
    a02 = a0[-1]+0.034
    print a00
    #a00=(VMean[-1]-VMean[-2])/(L[-1]-L[-2])
    for i in range(len(VMean)):
        find= 1- a0[i]/(a00)
        finder.append(find)
        finda= 1- a0[i]/(a01)
        findb= 1- a0[i]/(a02)
        findera.append(finda)
        finderb.append(findb)
    #creating points to align
    logf =np.log(np.array(finder))
    logL =np.log(np.array(L))
    plt.plot(logL, logf)
    fit = np.polyfit(logL, logf, 1)
    fitt =(np.poly1d(fit))
    fittt= fitt(range(6))
    # plt.plot(fittt)
    print(fitt)
    logfa =np.log(findera)
    logfb= np.log(finderb)
    plt.plot(logL, logfa)
    plt.plot(logL, logfb)
    # plt.plot(L, a0)  # a0 is the average slope as L -> inf
    plt.xlabel('System Size L')
    plt.ylabel('L(^-1)*a0*L(1-a1*L(^-w) +...)')
    plt.show()
    LL = np.array(L)
    w= fit[0]
    approx =a00*(1 -np.exp(fit[1])*(LL**w))
    #a1= 0.25 w= 0.63
    plt.plot(L, a0)
    plt.plot(L, approx, 'r--')
    plt.show()

elif input is 'z':
    n=1
    m=8
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    L = []
    Vsd = []
    for i in range(len(height)):
        L.append(2 ** (n + i))
        summ = 0
        summsq = 0
        for j in range(len(height[i])):  ####CHANGE HERE FOR Ls##########
            if j > tc[i]:
                summ += height[i][j]
                summsq += (height[i][j]**2)
        Mean = summ / (len(height[i]) - tc[i])
        Square = summsq/ (len(height[i])-tc[i])
        sd= (Square - (Mean)**2)**.5
        Vsd.append(sd)
    plt.title(' Standard Deviation of Height vs. System Size')
    plt.xlabel('System size L')
    plt.ylabel('Standard Deviation of Height')
    plt.plot(L, Vsd)
    plt.show()
    logsd= np.log(Vsd[3:])
    logL= np.log(L[3:]) #DOESNT TAKE INTO ACCOUNT SCALING ERROR ##########
    plt.plot(logL, logsd)
    plt.show()
    fit= np.polyfit(logL, logsd, 1)
    fitt= np.exp(fit[1])*(np.array(L))**fit[0]
    plt.plot(L, fitt, 'r--', zorder=2)
    plt.plot(L, Vsd, zorder=1)
    print fit
    plt.show()
    diff = fitt-Vsd
    plt.plot(diff)
    plt.show()
##################################################################
#2d)
elif input is 'd':
    import matplotlib.mlab as mlab
    aval, height, tc = Import_data(0.5, 3, 6, 1000000)
    for i in range(len(height)):
        L= 2**(3+i)
        Vmean, Vsd = standard_dev()
        standard = (np.array(height[i][tc[i]:]) - Vmean[i])/Vsd[i]
        bins0= range(L,2*L+1)
        bins= (np.array(bins0) - Vmean[i])/Vsd[i]
        ax1 = plt.subplot(131)
        ax1.hist(standard, normed=True, bins=bins, histtype='step', range=[-8, 8], align= 'left')
        x = np.linspace(-3, 3, 100)
        ax1.plot(x, mlab.normpdf(x, 0, 1))
        ax1.set_xlim([-8,8])
    plt.show()