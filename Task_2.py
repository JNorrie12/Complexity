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
    plt.xlabel('Time(Number of Grains Dropped)')
    plt.ylabel('Height(Grains)')

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
     #CHANGE W FOR SMOOTHNESS
    aval, height, tc = Import_data(0.5, 3, 6, 1000000)
    for i in range(len(height[:7])):
        #CHANGE WHICH TO PROCESS HERE
        W= (2**(3+i))
        Mov= [sum(height[i][j-W:j+W])/(2*W+1) for j in range(W,len(height[i][:75000])-W)] #CHANGE TIME HERE
        plt.plot(Mov)
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
        # Tsqrt = [np.sqrt(k) for k in range(W, len(height[i]) - W)]
        # plt.plot(Tshrink, Tsqrt )
        if i == m-1:
            Tsqrt=1.85*np.sqrt(Tshrink)
            ax1.plot(Tshrink, Tsqrt, 'r--')
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
    for i in range(len(VMean)):
        a= VMean[i]/L[i]
        a0.append(a)

    a00 = a0[-1]+0.017
    #a00=(VMean[-1]-VMean[-2])/(L[-1]-L[-2])
    for i in range(len(VMean)):
        find= 1- a0[i]/(a00)
        finder.append(find)
    #creating points to align
    logf =np.log(np.array(finder))
    logL =np.log(np.array(L))
    plt.plot(logL, logf)
    fit = np.polyfit(logL, logf, 1)
    fitt =(np.poly1d(fit))
    fittt= fitt(range(6))
    plt.plot(fittt)
    print(fitt)
    # plt.plot(L, a0)  # a0 is the average slope as L -> inf
    plt.xlabel('System Size L')
    plt.ylabel('L(^-1)*a0*L(1-a1*L(^-w) +...)')
    plt.show()
    LL = np.array(L)
    w= fit[0]
    approx =a00*(1 -np.exp(fit[1])*(LL**w))
    #a1= 0.25 w= 0.63
    plt.plot(L, a0)
    plt.plot(L, approx)
    plt.show()

    #The fact that the correction increases from 0, might need to add linear term
    # for i in range(len(height)):
    #     corr= dist[i] - dist[-1]
    #     distcorr.append(corr)
    #
    # plt.plot(L, dist)
    # plt.xlabel('System Size')
    # plt.ylabel('(Mean Height) - a0L')
    # plt.show()
    # plt.plot(L,distcorr)
    # plt.xlabel('System Size')
    # plt.ylabel('(Mean Height) - (a0L + b)')
    # plt.plot()

elif input is 'z':
    aval, height, tc = Import_data(0.5, 3, 6, 1000000)
    L = []
    Vsd = []
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
    plt.title(' Standard Deviation of Height vs. System Size')
    plt.xlabel('System size L')
    plt.ylabel('Standard Deviation of Height')
    plt.plot(L, Vsd)
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