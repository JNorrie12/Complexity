import numpy as np
import random as rd
import matplotlib.pyplot as plt
import pylab as py
from log_bin import *
from Oslo_2 import *
import sys
from data import *
import matplotlib.lines as mlines
input = sys.argv[1]
################################################################################
#REMEMBER TO CHANGE M N IN DATA
def standard_dev(n, m):
    L = []
    Vsd = []
    Vmean = []
    for i in range(m):
        L.append(2 ** (n + i))
        # summ = 0
        # summsq = 0
        # for j in range(len(height[i])):  ####CHANGE HERE FOR Ls##########
        #     if j > tc[i]:
        #         summ += height[i][j]
        #         summsq += (height[i][j]**2)
        m = np.mean(height[i][tc[i]+100:])
        s = np.std(height[i][tc[i]+100:])
        # Mean = summ / (len(height[i]) - (tc[i]))
        # Square = summsq/ (len(height[i])-(tc[i]))
        # sd= (Square - (Mean)**2)**.5
        Vsd.append(s)
        Vmean.append(m)
    return L, Vmean, Vsd

if input is 'a':
    n=4
    m=5
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    for i in range(len(height)):
        L= 2**(n+i)
        plt.plot(height[i][:100000], zorder=1)
        plt.annotate('(' + str(tc[i]) + ',' + str(int(height[i][tc[i]])) + ')', xy=(tc[i], height[i][tc[i]]), xytext=(tc[i] + 0.01 *L, height[i][tc[i]] + 0.05 * L))
        b = plt.scatter(tc[i], height[i][tc[i]], color='black', zorder=2, label='Respective Crossover Times')
        plt.legend(handles=[b])
    blue_line = mlines.Line2D([], [], color='blue', label='L=16')
    orange_line = mlines.Line2D([], [], color='orange', label='L=32')
    green_line = mlines.Line2D([], [], color='green', label='L=64')
    red_line = mlines.Line2D([], [], color='red', label='L=128')
    purple_line = mlines.Line2D([], [], color='purple', label='L=256')

    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line ])
    plt.title('Total Heights for system sizes L=16,32,64,128,256')
    plt.xlabel('Time')
    plt.ylabel('Height')

    plt.show()
elif input is 'y':
    n=4
    m=5
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    L=[]
    for i in range(len(tc)):
        L.append(2**(n+i))
    a= plt.scatter(L, tc, color='black', zorder=3, label='Crossover Times for System Size L')
    coeff =np.polynomial.polynomial.polyfit(L, tc,[2] )
    coeffrev= [coeff[2], 0, 0] ####reverses the list #####
    Quad = np.poly1d(coeffrev)
    Fit = Quad(range(256))
    red_line = mlines.Line2D([], [], color='red', label='Quadratic fit, T='+ str(round(coeff[2],2)) + '*L^2')
    x=np.linspace(0,256, 256)
    y=np.linspace(0,240,240 )
    plt.plot(x, 0.5*(x**2), 'b--',  zorder=1)
    plt.plot(y, y**2, 'g--', zorder=1)
    blue_line = mlines.Line2D([], [], color='blue', linestyle='--', label='T= 0.5*L^2')
    green_line = mlines.Line2D([], [], color='green', linestyle='--', label='T=L^2')
    plt.plot(Fit, 'r', zorder=2)
    plt.legend(handles=[red_line, blue_line, green_line, a])

    plt.title('Crossover Times')
    plt.xlabel('System size L')
    plt.ylabel('Time')
    plt.show()
    for i in range(len(height)):
        plt.plot(np.array(height[i])/L[i])
    z= np.ones(1000000)*1.715
    plt.plot(z, color='black', linestyle='--')
    blue_line = mlines.Line2D([], [], color='blue', label='L=16')
    orange_line = mlines.Line2D([], [], color='orange', label='L=32')
    green_line = mlines.Line2D([], [], color='green', label='L=64')
    red_line = mlines.Line2D([], [], color='red', label='L=128')
    purple_line = mlines.Line2D([], [], color='purple', label='L=256')
    black_line = mlines.Line2D([], [], color='black', linestyle='--', label='Heights/L=1.715')
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, black_line])
    plt.title('Heights/System Size ')
    plt.xlabel('Time')
    plt.ylabel('Heights/L')
    plt.show()

#################################################################
#2b)
elif input is 'b':
    n=3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    ax1 = plt.subplot(111)
    ax1.set_xlim([0,2])
    ax1.set_ylim([0.1, 2.3])
    for i in range(m):
        L = (2 ** (n + i))
        W = 2*L
        Tshrink = [float(k)/(L**2) for k in range(W,len(height[i])- W)]#NEED TO CHANGE TIME DEPENDENT ON L
        Mov = [sum(height[i][j - W:j + W]) /(L*(2 * W + 1)) for j in range(W, len(height[i]) - W)]  # CHANGE TIME HERE
        ax1.plot(Tshrink, Mov)
        if i == m-1:
            Tsqrt=1.85*np.sqrt(Tshrink)
            Tconst= 1.736*np.ones(len(Tshrink))
            ax1.plot(Tshrink, Tsqrt, 'r--')
            ax1.plot(Tshrink, Tconst, 'r--')
    blue_line = mlines.Line2D([], [], color='blue', label='L=8')
    orange_line = mlines.Line2D([], [], color='orange', label='L=16')
    green_line = mlines.Line2D([], [], color='green', label='L=32')
    red_line = mlines.Line2D([], [], color='red', label='L=64')
    purple_line = mlines.Line2D([], [], color='purple', label='L=128')
    brown_line = mlines.Line2D([], [], color='brown', label='L=256')

    plt.annotate( 'F=k*(T/L^2)^0.5', color='red', xy=(0.05, 1.3),xytext=(0.05, 1.3), rotation= 45)
    plt.annotate('F=constant', color='red', xy=(1.5, 1.8), xytext=(1.5, 1.8))
    plt.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, brown_line])
    plt.title('Height Moving Average Data Collapse')
    plt.xlabel('Scaled Time(t/L**2)')
    plt.ylabel('Scaled Height(h/L)')
    plt.show()

########################################################################################################
# #2c) ~~~~~WORKING~~~~~Making more streamline w/ sd function
if input is 'c':
    n= 3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000) #######Include 2 and 4 to see signs of scaling################
    L, VMean, sd = standard_dev(n,m)
    a0=[]
    finder =[]
    findera=[]
    finderb=[]
    for i in range(len(VMean)):
        a= VMean[i]/L[i]
        a0.append(a)

    a00 = 1.736
    a01 = a0[-1]
    a02 = a0[-1]+0.034
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
    plt.plot(logL, logf, 'b-')
    fit = np.polyfit(logL, logf, 1)
    fitt =(np.poly1d(fit))
    fittt= fitt(range(6))

    logfa =np.log(findera)
    logfb= np.log(finderb)
    plt.plot(logL, logfa, color='orange')
    plt.plot(logL, logfb, 'g-')
    # a0 is the average slope as L -> inf
    blue_line = mlines.Line2D([], [], color='blue', label='a0=' + str(round(a00,3)))
    orange_line = mlines.Line2D([], [], color='orange', label='a0=' + str(round(a01,3)) + '(Too small)')
    green_line = mlines.Line2D([], [], color='green', label='a0=' + str(round(a02, 3)) + '(Too large)')
    plt.legend(handles=[blue_line, orange_line, green_line])
    plt.title('Approximation of a0')
    plt.xlabel('Log(L)')

    plt.ylabel('Log(<h>/L)')
    plt.show()
    LL = np.array(L)
    w= fit[0]
    y = np.linspace(0, 300, 300)
    approx =lambda x: a00*(1 -np.exp(fit[1])*(x**w))
    s = plt.scatter(L, a0, color='black', zorder='2', label='Data for L=8,16,32,128,256')
    plt.plot(y, approx(y), 'r-', zorder='1')
    z=a00*np.ones(300)
    plt.plot(z, color='gray', linestyle=':')
    red_line = mlines.Line2D([], [], color='red', label='Scale Correction')
    grey_line = mlines.Line2D([], [], color='grey', linestyle='--', label='Limit of <h>/L as L-> inf')
    plt.legend(handles=[red_line, grey_line, s])
    plt.title('Fit of Scale Correction, with a0='+str(round(a00,3))+', a1='+str(round(np.exp(fit[1]), 3))+', w1='+str(round(-w,3)) )
    plt.xlabel('System size L')
    plt.ylabel('Mean Height/L')
    plt.show()
    fig = plt.figure(figsize=(6, 4))
    ax2 = fig.add_subplot(1,1,1)
    ax2.set_xlim([0, 35])
    ax2.set_ylim([0, 60])
    plt.title('Correction to scaling of Mean Height')
    plt.xlabel('System size L')
    plt.ylabel('Mean Height')
    s =plt.scatter(L, VMean, color='black' , zorder=3, label='Data points')
    x=np.linspace(0,64,64)
    lin0 = np.polyfit(L[4:],VMean[4:], 1)
    lin= lin0[0]*x +lin0[1]
    ax2.plot(x,  lin, color='gray', linestyle=':', zorder=2)
    ax2.plot(x, approx(x)*x, 'r-', zorder=1)
    red_line = mlines.Line2D([], [], color='red', label='Scale Correction')
    grey_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Limit of <h>/L as L-> inf')
    plt.legend(handles=[red_line, grey_line, s])
    plt.show()
elif input is 'z':
    n=3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    L, Vmean, Vsd = standard_dev(n, m)
    plt.title(' Standard Deviation of Height vs. System Size')
    plt.xlabel('System size L')
    plt.ylabel('Standard Deviation of Height')
    plt.scatter(L, Vsd, color='black')
    plt.show()
    logsd= np.log(Vsd)
    logL= np.log(L) #DOESNT TAKE INTO ACCOUNT SCALING ERROR ##########
    a = plt.scatter(logL, logsd, color='black' , label='Data points',zorder=2)
    fit= np.polyfit(logL[-2:], logsd[-2:], 1)
    x =np.linspace(2,6, 20)
    plt.plot(x, fit[0]*x + fit[1], 'b-', zorder=1)
    plt.title('Fit of Loglog Plot')
    plt.xlabel('Log(System size)')
    plt.ylabel('Log(Standard Deviation)')
    blue_line = mlines.Line2D([], [], color='blue', linestyle='-', label='Linear Fit')
    plt.legend(handles=[blue_line, a])
    plt.show()
    print fit[0]
    y=np.linspace(4,256,256)
    fitt= np.exp(fit[1])*y**fit[0]
    plt.plot(y, fitt, 'r--', zorder=2)
    a= plt.scatter(L, Vsd, color='black', label='Data points', zorder=1)
    plt.title('Fit of Loglog Plot')
    plt.xlabel('Log(System size)')
    plt.ylabel('Log(Standard Deviation)')
    blue_line = mlines.Line2D([], [], color='red', linestyle='--', label='Fit='+str(round(np.exp(fit[1]),2))+'x^'+str(round(fit[0],2)))
    plt.legend(handles=[blue_line, a])
    plt.show()

elif input is 'g':
    n=3
    m=6
    a= 0.68
    b= 0.21
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    L, Vmean, Vsd = standard_dev(n, m)
    ml= np.array(Vsd)/np.array(L)**b
    plt.plot(L,ml, 'k ', marker='o' )
    plt.title('Plot of Scaled Standard Deviation')
    plt.xlabel('System size')
    plt.ylabel('Standard Deviation/L^'+str(b))
    x = a*np.ones(len(L))
    plt.plot(L, x, 'g:' )
    black_line = mlines.Line2D([], [], color='black', linestyle=' ', marker='o', label='Data points')
    grey_line= mlines.Line2D([], [], color='gray', linestyle=':', label='Approximate asymptote at of Stadard deviation('+str(a)+')')
    plt.legend(handles=[black_line, grey_line])

    plt.show()
##################################################################
#2d)
elif input is 'd':
    import matplotlib.mlab as mlab
    n=3
    m=6
    aval, height, tc = Import_data(0.5, n, m, 1000000)
    L, Vmean, Vsd = standard_dev(n, m)
    f1 = plt.figure()
    f2 = plt.figure(figsize=(8, 6))
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)
    ax2.set_xlim([-8, 8])

    for i in range(len(height)):
        bins0= range(L[i],2*(L[i]+2))
        bins= (np.array(bins0) - Vmean[i])/Vsd[i]
        a, b = np.histogram(height[i], bins=bins0, range=None, normed=True, weights=None, density=None)
        scaled = Vsd[i]*a

        ax1.plot(b[:-1], a)

        ax2.plot(bins[:-1], scaled)
    x = np.linspace(-3, 3, 100)
    ax2.plot(x, mlab.normpdf(x, 0, 1), 'r--')

    blue_line = mlines.Line2D([], [], color='blue', label='L=8')
    orange_line = mlines.Line2D([], [], color='orange', label='L=16')
    green_line = mlines.Line2D([], [], color='green', label='L=32')
    red_line = mlines.Line2D([], [], color='red', label='L=64')
    purple_line = mlines.Line2D([], [], color='purple', label='L=128')
    brown_line = mlines.Line2D([], [], color='brown', label='L=256')
    n =mlines.Line2D([], [], color='red', linestyle='--', label='pdf of N(1,0) distribution')
    ax1.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, brown_line])
    ax2.legend(handles=[blue_line, orange_line, green_line, red_line, purple_line, brown_line, n])
    ax1.set_title('Total Height Distributions')
    ax2.set_title('Data Collapsed Height Distributions')
    ax1.set_ylabel('P(Total Height Occurring)')
    ax1.set_xlabel('Total Height')
    ax2.set_ylabel('standdev*P(Height Occurring)')
    ax2.set_xlabel('Transformed Total Height')
    plt.show()