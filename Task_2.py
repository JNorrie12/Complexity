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
if input is 'a':
    aval, height, tc = Import_data(0.5, 3, 5)
    for i in height:
        plt.plot(i[:5000])
    plt.title('Heights')
    plt.xlabel('Time(Number of Grains Dropped)')
    plt.ylabel('Height(Grains)')

    plt.show()
elif input is 'y':
    aval, height, tc = Import_data(0.5, 3, 5)
    L=[]
    for i in range(len(tc)):
        L.append(2**(3+i))
    plt.plot(L, tc, )
    coeff =np.polyfit(L, tc, 2)
    Quad = np.poly1d(coeff)
    print Quad
    Fit = Quad(range(256))
    plt.plot(Fit)
    plt.title('Crossover Times')
    plt.xlabel('System size L')
    plt.ylabel('Time')
    plt.show()





    #CHANGES CUT OFF  OF DATA, 75,000 good for L=25:

# print float(tc/L) #Ratio roughly doubles when L doubles  => tc=kL^2
#
#
# # plt.plot(hsave)
# # plt.show()
# hnorm=np.array(hsave)/L #Probability of heighth
# plt.plot(hnorm)
# plt.show()



# for i in np.arange(L):
#     meanh=(np.array(hsave).cumsum()/L)
# plt.plot(meanh)
# plt.show()

#################################################################
#2b)
elif input is 'b':
     #CHANGE W FOR SMOOTHNESS
    aval, height, tc = Import_data(0.5, 3, 5)
    for i in range(len(height[:7])):
        #CHANGE WHICH TO PROCESS HERE
        W= (2**(3+i))
        Mov= [sum(height[i][j-W:j+W])/(2*W+1) for j in range(W,len(height[i][:75000])-W)] #CHANGE TIME HERE
        plt.plot(Mov)
    plt.show()

elif input is 'x':
    aval, height, tc = Import_data(0.5, 3, 5)
    for i in range(len(height[:7])):
        L = (2 ** (3 + i))
        W = 100
        heightsq = [j for j in height[i]]
        Tshrink = [k/(L**2) for k in range(W,len(height[i][:20000])- W)] #NEED TO CHANGE TIME DEPENDENT ON L
        Mov = [sum(height[i][j - W:j + W]) /(L*(2 * W + 1)) for j in
               range(W, len(height[i][:20000]) - W)]  # CHANGE TIME HERE
        plt.plot(Tshrink , Mov)
    plt.show()
########################################################################################################

###################################################################
# #2c) ~~~~~WORKING~~~~~
if input is 'c':
# n= 5
# m= 3
# p = 0.5
# L=[2**x for x in range(n,n+m)]

    aval, height, tc = Import_data(0.5, 3, 5)
    L= []
    VMean=[]
    for i in range(len(height)):
        L.append(2 ** (3 + i))
        summ = 0
        for j in range(len(height[i])): ####CHANGE HERE FOR Ls##########
            if j > tc[i] :
                summ += height[i][j]
            else:
                pass

        Mean= summ/(len(height[i])-(tc[i]))
        VMean.append( Mean)
    coeff = np.polyfit(L, VMean, 1)
    Fit=np.poly1d(coeff)
    print Fit
    a0 = coeff[0]
    b = coeff[1]
    a1 = b/a0
    print a0, a1
    plt.title('Mean Height vs. System Size')
    plt.xlabel('System size L')
    plt.ylabel('Mean Height')
    plt.plot(L, VMean)
    plt.show()

elif input is 'z':
    aval, height, tc = Import_data(0.5, 3, 5)
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
        # summ = sum(height[i-tc[i]:])
        # summsq= height[i]**2
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


if input is 'd':
    import matplotlib.mlab as mlab
    aval, height, tc = Import_data(0.5, 3, 5)
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




























    # Mean = [sum(hsave[0:j])/j for j in range(1, len(hsave)+1)]
        # Mean = sum(hsave) / pile.time
        # TMean.extend([Mean])
            # hsq= [sum([i**2 for i in hsave][0:j])/j for j in range(1, len(hsave)+1)]
            # sd= [(hsq[i]- Mean[i]**2)**0.5 for i in range(0, len(hsave))]
        # plt.plot(np.array(range(len(Mean))), Mean)
        # hsq = [sum(i ** 2 for i in hsave) / pile.time]
        # sd = (hsq - Mean ** 2) ** .5
        # Tsd.extend(sd)


####################################################################
#2d


















