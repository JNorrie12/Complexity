import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
import sys
from data import *
input = sys.argv[1]

# n= 4
# m= 4
# p = 0.5
# L=[2**x for x in range(n,n+m)]
# print L
# piles= []
# Tssave=[]
# Thsave=[]
# Ttc=[]
#
# for i in range(len(L)):
#     piles.extend([oslo(L[i], p)])
# for pile in piles:
#         ssave, hsave, tc = pile.transrec(10000, True)
#         Thsave.extend(hsave)
#         Ttc.extend([tc])
#         plt.plot(hsave[-1])
# print Ttc
# plt.show()
#
# plt.plot(Ttc)
# plt.show()
# Ttcscale= np.array(Ttc)/np.array(L)
# print Ttcscale
# plt.plot(Ttcscale)
# plt.show()

################################################################################
#REMEMBER TO CHANGE M N IN DATA
if input is 'a':
    aval, height, tc = Import_data(True, True, True, 0.5)
    for i in height:
        plt.plot(i[:5000])
    plt.title('Heights')
    plt.xlabel('Time(Number of Grains Dropped)')
    plt.ylabel('Height(Grains)')

    plt.show()
elif input is 'y':
    aval, height, tc = Import_data(True, True, True, 0.5)
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
    aval, height, tc = Import_data(True, True, True, 0.5)
    for i in range(len(height[:7])):
        #CHANGE WHICH TO PROCESS HERE
        W= (2**(3+i))
        Mov= [sum(height[i][j-W:j+W])/(2*W+1) for j in range(W,len(height[i][:75000])-W)] #CHANGE TIME HERE
        plt.plot(Mov)
    plt.show()

elif input is 'x':
    aval, height, tc = Import_data(True, True, True, 0.5)
    for i in range(len(height[:7])):
        L = (2 ** (3 + i))
        W = 100
        heightsq = [j for j in height[i]]
        Tshrink = [k/(L**2) for k in range(W,len(height[i][:20000])- W)] #NEED TO CHANGE TIME DEPENDENT ON L
        Mov = [sum(height[i][j - W:j + W]) /(L*(2 * W + 1)) for j in
               range(W, len(height[i][:20000]) - W)]  # CHANGE TIME HERE
        plt.plot(Tshrink , Mov)
    plt.show()
################################################################################################################

#
#
# hsq = [i**0.5 for i in hsave]
#         W=100
#         Mov = [sum(hsave[j-W: j+W])/pile.size for j in range(W, len(hsave)-W)] #how do I shrink the graph??? Thinking h**2
#         plt.plot(np.array(range(len(Mov)))/pile.size**2, Mov) #time scales with L^2
#
# plt.show()

###################################################################
# #2c) ~~~~~WORKING~~~~~
if input is 'c':
# n= 5
# m= 3
# p = 0.5
# L=[2**x for x in range(n,n+m)]

    aval, height, tc = Import_data(True, True, True, 0.5)
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
    aval, height, tc = Import_data(True, True, True, 0.5)
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
    aval, height, tc = Import_data(True, True, True, 0.5)
    for i in range(len(height)):
        L= 2**(3+i)
        #freq = np.bincount(np.array([int(height[i][j]) for j in range(tc[i], len(height[i]))]))
        #prob = freq/float(len(height[i])-tc[i])
        #print sum(prob)
        #width = 1.5
       # plt.plot(prob)
    #plt.show()
        Vmean, Vsd = standard_dev()
        standard = (np.array(height[i][tc[i]:]) - Vmean[i])/Vsd[i]
        # norm , bins= np.histogram(standard, bins=16, density = True)
        # width = np.diff(bins)
        # center = (bins[:-1] + bins[1:]) / 2
        # plt.bar(center, norm, align='center', width=width)
        # plt.show()
        plt.hist(standard, normed= True, bins=15, histtype= 'step' ,range=[-8,8])
        x = np.linspace(-3, 3, 100)
        plt.plot(x, mlab.normpdf(x, 0, 1))
        plt.show()

        #standfreq = np.bincount(np.array([int(standard[j]) for j in range(tc[i], len(height[i]))]))
        #standprob= standfreq/float(float(len(height[i])-tc[i]))
        #plt.plot(standardprob)




























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


















