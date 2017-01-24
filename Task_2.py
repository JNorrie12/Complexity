import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
# n= 5
# m= 3
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
#         ssave, hsave, tc = pile.transrec(10000, False)
#         Thsave.extend(hsave)
#         Ttc.extend([tc])
#         plt.plot(hsave)
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

##################################################################
#2b)
# n= 3
# m= 3
# p = 0.5
# L=[2**x for x in range(n,n+m)]
# print L
# piles= []
# Thsave=[]
# TMov = []
# for i in range(len(L)):
#     piles.extend([oslo(L[i], p)])
# for pile in piles:
#         ssave, hsave, tc = pile.transrec(10000, True)
#         # Thsave.extend(hsave)
#         hsq = [i**0.5 for i in hsave]
#         W=100
#         Mov = [sum(hsave[j-W: j+W])/pile.size for j in range(W, len(hsave)-W)] #how do I shrink the graph??? Thinking h**2
#         plt.plot(np.array(range(len(Mov)))/pile.size**2, Mov) #time scales with L^2
#
# plt.show()

###################################################################
# #2c)
n= 5
m= 3
p = 0.5
L=[2**x for x in range(n,n+m)]
print L
piles= []
for i in range(len(L)):
    piles.extend([oslo(L[i], p)])
for pile in piles:
    ssave, hsave, tc = pile.transrec(10000, False)
    Mean = [sum(hsave[0:j])/j for j in range(1, len(hsave)+1)]
#
#     # hsq= [sum([i**2 for i in hsave][0:j])/j for j in range(1, len(hsave)+1)]
#     # sd= [(hsq[i]- Mean[i]**2)**0.5 for i in range(0, len(hsave))]
    plt.plot(np.array(range(len(Mean))), Mean)
plt.show()

####################################################################
#2d


















