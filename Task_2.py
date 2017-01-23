import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
n= 3
m= 5
p = 0.5
L=[2**x for x in range(n,n+m)]
print L
piles= []
Tssave=[]
Thsave=[]
Ttc=[]

for i in range(len(L)):
    piles.extend([oslo(L[i], p)])
for pile in piles:
        ssave, hsave, tc = pile.transrec(10000)
        Thsave.extend(hsave)
        Ttc.extend([tc])

# print Ttc
# plt.plot(Ttc)
# plt.show()
Ttcscale= np.array(Ttc)/np.array(L)
print Ttcscale
plt.plot(Ttcscale)
plt.show()

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


