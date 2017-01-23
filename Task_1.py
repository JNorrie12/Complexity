import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *
from Oslo_2 import *
L = 16
p = 0.5
pile = oslo(L, p)
ssave, hsave, tc = pile.transrec(10e5)


print(ssave)
b, c = log_bin(ssave)
plt.loglog(b,c )
plt.show()
#
# while True:
#     pile.drive()
#     pile.relax()
#     pile.draw()
#     raw_input()


