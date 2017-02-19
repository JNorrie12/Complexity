# Complexity Project 1/17 - 2/17
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from log_bin import *

#####ACTUAL MODEL######

class oslo():
    def __init__(self, L, p):
        self.size = L #system size
        self.p = p #prob of threshold
        self.height = np.zeros(self.size)
        self.thresh = np.array([np.random.binomial(1, self.p, None) + 1 for x in self.height])
        #randomising threshold

    def draw(self):
        for i in range(self.size):
            print '[{}, {}]'.format(int(self.height[i]),int(self.thresh[i])) + '#'* np.int(self.height[i])

    def drive(self):
        # add rice grain to lefthand side
        self.height[0] += 1

    def transrec(self, T): #T= number of grains dropped
        ssave= []
        hsave= []
        tc =0
        y= 0
        self.time= T
        while T >0 :
            self.drive()
            a, h, x = self.relax()
            T -= 1
            y += x
            if y <= 1 :
                tc +=1
            ssave.extend([a])
            hsave.extend([h])


        return ssave, hsave, tc,


    def relax(self):
        # move through each element of heights
        # compute slope
        shift = np.roll(self.height, -1)
        shift[-1] = 0
        slope = self.height - shift
        truefalse = (slope) > self.thresh #When slope>thresh = 1
        s=0
        x=0
        #Algorithm
        while np.sum(truefalse) > 0 :
            for i in np.arange(self.size):
                 if truefalse[i] == 1 :
                    if i==self.size -1 :
                        self.height[self.size - 1] -= 1
                        x += 1
                    else:
                        self.height[i] -= 1
                        self.height[i+1] += 1
                    s +=1
                    self.thresh[i] = np.random.binomial(1, self.p, None) + 1
            #Update condition
            shift = np.roll(self.height, -1)
            shift[-1] = 0
            slope = self.height - shift
            truefalse = (slope) > self.thresh
        return s, self.height[0], x

