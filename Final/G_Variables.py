import time as t 
import numpy as np
import math 

# Class 
class random_variables:
    def __init__(self, a, m, c):
        self.a = a
        self.c = c
        self.m = m
        self.seed = int(round(t.time()))

    def generateSeed(self):
        self.seed = (self.a * self.seed + self.c) % self.m # Metodo
        return self.seed

    def lcg(self, max, min): # Linear congruence method
        while True:
            seed = self.generateSeed()
            uniform = seed / self.m # normalizar resultado
            # Generar numero pseudo
            pseudo = min + ((max-min) * uniform)

            if pseudo <= max:
                return pseudo
            else:
                self.seed = t.time()
                
    def exponential(self, media):
        uniform = self.lcg(1,0.1)
        result = -(media)* np.log(1 - uniform)
        result = round(result, 2)
        return result

rv = random_variables(524287, 46340, 2147483647)
