# Library
import time as t
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# Variables
uniform = []
random_numbers = []
pseudo = []
res_expo = []
res_poiss = []

seed = int(t.time())    # Semilla
print("Semilla = ", seed)
a = 1103515245          # Multiplicador (a)
c = 12345               # Incremento (c)
m = 24582361784         # Modulo (m)

min = int(input("Limite Inferior: "))
max = int(input("Limite Superior: "))
n = int(input("Cantidad de numeros a generar: "))     # Cantidad de Numeros Pseudoaleatorios (n)
 
#! Function ---------------------------------------------------------------------------------------
def lcg(seed, a, c, m, n):  #? Linear Congruential method
    rep = []
    random_numbers.append(seed)

    for i in range(n):
        seed = (a * seed + c) % m # Metodo
        random_numbers.append(seed) # Almacenar resultado
        uniform.append(round((seed/m),4)) # variable uniforme el resultado rango (0,1)

        # Generar numero pseudo
        pseudo.append((round((uniform[i] * (max-min) + min),4)))

        if seed not in rep:
            rep.append(seed)
        
    return uniform, pseudo, rep

def halfSquare(seed,n): #? half square
    resul = []
    for i in range(n):
        x1 = (1103515245 * seed + 12345) % 32768
        seed = x1
        resul.append(x1)
    
    return resul

def disPoiss(lam, u, min, max): #? poisson distribution
    e = np.e #euler
    proba = []
    acum = []
    val_Poison=[]
    for i in range(min, max+1):
        val_Poison.append(i)

    #Calcular la probabilidad 
    for i in val_Poison:
        proba.append(((e ** -lam)*(lam ** (i))) / (math.factorial(i)))

    #Calcular el acumulado
    sum = 0
    for i in range(len(proba)):
        if i ==0:
            acum.append(proba[i])
        else:
            acum.append(proba[i]+acum[i-1])
        
        print(acum)

    # Numeros con poisson
    limite =len(acum)-1
    va =[]
    for i in range(len(u)):
        rango = 0
        for j in range(len(acum)):
            if acum[j] < u[i]: 
                if rango < limite:
                    rango = rango+1
 
        res_poiss.append(rango) 
    return res_poiss


def disExpo(lam,r): #? exponencial distribution
    return -(1/lam)*np.log(r)

#! Implementation ----------------------------------------------------------------------------------
methods = [
    ("Metodo Congruencial Lienal", lcg(seed,a,c,m, n)),
    ("Metodo Cuadrado Medio", halfSquare(seed,n)),
]

#! Graph -------------------------------------------------------------------------------------------
for name, method in methods:
    if name == "Metodo Congruencial Lienal":
        #Mostrar Info
        m_unifor, n_pseudo, rep = method
        print(f'\n{name} \n\nuniformes \n{uniform} \n\nNumeros pseudo Aleatorios \n{n_pseudo} \n\nPeriodo {len(rep)}')

        # Histogram
        plt.figure(figsize=(7, 4))
        plt.hist(m_unifor, bins=30, color='Green')
        plt.title(f'Histograma de números aleatorios: {name}')
        plt.xlabel('Número aleatorio')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        # Dispersion
        plt.figure(figsize=(7, 4))
        plt.scatter(range(len(m_unifor)), m_unifor, color='Green', marker='o')
        plt.title(f'Gráfico de dispersión: {name}')
        plt.xlabel('Índice')
        plt.ylabel('Número aleatorio')
        plt.grid(True)
        plt.show()
        
    if name == "Metodo Cuadrado Medio":
        print(f'\n{name} \n{method}')
        plt.figure(figsize=(7, 4))
        plt.scatter(range(len(method)), method, color='Green', marker='o')
        plt.title(f'Gráfico de dispersión: {name}')
        plt.xlabel('Índice')
        plt.ylabel('Número aleatorio')
        plt.grid(True)
        plt.show()

#! Distribuciones de probabilidad ------------------------------------------------------------------------
#? Distribucion de probabilidad Exponencial ---------------
# Ejercicio, Calcular la cantidad de autos que entran a una autopista por hora. EI tiempo de llegada de autos sigue una distribución exponencial con una media de 0.2 horas.

mu = 0.2   #media
lam = 1/mu #lamda

for i in range(n):
    res_expo.append(disExpo(lam, uniform[i]))

#grafica
plt.figure(figsize=(7, 4))
plt.bar(range(n), res_expo, color='Green')
plt.title(f'Valores simulador de variable aleatoria exponencial')
plt.xlabel('Tiempo de llegada')
plt.ylabel('Numero de autos')
plt.show()

#? Distribucion de probabilidad de Poisson --------------
lamp = 100
minp = int(input("Valor minimo para poisson: "))
maxp = int(input("Valor maximo para poisson: "))
print("poisson ", disPoiss(lamp, pseudo,minp, maxp))