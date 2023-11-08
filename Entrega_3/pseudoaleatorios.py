# Library
import matplotlib.pyplot as plt

# Variables
seed = 5        # Semilla
a = 3        # Multiplicador (a) 
c = 3       # Incremento (c)
m = 7       # Modulo (m)
n = 10      # Cantidad de Numeros Pseudoaleatorios (n)
 
# Function ---------------------------------------------------------------------------------------
def lcg(seed, a, c, m, n):  # Linear Congruential
    random_numbers = []
    random_numbers.append(seed)

    for i in range(n):
        seed = (a * seed + c) % m
        random_numbers.append(seed)
    return random_numbers



# Implementation ----------------------------------------------------------------------------------
LinearCongruential = lcg(seed,a,c,m, n)

print(f'Metodo Congruencial Lienal {LinearCongruential}')


# Graph -------------------------------------------------------------------------------------------

# Sequence graph
plt.figure(figsize=(6,5))
plt.plot(LinearCongruential, color= 'Green')
plt.title('Secuencia de números aleatorios')
plt.xlabel('Índice')
plt.ylabel('Número aleatorio')
plt.grid(True)


# Histogram
plt.figure(figsize=(6,5))
plt.hist(LinearCongruential, bins=30, color= 'Green')
plt.title('Histograma de números aleatorios')
plt.xlabel('Número aleatorio')
plt.ylabel('Frecuencia')
plt.grid(True)


# ACF(Autocorrelation)
plt.figure(figsize=(6,5))
plt.acorr(LinearCongruential, maxlags=10, color= 'Green')
plt.title('Gráfico de autocorrelación')
plt.xlabel('Retraso')
plt.ylabel('Autocorrelación')
plt.grid(True)
plt.show()
