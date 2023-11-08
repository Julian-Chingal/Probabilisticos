# Library
import matplotlib.pyplot as plt
import sys

# Variables
seed = 5        # Semilla
a = 2        # Multiplicador (a) 
c = 0       # Incremento (c)
m = 9       # Modulo (m)
n = 10      # Cantidad de Numeros Pseudoaleatorios (n)
 
# Function ---------------------------------------------------------------------------------------
def lcg(seed, a, c, m, n):  # Linear Congruential
    if a < m:
        random_numbers = []
        random_numbers.append(seed)

        for i in range(n):
            seed = (a * seed + c) % m
            random_numbers.append(seed)
        return random_numbers
    else:
        print("El multiplicador (a) no debe ser mayor al modulo  (m)")
        sys.exit(1) 

# Implementation ----------------------------------------------------------------------------------
methods = [
    ("Metodo Congruencial Lienal", lcg(seed,a,c,m, n))
]

# Graph -------------------------------------------------------------------------------------------
for name, method in methods:
    print(f'{name} {method}')

    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    # Sequence graph
    axs[0].plot(method, color='Green')
    axs[0].set_title(f'Secuencia de números aleatorios: {name}')
    axs[0].set_xlabel('Índice')
    axs[0].set_ylabel('Número aleatorio')
    axs[0].grid(True)
    # Histogram
    axs[1].hist(method, bins=30, color='Green')
    axs[1].set_title(f'Histograma de números aleatorios: {name}')
    axs[1].set_xlabel('Número aleatorio')
    axs[1].set_ylabel('Frecuencia')
    axs[1].grid(True)
    # ACF(Autocorrelation)
    axs[2].acorr(method, maxlags=10, color='Green')
    axs[2].set_title(f'Gráfico de autocorrelación: {name}')
    axs[2].set_xlabel('Retraso')
    axs[2].set_ylabel('Autocorrelación')
    axs[2].grid(True)
    plt.subplots_adjust(wspace=0, hspace=1)
    plt.show()