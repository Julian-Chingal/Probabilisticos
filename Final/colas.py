# Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from G_Variables import rv

#! Input Variables-------------------------------------------------------------------------------------------
# Variables inciales
arrivalList = []
serviceList = []
catered = []

# Variables para guardar tabla
df = pd.DataFrame()
df = pd.DataFrame(columns=['id', 'Tiempo Llegada', 'Tiempo Atencion', 'Tiempo Espera', 'Tiempo Atencion Cliente','Tiempo Ocio Server'])

# Variables de entrada
t_server = int(input("Indique las horas de trabajo del servidor (H): "))
m_arri =  int(input("Media de llegada de los cliente (m): ")) # para pruebas t corto
m_atten = int(input("Media de atencion de los clientes (m): ")) # para pruebas t largo luego se intercambian  
n_servers = int(input("Cantidad de servidores a trabajar: "))

#! Functions -------------------------------------------------------------------------------------------------
def arrivalClient(half): # Generar variables aleatorias para la llegada con distribucion exponencial
    global arrivalList
    r = rv.exponential(media=half)
    arrivalList.append(r)
    return arrivalList

def serviceTime(half):
    global serviceList
    r = rv.exponential(half)
    serviceList.append(r)
    return serviceList

#! Class ------------------------------------------------------------------------------------------------------
class server:  # Server class
    def __init__(self, id):
        self.id = id
        self.busyTime = 0     # Tiempo ocupado
        self.serviceTime = 0  # Tiempo de servicio
        self.leisureTime = 0  # Tiempo de ocio

class client: 
    def __init__(self, id):
        self.id = id
        self.arrivalTime = 0  # Tiempo de llegada
        self.serviceTime = 0  # Tiempo de servicio

# Tiempo de hocio cualquier numero debajo de 0 se debe poner 0, se puede trabajar tiempo de hocio por servidor o total

for i in range(0, 50):
    a = arrivalClient(m_arri)
    print(a)
