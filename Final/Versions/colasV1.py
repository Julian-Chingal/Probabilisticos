# Library
import pandas as pd
import matplotlib.pyplot as plt
from G_Variables import rv

#! Input Variables-------------------------------------------------------------------------------------------
# Variables inciales
arrivalList = []     #? Almacenar tiempos de llegada
serviceList = []     #? Almacenar tiempo de servicio
served = []         #? Almacenar clientes atendidos

current_time = 0     #? tiempo actual minutos
cola = []            #? Almacena clientes que se estan atendiendo

# Variables para guardar tabla
pathMatrix = "./Final/Data/matrix.xlsx"
df = pd.DataFrame()
df = pd.DataFrame(columns=['id Cliente', 'Tiempo Llegada', 'Tiempo Atencion', 'Tiempo Espera', 'id Server'])

# Variables de entrada
print("\nVariables inciales:", "\n----------------------")
simulation_t = int(input("Indique las horas de trabajo del servidor (H): "))
m_arri =  int(input("Media de llegada de los cliente (m): ")) #? para pruebas t corto
m_atten = int(input("Media de atencion de los clientes (m): ")) #? para pruebas t largo luego se intercambian  
n_servers = int(input("Cantidad de servidores a trabajar: "))

#! Class ------------------------------------------------------------------------------------------------------
class server:  # Server class
    def __init__(self, id):
        self.id = id
        self.busyTime = 0     #? Tiempo ocupado
        self.serviceTime = 0  #? Tiempo de servicio
        self.leisureTime = 0  #? Tiempo de ocio

servers = [server(i+1) for i in range(n_servers)]  #? Inicializar servidores 
print(f'\nCantidad de Servidores inicializados para trabajar: {len(servers)}',
      "\n---------------------------------------------------------")

class client: 
    def __init__(self, id):
        self.id = id
        self.arrivalTime = 0  # Tiempo de llegada
        self.serviceTime = 0  # Tiempo de servicio

#! Functions -------------------------------------------------------------------------------------------------
def arrivalClient(half): #? Variables aleatorias llegada distribucion exponencial
    global arrivalList
    val = rv.exponential(half)
    val = round(val,2)
    arrivalList.append(val)
    return val

def serviceTime(half): #? Variables aleatorias tiempo servicio distribucion exponencial
    global serviceList
    val = rv.exponential(half)
    val = round(val,2)
    serviceList.append(val)
    return val

def conv_hours(m): #? covertir a horas
    h = m/60
    return  round(h,1)

def conv_mins(h): #? Convertir a minutos
    return h * 60

def addRow(id, arri_t, serv_t, wait_t, id_ser): #? Agregar fila al excel
    global df

    newRow = {'id Cliente': id,
            'Tiempo Llegada': round(arri_t,2),
            'Tiempo Atencion': serv_t, 
            'Tiempo Espera': wait_t, 
            'id Server': id_ser}
    
    # if not df.empty:
    #     df = pd.concat([df, pd.DataFrame([newRow])], ignore_index= True)
    # else:
    #     df = pd.DataFrame([newRow])

    df = pd.concat([df, pd.DataFrame([newRow])], ignore_index= True)
    
    
    

def simulate(time):
    global current_time, cola, m_atten, m_arri

    id_client = 1  #? id cliente
    simulation_t = conv_mins(time)
    print(simulation_t)
    i = 0
    while current_time <= simulation_t or len(cola) > 0 : #? bucle funcionamiento 
        Client = client(id=id_client) # Crea un objetos cliente 
        print(i)
        i += 1
        

        if current_time < simulation_t - m_atten: #? Asegurarse que haya tiempo para atender un cliente nuevo
            Client.arrivalTime = current_time + arrivalClient(m_arri) # Genera variables de llegada  y le suma el tiempo actual 
            Client.serviceTime = serviceTime(m_atten) # Genera variables de atencion al cliente 
            cola.append(Client)  # Agregar el cliente a la cola 
            id_client += 1  # Agregar clientes
        
        if len(cola) > 0: #? Verifica que haya clientes para atender
            for server in servers:
                if server.busyTime <= current_time and len(cola) > 0:
                    if cola[0].arrivalTime  <= current_time: # verifica que el tiempo de llegada sea menor o igual al tiempo actual
                        server.serviceTime += cola[0].serviceTime  
                        server.busyTime = server.busyTime + cola[0].serviceTime  
                        wait_time = round(current_time - cola[0].arrivalTime, 2)

                        addRow(cola[0].id, cola[0].arrivalTime, current_time, wait_time, server.id) # Agregar nueva Fila al excel 

                        served.append(cola[0]) # Agregar clientes atendidos
                        cola.pop(0)            # Reiniciar variable

                current_time += 1 # Se agrega un minuto

#! Init -------------------------------------------------------------------------------------------------------
print("\nSimulacion inciada.........")
simulate(simulation_t)
print("\nGuardando Excel.....")
df.to_excel(pathMatrix, index=False)
print("\nSimulacion terminada\n")

#? servers info
promleisure = 0
for serve in servers:
    if current_time - serve.serviceTime >= 0:
        serve.leisureTime =  current_time - serve.serviceTime
    promleisure += serve.leisureTime # tiempo toatla de ocio

    print(
        f"\n----------- Servers {serve.id} info -----------",
        f'\nTiempo de servicio: {conv_hours(serve.serviceTime)} Horas',
        f'\nTiempo de ocio: {conv_hours(serve.leisureTime)} Horas',
    )

#? info
wait_t = df["Tiempo Espera"].mean()
pro = promleisure/len(served)
print(
    '\n-------------------- Info ----------------------',
    f'\nNumero de clientes atendidos: {len(served)}',
    f'\nTiempo medio de espera de los clientes (Horas): {conv_hours(wait_t)}',
    f'\nTiempo ocio promedio por cliente (Horas): {conv_hours(pro)}',
    f'\nTiempo ocio total (min): {promleisure}',
    f'\nTiempo ocio total (horas): {conv_hours(promleisure)}',
    f'\nTiempo total simulacion (min): {current_time}',
    f'\nTiempo total simulacion (horas): {conv_hours(current_time)}',
)

#! Graph ------------------------------------------------------------------------------------------------------
# Histograma tiempo de llegada cliente
plt.hist(arrivalList, bins=10, color= 'Green')
plt.xlabel('Val')
plt.ylabel('Frencuencia')
plt.title("Histograma llegada clientes")
plt.show()

# histograma tiempo de Servicio cliente
plt.hist(serviceList, bins=10, color= 'Green')
plt.xlabel('Val')
plt.ylabel('Frencuencia')
plt.title("Histograma servicio clientes")
plt.show()