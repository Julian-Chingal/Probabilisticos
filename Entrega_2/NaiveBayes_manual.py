import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms

#Variables
N_Attributes = 0
media = 0
sigma = 0
pi = 0
# distribucion de probabilidad, para numeros continuos y apartir de esa distribucion es decir cual es la probabilidad de ocurrencia de ese data

# x_train = train[features]
# y_train = train[target]
# x_test = test[features]
# y_test = test[target]


# Definicion del modelo
def fit(Class, Data):
    global N_Attributes,media, sigma,pi
    
    #Se inciializan las variables
    N_Train_Instances, N_Attributes = Data.shape
    N_Classes = np.max(Class) + 1

    media, sigma = np.zeros((N_Classes,N_Attributes)), np.zeros((N_Classes, N_Attributes))
    N_Instances = np.zeros(N_Classes)

    for c in range(N_Classes):
        Data_Class = Data[Class == c]  # Datos de entrada que pertenecen a una misma clase
        N_Instances[c] = Data_Class.shape[0]
        media [c,:] = np.mean(Data_Class,0)
        sigma [c, :] = np.std(Data_Class, 0)
    
    pi = (N_Instances + 1) / (N_Train_Instances + N_Classes) 

# suma exponencial de un conjunto de valores en la matriz 

# Prediccion