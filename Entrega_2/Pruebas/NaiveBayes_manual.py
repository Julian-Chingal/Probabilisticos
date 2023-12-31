# Libaries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder 
from yellowbrick.classifier import confusion_matrix as cm

# metodos para gaussian naive bayes
class GaussianNaiveBayes:
    
    def __init__(self):
        return
    
    def fit(self, x, y):
        N, D = x.shape
        C = np.max(y) + 1
        # un parámetro para cada característica condicionada a cada clase
        mu, sigma = np.zeros((C, D)), np.zeros((C, D))
        Nc = np.zeros(C) # número de instancias en la clase c

        # para cada clase, obtenemos el MLE para la media y la desviación estándar
        for c in range(C):
            x_c = x[y == c]                           # seleccionar todos los elementos de la clase c
            Nc[c] = x_c.shape[0]                      # obtener el número de elementos de la clase c
            mu[c, :] = np.mean(x_c, axis=0)           # media de características de la clase c
            sigma[c, :] = np.std(x_c, axis=0)         # desviación estándar de características de la clase c
            
        self.mu = mu                                  # C x D
        self.sigma = sigma                            # C x D
        self.pi = (Nc + 1) / (N + C)                  # Suavizado de Laplace (usando alpha c=1 para todas las c); puedes derivarlo usando la distribución de Dirichlet
    
    def logsumexp(self, Z):  
        Zmax = np.max(Z, axis=0)[None, :]  # máximo sobre C
        log_sum_exp = Zmax + np.log(np.sum(np.exp(Z - Zmax), axis=0))
        return log_sum_exp

    def predict(self, xt):
        Nt, D = xt.shape
        log_prior = np.log(self.pi)[:, None]
        log_likelihood = -.5 * np.log(2*np.pi) - np.log(self.sigma[:,None,:]) -.5 * (((xt[None,:,:] - self.mu[:,None,:])/self.sigma[:,None,:])**2)
        log_likelihood = np.sum(log_likelihood, axis=2)
    
    # posterior calculation
        log_posterior = log_prior + log_likelihood
        posterior = np.exp(log_posterior - self.logsumexp(log_posterior))
        return posterior.T  
    
#Variables
label = LabelEncoder()
model = GaussianNaiveBayes()

# Data train
path = os.path.abspath("Entrega_2/Data/Titanic.csv")    #https://www.kaggle.com/competitions/titanic/data?select=train.csv
path2 = os.path.abspath("Entrega_2/Data/TitanicTest.csv")    

train = pd.read_csv(path) 
test = pd.read_csv(path2)

data = pd.concat([train, test], ignore_index=True, sort=False)

# Preprocess ------------------------------------------------------------------------------------------------------------------
features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
name_target = 'Survived'

#imputar valores faltantes
data['Age'].fillna(data['Age'].median(), inplace=True)  
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Transformar los valores categoricos
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])

#Discretizar la informacion, en contenedores o bins para discretizar la variable continua.
data['Age'] = pd.qcut(data['Age'], 4, labels=False, duplicates='drop') 
data['Fare'] = pd.qcut(data['Fare'], 4, labels=False, duplicates='drop')
data = data.dropna()

#convertir a entero los datos
data['Fare'] = data['Fare'].astype(int)
data['Survived'] = data['Survived'].astype(int)

# Trainin train Titanic ---------------------------------------------------------------------------------------------------------
train = data[features]
target = data[name_target]

x_train, x_test, y_train, y_test = tts(train, target, test_size=0.2, random_state=42) 

# fit ---------------------------------------------------------------------------------------------------------------------------
model.fit(x_train,y_train) #entrenar el modelo titanic

#test
prediction = model.predict(x_test) #Predicciones ustilziando el conjunto de prueba
accuracy = accuracy_score(y_test, prediction) #precision del modelo

# Perform for class
perfom_class = classification_report(y_test, prediction)
print("Clasification report: \n", perfom_class)

print("------------------------------------",
      f"\nPrecision del modelo Titanic: {accuracy}",
      "\nCantidad de Instancias : ", len(y_train),
      "\nCantidad de Prueba: ", len(y_test), 
      "\n------------------------------------")

# graphic -----------------------------------------------------------------------------------------------------------------------
Clases = ["Not Survival", "Survival"]

Matriz= cm(model,x_train,y_train,x_test,y_test, classes= Clases,cmap="Greens")
Matriz.poof()

Matriz= cm(model,x_train,y_train,x_test,y_test,percent=True, cmap="Greens")
Matriz.poof()
