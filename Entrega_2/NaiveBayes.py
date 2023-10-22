# Libaries
import os
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from yellowbrick.classifier import confusion_matrix as cm

#Variables
label = LabelEncoder()
model = GaussianNB()

# Data file
path = os.path.abspath("Entrega_2/Data/Titanic.csv")    #https://www.kaggle.com/competitions/titanic/data?select=train.csv
file = pd.read_csv(path) 

# Preprocess
print(file.isnull().sum()) # Identificar valores faltantes

file['Age'].fillna(file['Age'].median(), inplace=True) #imputar valores faltantes
file['Embarked'].fillna(file['Embarked'].mode()[0], inplace=True)
file.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True) #Eliminar Atributos innecesarios
file['Sex'] = label.fit_transform(file['Sex']) # Transformar las etiquetas del sexo
file['Embarked'] = label.fit_transform(file['Embarked'])

#--------------------------------------------------------------------------------------------------------------------------------
# Trainin file Titanic
input = file.drop(columns=['Survived']) #eliminar la variable objetivo para entrenar le modelo con los atributos
target = file['Survived'] # Variable objetivo
x_train, x_test, y_train, y_test = tts(input, target, test_size=0.3, random_state=42) 

#--------------------------------------------------------------------------------------------------------------------------------
# fit
model.fit(x_train,y_train) #entrenar el modelo titanic

#test
prediction = model.predict(x_test) #Predicciones ustilziando el conjunto de prueba
accuracy = accuracy_score(y_test, prediction) #precision del modelo
print("------------------------------------",
      f"\nPrecision del modelo Titanic: {accuracy}",
      "\nCantidad de Instancias : ", len(y_train),
      "\nCantidad de Prueba: ", len(y_test), 
      "\n------------------------------------")

#--------------------------------------------------------------------------------------------------------------------------------
# graphic
Clases = ["Die","Survival"]

Matriz= cm(model,x_train,y_train,x_test,y_test, classes= Clases,cmap="Greens")
Matriz.poof()

Matriz= cm(model,x_train,y_train,x_test,y_test, classes= Clases, percent=True, cmap="Greens")
Matriz.poof()

