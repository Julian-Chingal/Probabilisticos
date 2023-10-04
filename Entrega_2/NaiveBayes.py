# Libaries
import os
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import confusion_matrix as cm
#dataset
from sklearn.datasets import load_wine

#Variables
label = LabelEncoder()
model = GaussianNB()

# Data file
path = os.path.abspath("Entrega_2/Data/water_potability.xlsx")    #https://www.kaggle.com/datasets/girishvutukuri/diabetes-binary-classification/
srcfile = pd.read_excel(path)
srcfile = srcfile.dropna()  # Eliminar instancias imcompletas

#data library
dataRe = load_wine()

#--------------------------------------------------------------------------------------------------------------------------------
# Trainin file
x = srcfile.drop(columns=['Potability']) #eliminar la variable objetivo para entrenar le modelo con los atributos
y = srcfile['Potability'] # Variable objetivo
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=42) 

#Train library
x_train, x_test, y_train, y_test = tts(dataRe['data'], dataRe['target'], test_size=0.3, random_state=0) 
#--------------------------------------------------------------------------------------------------------------------------------

model.fit(x_train,y_train) #entrenar el modelo

#test
prediction = model.predict(x_test) #Predicciones ustilziando el conjunto de prueba
accuracy = accuracy_score(y_test, prediction) #precision del modelo

print(f"----------------------------------\nPrecision de la prediccion:\n----------------------------------\n{accuracy}\n----------------------------------")

#graphic
classes = ["No Potable", "Potable"]   #File
classes = dataRe.target_names         #library

Matriz= cm(model,x_train,y_train,x_test,y_test, classes=classes, cmap="Greens")
Matriz.show()

#porcentaje
Matriz= cm(model,x_train,y_train,x_test,y_test, classes=classes, percent=True, cmap="Greens")
Matriz.show()