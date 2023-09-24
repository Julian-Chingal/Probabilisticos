# Libaries
import os
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import confusion_matrix as cm

#Variables
label = LabelEncoder()
model = GaussianNB()

# Data
path = os.path.abspath("Entrega_2/Data/water_potability.xlsx")
srcfile = pd.read_excel(path)

# Preprocess
srcfile = srcfile.dropna()  # Eliminar instancias imcompletas

# Trainin
x = srcfile.drop(columns=['Potability']) #eliminar la variable objetivo para entrenar le modelo con los atributos
y = srcfile['Potability'] # Variable objetivo
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0) # random_state https://grsahagian.medium.com/what-is-random-state-42-d803402ee76b

model.fit(x_train,y_train) #entrenar el modelo

#test
prediction = model.predict(x_test) #Predicciones ustilziando el conjunto de prueba
accuracy = accuracy_score(y_test, prediction) #precision del modelo

print(f"----------------------------------\nPrecision de la prediccion:\n----------------------------------\n{accuracy}\n----------------------------------")

#graphic
classes = ["No Potable", "Potable"]
Matriz= cm(model,x_train,y_train,x_test,y_test, classes= classes, cmap="Greens")
Matriz.show()

#porcentaje
Matriz= cm(model,x_train,y_train,x_test,y_test, classes= classes, percent=True, cmap="Greens")
Matriz.show()