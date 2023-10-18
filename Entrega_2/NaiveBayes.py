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
model2 = GaussianNB()

# Data file
path = os.path.abspath("Entrega_2/Data/Titanic_survival.xlsx")    #https://www.kaggle.com/competitions/titanic/data?select=train.csv
srcfile = pd.read_excel(path, header= 0)
srcfile = srcfile.dropna()  # Eliminar instancias imcompletas
srcfile['Sex'] = label.fit_transform(srcfile['Sex']) # Transformar a binario las etiquetas del sexo

path2 = os.path.abspath("Entrega_2/Data/water_potability.xlsx")    #https://www.kaggle.com/datasets/girishvutukuri/diabetes-binary-classification/
srcfile2 = pd.read_excel(path2)
srcfile2 = srcfile2.dropna()  # Eliminar instancias imcompletas

#--------------------------------------------------------------------------------------------------------------------------------
# Trainin file water
x = srcfile.drop(columns=['Survived']) #eliminar la variable objetivo para entrenar le modelo con los atributos
y = srcfile['Survived'] # Variable objetivo
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3, random_state=42) 

#train file Titanic Survival
x2 = srcfile2.drop(columns=['Potability']) #eliminar la variable objetivo para entrenar le modelo con los atributos
y2 = srcfile2['Potability'] # Variable objetivo
x_train2, x_test2, y_train2, y_test2 = tts(x2, y2, test_size=0.3, random_state=42) 

#--------------------------------------------------------------------------------------------------------------------------------
# fit
model.fit(x_train,y_train) #entrenar el modelo titanic
model2.fit(x_train2,y_train2) #entrenar el modelo water

#test
prediction = model.predict(x_test) #Predicciones ustilziando el conjunto de prueba
accuracy = accuracy_score(y_test, prediction) #precision del modelo
print(f"Precision del modelo Titanic: {accuracy}\n---------------------------------")

prediction2 = model2.predict(x_test2) #Predicciones ustilziando el conjunto de prueba
accuracy2 = accuracy_score(y_test2, prediction2) #precision del modelo
print(f"Precision del modelo Water: {accuracy2}\n---------------------------------")




# graphic
#Titanic
Clases = ["Die","Survival"]
Matriz= cm(model,x_train,y_train,x_test,y_test, classes= Clases,cmap="Greens")
Matriz.set_title("Matriz de Confusión - Titanic")
Matriz.poof()

Matriz= cm(model,x_train,y_train,x_test,y_test,classes= Clases, percent=True, cmap="Greens")
Matriz.set_title("Matriz de Confusión - Titanic")
Matriz.poof()

# Water
Clases = ["No Potable","Potable"]
Matriz2= cm(model2,x_train2,y_train2,x_test2,y_test2, classes= Clases,cmap="Greens")
Matriz2.poof()

Matriz2= cm(model2,x_train2,y_train2,x_test2,y_test2,classes= Clases, percent=True, cmap="Greens")
Matriz2.show()