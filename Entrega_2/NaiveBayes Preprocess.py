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

# Data train
path = os.path.abspath("Entrega_2/Data/Titanic.csv")    #https://www.kaggle.com/competitions/titanic/data?select=train.csv
path2 = os.path.abspath("Entrega_2/Data/TitanicTest.csv")    

train = pd.read_csv(path) 
test = pd.read_csv(path2)

train['train'] = 1
test['train'] = 0

data = pd.concat([train, test], ignore_index=True, sort=False)

# Preprocess ------------------------------------------------------------------------------------------------------------------
features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

#imputar valores faltantes
data['Age'].fillna(data['Age'].median(), inplace=True)  
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Transformar los valores categoricos
data['Sex'] = label.fit_transform(data['Sex']) 
data['Embarked'] = label.fit_transform(data['Embarked'])

data['Age'] = pd.qcut(data['Age'], 10, labels=False, duplicates='drop') #Discretizar la edad en 10 intervalos 

# Trainin train Titanic ---------------------------------------------------------------------------------------------------------
train = data.query('train == 1')
test = data.query('train == 0')

x_train, x_test, y_train, y_test = tts(train[features], train[target], test_size=0.3, random_state=42) 

# fit ---------------------------------------------------------------------------------------------------------------------------
model.fit(x_train,y_train) #entrenar el modelo titanic

#test
prediction = model.predict(x_test) #Predicciones ustilziando el conjunto de prueba
accuracy = accuracy_score(y_test, prediction) #precision del modelo
print("------------------------------------",
      f"\nPrecision del modelo Titanic: {accuracy}",
      "\nCantidad de Instancias : ", len(y_train),
      "\nCantidad de Prueba: ", len(y_test), 
      "\n------------------------------------")

# graphic -----------------------------------------------------------------------------------------------------------------------
Clases = ["Not Survival", "Survival"]

Matriz= cm(model,x_train,y_train,x_test,y_test, classes= Clases,cmap="Greens")
Matriz.poof()

Matriz= cm(model,x_train,y_train,x_test,y_test, classes= Clases,percent=True, cmap="Greens")
Matriz.poof()

