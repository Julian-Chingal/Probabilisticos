# Libaries
import os
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
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


export = pd.DataFrame(pd.concat([x_test,y_test] , axis=1))
pd.set_option('display.max_rows', None)
print(export)