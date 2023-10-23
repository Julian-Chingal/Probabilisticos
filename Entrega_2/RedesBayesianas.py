import os
import pandas as pd
from pgmpy.models import BayesianNetwork as bn
from pgmpy.inference import VariableElimination as ve
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import confusion_matrix as cm

#Variables
label = LabelEncoder()

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

# Training Titanic
train = data.query('train == 1')
test = data.query('train == 0')