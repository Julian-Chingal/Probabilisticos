import os
import pandas as pd
from pgmpy.models import BayesianNetwork as bn
from pgmpy.inference import VariableElimination as ve
from pgmpy.estimators import MaximumLikelihoodEstimator as mle
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
data['Sex'] = label.fit_transform(data['Sex']) # 1 masculino,  femenino
data['Embarked'] = data['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])

#Discretizar la informacion 
data['Age'] = pd.qcut(data['Age'], 10, labels=False, duplicates='drop') 

# Training Titanic ----------------------------------------------------------------------------------------------------------------
train = data[data['train'] == 1]
test = data[data['train'] == 0]

# Limpieza adicional a datos inconclusos del dataset Test
test = test[features].dropna() # valores incompletos
test.drop(test[(test['Fare'] == 7) | (test['Fare'] == 9.6875)].index, inplace=True)

# Definir la estructura del modelo de la red bayesiana
model = bn([('Age', 'Survived'), ('Sex', 'Survived'), ('Pclass', 'Survived'), ('Fare', 'Pclass'), ('Embarked', 'Pclass'), ('Parch', 'Survived'), ('SibSp', 'Survived')])
model.fit(train[features + [target]], estimator=mle) #Entrenar modelo

# inferencia de la red bayesiana
inference = ve(model)

# Consultas---------------------------------------------------------------------------------------------------------------------------
prueba = []
aux = 0

for index, row in test.iterrows():
    print(aux)
    evidence = {
        'Age': row['Age'],
        'Embarked': row['Embarked'],
        'Fare': row['Fare'],
        'Parch': row['Parch'],
        'Pclass': row['Pclass'],
        'Sex': row['Sex'],
        'SibSp': row['SibSp']
    }
    prueba.append(evidence)

    aux += 1


pru = prueba[8]
print(pru)
result = inference.query(variables=['Survived'], evidence=pru)
print(result)
# al mayor porcentaje agregar una categoria para saber si se muere o vive 
#pasar la base de datos a una funcion para hacer las coorelaciones automaticamente 