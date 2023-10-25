import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork as bn
from pgmpy.inference import VariableElimination as ve
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score 

#Variables
label = LabelEncoder()
prediction = []
real_value = []

# Data train
path = os.path.abspath("Entrega_2/Data/Titanic.csv")    #https://www.kaggle.com/competitions/titanic/data?select=train.csv 
data = pd.read_csv(path) 

# Preprocess ------------------------------------------------------------------------------------------------------------------
features = ['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp']
target = 'Survived'

#imputar valores faltantes
data['Age'].fillna(data['Age'].median(), inplace=True)  
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Transformar los valores categoricos
data['Sex'] = label.fit_transform(data['Sex']) # 1 masculino,  femenino
data['Embarked'] = data['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2])

#Discretizar la informacion, en 10 contenedores o bins para discretizar la variable continua.
data['Age'] = pd.qcut(data['Age'], 10, labels=False, duplicates='drop') 
data['Fare'] = pd.qcut(data['Fare'], 10, labels=False, duplicates='drop')
data['Parch'] = pd.qcut(data['Parch'], 2, labels=False, duplicates='drop')

data['Age'] = data['Age'].astype(str)
data['Embarked'] = data['Embarked'].astype(str)
data['Fare'] = data['Fare'].astype(str)
data['Parch'] = data['Parch'].astype(str)
data['Pclass'] = data['Pclass'].astype(str)
data['Sex'] = data['Sex'].astype(str)
data['SibSp'] = data['SibSp'].astype(str)
data['Survived'] = data['Survived'].astype(str)

# Training Titanic ----------------------------------------------------------------------------------------------------------------

# Definir la estructura del modelo de la red bayesiana
estimator = HillClimbSearch(data[features + [target]])
best_model = estimator.estimate(scoring_method=BicScore(data[features + [target]]))
model = bn(best_model.edges())
model.fit(data[features + [target]], estimator=MaximumLikelihoodEstimator) #Entrenar modelo

# inferencia de la red bayesiana
inference = ve(model)

# Consultas---------------------------------------------------------------------------------------------------------------------------
for index, row in test.iterrows():
    evidence = {
        'Age': row['Age'],
        'Embarked': row['Embarked'],
        'Fare': row['Fare'],
        'Parch': row['Parch'],
        'Pclass': row['Pclass'],
        'Sex': row['Sex'],
        'SibSp': row['SibSp']
    }
    probability = inference.query(variables=['Survived'], evidence=evidence)

    if probability.values[0] > probability.values[1]:
        pre = 0
        print(f"Tiene {probability.values[0]*100:.2f}% probabilidad de morir")
    else:
        pre = 1 
        print(f"Tiene {probability.values[1]*100:.2f}% probabilidad de sobrevivir")

    prediction.append(pre)
    real_value.append(row['Survived'])


# Precision del modelo
precision = accuracy_score(real_value,prediction)
print("------------------------------------",
      f"\nPorcentaje de rendimiento:: {precision* 100:.2f}%",
      "\nCantidad de Instancias : ", len(train),
      "\nCantidad de Prueba: ", len(test), 
      "\n------------------------------------")

# Matriz de confusion ---------------------------------------------------------
Matriz = confusion_matrix(prediction,real_value)
plt.figure(figsize=(8, 6))
sns.heatmap(Matriz, annot=True,cmap="Greens", fmt="d")
plt.xlabel("Predicción")
plt.ylabel("Valor verdadero")
plt.title("Matriz de Confusión")
plt.show()


# al mayor porcentaje agregar una categoria de desicion para saber si se muere o vive, mostrar mensaje si tiene tanto porciento escribir tiene tanto porcentaje de sobrevivir o tanto de no sobrevivir
# probabilidad marginal
# pasar la base de datos a una funcion para hacer las coorelaciones automaticamente,  Definir la estructura del modelo de la red bayesiana