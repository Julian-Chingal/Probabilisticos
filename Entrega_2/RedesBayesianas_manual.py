import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pgmpy.models import BayesianNetwork as bn
from pgmpy.inference import VariableElimination as ve
from pgmpy.estimators import MaximumLikelihoodEstimator as mle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix, accuracy_score 

#Variables
label = LabelEncoder()
prediction = [] #Almacenar la prediccion del modelo
real_value = [] # Almacenar la respuseta real del modelo
prob_survived = []  # Lista para almacenar las probabilidades de supervivencia
prob_die = []  # Lista para almacenar las probabilidades de no supervivencia

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

# Training Titanic ----------------------------------------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = tts(data[features], data[target], test_size=0.2, random_state=42) 

train = pd.concat([x_train, y_train], axis=1)
test =  pd.concat([x_test, y_test], axis=1)

# Definir la estructura del modelo de la red bayesiana
model = bn([('Age', 'Survived'), ('Sex', 'Survived'), ('Pclass', 'Survived'), ('Fare', 'Pclass'), ('Embarked', 'Pclass'), ('Parch', 'Survived'), ('SibSp', 'Survived')])
model.fit(train, estimator=mle) #Entrenar modelo

# Graph BN----------------------------------------------------------------------------------------------------------------------------
pos = nx.circular_layout(model)
plt.figure(figsize=(10, 6))
nx.draw(model, pos, with_labels= True)
plt.show()

#calculate --------------------------------------------------------------------------------------------------------------------------
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
    else:
        pre = 1 

    prediction.append(pre)
    real_value.append(row['Survived'])

    #Probabilidad marginal de Sobrevivir
    prob_survived.append(probability.values[1])
    prob_die.append(probability.values[0])
 
# Precision del modelo -----------------------------------------------------------------------------
proMarginal_survived = sum(prob_survived) / len(prob_survived) # Calcula la probabilidad marginal promedio
proMarginal_die = sum(prob_die) / len(prob_die) 
precision = accuracy_score(real_value,prediction) # Presicion del modelo

print("-----------------------------------------------------------",
      f"\nPorcentaje de rendimiento:: {precision*100:.2f}%",
      "\nCantidad de Instancias : ", len(train),
      "\nCantidad de Prueba: ", len(test), 
      f"\nProbabilidad Marginal Promedio de Sobrevivir: {proMarginal_survived*100:.2f}%", 
      f"\nProbabilidad Marginal Promedio de No Sobrevivir: {proMarginal_die*100:.2f}%", 
      "\n------------------------------------------------------")

# Matriz de confusion ---------------------------------------------------------
Matriz = confusion_matrix(prediction,real_value)
plt.figure(figsize=(8, 6))
sns.heatmap(Matriz, annot=True,cmap="Greens", fmt="d")
plt.xlabel("Predicción")
plt.ylabel("Valor verdadero")
plt.title("Matriz de Confusión")
plt.show()