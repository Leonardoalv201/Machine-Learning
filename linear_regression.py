'''
  Implementacion del algoritmo de regresion lineal
  Leonardo Alvarado Menéndez A01705998
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Nombres de las columnas del dataset
columns = ["Class", "Alcohol","Malic acid","Ash","Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

df = pd.read_csv('wine.data',names = columns)

'''
    Definimos nuestras "x" y nuestra "y"
    del dataset "wine" e inicializamos 
    nuestos pesos en ceros
'''
y_ = df['Total phenols'].to_numpy().tolist()
xs_ = df[["Flavanoids", "Nonflavanoid phenols"]].to_numpy().tolist()
t = np.zeros(len(xs_[0])+1).tolist()


#Separamos nuestras x y y en train y test
xs, X_test, y, y_test = train_test_split(xs_, y_, random_state=1)

#Declaramos nuestro learning rate, e inicializamos epocas y el rreglo de errores
__errors__ = []
epoch = 0
lr = 0.01


'''
    Calculamos cada hipotesis con nuestro respectivos pesos y x
'''
def Hyp(t, x):
  n = len(t)
  acum = 0
  for i in range(n):
    acum = acum + t[i] * x[i]
  return acum


'''
    Calculamos el gradiente descendiente para cada peso
    y una vez calculado actualizamos los pesos
'''
def GD(t, xs, y, lr):
  m = len(t)
  n = len(xs)
  temp = []
  for i in range(m):
    acum = 0
    for j in range(n):
      h = Hyp(t, xs[j])
      error = (h - y[j]) * xs[j][i]
      acum = acum + error
    temp.append(t[i] - acum * (lr / n))
  return temp


'''
    Obtenemos el Mean Square Error entre las "y" predichas con el modelo
    y las "y" reales, este resultado se guarda dentro de una lista para 
    despues plotear los errores
'''
def MSE(t, xs, y):
  global __errors__
  n = len(xs)
  acum = 0
  for i in range(n):
    error = (Hyp(t, xs[i]) - y[i])**2
    acum = acum + error
  __errors__.append(acum / n)


'''
    Agregamos un uno al final de cada conjunto de "x" 
    para poder realizar operaciones de manera mas sencilla.
    Tambien se puede ver como agregar una columna de unos a 
    la matriz de x que tenemos para poder realizar la opetacion
    punto entre las "x" y los pesos 
'''
def AddOne(xs):
  for i in range(len(xs)):
    xs[i].append(1)
  return xs


'''
    Calculamos el mean square error entre los valores predichos y
    las y_train
'''
def MSET (y_test, y_pred):
  acum = 0
  for i in range(len(y_test)):
    acum = acum + (y_test[i] - y_pred[i]) ** 2
  return acum


'''
    Corremos las funciones para la regresion lineal
'''
xs = AddOne(xs)
while True:
  oldt = t
  t = GD(t, xs, y, lr)
  MSE(t, xs, y)
  if(oldt == t or __errors__[epoch] <= 0.01 or epoch == 20000):
    print('error')
    print(__errors__[epoch])
    print('xs')
    print(xs)
    print('y')
    print(y)
    print('params')
    print(t)
    break;
  epoch = epoch + 1;



#Plot de la curva de error por cada epoca
plt.plot(__errors__)
plt.show()


'''
    Con los params obtenidos con los datos del train y
    las x_train obtenemos predicciones de y
'''
X_test = AddOne(X_test)
y_pred = [np.dot(x, t)for x in X_test]


# Muestra del error obtenido
print(MSET(y_test, y_pred))


#Plot de las diferencias entre nuestras y_test y y_pred
plt.plot([x-y for x, y in zip(y_test, y_pred)])