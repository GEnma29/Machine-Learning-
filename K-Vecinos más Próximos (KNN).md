# K-Vecinos más Próximos (KNN)

Este algoritmo se basa en descubrir a que grupo pertenece un elemento al conocer el grupo de los elementos 
cercanos a el. 
se necesitan las siguientes libtrerías 
``` python
   import itertools
   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.ticker import NullFormatter
   import pandas as pd
   import numpy as np
   import matplotlib.ticker as ticker
   from sklearn import preprocessing
   %matplotlib inline```
   
   descargamos los set de datos y los cargamos a nuestro archivo 
   
   
   df = pd.read_csv('setdeDatos.csv')
   df.head()
   
   from sklearn.neighbors import KNeighborsClassifier
   
   k = 4
   #Entrenar el Modelo y Predecir  
   neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
   neigh
   
   yhat = neigh.predict(X_test)
   yhat[0:5]
   
   from sklearn import metrics
   print("Entrenar el set de Certeza: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
   print("Probar el set de Certeza: ", metrics.accuracy_score(y_test, yhat))
