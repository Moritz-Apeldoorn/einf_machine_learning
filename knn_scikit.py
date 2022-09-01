import numpy
import pandas as pd
import sklearn.neighbors as neigh
column_names = ["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"]
iris_data_set = pd.read_csv("iris.csv")
# Bestimmten Bereich auswählen von den Daten
# values damit es zu numpy.array wird
matrix_x = iris_data_set.iloc[0:150, 0:4].values # features der Blumen, da sich die Vektoren aus 4 features zusammensetzen
vector_y = iris_data_set.iloc[0:150, 4].values #labels der Blumen, da es 4 features hat und nachher das Label kommt.

#print(iris_data_set.head())
#print(iris_data_set.tail())
#print(iris_data_set.describe())
# Klassifizieren, sodass man sortieren kann nach nächstem Nachbarn
classifier = neigh.KNeighborsClassifier(n_neighbors=3)
# Hier wird das Programm aufgebaut
classifier.fit(matrix_x, vector_y)
#Hier wird das Programm aktiviert
print(classifier.predict(numpy.array([1, 2, 3, 4]).reshape((1, 4)))[0])