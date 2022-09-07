import numpy
import pandas as pd
import sklearn.neighbors as neigh
import sklearn.model_selection as modsel
column_names = ["sepal_lenght", "sepal-width", "petal-length", "petal-width", "class"]
iris_data_set = pd.read_csv("iris.csv")
#print(iris_data_set.head())
#print(iris_data_set.tail())
#print(iris_data_set.describe())
# Bestimmten Bereich auswählen von den Daten
# values damit es zu numpy.array wird
matrix_x = iris_data_set.iloc[0:150, 0:4].values # features der Blumen, da sich die Vektoren aus 4 features zusammensetzen
vector_y = iris_data_set.iloc[0:150, 4].values #labels der Blumen, da es 4 features hat und nachher das Label kommt.


# random_state: Gleich anfangen
# stratify: Von jeder Gruppe ungefähr gleich viele
x_train, x_test, y_train, y_test = modsel.train_test_split(matrix_x, vector_y, test_size=0.2, random_state=0, stratify=vector_y)
#print(iris_data_set.head())
#print(iris_data_set.tail())
#print(iris_data_set.describe())
# Klassifizieren, sodass man sortieren kann nach nächstem Nachbarn
classifier = neigh.KNeighborsClassifier(n_neighbors=3)
# Hier wird das Programm aufgebaut
classifier.fit(x_train, y_train)


accuracy_1 = numpy.array([0, 0])
error = 0
for index in range(len(x_test)):
    if classifier.predict(numpy.array(x_test[index, :]).reshape((1, 4)))[0] == y_test[index]:
        accuracy_1 += 1
    else:
        accuracy_1[1] += 1
        error += 1
print(f"my calculated accuracy: {accuracy_1[0] / accuracy_1[1]}")
print(f"my calculated accuracy: {(len(x_test) - error) / len(x_test)}")

#Teste die samples mit dem Klassifizierer
# Vergleiche Vorhersage mit dem tatsächlichen Resultat
accuracy = classifier.score(x_test, y_test)
print(f"accuracy: {accuracy}")
x=numpy.array([5, 3, 4, 1])
x_row_vector =  x.reshape((1, 4))
#Hier wird das Programm aktiviert
predicted_class = classifier.predict(x_row_vector)
print(f"classification: {predicted_class[0]}")