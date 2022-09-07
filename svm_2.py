import sklearn.svm as svm
import pandas as pd
import sklearn.model_selection as modsel
data_set = pd.read_csv("abstrakt.csv")

matrix_x = data_set.iloc[0:12, 0:2].values
matrix_y = data_set.iloc[0:12, 2].values

x_train, x_test, y_train, y_test = modsel.train_test_split(matrix_x, matrix_y, test_size=0.2, random_state=0, stratify=matrix_y)

svclassifier = svm.SVC(kernel='rbf')
svclassifier.fit(x_train, y_train)
print(svclassifier.score(x_test, y_test))
print(svclassifier.get_params())