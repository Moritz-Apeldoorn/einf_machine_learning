import sklearn.svm as svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_set = pd.read_csv("abstrakt.csv")

matrix_x = data_set.iloc[0:12, 0:2].values
matrix_y = data_set.iloc[0:12, 2].values

svclassifier = svm.SVC(kernel='linear')
svclassifier.fit(matrix_x, matrix_y)
#print(f"w={svclassifier.coef_}")
#print(f"b={svclassifier.intercept_}")


plt.scatter(data_set.iloc[0:12, 0].values, data_set.iloc[0:12, 1].values)
x=np.linspace(0, 20, 100)
y = -(svclassifier.intercept_ + (svclassifier.coef_[0][0] * x)) / svclassifier.coef_[0][1]
plt.plot(x, y, 'r')
plt.show()


print(svclassifier.predict([[8, 8]]))
if -(svclassifier.intercept_ + (svclassifier.coef_[0][0] * 8)) / svclassifier.coef_[0][1] < 8:
    print('K2')
else:
    print('K1')
print(svclassifier.predict([[14, 14]]))
if -(svclassifier.intercept_ + (svclassifier.coef_[0][0] * 14)) / svclassifier.coef_[0][1] < 14:
    print('K2')
else:
    print('K1')