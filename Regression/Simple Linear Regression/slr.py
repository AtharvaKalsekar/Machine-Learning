import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,0].values
X=X.reshape(-1,1)
Y=dataset.iloc[:,1].values
Y=Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
'''
print(X_train)
print(X_test)
print(Y_train)
'''
print(Y_test)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)
print(Y_pred)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Exp (training set)')
plt.xlabel('Exp')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Exp (test set)')
plt.xlabel('Exp')
plt.ylabel('salary')
plt.show()