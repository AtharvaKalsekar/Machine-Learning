import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
#poly_reg.fit(X_poly,Y)

l_reg=LinearRegression()
l_reg.fit(X_poly,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,l_reg.predict(poly_reg.fit_transform(X)),color='blue')

plt.show()