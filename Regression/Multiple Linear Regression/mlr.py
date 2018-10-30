import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

def backElem(x,y,sl):
    numVars=len(x[0])
    regressor_OLS=sm.OLS(endog=y,exog=x).fit()
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(endog=y,exog=x).fit()
        maxP_val=max(regressor_OLS.pvalues).astype(float)
        if(maxP_val>sl):
            for j in range(0,numVars-i):
                if(regressor_OLS.pvalues[j].astype(float)==maxP_val):
                    x=np.delete(x,j,axis=1)
    print(regressor_OLS.summary())
    return x

dataset=pd.read_csv("50_Startups.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

label_encoder_X=LabelEncoder()
X[:,3]=label_encoder_X.fit_transform(X[:,3])

ohe=OneHotEncoder(categorical_features=[3])
X=ohe.fit_transform(X).toarray()
#print(X)
#dummy var trap
X=X[:,1:]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)
'''
plt.scatter(X_train[:,2],Y_train,color='red')

plt.scatter(X_train[:,3],Y_train,color='yellow')
plt.scatter(X_train[:,4],Y_train,color='green')

plt.plot(X_train[:,2].reshape(-1,1),regressor.predict(X_train[:,2].reshape(-1,1)))
plt.show()
#print(X_train,Y_train)
'''

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
X_opt=backElem(X_opt,Y,0.05)
'''
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
print(regressor_OLS.summary())
'''