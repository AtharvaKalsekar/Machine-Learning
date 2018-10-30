import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import defs

dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

label_encoder_x=LabelEncoder()
X[:,1]=label_encoder_x.fit_transform(X[:,1])
X[:,2]=label_encoder_x.fit_transform(X[:,2])

ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()

X=X[:,1:]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=KerasClassifier(build_fn=defs.build_classifier)
parameters={'batch_size':[25,30],
'nb_epoch':[200,500],
'optimizer':['adam','rmsprop']
}
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,cv=10,scoring='accuracy')
grid_search=grid_search.fit(X_train,Y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)