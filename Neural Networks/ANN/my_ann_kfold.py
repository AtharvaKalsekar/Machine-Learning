import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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

classifier=KerasClassifier(build_fn=defs.build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier, X=X_train,y=Y_train,cv=10,n_jobs=-1)
mean=accuracies.mean()
vari=accuracies.std()
print(accuracies)
print(mean)
print(vari)

