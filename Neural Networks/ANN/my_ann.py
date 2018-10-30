import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

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

#print(X_train[1:10,0:7])

classifier=Sequential()

classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=100)

Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)
new_pred=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>0.5)
print(new_pred)

cm=confusion_matrix(Y_test,Y_pred)
print(cm)
