import pandas as pd
import numpy as np

dataset = pd.read_csv("Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10)
print(acc.mean())
