import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_classifier(opti):
        classifier=Sequential()
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
        classifier.add(Dropout(rate=0.1))
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
        classifier.add(Dropout(rate=0.1))
        classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
        classifier.compile(optimizer=opti,loss='binary_crossentropy',metrics=['accuracy'])
        return classifier