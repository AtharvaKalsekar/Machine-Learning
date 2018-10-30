import keras
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(opti):
        classifier=Sequential()
        classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
        classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
        classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
        classifier.compile(optimizer=opti,loss='binary_crossentropy',metrics=['accuracy'])
        return classifier