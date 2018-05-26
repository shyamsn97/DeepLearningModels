import numpy as np
import tensorflow as tf
from keras import *
from keras.layers import *
from keras import backend as K
from keras.utils import np_utils 
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.models import *
from keras.optimizers import SGD
import pydot
import graphviz

class VGG16():
    """
    VGG16 net implemented with keras
    introduced in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    https://arxiv.org/abs/1409.1556
    Parameters:
        X: numpy array data matrix
        y: numpy array of labels, to_categorical changes it to a sparse binary matrix
        weights: name of file that denotes weights to load in
    """
    def __init__(self,X,y,weights="None"):
        
        self.X = X
        self.y = to_categorical(y)
                
        self.weights = weights
        self.model = None
        
        if self.weights == "None":
            self.initialize()
        else:
            self.model = load_model(weights)
        
    def initialize(self):
        
        K.clear_session()
        
        n_outputs = self.y.shape[1]
        height = self.X.shape[1]
        width = self.X.shape[2]
        
        inp = Input(shape=(height,width,3))
        conv1 = Conv2D(64,kernel_size=3,border_mode='same',activation="relu")(inp)
        conv2 = Conv2D(64,kernel_size=3,border_mode='same',activation='relu')(conv1)
        max1 = MaxPool2D(2,border_mode='same')(conv2)
        dropout1 = Dropout(0.5)(max1)
        conv3 = Conv2D(128,kernel_size=3,border_mode='same',activation='relu')(dropout1)
        conv4 = Conv2D(128,kernel_size=3,border_mode='same',activation='relu')(conv3)
        max2 = MaxPool2D(2,border_mode='same')(conv4)
        dropout2 = Dropout(0.5)(max2)
        conv5 = Conv2D(256,kernel_size=3,border_mode='same',activation='relu')(dropout2)
        conv6 = Conv2D(256,kernel_size=3,border_mode='same',activation='relu')(conv5)
        conv7 = Conv2D(256,kernel_size=3,border_mode='same',activation='relu')(conv6)
        max3 = MaxPool2D(2,border_mode='same')(conv7)
        dropout3 = Dropout(0.5)(max3)
        conv8 = Conv2D(512,kernel_size=3,border_mode='same',activation='relu')(dropout3)
        conv9 = Conv2D(512,kernel_size=3,border_mode='same',activation='relu')(conv8)
        conv10 = Conv2D(512,kernel_size=3,border_mode='same',activation='relu')(conv9)
        max4 = MaxPool2D(2,border_mode='same')(conv10)
        dropout4 = Dropout(0.5)(max4)
        conv11 = Conv2D(512,kernel_size=3,border_mode='same',activation='relu')(dropout4)
        conv12 = Conv2D(512,kernel_size=3,border_mode='same',activation='relu')(conv11)
        conv13 = Conv2D(512,kernel_size=3,border_mode='same',activation='relu')(conv12)
        max5 = MaxPool2D(2,border_mode='same')(conv13)
        dropout5 = Dropout(0.5)(max5)
        flatten = Flatten()(dropout5)
        dense1 = Dense(4096,activation='relu')(flatten)
        dropout6 = Dropout(0.5)(dense1)
        dense2 = Dense(4096,activation='relu')(dropout6)
        dropout7 = Dropout(0.5)(dense2)
        dense3 = Dense(n_outputs,activation='linear')(dropout7)
        softmax = Softmax(n_outputs)(dense3)
        
        model = Model(inputs=inp,outputs=softmax)
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        self.model = model
        print(self.model.summary())
        
    def save_picture(self,filename):
        plot_model(self.model, to_file=filename)
        
    def train(self,epochs,save=True):

        self.model.fit(self.X, self.y ,validation_split=0.1, epochs=epochs,verbose=1)
        if save == True:
            self.model.save('saved_models/VGG16.h5')
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
    def predict(self,X):
        
        if len(X.shape) == 3:
            X = X.reshape(1,X.shape[0],X.shape[1],X.shape[2])
        predictions = self.model.predict(X)
        return np.argmax(predictions)
        