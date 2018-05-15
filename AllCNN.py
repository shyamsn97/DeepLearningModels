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

class AllCNN():
    """
    All-CNN-C model introduced in the paper "Striving for Simplicity: The All Convolutional Net
    https://arxiv.org/abs/1412.6806
    Replaces pooling layers with conv layers with stride 2
    uses Global Average Pooling instead of a dense layer as well
    Specifically used with images as input
    Parameters:
        X: numpy array data matrix of image pixel intensities
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
        conv1 = Conv2D(filters=96,kernel_size=3,border_mode='same',activation="relu")(inp)
        conv2 = Conv2D(filters=96,kernel_size=3,border_mode='same',activation="relu")(conv1)
        conv3 = Conv2D(filters=96,kernel_size=3,border_mode='same',strides=2,activation="relu")(conv2)
        dropout1 = Dropout(0.5)(conv3)
        conv4 = Conv2D(filters=192,kernel_size=3,border_mode='same',activation="relu")(dropout1)
        conv5 = Conv2D(filters=192,kernel_size=3,border_mode='same',activation="relu")(conv4)
        conv6 = Conv2D(filters=192,kernel_size=3,border_mode='same',strides=2,activation="relu")(conv5)
        dropout2 = Dropout(0.5)(conv6)
        conv7 = Conv2D(filters=192,kernel_size=3,border_mode='same',activation="relu")(dropout2)
        conv8 = Conv2D(filters=192,kernel_size=1,border_mode='valid',activation="relu")(conv7)
        conv9 = Conv2D(filters=n_outputs,kernel_size=1,border_mode='valid',activation="relu")(conv8)
        global_averaging = GlobalAveragePooling2D()(conv9)
        softmax = Softmax(n_outputs)(global_averaging)
        model = Model(inputs=inp, outputs=softmax)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        print(model.summary())
        self.model = model
        
    def save_picture(self,filename):
        plot_model(self.model, to_file=filename)
        
    def train(self,epochs):

        self.model.fit(self.X, self.y ,validation_split=0.1, epochs=epochs,verbose=1)
        self.model.save('saved_models/allcnn.h5')
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
    def predict(self,X):
        
        if len(X.shape) == 3:
            X = X.reshape(1,X.shape[0],X.shape[1],X.shape[2])
        predictions = self.model.predict(X)
        return np.argmax(predictions)
    
    
        

