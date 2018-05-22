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
from keras.datasets import mnist
# import pydot
# import graphviz

class LeNet5():
    """
    LeNet5 implemented with Keras
    introduced in the paper "Gradient-Based Learning Applied to Document Recognition"
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf    
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
            
    def reshape_matrix(self,X):
        
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
            return X
        else:
            return X
        
    def initialize(self):
        
        K.clear_session()
        
        n_outputs = self.y.shape[1]
        height = self.X.shape[1]
        width = self.X.shape[2]
        
        if len(self.X.shape) == 3:
            inp = Input(shape=(height,width,1))
            self.X = self.reshape_matrix(self.X)
        else:
            inp = Input(shape=(height,width,3))
            
        conv1 = Conv2D(6,kernel_size=5,border_mode='same',activation='relu')(inp)
        max1 = MaxPool2D(2,border_mode='same')(conv1)
        conv2 = Conv2D(16,kernel_size=5,border_mode='same',activation='relu')(max1)
        max2 = MaxPool2D(2,border_mode='same')(conv2)
        flatten = Flatten()(max2)
        dense1 = Dense(120,activation='tanh')(flatten)
        dense2 = Dense(84,activation='tanh')(dense1)
        dense3 = Dense(n_outputs,activation='linear')(dense2)
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
            self.model.save('saved_models/LeNet5.h5')
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
    def predict(self,X):
        
        if len(X.shape) == 3:
            X = X.reshape(1,X.shape[0],X.shape[1],X.shape[2])
        predictions = self.model.predict(X)
        return np.argmax(predictions)
        
if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    lenet = LeNet5(X_train,y_train)
    lenet.train(10)
    lenet.predict(X_train[0])