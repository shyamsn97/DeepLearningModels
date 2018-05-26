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

class GoogLeNet():
    """
    GoogLeNet(also known as Inceptionv1) implemented with Keras
    introduced in the paper "Going deeper with convolutions"
    https://arxiv.org/pdf/1409.4842.pdf
    Parameters:
        X: numpy array data matrix of image pixel intensities
        y: numpy array of labels, to_categorical changes it to a sparse binary matrix
        weights: name of file that denotes weights to load in
    """
    def __init__(self,X,y,weights="None"):
        
        self.X = X
        self.y = to_categorical(y)
        self.n_outputs = self.y.shape[1]
                
        self.weights = weights
        self.model = None
        
        if self.weights == "None":
            self.initialize()
        else:
            self.model = load_model(weights)
    
    def create_Inception(self,params,prev_layer):
        """
            creates a dimensionality reduced inception layer
            params is a list that denotes the number of filters in the corresponding layers:
                leftmost 1x1 conv 
                1x1 conv layer before  3x3
                3x3 conv layer
                1x1 conv layer before  5x5
                5x5 conv layer
                1x1 conv layer after max-pool      
        """
        lay1 = Conv2D(filters=params[0],kernel_size=1,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(prev_layer)
        lay2 = Conv2D(filters=params[1],kernel_size=1,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(prev_layer)
        lay2 = Conv2D(filters=params[2],kernel_size=3,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(lay2)
        lay3 = Conv2D(filters=params[3],kernel_size=1,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(prev_layer)
        lay3 = Conv2D(filters=params[4],kernel_size=5,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(lay3)
        max1 = MaxPool2D(3,strides=1,border_mode='same')(prev_layer)
        lay4 = Conv2D(filters=params[5],kernel_size=1,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(max1)
        layers = [lay1,lay2,lay3,lay4]
        concat = concatenate(layers)
        return concat
    
    def create_auxoutput(self,prev_layer):
        """
            creates auxillary output
        """
        avg = AveragePooling2D(5,strides=3,border_mode='valid')(prev_layer)
        conv1 = Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(avg)
        flat = Flatten()(conv1)
        dense1 = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(flat)
        dropout1 = Dropout(0.7)(dense1)
        dense2 = Dense(self.n_outputs,activation='linear')(dropout1)
        softmax = Softmax(self.n_outputs)(dense2)
        return softmax
        
        
    def initialize(self):
        
        K.clear_session()
        
        n_outputs = self.n_outputs
        input_shape = list(self.X.shape[1:])
        
        inp = Input(shape=[self.X.shape[1],self.X.shape[2],3])
        #block 1
        conv1 = Conv2D(filters=64,kernel_size=7,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(inp)
        max1 = MaxPool2D(3,strides=2,border_mode='same')(conv1)
        norm1 = BatchNormalization()(max1)
        conv2 = Conv2D(filters=64,kernel_size=1,strides=1,border_mode='valid',kernel_regularizer=l2(0.0002),activation='relu')(norm1)
        conv3 = Conv2D(filters=192,kernel_size=3,strides=1,border_mode='same',kernel_regularizer=l2(0.0002),activation='relu')(conv2)
        norm2 = BatchNormalization()(conv3)
        max2 = MaxPool2D(3,strides=2,border_mode='same')(norm2)
        inception_3a = self.create_Inception([64,96,128,16,32,32],max2)
        inception_3b = self.create_Inception([128,128,192,32,96,64],inception_3a)
        max3 = MaxPool2D(3,strides=2,border_mode='same')(inception_3b)
        
        #block 2
        inception_4a = self.create_Inception([192,96,208,16,48,64],max3)
        #aux output 0
        output0 = self.create_auxoutput(inception_4a)
        inception_4b = self.create_Inception([160,112,224,24,64,64],inception_4a)
        inception_4c = self.create_Inception([128,128,256,24,64,64],inception_4b)
        inception_4d = self.create_Inception([112,144,288,32,64,64],inception_4c)
        #aux output 1
        output1 = self.create_auxoutput(inception_4d)
        inception_4e = self.create_Inception([246,160,320,32,128,128],inception_4d)
        max4 = MaxPool2D(3,strides=2,border_mode='same')(inception_4e)
        
        #block 3
        inception_5a = self.create_Inception([256,160,320,32,128,128],max4)
        inception_5b = self.create_Inception([384,192,384,48,192,192],inception_5a)
        
        avgpool = AveragePooling2D(7,strides=1,border_mode='valid')(inception_5b)
        flatten = Flatten()(avgpool)
        dropout = Dropout(0.4)(flatten)
        dense = Dense(n_outputs)(dropout)
        output2 = Softmax(n_outputs)(dense)
        
        model = Model(inputs=inp,outputs=[output0,output1,output2])
        sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        self.model = model
        print(self.model.summary())
        
    def save_picture(self,filename):
        plot_model(self.model, to_file=filename)
        
    def train(self,epochs,save=True):

        self.model.fit(self.X, [self.y,self.y,self.y] ,validation_split=0.1, epochs=epochs,verbose=1)
        if save == True:
            self.model.save('saved_models/GoogLeNet.h5')
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
    def predict(self,X):
        
        if len(X.shape) == 3:
            X = X.reshape(1,X.shape[0],X.shape[1],X.shape[2])
        predictions = self.model.predict(X)[2]
        return np.argmax(predictions)
        