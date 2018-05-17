import numpy as np
import pandas as pd
import tensorflow as tf
from keras import *
from keras.layers import *
from keras import backend as K
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values


class MultiChannel2DCNN():
    """
    MultiChannelCNN that has different kernel sizes to account for different scales of local dependencies
    """
    saved = 0
    
    def __init__(self, X,y,models=[]):
        #self.X = np.expand_dims(X, axis=2)
        self.X = X
        self.y = y
        self.models = models
        
    #channels is an integer, number of channels
    #kernel size is a list of dimensions for the kernels
    def createmodel(self,channels,kernel_size,num_filters):
        
        K.clear_session()

        inputlayers = {}
        layers = {}
        flats = {}
        height = self.X.shape[1]
        length = self.X.shape[2]
        print(length)
        for i in range(channels):
            inputlayers["input"+ str(i)] = Input(shape=(height,length,1))
            layers["conv" + str(i)] = Conv2D(filters=num_filters,input_shape=(height,length), kernel_size=(kernel_size[i],kernel_size[i]), activation='relu')(inputlayers["input" + str(i)])                        
            layers["dropout" + str(i)] =  Dropout(0.5)(layers["conv" + str(i)])
            layers["pool" + str(i)] = MaxPooling2D(pool_size=(2, 2))(layers["dropout" + str(i)])
            flats["flat" + str(i)] = Flatten()(layers["pool" + str(i)])
        
        merge = concatenate(list(flats.values()))
        dense = Dense(25, activation='relu')(merge)
        outputs = Dense(self.y.shape[1], activation='sigmoid')(dense)
        model = Model(inputs=list(inputlayers.values()), outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.models.append(model)
    
    #train using custom params
    def train(self,model,epochs,channels,batch_size):
        inp = []
        X = self.X.reshape(self.X.shape[0],self.X.shape[1],self.X.shape[2],1)
        for i in range(channels):
            inp.append(X)
        
#         model.fit(inp, self.y,validation_split=0.1, epochs=epochs, batch_size=batch_size,verbose=1)
        model.fit(inp, self.y,validation_split=0.1, epochs=epochs,verbose=1)

        
        if MultiChannelCNN.saved < 1:
            model.save('multichannelcnn.h5')
        else:
            print("Already Saved")
        loss, acc = model.evaluate(inp, self.y, verbose=0)
        print('Train Accuracy: %f' % (acc*100))
        
        return model
    
    #predict
    def predict(self,model,data):
        
        #model = load_model('multichannelcnn.h5')
        predicts = model.predict(data)

        return predicts 