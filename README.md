
## Popular Deep Learning models implemented from the ground up in Keras and Tensorflow

### Models Implemented:
  - [AlexNet](Models/AlexNet.py)
  - [All-CNN-C](Models/AllCNN.py)
  - [General DeepNeuralNetwork](Models/DNN_tf.py)
  - [General 1D MultiChannel CNN](Models/MultiChannelCNN1d.py)
  - [General 2D MultiChannel CNN](Models/MultiChannelCNN2d.py)
  - [GoogLeNet (Inception v1)](Models/GoogLeNet.py)
  - [LeNet-5](Models/LeNet5.py)
  - [VGG16](Models/VGG16.py)
  - [VGG19](Models/VGG19.py)
  - [ZFNet](Models/ZFNet.py)

## How to use 
  #### Train the AllCNN Model on MNIST dataset for 100 epochs:
      (X_train, y_train), (X_test, y_test) = mnist.load_data()
      model = AllCNN(X_train,y_train)
      model.train(100)
  #### Predict input:
      predictions = model.predict(input)

### [Examples](Applications/)



