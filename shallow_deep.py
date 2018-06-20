
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,GaussianNoise
from keras import regularizers,losses
from keras import backend as K
import os
import numpy as np

#extra import
from keras.layers import Input,Add
from keras.models import Model


'''
First we train a shallow network, with 2 conv layers, followed by an FC and then a softmax
We will calculate entropy of the output by using - sum(p_i log p_i)
We also train the regular model and find the entropy at the final layer, this is an extra output
'''
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

#make the composite model
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

def shallow_model(input_shape,depth=2,num_classes=10):
    """
    returns a shallow model computed by taking softmax after _depth_ conv layers
    HARDCODED for now
    we start with a 32x32 net, and hence 
    """
    conv = lambda x,y=2 : Conv2D(x, (y, y), padding='same',activation="relu")
    max_pool = lambda z=2 : MaxPooling2D(pool_size=(z,z))
    x = Conv2D(filters=32,kernel_size=(3,3),padding="same",input_shape=input_shape)(inputs)
    for i in range(depth-1):
        x = conv(64,2)(x)
    x = max_pool(2)(x)
    x = Flatten()(x)
    #final softmax layer
    output = Dense(num_classes,activation="softmax",name="cifar_shallow_out") (x)
    return output

class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class entpy(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.log(x))
        
        return regularization

    def get_config(self):
        return {'l1': float(self.l1)}



mdl = Model(inputs=input,outputs=shallow_model(input_shape))
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# Let's train the model using RMSprop
mdl.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
predictions = mdl.predict(x_train,batch_size=32)
#estimating entropy
entropies= -1*np.sum(np.multiply(np.log2(predictions),predictions),axis=1)    


inputs = Input(shape=input_shape)
x = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(inputs)
x = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu") (x)
x = MaxPooling2D(pool_size=(2,2))(x)
#dense shallow model
x1 = Flatten()(x)
x1 = Dense(100,activation="relu")(x1)
x1 = Dropout(0.5)(x1)
o1 = Dense(num_classes,activation="softmax")(x1)

# deep model
x = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x)
# adding a few residual layers for the lulz 
x_ = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x)
x2 = Add()([x_,x])
x_ = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x2)
x2 = Add()([x_,x2])
x_ = Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x2)
x2 = Add()([x_,x2])

x2 = MaxPooling2D(pool_size=(2,2))(x2)
x2 = Flatten()(x2)
x2 = Dense(100,activation="relu")(x2)
x2 = Dropout(0.25)(x2)
o2 = Dense(num_classes,activation="softmax")(x2)
composite_model = Model(inputs=inputs,outputs=[o1,o2])
composite_model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.RMSprop(lr=0.0001,decay=1e-6),metrics=["accuracy"])

