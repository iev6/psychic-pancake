from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy as np
from keras.models import load_model,clone_model

np.random.seed(1) #setting a seed for  reproducibility


#bitwise masking
#generate mask array first

def maskArray(width=16):
    #width = size of integer datatype, default 16bits HARDCODED
    baseMask0 = np.int16(0)
    baseMask1 = np.int16(1)
    masks = [baseMask0,baseMask1]
    for i in range(1,width): #shift for 15times
        newMask = np.left_shift(baseMask1,i)
        masks.append(newMask)
    #since we need the inverted masks we do
    masks = np.invert(masks,dtype=np.int16)
    return masks
    

#use spew out mask value for given random number

def makeTensorMask(t_size,p_retain,width=16):
    p = [p_retain] + (width)*[(1-p_retain)/width]
    masks = maskArray(width)
    return np.random.choice(a=masks,size=t_size,p=p)


#fill tensor after masking

def makeTrainTest(batch_size=128,epochs = 12):
    batch_size = 128
    num_classes = 10
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return {"train":(x_train,y_train),"test":(x_test,y_test)}
    

def makeModel(model_h5py="/home/giridhur/Expts/MNIST_991.h5"):
    model = load_model(model_h5py)
    return model


def changeModel(model,p_retain):
    weights = model.get_weights()
    new_weights = weights.copy() #deep
    for i in range(len(weights)):
        w_i = weights[i]
        w_i = np.int16(w_i*10000) #convert to 4 digit
        mask_i = makeTensorMask(w_i.shape,p_retain=p_retain)
        new_weights[i] = np.bitwise_and(w_i,mask_i)
        new_weights[i] = np.float32(new_weights[i]/10000)
    model.set_weights(new_weights)
    return model


def maskExpt(model,p_retains,n_expts=10):
    orig_weights = model.get_weights()
    scores = np.zeros((len(p_retains),n_expts))
    for i in range(len(p_retains)):
        for j in range(n_expts):
            model.set_weights(orig_weights)
            model = changeModel(model,p_retains[i])
            loss,score = model.evaluate(x_test,y_test,verbose = 0)
            print(i,j,score,sep=",")
            scores[i,j] = score
    return scores
            


def addNoiseWeights(model,sigma_n):
    weights = model.get_weights()
    new_weights = weights.copy()
#    means_ = []
#    for i in range(len(weights)):
#        means_.append(np.mean(weights[i]))
#    mean_ = np.mean(means_)
    for i in range(len(weights)):
        wi = weights[i]
        new_weights[i] = wi + np.random.normal(scale = sigma_n*np.abs(wi),size=wi.shape)
    model.set_weights(new_weights)
    return model
    #returns model and new weights

        

#for testing 
loss,score = model.evaluate(x_test,y_test,verbose = 0)
dict_dataset = makeTrainTest()
(x_test,y_test) = dict_dataset["test"]


def loopExpt(model,sigmas,n_expts=10):
    orig_weights = model.get_weights()
    scores = np.zeros((len(sigmas),n_expts))
    for sig in range(len(sigmas)):
        for i in range(n_expts):
            model.set_weights(orig_weights)
            model = addNoiseWeights(model,sig)
            loss,score = model.evaluate(x_test,y_test,verbose = 0)
            scores[sig,i] = score
            print(sig,i,score)
    return scores
