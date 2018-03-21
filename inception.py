import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#KTF.set_session(get_session())


import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model

#from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
import numpy as np
import h5py
#import cv2

#table = ['beans', 'cake', 'candy', 'cereal', 'chips', 'chocolate', 'coffee', 'corn', 'fish',
# 'flour', 'honey', 'jam', 'juice', 'milk', 'nuts', 'oil', 'pasta', 'rice', 'soda',
# 'spices', 'sugar', 'tea', 'tomatosauce', 'vinegar', 'water']

#x_train = np.load('x_train.npy').astype(np.float32)
#y_train = np.load('y_train.npy')
#x_val = np.load('x_val.npy').astype(np.float32)
#y_val = np.load('y_val.npy')

#train= np.load('train_dl2_256.npy')

#label = np.load('trainlabel_and_5223auglabel_arr.npy')
#train2 = np.load('train_299_part2.npy')
#train=np.concatenate((train1,train2),axis=0)
#x_val = np.load('x_val.npy').astype(np.float32)
#y_val = np.load('y_val.npy')
#x_train=np.load('newtrain_img_arr.npy')
#x_val=np.load('newval_img_arr.npy')
a = np.load('train_label_dl2.npy')
label=np.zeros((22777,14))
for i in range(18577):
	label[i,a[i]]=1
label[18577:19077,0]=1
label[19077:19377,6]=1
label[19377:19977,12]=1
label[19977:20977,13]=1
label[20977:21877,1]=1
label[21877:22077,2]=1
label[22077:22377,4]=1
label[22377:22777,5]=1
#label2 = np.load('test_label_arr3.npy')
#y_train=np.concatenate((label1[0:2915], label2), axis = 0)
#y_val=label1[2915:3215]
x_train -= 0.5
x_train *= 2.0
trainx=x_train[577:22777]
valx=x_train[0:577]
trainy=label[577:22777]
#y_train2=label[3215:8438]
valy=label[0:577]
#trainx2=x_train[18577:22777]
#valx=x_train[18000:18577]
#trainy2=label[18577:22777]

#trainx=np.concatenate((trainx1,trainx2),axis=0)
#trainy=np.concatenate((trainy1,trainy2),axis=0)
#preprocess data accordingly
#x_train =x_train.astype(np.float32)/ 255.0
#x_train -= 0.5
#x_train *= 2.0

#x_val =x_val/ 255.0
#x_val -= 0.5
#x_val *= 2.0

#def myGenerator():


#	while True:
#		for i in range(1000):
#			a=x_train[i*18:(i+1)*18]

#			b=y_train[i*18:(i+1)*18]
#			a =a/ 255.0
#			a -= 0.5
#			a *= 2.0
#			yield a,b

base_model = keras.applications.xception.Xception(include_top=False, input_shape=(512, 512, 3))
for layer in base_model.layers:
    layer.trainable = False

last = base_model.output
x = GlobalAveragePooling2D()(last)
pred = Dense(14, activation='softmax')(x)
model=Model(base_model.input, pred)

#optimizer = SGD(lr=0.01,momentum=0.9, decay=0.0005, nesterov=False)
optimizer = SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = 30
batch_size = 16

#datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
decrease_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,vertical_flip=True)
val_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,vertical_flip=True)
#checkpoint=ModelCheckpoint('xcep_dense_only.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#history = model.fit_generator(myGenerator(),
#                                    steps_per_epoch=1000,
#                                   epochs=epochs, validation_data=(x_val, y_val),verbose=1, callbacks=[decrease_learning_rate])


history = model.fit_generator(datagen.flow(trainx, trainy, batch_size=batch_size),
                                    steps_per_epoch=(trainx.shape[0]/batch_size),
                                    epochs=epochs, validation_data=val_datagen.flow(valx, valy, batch_size=batch_size),validation_steps=(valx.shape[0]/batch_size), verbose=1, callbacks=[decrease_learning_rate])

model.save('xception_dense_only512_withaug_dl2.h5')
'''
base_model.summary()

layer_names = [layer.name for layer in model.layers]
for i, layer in enumerate(model.layers):
    print i, layer.trainable, str(layer.name)
'''
