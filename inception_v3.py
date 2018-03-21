
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.35):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

import keras


from PIL import Image
import numpy as np
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
#import sklearn

#table = ['beans', 'cake', 'candy', 'cereal', 'chips', 'chocolate', 'coffee', 'corn', 'fish',
# 'flour', 'honey', 'jam', 'juice', 'milk', 'nuts', 'oil', 'pasta', 'rice', 'soda',
# 'spices', 'sugar', 'tea', 'tomatosauce', 'vinegar', 'water']
#train= np.load('train_dl2_256.npy')

a = np.load('train_label_dl2.npy')
label=np.zeros((18577,14))
for i in range(18577):
	label[i,a[i]]=1
#train  =train * 255
#train=train.astype('uint8')
#x_val = np.load('x_val.npy').astype(np.float32)
#y_val = np.load('y_val.npy')
#train = np.load('train_and_5223augimg_arr.npy')
#x_val = np.load('val_img_arr.npy')
#label = np.load('trainlabel_and_5223auglabel_arr.npy')

x_train -= 0.5
x_train *= 2.0
trainx=x_train[0:18000]
valx=x_train[18000:18577]
trainy=label[0:18000]
#y_train2=label[3215:8438]
valy=label[18000:18577]
#x_train=train[0:18000]
#x_val=train[18000:18577]
#y_train=label[0:18000]
#y_val=label[18000:18577]
#x_train=np.load('newtrain1_img_arr.npy')
#x_val=np.load('newval1_img_arr.npy')
#label1 = np.load('train_label_arr.npy')
#label2 = np.load('test_label_arr3.npy')
#y_train=np.concatenate((label1[0:3000], label2), axis = 0)
#y_val=label1[3000:3215]
#train_class_wt = sklearn.utils.class_weight.compute_class_weight('balanced', np.arange(25), np.argmax(y_train, axis = 1))
#values = []
#for i in range(25):
#	values.append(train_class_wt[i])
#dict_weights = dict(list(enumerate(values)))
#x_train=train[0:3000]
#x_val=x_val[0:215]
#y_train=label[0:3000]
#y_train2=label[3215:4000]
#y_val=label[3000:3215]
#y_train=np.concatenate((y_train1,y_train2),axis=0)
#preprocess data accordingly

#x_train /= 255.0
#x_train -= 0.5
#x_train *= 2.0

#x_val = x_val/255.0
#x_val -= 0.5
#x_val *= 2.0
print ('data loaded.')



model = load_model('xception_dense_only512_dl2.h5')
#def preprocess(a):


	
#	a=a/255.
#	a -= 0.5
#	a *= 2.0
#	return a

index = 85
for layer in model.layers[index:]:
    layer.trainable = True
'''
for i, layer in enumerate(model.layers):
    print i, layer.trainabam_le, str(layer.name)
'''
#optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0005, nesterov=False)
optimizer = SGD(lr=0.1)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

epochs = 50
batch_size = 16

#datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
decrease_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,vertical_flip=True)
val_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,vertical_flip=True)

#datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,vertical_flip=True)
#val_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,vertical_flip=True)
checkpoint=ModelCheckpoint('xception_deeptraining(with512)_dense_only_dl2.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit_generator(datagen.flow(trainx, trainy, batch_size=batch_size),
                                    steps_per_epoch=(trainx.shape[0]/batch_size),
                                    epochs=epochs, validation_data=val_datagen.flow(valx, valy, batch_size=batch_size),
                                    validation_steps=(valx.shape[0]/batch_size), verbose=1, callbacks=[decrease_learning_rate,checkpoint])



#history=model.fit(x_train,y_train,batch_size=30,epochs=60,validation_data=(x_val,y_val),verbose=1,callbacks=[decrease_learning_rate,checkpoint])


model.save('xception_deeptraining(with512)_dense_only_dl2(last).h5')
#model.summary()
