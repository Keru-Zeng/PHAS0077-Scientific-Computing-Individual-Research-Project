# model build by PLA data

# import useful libraries
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functions import *

# set useful parameters
BATCH_SIZE = 10 # the higher the better
IMG_WIDTH = 128 # for faster computing on kaggle
IMG_HEIGHT = 128 # for faster computing on kaggle
IMG_CHANNELS = 3 
TRAIN_PATH = './train_set/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

#load all the file name into training and testing directory
train_ids = next(os.walk(TRAIN_PATH))[1]

# build train set
n_num = num(dirname='tifs_unanalysed/')
X_train = np.zeros((n_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((n_num, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
n=0
for id_ in tqdm(train_ids, total=len(train_ids)):
    path = TRAIN_PATH + id_+'/images/'
    path2 = TRAIN_PATH + id_+'/masks/'
    dir_1 = os.listdir(path)
    dir_2 = os.listdir(path2)
    k = len(dir_1)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for i in range(0,k): 
        img = imread(path +dir_1[i])[:,:,:IMG_CHANNELS] # look for all pictures with name Series001
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask_= imread (path2 +dir_2[i]) 
        mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
        Y_train[n] = mask_
        n+=1

#build data augmentation and generate train and validation set
x,y,x_val,y_val=data_aug(X_train,Y_train)
train_generator = zip(x, y)
val_generator = zip(x_val, y_val)

# train the model        
model = trainU_net(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# fit the model
#Stop training when a monitored metric has stopped improving.
earlystopper = EarlyStopping(patience=2, verbose=1)
# save a model or weights (in a checkpoint file) at some interval, 
#so the model or weights can be loaded later to continue the training from the state saved.
checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
#default batch size is 32. We use fit_generator(), since there are data argumentation
results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=100,
                              epochs=20, callbacks=[earlystopper, checkpointer])
results.history

#save loss data
mse = np.array((results.history['loss']))
val_mse=np.array((results.history['val_loss']))
np.save('loss.npy', mse)
np.save('val_loss.npy', val_mse)
acc = np.array((results.history['accuracy']))
val_acc=np.array((results.history['val_accuracy']))
np.save('acc.npy', acc)
np.save('val_acc.npy', val_acc)

# summarize history and plot the relevant data
plt.figure(figsize=(12,12))
plt.subplot(211)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy',fontsize=14)
plt.ylabel('accuracy',fontsize=14)
plt.xlabel('epoch',fontsize=14)
plt.legend(['train_set', 'val_set'], loc='upper right',fontsize=14)
plt.subplot(212)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss',fontsize=14)
plt.ylabel('loss',fontsize=14)
plt.xlabel('epoch',fontsize=14)
plt.legend(['train_set', 'val_set'], loc='upper right',fontsize=14)
plt.show()
plt.savefig('summary.png')
