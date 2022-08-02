#import useful libraries
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from keras.models import load_model
from functions import *

# set parameter
BATCH_SIZE = 10 # the higher the better
IMG_WIDTH = 128 # for faster computing on kaggle
IMG_HEIGHT = 128 # for faster computing on kaggle
IMG_CHANNELS = 3 
TEST_PATH='./test_set/'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
n_num=num(dirname='tifs_unanalysed/')

#build PLA test set
test_ids = next(os.walk(TEST_PATH))[1]
#just to ensure that len of X_test is equal to len of sizes_test
X_test = np.zeros((n_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
sys.stdout.flush()
n=0
for id_ in tqdm(test_ids, total=len(test_ids)):
    path = TEST_PATH + id_+'/images/'
    dir_3=os.listdir(path)
    k=len(dir_3)
    for i in range(0,k): 
        img = imread(path +dir_3[i])[:,:,:IMG_CHANNELS] # look for all pictures with name Series001
        sizes_test.append([img.shape[0], img.shape[1]])# append sizes of figures i.e. 256*256 pixel: [256,256]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n]=img
        n+=1

#load the model and use it to make predicted results
model = load_model('model.h5', custom_objects={'accuracy': 'accuracy'})
preds_test = model.predict(X_test, verbose=1)

#Threshold predictions
preds_test_t = (preds_test > 0).astype(np.uint8)

# show the result of prediction
ID_dir=count(test_ids,TEST_PATH)  
for i in range(0,len(X_test)):
    plt.figure(figsize=(12,12))
    plt.imshow(preds_test[i])
    plt.axis('off')
    plt.tight_layout()
    figure_save_path ='pred_set/'
    if not os.path.exists(figure_save_path):# if directory not exist, create it
        os.makedirs(figure_save_path)   
    plt.savefig(os.path.join(figure_save_path , '{}'.format(ID_dir[i])),bbox_inches='tight', pad_inches = -0.1)
    plt.close()    

# show the cell segmentation in prediction
path = './pred_set/'
dirs=os.listdir(path)
kk=len(dirs)
for i in range(0,kk):
    name=path+str(dirs[i])
    sep_count_cells(filename=name)
    figure_save_path ='sep_set/'
    if not os.path.exists(figure_save_path):# if directory not exist, create it
        os.makedirs(figure_save_path)   
    plt.savefig(os.path.join(figure_save_path , '{}'.format(dirs[i])),bbox_inches='tight', pad_inches = -0.1)
    plt.close()
