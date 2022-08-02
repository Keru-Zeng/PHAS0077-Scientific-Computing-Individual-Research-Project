#import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage import io
import pims
import imutils
import cv2
from scipy import ndimage 
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from keras.preprocessing import image
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.models import Model

# import a tif as stack
def stackloader(filename, dir_in='',plot=True):
    """
    load and seperate 4D tif pictures into 20 2D figures and then plot them if needed
    
    Parameters
    ----------
    filename: string
        The name of the files
    dir_in: string
        Specific directory of the files, defaults to empty
    plot: boolean
        True to plot the figures, False only to load the data rather than plot the figures
    
    Returns
    -------
    There are 4 returns: value of nuclei and nps as well as number of rows and columns: nrows,ncols
    """
    #load data
    if dir_in=='':
        data=pims.TiffStack('./'+filename)# if the tif in the same folder as the code does.
    else:
        data=pims.TiffStack(dir_in+filename)#if the dir is not empty, use the dir to open the tif
    
    nuclei=np.array(data[1::2])#start from 1 with step 2, i.e.1,3,5,7,9...
    nps=np.array(data[::2])#start from 0 with step size 2, i.e. 0,2,4,6,8...
    z_size,y_size,x_size=np.shape(nuclei)# shape on nps, nuclei, data like a square matrix
    #print(np.shape(data)) #(40,512,521),512 entries in x y axes,with 20 in Z and 2 channel,total 40
    nrows=np.int(np.ceil(np.sqrt(z_size)))
    ncols=np.int(z_size//nrows+1)
    if plot==True:
        fig,axes=plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows))
        for n in range(z_size):
            i=n//ncols
            j=n%ncols
            axes[i,j].imshow(nuclei[n],interpolation='nearest',cmap='gray')#total 20 necleis,total 20 pictures
        for ax in axes.ravel():# returns contiguous flattened array(ravel() return an array to 1D) 
            if not(len(ax.images)):
                fig.delaxes(ax)# remove the Axes ax from its figure.
        fig.tight_layout()
        plt.show()
        plt.close()# reduce memory after function.
        
    return nuclei, nps, nrows,ncols


#Change tif into 2d picture and plot it
def plotpic(filename, dir_in='',plot=True):
    """
    transfor 4D tif pictures into 2D figures and then plot the first 2d pictures
    
    Parameters
    ----------
    filename: string
        The name of the files
    dir_in: string
        Specific directory of the files, defaults to empty
    plot: boolean
        True to plot the figures
    """
    
    if dir_in=='':
        data='./'+filename# if the tif in the same folder as the code does.
    else:
        data=dir_in+filename# if dir_in not empty, use dir to open the picture
    IM= io.imread(data)[10]#read 1th 2d-picture
    IM_MAX= np.max(IM, axis=0)
    IM_MAX= resize(IM_MAX, (512,512), mode='constant', preserve_range=True)
    if plot==True:
        plt.imshow(IM_MAX,cmap='gray')
        plt.axis('off')
        
        
#In folders, change all tifs into 2d pictures and save them.
def filefolder(dirname='',plot=True):
    """
    load, plot and save 4D tif pictures into 2D figures in given folders 
    
    Parameters
    ----------
    dirname: string
        General directory of the file, defaults to empty   
    plot: boolean
        True to plot and save the figures in a given folder
    """
    
    if dirname!='':# if dirname is not empty, use given dir
            dirname=dirname
    else:# if dirname is empty
            dirname='./'+dirname
    dirs=os.listdir(dirname)# read all sub-folders' names in dirname folder
    n_files=len(dirs)
    for i in range(0,n_files):
        if dirs[i]!='.DS_Store':
            dataname=dirname+dirs[i]+'/'
            data=dataname+'tifs/'# for tifs subsub-folder in sub-folders,
            dir_1=os.listdir(data)# read all tifs pictures' names
            k=len(dir_1)
            for j in range(0,k):
                if dir_1[j]!='.DS_Store':#check whether it is tifs folders
                    name=data+dir_1[j]
                    IM= io.imread(name)[0]
                    IM_MAX= np.max(IM, axis=0)
                    IM_MAX= resize(IM_MAX, (512,512), mode='constant', preserve_range=True)
                    if plot==True:
                        plt.imshow(IM_MAX,cmap='gray')
                        plt.axis('off')
                        figure_save_path ='test_set/'+dirs[i]+'/images/'# dir to save pictures
                        figure_save_path2 ='train_set/'+dirs[i]+'/images/'# dir to save pictures
                        if not os.path.exists(figure_save_path):# if directory not exist, create it
                            os.makedirs(figure_save_path)   
                        plt.savefig(os.path.join(figure_save_path , '{}'.format(dir_1[j])),bbox_inches='tight', pad_inches = -0.1)
                        if not os.path.exists(figure_save_path2):# if directory not exist, create it
                            os.makedirs(figure_save_path2)   
                        plt.savefig(os.path.join(figure_save_path2 , '{}.png'.format(dir_1[j])),bbox_inches='tight', pad_inches = -0.1)
                        plt.close()#close the figures, so the plot will not display

# binarize the images
def improve_reso(k):
    """
    In order to do a binary classification, convert all values in images above 0 to 1.
    So, picture only have two values {0,1}, which have accurate resolution.

    Parameters
    ----------
    k: number
        input a iterate number   
    """
    path="train_set/"
    dirs=os.listdir(path) #walk through all files in path
    n=len(dirs)
    if k<n:
        N=k
    else:
        N=n
    for i in range(0,N):#walk through specified number of files
        name=path+dirs[i]+"/masks/"
        dirss=os.listdir(name)
        nn=len(dirss)
        for j in range(0,nn):
            names=name+dirss[j]
            mask=io.imread(names) # read images
            # need to binarize the images, convert all values above 0 to 1 to assign a pixel value of 1 for class
            masked=np.where(mask>0,1,mask) 
            plt.imshow(masked,cmap='gray')
            plt.axis('off')
            figure_save_path ='train_set/'+dirs[i]+'/masks/'   
            plt.savefig(os.path.join(figure_save_path , '{}'.format(dirss[j])),bbox_inches='tight', pad_inches = -0.1)
            plt.close() # reduce memory

#count the number of pictures                        
def num(dirname=''):
    """
    count the number of the tifs in the directory
    
    Parameters
    ----------
    dirname: string
        The dir of the files
    
    Returns
    ----------
    The total number of the tifs in the whole dir
    """
    
    if dirname!='':# if dirname is not empty, use given dir
        dirname=dirname
    else:# if dirname is empty
        dirname='./'+dirname
    dirs=os.listdir(dirname)# read all sub-folders' names in dirname folder
    n_files=len(dirs)
    n=0
    for i in range(0,n_files):
        if dirs[i]!='.DS_Store':
            dataname=dirname+dirs[i]+'/'
            data=dataname+'tifs/'# for tifs subsub-folder in sub-folders,
            dir_1=os.listdir(data)# read all tifs pictures' names
            k=len(dir_1)
            for j in range(0,k):
                if dir_1[j]!='.DS_Store':
                    n+=1
    return n

def sep_count_cells(filename=''):
    """
    separate the cells that overlap with each other and count the number of cells 
    
    Parameters
    ----------
    filename: string
        The name of the files
    
    """
    paths=filename
	# construct the argument parse and parse the arguments
	# load the image and perform pyramid mean shift filtering
	# to aid the thresholding step
    image = cv2.imread(paths)
    shifted = cv2.pyrMeanShiftFiltering(image, 20, 55)
	# convert the mean shift image to grayscale, then apply
	# Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
	# compute the exact Euclidean distance from every binary
	# pixel to the nearest zero pixel, then find peaks in this
	# distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=50, labels=thresh)
	# perform a connected component analysis on the local peaks,
	# using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    #print("Note: {} unique segments found".format(len(np.unique(labels)) - 1))
    plt.figure(figsize=(10,10))
    plt.imshow(labels,cmap=plt.cm.nipy_spectral)
    plt.axis('off') 
	
def data_aug(X_train,Y_train):
    """
    build data augmentation to avoid the overfitting of a model 
    and also to increase the ability of model to generalize.
    
    Parameters
    ----------
    X_train: numpy array
        The X train set of data
    Y_train: numpy array
        The Y train set of data
    Returns
    ----------
    The augmented data of x,y,x validation and y validation data sets
    """
    # Creating the training Image and Mask generator (zoom in range[0.8,1.2])
    image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

    # Keep the same seed for image and mask generators so they fit together
    image_datagen.fit(X_train[:int(X_train.shape[0]*0.9)], augment=True, seed=42)#90% train data to fit
    mask_datagen.fit(Y_train[:int(Y_train.shape[0]*0.9)], augment=True, seed=42)

    # flow method generate batch of augmented data
    x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.9)],batch_size=10,shuffle=True, seed=42)
    y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.9)],batch_size=10,shuffle=True, seed=42)

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(X_train[int(X_train.shape[0]*0.9):], augment=True, seed=42)#other 10% apart from 90%
    mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*0.9):], augment=True, seed=42)

    x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.9):],batch_size=10,shuffle=True, seed=42)
    y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.9):],batch_size=10,shuffle=True, seed=42)

    return x,y,x_val,y_val

def trainU_net(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """
    build the U-net model
    
    Parameters
    ----------
    IMG_HEIGHT: number
        The input image height of parameter
    IMG_WIDTH: number
        The input image width of parameter
    IMG_CHANNELS: number
        The input image channels of parameter
    Returns
    ----------
     The build-in U-net model
    """


    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)#axis?
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    return Model(inputs=[inputs], outputs=[outputs])


