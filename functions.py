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

# import a tif as stack
def stackloader(filename, dir_in='',plot=True):
    """"""
    #load and seperate 4D tif pictures into 20 2D figures and then plot them if needed
    
    #Parameters
    #----------
    #filename: string
    #    The name of the files
    #dir_in: string
    #    Specific directory of the files, defaults to empty
    #plot: boolean
    #    True to plot the figures, False only to load the data rather than plot the figures
    
    #Returns
    #-------
    #There are 4 returns: value of nuclei and nps as well as number of rows and columns: nrows,ncols
    """"""
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
    """"""
    #transfor 4D tif pictures into 2D figures and then plot the first 2d pictures
    
    #Parameters
    #----------
    #filename: string
    #    The name of the files
    #dir_in: string
    #    Specific directory of the files, defaults to empty
    #plot: boolean
    #    True to plot the figures
    """"""
    
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
    """"""
    #load, plot and save 4D tif pictures into 2D figures in given folders 
    
    #Parameters
    #----------
    #dirname: string
    #    General directory of the file, defaults to empty   
    #plot: boolean
    #    True to plot and save the figures in a given folder
    """"""
    
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
                        plt.savefig(os.path.join(figure_save_path2 , '{}'.format(dir_1[j])),bbox_inches='tight', pad_inches = -0.1)
                        plt.close()#close the figures, so the plot will not display

#count the number of pictures                        
def num(dirname=''):
    """"""
    #count the number of the tifs in the directory
    
    #Parameters
    #----------
    #dirname: string
    #    The dir of the files
    
    #Returns
    #----------
    # The total number of the tifs in the whole dir
    """"""
    
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
    """"""
    #separate the cells that overlap with each other and 
    # count the number of cells 
    
    #Parameters
    #----------
    #filename: string
    #    The name of the files
    
    """"""
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
    print("Note: {} unique segments found".format(len(np.unique(labels)) - 1))
    plt.figure(figsize=(10,10))
    plt.imshow(labels,cmap=plt.cm.nipy_spectral)
    plt.axis('off') 
	
